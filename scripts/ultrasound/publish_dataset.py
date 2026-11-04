"""Publish ONE frozen dataset version sequentially using direct signed S3.

No bucket listing, multipart upload, general sync, or source deletion. A package
is not published until its entire remote payload has passed SHA256 readback.
"""

import argparse
import configparser
import datetime
import fcntl
import hashlib
import hmac
import http.client
import json
import zipfile
from pathlib import Path
from urllib.parse import quote

BUCKET = "research-storage"
HOST = "55c1cd021bfa87965a7b6ae1d6fc1fb1.r2.cloudflarestorage.com"
MIGRATION = Path("/home/yibo/paper1/migration")
LOCKS = Path("/home/yibo/shared/cache/locks")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def reserve(a, b):
    root = Path("/home/yibo/shared/cache/r2")
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / f"request-budget-{datetime.datetime.now(datetime.timezone.utc):%Y-%m}.json"
    with (root / ".request-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(ledger.read_text()) if ledger.exists() else {"paper1_reserved": {"class_a": 0, "class_b": 0}}
        other = json.loads(Path("/home/yibo/paper2/migration/request-budget.json").read_text())
        state["known_other_reservations"] = {"paper2": other}
        totals = state["paper1_reserved"]
        if (
            totals["class_a"] + a + other["class_a_reserved"] > 100000
            or totals["class_b"] + b + other["class_b_reserved"] > 100000
        ):
            raise RuntimeError("Conservative shared request cap reached; stop cloud I/O")
        totals["class_a"] += a
        totals["class_b"] += b
        state["note"] = (
            "Conservative reservations, NOT Cloudflare account-wide billing telemetry. Shared free allowances are not independently allocated to this task."
        )
        save(ledger, state)
    return str(ledger)


class S3:
    def __init__(self):
        config = configparser.RawConfigParser()
        config.read(Path.home() / ".aws/credentials")
        self.access = config["r2"]["aws_access_key_id"]
        self.secret = config["r2"]["aws_secret_access_key"]
        self.requests = {"class_a": 0, "class_b": 0}

    def request(self, method, key, payload=b"", headers=None):
        if not key.startswith("shared/datasets/"):
            raise ValueError("Dataset keys only")
        path = "/" + BUCKET + "/" + quote(key, safe="/")
        for attempt in range(2):
            now = datetime.datetime.now(datetime.timezone.utc)
            stamp = now.strftime("%Y%m%dT%H%M%SZ")
            day = now.strftime("%Y%m%d")
            digest = hashlib.sha256(payload).hexdigest()
            values = {"host": HOST, "x-amz-date": stamp, "x-amz-content-sha256": digest, **(headers or {})}
            names = ";".join(sorted(values))
            canonical = "".join(name + ":" + values[name].strip() + "\n" for name in sorted(values))
            request = "\n".join([method, path, "", canonical, names, digest])
            scope = day + "/auto/s3/aws4_request"
            string = "AWS4-HMAC-SHA256\n" + stamp + "\n" + scope + "\n" + hashlib.sha256(request.encode()).hexdigest()

            def sign(key, msg):
                return hmac.new(key, msg.encode(), hashlib.sha256).digest()

            signing = sign(sign(sign(sign(("AWS4" + self.secret).encode(), day), "auto"), "s3"), "aws4_request")
            signature = hmac.new(signing, string.encode(), hashlib.sha256).hexdigest()
            values["authorization"] = (
                f"AWS4-HMAC-SHA256 Credential={self.access}/{scope}, SignedHeaders={names}, Signature={signature}"
            )
            self.requests["class_a" if method == "PUT" else "class_b"] += 1
            connection = http.client.HTTPSConnection(HOST, timeout=120)
            try:
                connection.request(method, path, body=payload if method == "PUT" else None, headers=values)
                response = connection.getresponse()
                status = response.status
                body = response.read()
                metadata = {name.lower(): value for name, value in response.getheaders()}
                if status >= 500 and attempt == 0:
                    continue
                if status not in (200, 201, 204, 404, 412):
                    raise RuntimeError(f"S3 {method} failed with HTTP {status}")
                return status, metadata, body
            except (OSError, http.client.HTTPException):
                if attempt:
                    raise
            finally:
                connection.close()
        raise RuntimeError("S3 retries exhausted")

    def immutable(self, key, payload):
        digest = hashlib.sha256(payload).hexdigest()
        status, _, _ = self.request("HEAD", key)
        if status == 404:
            status, _, _ = self.request(
                "PUT",
                key,
                payload,
                {"if-none-match": "*", "x-amz-meta-sha256": digest, "x-amz-storage-class": "STANDARD"},
            )
            if status not in (200, 201, 204, 412):
                raise RuntimeError("Immutable upload failed")
        status, _, body = self.request("GET", key)
        if status != 200 or hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("Remote readback checksum mismatch")
        return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("synthetic_liver", "spine_phantom", "tus_rec_2024_qc", "tdsc_abus_2023"), required=True
    )
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / "paper1-r2-transfer.lock").open("a") as transfer_lock, (
        LOCKS / f"publish-{args.dataset}-{args.version}.lock"
    ).open("a") as lock:
        fcntl.flock(transfer_lock, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path("/home/yibo/shared/datasets") / args.dataset / args.version
        catalog = json.loads((root / "dataset.json").read_text())
        packages = catalog["packages"]
        prefix = f"shared/datasets/{args.dataset}/{args.version}/"
        ledger = reserve(2 * (len(packages) + 4), 4 * (len(packages) + 4) + 8)
        client = S3()
        published = []
        receipt = MIGRATION / "dataset-publications" / f"{args.dataset}-{args.version}.json"
        previous_attempt = json.loads(receipt.read_text()) if receipt.exists() else None
        prior_requests = (
            previous_attempt.get("requests", {"class_a": 0, "class_b": 0})
            if previous_attempt
            else {"class_a": 0, "class_b": 0}
        )
        if previous_attempt:
            audit = receipt.with_name(
                receipt.stem + f".attempt-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}.json"
            )
            save(audit, previous_attempt)
        state = {
            "status": "publishing",
            "dataset": args.dataset,
            "version": args.version,
            "request_reservation_ledger": ledger,
            "packages": published,
            "source_preserved": True,
            "disposable": False,
        }
        try:
            for package in packages:
                path = Path(package["package"])
                if sha(path) != package["sha256"]:
                    raise ValueError("Local ZIP checksum mismatch")
                key = prefix + path.name
                verified = client.immutable(key, path.read_bytes())
                published.append(
                    {
                        "sweep": package["sweep"],
                        "key": key,
                        "sha256": verified,
                        "size_bytes": path.stat().st_size,
                        "files": package["files"],
                    }
                )
                state["requests"] = {name: prior_requests[name] + count for name, count in client.requests.items()}
                save(receipt, state)
                print("VERIFIED", args.dataset, path.name, verified, flush=True)
            auxiliary = catalog["auxiliary_files"]
            aux_package = None
            if auxiliary:
                path = Path(packages[0]["package"]).parent / "dataset-auxiliary.zip"
                if not path.exists():
                    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                        for item in auxiliary:
                            if sha(root / item["path"]) != item["sha256"]:
                                raise ValueError("Auxiliary source checksum differs")
                            archive.write(root / item["path"], item["path"])
                        archive.writestr(
                            "manifest.json",
                            json.dumps(
                                {"dataset": args.dataset, "version": args.version, "files": auxiliary}, sort_keys=True
                            ),
                        )
                with zipfile.ZipFile(path) as archive:
                    for item in auxiliary:
                        if hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]:
                            raise ValueError("Auxiliary ZIP differs")
                key = prefix + path.name
                digest = client.immutable(key, path.read_bytes())
                aux_package = {"key": key, "sha256": digest, "size_bytes": path.stat().st_size}
            manifest = {
                "schema_version": 1,
                "dataset": args.dataset,
                "version": args.version,
                "packages": published,
                "auxiliary_package": aux_package,
                "source_index_sha256": catalog["source_index_sha256"],
                "storage_class": "STANDARD",
            }
            content = (json.dumps(manifest, indent=2) + "\n").encode()
            manifest_sha = client.immutable(prefix + "manifest.json", content)
            # Publish the current pointer last, with compare-and-swap if present.
            pointer = f"shared/datasets/{args.dataset}/current.json"
            status, headers, previous = client.request("GET", pointer)
            if status == 200:
                save(
                    MIGRATION / "dataset-publications" / f"{args.dataset}-previous-current.json",
                    {"payload": json.loads(previous), "etag": headers.get("etag")},
                )
                condition = {"if-match": headers["etag"]}
            else:
                condition = {"if-none-match": "*"}
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "version": args.version,
                    "manifest_key": prefix + "manifest.json",
                    "manifest_sha256": manifest_sha,
                },
                sort_keys=True,
            ).encode()
            status, _, _ = client.request("PUT", pointer, payload, {**condition, "x-amz-storage-class": "STANDARD"})
            if status == 412:
                raise RuntimeError("Another publisher changed the current pointer; leave verified version intact")
            status, _, body = client.request("GET", pointer)
            if status != 200 or body != payload:
                raise ValueError("Pointer readback mismatch")
            state.update(
                status="published_verified",
                manifest_key=prefix + "manifest.json",
                manifest_sha256=manifest_sha,
                auxiliary_package=aux_package,
                current_pointer_verified=True,
                disposable=True,
            )
            for package in packages:
                save(
                    root / package["sweep"] / "publication.json",
                    {
                        "status": "published_verified",
                        "disposable": True,
                        "package_key": prefix + Path(package["package"]).name,
                        "package_sha256": package["sha256"],
                        "receipt": str(receipt),
                    },
                )
        finally:
            state["requests"] = {name: prior_requests[name] + count for name, count in client.requests.items()}
            save(receipt, state)


if __name__ == "__main__":
    main()
