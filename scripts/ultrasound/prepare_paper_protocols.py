"""Prepare all retained final-paper protocols using native config dataclasses."""

import json
from pathlib import Path

from nerfstudio.research_artifacts.paths import evidence_root
from nerfstudio.scripts.ultrasound_resources import resources


def main():
    root = Path("/home/yibo/paper1")
    dest = root / "nerfstudio/experiments/paper1/native-protocols"
    dest.mkdir(exist_ok=True)
    rows = []
    with resources(gpu=False):
        source = evidence_root() / "audits/resolved_configs"
        import yaml  # noqa: PLC0415 - import under the shared-runtime lock

        from nerfstudio.configs.paper1_frozen_protocol import native_frozen_config  # noqa: PLC0415

        for path in sorted(source.glob("*.yaml")):
            cfg, receipt = native_frozen_config(path, root / "outputs/paper-protocol-reproduction")
            cfg.timestamp = "prepared-" + path.stem
            target = dest / path.name
            target.write_text(yaml.dump(cfg))
            receipt.update(
                native_config=str(target),
                source_id=path.stem,
                pose_mode=cfg.pipeline.datamanager.dataparser.pose_mode,
                canonical_spacing_mode=cfg.pipeline.datamanager.dataparser.canonical_spacing_mode,
            )
            rows.append(receipt)
    (dest / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("PREPARED_NATIVE_PROTOCOLS", len(rows), "NO_TRAINING")


if __name__ == "__main__":
    main()
