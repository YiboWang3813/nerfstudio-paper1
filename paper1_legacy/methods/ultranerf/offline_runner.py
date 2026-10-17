"""Deprecated UltraNeRF runner shim.

UltraNeRF now uses the repository-wide Engine with
``UltraNeRFOfflineState`` and registered callbacks. Keep this module importable
so old references fail with an actionable message instead of importing the
removed ``common.*`` prototype stack.
"""


class OfflineRunner:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "methods.ultranerf.offline_runner.OfflineRunner is deprecated. "
            "Use `python run.py experiment=preproduction/baseline_reproduction/ultranerf/fullres_30k`."
        )
