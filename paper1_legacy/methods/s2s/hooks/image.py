# methods/hooks/image.py

from common.debug.debug_scope import DebugScope
from common.debug.debug_utils import range_stats


class ImageDebugHook:
    @staticmethod
    def on_images(pred_img, gt_img):
        if not DebugScope.enabled():
            return

        stats = {}
        stats.update(range_stats(pred_img, "debug/img/pred"))
        stats.update(range_stats(gt_img, "debug/img/gt"))

        DebugScope.update(stats)
