import cv2
import numpy as np
from typing import Optional

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """
    它假设 'img_path' (RGB图像路径) 已经存在于 results 字典中。
    """
    def __init__(self,
                 depth_path_suffix: str = '.png',
                 depth_path_prefix: Optional[str] = None) -> None:
        self.depth_path_suffix = depth_path_suffix
        self.depth_path_prefix = depth_path_prefix

    def transform(self, results: dict) -> Optional[dict]:
        """
        根据 RGB 图像路径推断并加载深度图。
        """
        img_path = results['img_path']

        base_filename = img_path.split('/')[-1].split('.')[0]
        depth_filename = f"{base_filename}_stereo_noisy{self.depth_path_suffix}"

        if self.depth_path_prefix:
            depth_path = f"{self.depth_path_prefix}/{depth_filename}"
        else:
            depth_path = img_path.rsplit('/', 1)[0] + "/" + depth_filename

        img_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if img_depth is None:
            raise FileNotFoundError(f"深度图未找到: {depth_path}")

        results['depth_map'] = img_depth
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f"depth_path_suffix='{self.depth_path_suffix}', "
                f"depth_path_prefix='{self.depth_path_prefix}')")


@TRANSFORMS.register_module()
class MergeRGBD(BaseTransform):
    """
    将 results['img'] (RGB) 和 results['depth_map'] (Depth) 合并成一个4通道图像。
    """
    def transform(self, results: dict) -> dict:
        img_rgb = results['img']
        img_depth = results['depth_map']

        if img_depth.ndim == 3 and img_depth.shape[2] == 3:
            img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
        
        img_depth = np.expand_dims(img_depth, axis=-1)

        img_rgbd = np.concatenate([img_rgb, img_depth], axis=-1)

        results['img'] = img_rgbd
        results['img_shape'] = img_rgbd.shape

        return results
