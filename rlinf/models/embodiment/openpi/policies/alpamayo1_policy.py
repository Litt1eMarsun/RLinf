import dataclasses
from typing import ClassVar

import einops
import numpy as np
from openpi import transforms

def make_franka_example() -> dict:
    """Creates a random input example for the Panda policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), ## alpamayo图像数据大小
        "observation/wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image