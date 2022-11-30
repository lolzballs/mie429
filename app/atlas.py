import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Atlas:
    def __init__(self, atlas_path: str):
        self._atlas_paths = {
            'M': os.path.join(atlas_path, 'male_atlas'),
            'F': os.path.join(atlas_path, 'female_atlas'),
        }
        self._atlas_nums = {sex: _get_atlas_nums(path)
                            for sex, path in self._atlas_paths.items()}
        self._font = ImageFont.truetype("Arial.ttf", 100)

    def render(self, original_xray: np.ndarray, sex: str,
               age: int) -> np.ndarray:
        less_val, more_val = self._get_closest_nums(sex, age)

        image = Image.fromarray(original_xray, 'L')
        draw = ImageDraw.Draw(image)
        age_years, age_months = divmod(age, 12)
        draw.text((50, 80), f'{age_years}y {age_months}m', fill="#fff",
                  font=self._font, stroke_width=5, stroke_fill="#000")

        less_img = Image.open(os.path.join(self._atlas_paths[sex], f'{less_val}.png'))
        less_img = less_img.resize((image.size[1], image.size[1]))
        more_img = Image.open(os.path.join(self._atlas_paths[sex], f'{more_val}.png'))
        more_img = more_img.resize((image.size[1], image.size[1]))

        # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
        images = [less_img, image, more_img]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        image_w_atlas = Image.new('L', (total_width, max_height))

        x_offset = 0
        for im in images:
            image_w_atlas.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        image_w_atlas = np.array(image_w_atlas)
        return image_w_atlas

    def _get_closest_nums(self, sex: str, age: int) -> Tuple[int, int]:
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        closest_idx = (np.abs(self._atlas_nums[sex] - age)).argmin()
        closest_val = self._atlas_nums[sex][closest_idx]
        if closest_val > age:
            more_idx = closest_idx
            less_idx = closest_idx - 1
        else:
            more_idx = closest_idx + 1
            less_idx = closest_idx
        less_val = self._atlas_nums[sex][less_idx]
        more_val = self._atlas_nums[sex][more_idx]
        return less_val, more_val


def _get_atlas_nums(path: str) -> np.ndarray:
    atlas_imgs = os.listdir(path)
    atlas_nums = np.array([int(x.split('.')[0])
                           for x in atlas_imgs if x.split('.')[1] == 'png'])
    atlas_nums.sort()
    return atlas_nums
