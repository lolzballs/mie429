import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Atlas:
    def __init__(self, atlas_path: str, font_path: str):
        self._atlas_paths = {
            'M': os.path.join(atlas_path, 'male_atlas'),
            'F': os.path.join(atlas_path, 'female_atlas'),
        }
        self._atlas_nums = {sex: _get_atlas_nums(path)
                            for sex, path in self._atlas_paths.items()}
        self._font = ImageFont.truetype(font_path, 100)

    # def case_am

    def render(self, original_xray: np.ndarray, sex: str,
               age: int) -> np.ndarray:

        even_lower, less_val, more_val = self._get_closest_nums(sex, age)
        # if prediction (age) = atlas age, send one lower and one higher (i.e: [lower, image, match] and [match, image, higher])
        # for the sake of simplicity, we will stack them on top of each other (6 imgs in 1 series)
        # print(f'{even_lower}, {less_val}, {age}, {more_val}')
        # if it matches, by default, the match will be in less_val so we have taken care of the [match, image, higher] case

        image = Image.fromarray(original_xray, 'L')
        draw = ImageDraw.Draw(image)
        age_years, age_months = divmod(age, 12)
        format_pred = f'{age_years}y {age_months}m' if age_months != 0 else f'{age_years}y'
        draw.text((50, 80), format_pred, fill="#fff",
                  font=self._font, stroke_width=5, stroke_fill="#000")

        if even_lower != None: # happens if we have a match
            lowest_img = Image.open(os.path.join(self._atlas_paths[sex], f'{even_lower}.png'))  
            lowest_img = lowest_img.resize((image.size[1], image.size[1]))
        else:
            lowest_img = None
        
        less_img = Image.open(os.path.join(self._atlas_paths[sex], f'{less_val}.png'))
        less_img = less_img.resize((image.size[1], image.size[1]))
        more_img = Image.open(os.path.join(self._atlas_paths[sex], f'{more_val}.png'))
        more_img = more_img.resize((image.size[1], image.size[1]))

        if lowest_img == None:
            return self._make_grid(less_img, image, more_img)
        else:
            lower = Image.fromarray(self._make_grid(lowest_img, image, less_img), 'L')
            higher = Image.fromarray(self._make_grid(less_img, image, more_img), 'L')
            combined_img = Image.new('L', (lower.width, lower.height + higher.height))
            combined_img.paste(lower, (0, 0))
            combined_img.paste(higher, (0, lower.height))
            return np.array(combined_img)
    
    def _make_grid(self, less_img, image, more_img):
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
        even_lower = None
        if (closest_val > age and closest_val != self._atlas_nums[sex][0]) or closest_val == self._atlas_nums[sex][-1]: # if closest value is the largest val in atlas
            more_idx = closest_idx
            less_idx = closest_idx - 1
        else:
            if closest_val == age and closest_val != self._atlas_nums[sex][0]: # we can only go lower if the closest val is not the min value in atlas
                lower_idx = closest_idx - 1
                even_lower = self._atlas_nums[sex][lower_idx]
            more_idx = closest_idx + 1
            less_idx = closest_idx
        less_val = self._atlas_nums[sex][less_idx]
        more_val = self._atlas_nums[sex][more_idx]
        return even_lower, less_val, more_val


def _get_atlas_nums(path: str) -> np.ndarray:
    atlas_imgs = os.listdir(path)
    atlas_nums = np.array([int(x.split('.')[0])
                           for x in atlas_imgs if x.split('.')[1] == 'png'])
    atlas_nums.sort()
    return atlas_nums
