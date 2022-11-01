import argparse
import os
import cv2 as cv

from image_matcher import ImageMatcher

# Combines template matching + threshold maxing + draws a rectangular bounding box
def image_matching(img):
    img_matcher = ImageMatcher(cv.imread('icon.png'))
    img_matcher.upload_search_image(img)
    return img_matcher.find_object(padding_X=125, padding_Y=175, draw_box=False)

def remove_image_label(input_dir, input_files, output_dir, save_matched_img):
    if len(input_files) == 0:
        files = [f for f in os.listdir(input_dir)]
    else:
        files = input_files

    for file in files:
        img = cv.imread(input_dir + "/" + file)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)	
        kp_img, result_img = image_matching(gray_img)
        cv.imwrite(f"{output_dir}/{file}", result_img)
        if save_matched_img:
            cv.imwrite(f"{output_dir}/matched_{file}", kp_img)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image signature removal for bone xray images",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--input-dir", help="Input directory which stores the input files", required=True)
    parser.add_argument("-f", "--input-files", nargs='+', help="Input files located within the input directory to be processed, \
                                                                use relative file paths with respect to the input directory. \
                                                                do not specify this argument if all files in directory should \
                                                                processed", default=[])
    parser.add_argument("-o", "--output-dir", help="Output directory to store processed images", required=True)
    parser.add_argument("-s", "--save-matched-img", help="Boolean to indicates whether or not to save the bounding box image \
                                                     in addition to the processed image after signature removal", default=False)
    args = parser.parse_args()
    config = vars(args)
    remove_image_label(config["input_dir"], config["input_files"],
                       config["output_dir"], config["save_matched_img"])