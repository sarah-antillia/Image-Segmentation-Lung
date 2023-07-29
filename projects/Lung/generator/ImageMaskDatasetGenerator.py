# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2023/07/28
# ImageMaskDatasetGenerator.py

import os
import glob
from re import A
import shutil
from PIL import Image, ImageOps, ImageFilter
import cv2
import traceback
import numpy as np

class ImageMaskDatasetGenerator:
  
  def __init__(self, resize=512):
    self.RESIZE   = resize
    self.blur_mask = False


  def augment(self, image, output_dir, filename):
    # Create mirrored image
    mirrored = ImageOps.mirror(image)
    output_filename = "mirrored_" + filename
    image_filepath = os.path.join(output_dir, output_filename)
    #cropped = self.crop_image(mirrored)
    mirrored.save(image_filepath)
    print("=== Saved {}".format(image_filepath))


  def resize_to_square(self, image):
     w, h  = image.size

     bigger = w
     if h > bigger:
       bigger = h

     background = Image.new("RGB", (bigger, bigger), (0, 0, 0))
    
     x = (bigger - w) // 2
     y = (bigger - h) // 2
     background.paste(image, (x, y))
     background = background.resize((self.RESIZE, self.RESIZE))

     return background
  

  def create(self, input_images_dir, input_masks_dir,  output_dir,
                            debug=False):
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    image_files = glob.glob(input_images_dir + "/*.png")
    
    if image_files == None or len(image_files) == 0:
      print("FATAL ERROR: Not found mask files")
      return

    for image_file in image_files:
      basename = os.path.basename(image_file)

      maskfile = basename.split(".")[0] + "_mask.png"
      mask_filepath = os.path.join(input_masks_dir, maskfile)
      if not os.path.exists(mask_filepath):
        print("Not found mask_file {} corresponding to {}".format(mask_filepath, image_file))
        continue

      image = Image.open(image_file).convert("RGB")
      mask  = Image.open(mask_filepath).convert("RGB")
 
      basename = basename.replace(".png", ".jpg")
      image_output_filepath = os.path.join(output_images_dir, basename)
      
      squared_image = self.resize_to_square(image)
      squared_image.save(image_output_filepath)
      print("--- Saved cropped_square_image {}".format(image_output_filepath))

      self.augment(squared_image, output_images_dir, basename)
   
      # Blur mask 
      if self.blur_mask:
        print("---blurred ")
        mask = mask.filter(ImageFilter.BLUR)
      
      if debug:
        mask.show()
        input("XX")   
  
      mask_output_filepath = os.path.join(output_masks_dir, basename)

      squared_mask = self.resize_to_square(mask)
      squared_mask.save(mask_output_filepath)

      self.augment(squared_mask, output_masks_dir, basename)


if __name__ == "__main__":
  try:
   
    input_images_dir = "./Lung Segmentation/CXR_png/"
    input_masks_dir  = "./Lung Segmentation/masks"
    
    generator = ImageMaskDatasetGenerator()
    
    output_dir = "./Lung-master"
    generator.create(input_images_dir, input_masks_dir, output_dir, debug=False)

  except:
    traceback.print_exc()
    pass
