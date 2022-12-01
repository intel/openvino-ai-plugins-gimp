"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2

import numpy as np

from openvino_common.inpainting import ImageInpainting
from openvino.inference_engine import IECore



class InpaintingGUI:
    def __init__(self, srcImg, maskImg, modelPath, device="CPU"):
   

        self.inpainter = ImageInpainting(IECore(), modelPath, device)

        self.img = cv2.resize(srcImg, (self.inpainter.input_width, self.inpainter.input_height))
 
        self.mask = cv2.resize(maskImg, (self.inpainter.input_width, self.inpainter.input_height)) 


    def run(self):
                finalmask = np.expand_dims(self.mask.astype(np.float32)[:, :, 0], axis=-1)
                #finalmask = finalmask / 255. 
                
                finalmask[finalmask > 0] = 1.

                print("final mask", finalmask.shape)
                cv2.imwrite("Finalmask-notwork.png",finalmask.astype(np.uint8)*255)

                self.img[np.squeeze(finalmask, -1) > 0] = 0
                cv2.imwrite("inpaint_test-mask.png", self.img)
                print("img shape before process", self.img.shape)
                print("final mask before process", finalmask.shape)
                self.img = self.inpainter.process(self.img, finalmask)
                cv2.imwrite("FINAL_TEST.png", self.img)
                #self.mask[:, :, :] = 0
                return self.img
    


if __name__ == "__main__":
    mask = cv2.imread(r"D:\git\GIMP-OV\testscases\sampleinput\image_mask_new.png") 
    #print("ORG mask-shape---", mask.astype(np.float32).shape)
    #mask = cv2.imread(r"D:\git\open_model_zoo\demos\image_inpainting_demo\python\inpaint_test_mask.png")
    #print("mask-shape---", mask[:, :, 0].shape)
    srcImg =  cv2.imread(r"D:\git\GIMP-OV\testscases\sampleinput\img.png")
    #print("srcImg-shape", srcImg.shape)
    
    out = InpaintingGUI(srcImg, mask, r"D:\omz-models\v10\public\gmcnn-places2-tf\FP16\gmcnn-places2-tf.xml", "CPU").run()
    
    #print("type = ", type(out))
    #print(out.shape)
    #cv2.imwrite("inpaint_test.png", out)




