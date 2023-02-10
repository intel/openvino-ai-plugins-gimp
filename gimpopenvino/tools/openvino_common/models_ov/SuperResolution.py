"""
 Copyright (c) 2021 Intel Corporation
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
import math
import numpy as np

from .model import Model

class SuperResolution(Model):
    def __init__(self, ie, model_path, input_image_shape, model_name):
        super().__init__(ie, model_path)
        self.model_name = model_name
        self.reshape(input_image_shape)
        if self.model_name == "esrgan" or self.model_name == "edsr":
            self.input_blob_name = self.prepare_inputs()
        else:
            self.input_blob_name , self.bicinput_blob_name  = self.prepare_inputs() #, self.bicinput_blob_name
        self.output_blob_name = self.prepare_outputs()

    def reshape(self, base_shape):
        print("base shape:", base_shape)
        if self.model_name == "edsr":
            h, w = base_shape
        else:
            h, w, _ = base_shape
   

        input_iter = iter(self.net.input_info)
        input_layer = next(input_iter)
        input_shape = self.net.input_info[input_layer].input_data.shape
        input_num = len(self.net.input_info)
        print("self.net.input_info:",self.net.input_info)

        if input_num == 2:
            output_num = len(self.net.outputs)
            output_blob_name = next(iter(self.net.outputs))
            output_blob = self.net.outputs[output_blob_name]
            print("output_blob :", output_blob.shape[2])
            coeff = output_blob.shape[2] / input_shape[2]

            bicinput_blob_name = next(input_iter)
            bicinput_blob = self.net.input_info[bicinput_blob_name]
            bicinput_size = bicinput_blob.input_data.shape
            bicinput_size[0] = 1
            bicinput_size[2] = coeff * h
            bicinput_size[3] = coeff * w

        input_shape[2]=h
        input_shape[3]=w

        if self.model_name == "esrgan" or self.model_name == "edsr":
           self.net.reshape({input_layer: input_shape})
        else:
            self.net.reshape({input_layer: input_shape, bicinput_blob_name: bicinput_size})

    def prepare_inputs(self):
        input_num = len(self.net.input_info)
        if input_num != 1 and input_num != 2:
            raise RuntimeError("The demo supports topologies with 1 or 2 inputs only")

        iter_blob = iter(self.net.input_info)
        input_blob_name = next(iter_blob)
        input_blob = self.net.input_info[input_blob_name]
        input_blob.precision = "FP32"

        input_size = input_blob.input_data.shape
        if len(input_size) != 4 and input_size[1] != 1 and input_size[1] != 3:
             raise RuntimeError("one or 3-channel 4-dimensional model's input is expected")
        else:
            self.n, self.c, self.h, self.w = input_size
        
        print("iter1",input_blob_name)

        if input_num == 2:
            bicinput_blob_name = next(iter_blob)
            #print("iter2",bicinput_blob_name)
            bicinput_blob = self.net.input_info[bicinput_blob_name]
            bicinput_blob.precision = "FP32"
            temp = 0
            #print("input_blob :", input_blob.input_data.shape)
            bicinput_size = bicinput_blob.input_data.shape
            #print("bic :", bicinput_blob.input_data.shape)
            if len(bicinput_size) != 4:
                raise RuntimeError("Number of dimensions for both inputs must be 4")
            if input_size[2] >= bicinput_size[2] and input_size[3] >= bicinput_size[3]:
             print("add later")
             input_blob_name = temp 
             input_blob_name = bicinput_blob_name
             bicinput_blob_name = temp

        #print("input_blob_name in pre inputs", input_blob_name)
       
        if self.model_name == "esrgan" or self.model_name == "edsr":
            return input_blob_name
        else:
            return input_blob_name,bicinput_blob_name

    def prepare_outputs(self):
        output_num = len(self.net.outputs)
        if output_num != 1:
            raise RuntimeError("Demo supports topologies only with 1 output")

        output_blob_name = next(iter(self.net.outputs))
        output_blob = self.net.outputs[output_blob_name]
        output_blob.precision = "FP32"

        output_size = output_blob.shape
        if len(output_size) != 4:
            raise Exception("Unexpected output blob shape {}. Only 4D output blob is supported".format(output_size))

        return output_blob_name

    def preprocess(self, inputs):
        image = inputs
        input_num = len(self.net.input_info)
    
        if self.model_name == "edsr":
            image = np.expand_dims(image, axis=-1)

        if image.shape[0] != self.h or image.shape[1] != self.w:

            self.logger.warn("Chosen model aspect ratio doesn't match image aspect ratio")
            resized_image = cv2.resize(image, (self.w, self.h))
        else:
            resized_image = image

        if input_num == 2:
            
            bicinput_blob = self.net.input_info[self.bicinput_blob_name]
            bicinput_size = bicinput_blob.input_data.shape
            width = bicinput_size[3]
            ht = bicinput_size[2]
            resized_image_bic = cv2.resize(image, (width, ht))
            resized_image_bic = resized_image_bic.transpose((2, 0, 1))
            resized_image_bic = np.expand_dims(resized_image_bic, 0)
        
 
        resized_image= resized_image.astype(np.float32)
        #print("resssimage------",resized_image[0][0])
        if self.model_name == "esrgan":
            resized_image = resized_image / 255.0 # for esrgan
    
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, 0)
        #print("resized_image",resized_image.shape)

        if self.model_name == "esrgan" or self.model_name == "edsr":
            dict_inputs = {self.input_blob_name: resized_image}
        else:
            dict_inputs = {self.input_blob_name: resized_image , self.bicinput_blob_name: resized_image_bic.astype(np.float32)} 
     
        return dict_inputs, image.shape[1::-1]

    def postprocess(self, outputs, dsize):
        print("outputs", outputs[self.output_blob_name].shape)
        if self.model_name == "edsr" :
            prediction = outputs[self.output_blob_name][0] #.squeeze()
            print("outputs", prediction.shape)
            prediction =  prediction.transpose((1, 2, 0))
        else:
            prediction = outputs[self.output_blob_name].squeeze()
            print("outputs", prediction.shape)
            prediction =  prediction.transpose((1, 2, 0))
            prediction *= 255 
     
        prediction = np.clip(prediction, 0, 255) #if not done then we get artifacts due to pixel overflow

        return prediction.astype(np.uint8)
