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

class StyleTransfer(Model):
    def __init__(self, ie, model_path, input_image_shape):
        super().__init__(ie, model_path)

        self.input_blob_name = self.prepare_inputs()
        self.output_blob_name = self.prepare_outputs()


    def prepare_inputs(self):
        input_num = len(self.net.input_info)
        if input_num != 1:
            raise RuntimeError("Demo supports topologies only with 1 input")

        input_blob_name = next(iter(self.net.input_info))
        input_blob = self.net.input_info[input_blob_name]
        input_blob.precision = "FP32"

        input_size = input_blob.input_data.shape
        if len(input_size) == 4 and input_size[1] == 3:
            self.n, self.c, self.h, self.w = input_size
        else:
            raise RuntimeError("3-channel 4-dimensional model's input is expected")

        return input_blob_name

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


        resized_image = cv2.resize(image, (self.w, self.h),cv2.INTER_CUBIC)
        print("resized image",resized_image.shape)
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, 0)

        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, image.shape[1::-1]

    def postprocess(self, outputs, dsize):
        prediction = outputs[self.output_blob_name].squeeze()
        print("prediction",prediction.shape)
        prediction = prediction.transpose((1, 2, 0))
        
      
        prediction = cv2.resize(prediction, dsize)
        print("result:", prediction.shape)
   
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        prediction = np.clip(prediction, 0, 255)
        return prediction.astype(np.uint8)
