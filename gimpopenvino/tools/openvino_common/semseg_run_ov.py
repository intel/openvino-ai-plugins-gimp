#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

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

import logging
import sys

from time import perf_counter

import cv2
import numpy as np
from openvino import AsyncInferQueue, Core, PartialShape, layout_helpers, get_version, Dimension
from adapters import create_core, OpenvinoAdapter


from models_ov.segmentation import  SegmentationModel 

from pipelines import get_user_config, AsyncPipeline

from performance_metrics import PerformanceMetrics
from models import OutputTransform


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


class SegmentationVisualizer:
    pascal_voc_palette = [
        (0,   0,   0),
        (128, 0,   0),
        (0,   128, 0),
        (128, 128, 0),
        (0,   0,   128),
        (128, 0,   128),
        (0,   128, 128),
        (128, 128, 128),
        (64,  0,   0),
        (192, 0,   0),
        (64,  128, 0),
        (192, 128, 0),
        (64,  0,   128),
        (192, 0,   128),
        (64,  128, 128),
        (255, 0, 0),
        (0,   64,  0),
        (128, 64,  0),
        (0,   192, 0),
        (128, 192, 0),
        (0,   64,  128)
    ]

    def __init__(self, colors_path=None):
        if colors_path:
            self.color_palette = self.get_palette_from_file(colors_path)
        else:
            self.color_palette = self.pascal_voc_palette
        self.color_map = self.create_color_map()

    def get_palette_from_file(self, colors_path):
        with open(colors_path, 'r') as file:
            colors = []
            for line in file.readlines():
                values = line[line.index('(')+1:line.index(')')].split(',')
                colors.append([int(v.strip()) for v in values])
            return colors

    def create_color_map(self):
        classes = np.array(self.color_palette, dtype=np.uint8)[:, ::-1] # RGB to BGR
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        classes_num = len(classes)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256-classes_num, 3))
        return color_map

    def apply_color_map(self, input):
        input_3d = cv2.merge([input, input, input])
        return cv2.LUT(input_3d, self.color_map)


class SaliencyMapVisualizer:
    def apply_color_map(self, input):
        saliency_map = (input * 255.0).astype(np.uint8)
        saliency_map = cv2.merge([saliency_map, saliency_map, saliency_map])
        return saliency_map


def render_segmentation(frame, masks, visualiser, only_masks=False):
    output = visualiser.apply_color_map(masks)
   
    return output
 


#def get_model(ie, model):
    
  #      return SegmentationModel(ie, model), SegmentationVisualizer(None)
    


def run(frame, model_path, device):
    
    log.info('Initializing Inference Engine...')
 
    

    plugin_config = get_user_config(device, '', None)
    model_adapter = OpenvinoAdapter(create_core(), model_path, device=device, plugin_config=plugin_config,max_num_requests=1, model_parameters={})
    model = SegmentationModel.create_model('segmentation', model_adapter, None)
    visualizer = SegmentationVisualizer(None)
    model.log_layers_info()

    #model, visualizer = get_model(ie, model_path)
    log.info('Loading network: %s',model_path )
    log.info('Device: %s',device)   
    pipeline = AsyncPipeline(model)
    log.info('Starting inference...')

    if pipeline.is_ready():
        start_time = perf_counter()
        pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    else:
            # Wait for empty request
            pipeline.await_any()

    if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]

    pipeline.await_all()
    # Process all completed requests
    results = pipeline.get_result(0)

    while results is None:
            log.info("WAIT for results")
            results = pipeline.get_result(0)

    if results:
            log.info('We got some results')
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            frame = render_segmentation(frame, objects, visualizer)
   
    
   
    return frame 

#img = cv2.imread(r'D:\sampleinput\img.png')[:, :, ::-1]
#mask = run(img, r'C:\GIMP-ML\weights\semseg\deeplabv3.xml',"NPU")
#print("type = ", type(mask))
#print(mask.shape)
#cv2.imwrite("cache_ov.png", mask)


