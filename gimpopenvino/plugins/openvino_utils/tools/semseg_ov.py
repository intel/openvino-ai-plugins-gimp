#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

import json
import os
import sys

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")])
sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..","tools")])
from tools_utils import get_weight_path


#from semseg_run import run
from semseg_run_ov import run
#import torch
import cv2
import os
import traceback


def get_seg(input_image, model_name="deeplabv3", device="CPU", weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()

    if model_name == "deeplabv3": 
        out = run(
                input_image, 
                os.path.join(weight_path, "semseg-ov", "deeplabv3.xml"),  
                device,
            )
    else:
        out = run(
                input_image, 
                os.path.join(weight_path, "semseg-ov", "semantic-segmentation-adas-0001.xml"),
                device,
            )

    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "r") as file:
        data_output = json.load(file)
    device = data_output["device_name"] #sys.argv[1]
    model_name = data_output["model_name"] #sys.argv[2]
    
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_seg(image, model_name=model_name, device=device, weight_path=weight_path)
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        data_output["inference_status"] = "success"
        with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "w") as file:
            json.dump(data_output, file)

        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, "..")
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))
        #sys.exit(0)

    except Exception as error:
        with open(os.path.join(weight_path, "..", "gimp_openvino_run.json"), "w") as file:
            json.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
            traceback.print_exception("DEBUG THE ERROR", file=file)
            # Uncoment below lines to debug
            #e_type, e_val, e_tb = sys.exc_info()
            #traceback.print_exception(e_type, e_val, e_tb, file=file)
        #sys.exit(1)