import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

import cv2
from styletransfer_run_ov import run
import torch
from gimpov.tools.tools_utils import get_weight_path
import traceback
import numpy as np

def get_style(img, model_name="mosaic", weight_path=None,device="CPU"):
    if weight_path is None:
        weight_path = get_weight_path()
    
    if model_name == "mosaic":
        model_path = os.path.join(weight_path, "fast-style-transfer-ov", "mosaic-8/fast-neural-style-mosaic-onnx.xml")
    elif model_name == "candy":
        model_path = os.path.join(weight_path, "fast-style-transfer-ov", "candy-8/fast-neural-style-candy8-onnx.xml")
    elif model_name == "pointilism":
        model_path = os.path.join(weight_path, "fast-style-transfer-ov", "pointilism-8/fast-neural-style-pointilism8-onnx.xml")
    elif model_name == "rain-princess":
        model_path = os.path.join(weight_path, "fast-style-transfer-ov", "rain-princess-8/fast-neural-style-rain-princess8-onnx.xml")
    else:
        model_path = os.path.join(weight_path, "fast-style-transfer-ov", "udnie-8/fast-neural-style-rain-udnie8-onnx.xml")

    out = run(img, model_path, device)       
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ov_run.pkl"), "rb") as file:
        data_output = pickle.load(file)

    device = data_output["device_name"]
    model_name = data_output["model_name"]

    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_style(image, model_name=model_name, weight_path=weight_path, device=device)
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        with open(os.path.join(weight_path, "..", "gimp_ov_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "success", "device_name": device }, file)

        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, "..")
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))

    except Exception as error:
        with open(os.path.join(weight_path, "..", "gimp_ov_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=file)
