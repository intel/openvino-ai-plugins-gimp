import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

import cv2
from stable_diffusion_run_ov import run
import torch
from gimpov.tools.tools_utils import get_weight_path
import traceback
import numpy as np

def get_sb(device="CPU", prompt="northern lights", weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    model_path = os.path.join(weight_path, "stable-diffusion-ov")
 
    out = run(device, prompt, model_path)
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ov_run.pkl"), "rb") as file:
        data_output = pickle.load(file)

    device = data_output["device_name"]

    prompt = data_output["prompt"]

    #prompt = data_output["model_name"]
    #image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_sb(device=device, prompt=prompt, weight_path=weight_path)
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
            traceback.print_exception("DEBUG THE ERROR", file=file)
            # Uncoment below lines to debug            
            #e_type, e_val, e_tb = sys.exc_info()
            #traceback.print_exception(e_type, e_val, e_tb, file=file)
