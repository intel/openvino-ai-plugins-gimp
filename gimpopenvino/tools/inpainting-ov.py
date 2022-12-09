import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common")
sys.path.extend([plugin_loc])

import torch
import cv2
import numpy as np
from inpainting_gui import InpaintingGUI
import random
import os

from gimpopenvino.tools.tools_utils import get_weight_path
import traceback


def get_inpaint(images, masks, device, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    out = InpaintingGUI(images, masks, os.path.join(weight_path, "inpainting-ov", "gmcnn-places2-tf.xml"), device).run()
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()

    n_drawables = int(sys.argv[2])
    
    device = sys.argv[1]

    image1 = cv2.imread(os.path.join(weight_path, "..", "cache0.png"))
    image2 = None
    if n_drawables == 2:
        image2 = cv2.imread(os.path.join(weight_path, "..", "cache1.png"))
 
    #model_name = data_output["model_name"]
    h, w, c = image1.shape
    #image1 = cv2.resize(image1, (256, 256))
    #image2 = cv2.resize(image2, (256, 256))

    try:
        output = get_inpaint(
            image2[:, :, ::-1],
            image1,
            device,
            weight_path=weight_path
            )

        output = cv2.resize(output, (w, h))
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, "..")
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))
        sys.exit(0)

    except Exception as error:
        with open(os.path.join(weight_path, "..", "gimp_openvino_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
            traceback.print_exception("DEBUG THE ERROR", file=file)
            # Uncoment below lines to debug            
            #e_type, e_val, e_tb = sys.exc_info()
            #traceback.print_exception(e_type, e_val, e_tb, file=file)
        sys.exit(1)
