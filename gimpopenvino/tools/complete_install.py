# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

"""
Script will create gimp_openvino_config.txt, and print path to be added to GIMP Preferences.
"""
import os
import sys
import json

import gimpopenvino
import platform
import subprocess
from openvino.runtime import Core


def setup_python_weights(install_location=None):
    if not install_location:
        install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
        
    if not os.path.isdir(install_location):
        os.mkdir(install_location)
    python_string = "python"
    if os.name == "nt":  # windows
        python_string += ".exe"
    python_path = os.path.join(os.path.dirname(sys.executable), python_string)
    weight_path = os.path.join(install_location, "weights")
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)

    step = 1
    print("\n##########")
    print(
        "{}>> Reference Models present in weights folder. You can also download them from open model zoo.".format(
            step
        )
    )
    step += 1
    print(
        "{}>> Please move the weights folder from the cloned repository: \n"
        "openvino-ai-plugins-gimp".format(
            step
        )
    )
    print("and place in: " + weight_path)
    step += 1

    plugin_loc = os.path.dirname(gimpopenvino.__file__)
    ie = Core()
    supported_devices = ie.available_devices
    py_dict = {
        "python_path" : python_path,
        "weight_path" : weight_path,
        "supported_devices": supported_devices
        }
    with open(os.path.join(plugin_loc, "tools", "gimp_openvino_config.json"), "w") as file:
        json.dump(py_dict,file)

    # For Linux, the python plugin scripts need to have executable permissions added.
    if platform.system() == "Linux":
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/superresolution-ov/superresolution-ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/stable-diffusion-ov/stable-diffusion-ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/semseg-ov/semseg-ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/inpainting-ov/inpainting-ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/fast-style-transfer-ov/fast-style-transfer-ov.py'])


    print(
        "{}>> Please add this path to Preferences --> Plug-ins in GIMP : ".format(step),
        os.path.join(plugin_loc, "plugins"),
    )
    print("##########\n")


if __name__ == "__main__":
    setup_python_weights()
