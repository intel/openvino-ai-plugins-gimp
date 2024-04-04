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
    print("\n##########\n")
    plugin_loc = os.path.dirname(gimpopenvino.__file__)
    ie = Core()
    supported_devices = ie.available_devices

    for i in supported_devices:
        if "Intel" not in ie.get_property(i, "FULL_DEVICE_NAME"):
            supported_devices.remove(i)
    
    
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
        "NOTE ! >> Please add this path to Preferences --> Plug-ins in GIMP : ",
        os.path.join(plugin_loc, "plugins"),
    )
    print("\n##########\n")


if __name__ == "__main__":
    setup_python_weights()
