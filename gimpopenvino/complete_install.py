#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0


"""
Script will create gimp_openvino_config.txt
"""
import os
import sys
import json
import uuid
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
    ZERO_UUID = uuid.UUID('00000000-0000-0000-0000-000000000000')

    for i in supported_devices:
        if "Intel" not in ie.get_property(i, "FULL_DEVICE_NAME"):
            supported_devices.remove(i)
        elif "DEVICE_UUID" in ie.get_property(i,"SUPPORTED_PROPERTIES") and uuid.UUID(ie.get_property(i,"DEVICE_UUID")) == ZERO_UUID:
            supported_devices.remove(i)
    
    py_dict = {
        "python_path" : python_path,
        "weight_path" : weight_path,
        "supported_devices": supported_devices
        }
    govconfig = os.path.join(plugin_loc, "plugins","openvino_utils", "tools", "gimp_openvino_config.json")    
    with open(govconfig, "w+") as file:
        json.dump(py_dict,file)

    # For Linux, the python plugin scripts need to have executable permissions added.
    if platform.system() == "Linux":
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/superresolution_ov/superresolution_ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/stable_diffusion_ov/stable_diffusion_ov.py'])
        subprocess.call(['chmod', '+x', plugin_loc + '/plugins/semseg_ov/semseg_ov.py'])


if __name__ == "__main__":
    setup_python_weights()
