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
import shutil
from pathlib import Path
from openvino.runtime import Core
from gimpopenvino.plugins.openvino_utils.tools.tools_utils import base_model_dir, config_path_dir


def install_base_models(base_model_dir, repo_weights_dir):
    for folder in os.scandir(repo_weights_dir):
        model = os.path.basename(folder)
        model_path = os.path.join(base_model_dir, model)
        if not os.path.isdir(model_path):
            print("Copying {} to {}".format(model, base_model_dir))
            shutil.copytree(Path(folder), model_path)

    print("Setup done for base models")

def setup_python_weights(install_location=None, repo_weights_dir=None):
    if not install_location:
        install_location = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
        
    if not os.path.isdir(install_location):
        os.mkdir(install_location)
    python_path = sys.executable
    weight_path = os.path.join(base_model_dir, "weights")
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)

    if repo_weights_dir:
        install_base_models(weight_path, repo_weights_dir)

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
    govconfig = os.path.join(config_path_dir, "gimp_openvino_config.json")

    with open(govconfig, "w+") as file:
        json.dump(py_dict,file)


if __name__ == "__main__":
    setup_python_weights()
