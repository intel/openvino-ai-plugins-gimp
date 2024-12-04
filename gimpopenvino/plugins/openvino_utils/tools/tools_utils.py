#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0
import os
import json

base_model_dir = (
    os.path.join(os.environ.get("GIMP_OPENVINO_MODELS_PATH"))
    if os.environ.get("GIMP_OPENVINO_MODELS_PATH") is not None
    else os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp")
)

config_path_dir = (
    os.path.join(os.environ.get("GIMP_OPENVINO_CONFIG_PATH"))
    if os.environ.get("GIMP_OPENVINO_CONFIG_PATH") is not None
    else os.path.join(os.path.dirname(__file__)) 
)

def get_weight_path():
    config_path = config_path_dir
    #data={}
    with open(os.path.join(config_path, "gimp_openvino_config.json"), "r") as file:
        data = json.load(file)

    weight_path=data["weight_path"]

    return weight_path



if __name__ == "__main__":
    wgt = get_weight_path()

