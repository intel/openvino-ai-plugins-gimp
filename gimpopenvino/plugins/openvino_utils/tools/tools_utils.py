#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0
import os
import json

def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    #data={}
    with open(os.path.join(config_path, "gimp_openvino_config.json"), "r") as file:
        data = json.load(file)

    weight_path=data["weight_path"]

    return weight_path



if __name__ == "__main__":
    wgt = get_weight_path()

