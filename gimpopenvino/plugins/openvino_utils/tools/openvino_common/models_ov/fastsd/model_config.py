from json import load, dump
from os import path 

class ModelConfig:
    def __init__(self, config_path):
        self.config_path = config_path
        self.default_models = [
        "rupeshs/sd-turbo-openvino",
        "rupeshs/sdxs-512-0.9-openvino",
        "rupeshs/hyper-sd-sdxl-1-step-openvino-int8",
        "rupeshs/SDXL-Lightning-2steps-openvino-int8",
        "rupeshs/sdxl-turbo-openvino-int8",
        "rupeshs/LCM-dreamshaper-v7-openvino",
        "Disty0/LCM_SoteMix",
        "rupeshs/sd15-lcm-square-openvino-int8",
        "OpenVINO/FLUX.1-schnell-int4-ov",
        "rupeshs/sana-sprint-0.6b-openvino-int4",
    ]

    def load(self):
        self.config = {"device_name": "CPU", "models": self.default_models.copy()}
        if path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as file:
                    self.config = load(file)
                    return self.config
            except Exception :
                print("Using default model configuration.")
        return self.config


    def save(self,key,value):
        with open(self.config_path, "w") as file:
            self.config[key]=value
            dump(self.config, file)
        
    def get_default_models(self):
        return self.default_models.copy()
