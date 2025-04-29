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

class SDOptionCache:
    def __init__(self, config_path):
        """
        Initialize the OptionCache with a configuration path and load options.

        Parameters:
            config_path_output (dict): Configuration containing the weight path.
        """
        self.default_options = dict(
            prompt="",
            negative_prompt="",
            num_images=1,
            num_infer_steps=20,
            num_infer_steps_turbo=5,
            guidance_scale=7.5,
            guidance_scale_turbo=0.5,
            model_name="",
            advanced_setting=False,
            power_mode="best power efficiency",
            initial_image=None,
            strength=0.8,
            seed="",
            inference_status="success",
            src_height=512,
            src_width=512,
            show_console=True,
        )
        self.cache_path = config_path
        self.options = self.default_options.copy()
        self.load()

    def load(self):
        """
        Load options from the JSON file if it exists, or use defaults.
        """
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as file:
                    json_data = json.load(file)
                    self.options.update(json_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {self.cache_path}: {e}. Using default options.")

    def get(self, key, default=None):
        """
        Get the value of a specific option.

        Parameters:
            key (str): The key to retrieve.
            default (any): The default value to return if the key is not found.

        Returns:
            The value of the option, or the default value.
        """
        return self.options.get(key, default)
    
    def set(self, key, value):
        """
        Set a specific key to a given value in the options.

        Parameters:
            key (str): The key to set.
            value (any): The value to set for the key.
        """
        if key not in self.default_options:
            raise KeyError(f"'{key}' is not a valid option key.")
        self.options[key] = value
    
    def update(self, updates):
        """
        Update options with a dictionary of key-value pairs.

        Parameters:
            updates (dict): A dictionary of key-value pairs to update.
        """
        if not isinstance(updates, dict):
            raise ValueError("Updates must be a dictionary.")
        self.options.update(updates)

    def save(self):
        """
        Save the current options to the JSON file.
        """
        try:
            with open(self.cache_path, "w") as file:
                json.dump(self.options, file, indent=4)
            #print(f"Options written to {self.cache_path} successfully.")
        except IOError as e:
            print(f"Error writing to {self.cache_path}: {e}")


if __name__ == "__main__":
    wgt = get_weight_path()

