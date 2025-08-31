"""
Copyright(C) 2022-2025 Intel Corporation
SPDX - License - Identifier: Apache - 2.0

"""

import os
import sys
from typing import List, Optional

cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
fastsd_dir = os.path.join(script_dir, "fastsd")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    sys.path.insert(1, fastsd_dir)

# FastSD imports
from fastsd.backend.device import get_device_name
from fastsd.models.interface_types import InterfaceType
from fastsd.state import get_context, get_settings


class StableDiffusionEngineFastSD:
    def __init__(
        self, model: str, model_name: str, device: List = ["GPU", "GPU", "GPU"]
    ):
        self.device = device[0] if device else "CPU"
        print(get_device_name())
        app_settings = get_settings(skip_file=True)
        self.context = get_context(InterfaceType.CLI)
        app_settings.settings.generated_images.save_image = True
        self.config = app_settings.settings
        self.config.lcm_diffusion_setting.openvino_lcm_model_id = model_name
        self.config.lcm_diffusion_setting.use_openvino = True
        print(f"Setting DEVICE to {self.device}")
        os.environ["DEVICE"] = self.device

    def _is_reshape_required(
        self,
        prev_width: int,
        cur_width: int,
        prev_height: int,
        cur_height: int,
    ) -> bool:
        reshape_required = False
        if prev_width != cur_width or prev_height != cur_height:
            print("Reshape and compile")
            reshape_required = True

        return reshape_required

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed: Optional[str] = None,
    ):
        prev_width = self.config.lcm_diffusion_setting.image_width
        prev_height = self.config.lcm_diffusion_setting.image_height
        print(f"Running Stable Diffusion with prompt: {prompt}")
        self.config.lcm_diffusion_setting.prompt = prompt
        self.config.lcm_diffusion_setting.image_height = height
        self.config.lcm_diffusion_setting.image_width = width
        self.config.lcm_diffusion_setting.inference_steps = num_inference_steps
        self.config.lcm_diffusion_setting.guidance_scale = guidance_scale
        self.config.lcm_diffusion_setting.seed = int(seed)
        self.config.lcm_diffusion_setting.use_seed = True
        images = self.context.generate_text_to_image(
            settings=self.config,
            device=self.device,
            reshape=self._is_reshape_required(
                prev_width,
                width,
                prev_height,
                height,
            ),
        )
        return images[0]
