"""This is an experimental pipeline used to test AI PC NPU and GPU"""

from pathlib import Path

from diffusers import EulerDiscreteScheduler, LCMScheduler
from huggingface_hub import snapshot_download
from PIL import Image
from backend.openvino.latent_consistency_engine_advanced import (
    LatentConsistencyEngineAdvanced,
)


class OvHcLatentConsistency:
    """
    OpenVINO Heterogeneous compute Latent consistency models
    For the current Intel Cor Ultra, the Text Encoder and Unet can run on NPU
    Supports following  - Text to image , Image to image and image variations
    """

    def __init__(
        self,
        model_path,
        device: list = ["NPU", "NPU", "GPU"],
    ):
        model_dir = Path(snapshot_download(model_path))

        self.scheduler = LCMScheduler(
            beta_start=0.001,
            beta_end=0.01,
        )
        self.ov_sd_pipleline = LatentConsistencyEngineAdvanced(
            model=model_dir,
            device=device,
        )

    def generate(
        self,
        prompt: str,
        neg_prompt: str,
        init_image: Image = None,
        num_inference_steps=4,
        strength: float = 0.5,
    ):
        image = self.ov_sd_pipleline(
            prompt=prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            scheduler=self.scheduler,
            seed=None,
        )

        return image
