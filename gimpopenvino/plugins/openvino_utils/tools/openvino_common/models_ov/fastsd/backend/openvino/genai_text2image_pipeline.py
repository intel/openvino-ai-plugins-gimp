
from PIL import Image
from dataclasses import dataclass
from openvino_genai import Text2ImagePipeline
from huggingface_hub import snapshot_download


@dataclass
class PipelineOutput:
    images: list[Image.Image]

class GenaiText2ImagePipeline:
    def __init__(self, model_id: str, device: str = "CPU"):
        """
        Wrapper around openvino_genai.Text2ImagePipeline with a callable API.
        
        Args:
            model_id (str): Path to the OpenVINO-converted Stable Diffusion model.
            device (str): Device to run inference ("CPU", "GPU", etc.).
        """
        model_path = snapshot_download(repo_id=model_id)
        self.pipe = Text2ImagePipeline(model_path, device)

    def __call__(self, prompt: str, num_images_per_prompt: int = 1, **kwargs):
        """
        Generate image(s) from prompt (similar to OVStableDiffusionPipeline).
        
        Args:
            prompt (str): The text prompt for image generation.
            num_images (int): Number of images to generate.
            **kwargs: Extra arguments (e.g., guidance_scale, seed, size).
        
        Returns:
            dict: {"images": [PIL.Image.Image, ...], "tensor": torch.Tensor}
        """
        if "guidance_scale" in kwargs and kwargs["guidance_scale"] <= 1:
            kwargs.pop("negative_prompt")
        print(kwargs)
        print(f"Prompt: {prompt}")
        results = []
        for _ in range(num_images_per_prompt):
            image_tensor = self.pipe.generate(prompt, **kwargs)
            image = Image.fromarray(image_tensor.data[0])
            results.append(image)

        return PipelineOutput(images=results)

