#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, LCMScheduler, EulerDiscreteScheduler
from openvino.runtime import Core

# Local imports
sys.path.extend(
    [
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "openvino_common"),
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tools"),
    ]
)

from gimpopenvino.plugins.openvino_utils.tools.tools_utils import (  # type: ignore
    SDOptionCache,  # noqa: F401 (import kept for compatibility)
    config_path_dir,
    get_weight_path,
)
from gimpopenvino.plugins.openvino_utils.tools.openvino_common.models_ov import (  # type: ignore
    controlnet_canny_edge,
    controlnet_cannyedge_advanced,
    controlnet_openpose,
    controlnet_openpose_advanced,
    controlnet_scribble,
    stable_diffusion_3,
    stable_diffusion_engine,
    stable_diffusion_engine_fastsd,
    stable_diffusion_engine_genai,
    stable_diffusion_engine_inpainting,
    stable_diffusion_engine_inpainting_advanced,
    stable_diffusion_engine_inpainting_genai,
)

from gimpopenvino.plugins.openvino_utils.tools.openvino_common.models_ov.fastsd.model_config import (  # type: ignore
    ModelConfig,
)

# ---- FastSD model catalog
fast_sd_models_config = ModelConfig(os.path.join(config_path_dir, "fastsd_models.json")).load()
fast_sd_models: List[str] = fast_sd_models_config.get("models", []) or []
fast_sd_models_up = [m.lower() for m in fast_sd_models]
fast_sd_models_map = {m.lower(): m for m in fast_sd_models}

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)


# ----------------------- System info helpers -----------------------

def get_bios_version() -> str:
    """Avoid sudo; best-effort cross-platform."""
    try:
        os_name = platform.system()
        if os_name == "Windows":
            try:
                import wmi  # type: ignore
                c = wmi.WMI()
                bios = c.Win32_BIOS()[0]
                return str(bios.SMBIOSBIOSVersion)
            except Exception:
                return "<unknown>"
        if os_name == "Linux":
            # Prefer sysfs
            for p in ("/sys/class/dmi/id/bios_version", "/sys/devices/virtual/dmi/id/bios_version"):
                if os.path.exists(p):
                    return Path(p).read_text(encoding="utf-8").strip()
            # Fallback to dmidecode if available (no sudo)
            try:
                result = subprocess.run(
                    ["dmidecode", "-t", "bios"], capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "Version:" in line:
                            return line.split("Version:")[1].strip()
            except Exception:
                pass
            return "<unknown>"
        return "<unsupported>"
    except Exception:
        return "<unknown>"


def get_windows_pcie_device_driver_versions():
    """Best-effort WMI; absent libs return []."""
    try:
        import win32com.client  # type: ignore
    except Exception:
        return []
    try:
        wmi = win32com.client.Dispatch("WbemScripting.SWbemLocator")
        service = wmi.ConnectServer(".", "root\\cimv2")
        drivers = service.ExecQuery("SELECT DeviceID, DriverVersion, Description FROM Win32_PnPSignedDriver")
        out = []
        for d in drivers:
            out.append(
                {"DeviceID": getattr(d, "DeviceID", None), "DriverVersion": getattr(d, "DriverVersion", None), "Description": getattr(d, "Description", None)}
            )
        return out
    except Exception:
        return []


def check_windows_device_driver_version(device_name: str, driver_info) -> Optional[str]:
    for info in driver_info or []:
        desc = info.get("Description") or ""
        if device_name and desc and device_name.lower() in desc.lower():
            return info.get("DriverVersion")
    return None


def print_system_info() -> None:
    log.info("System Information")
    log.info("==================")
    log.info(f"System: {platform.system()}")
    log.info(f"Node Name: {platform.node()}")
    log.info(f"Python Version: {platform.python_version()}")
    log.info(f"Platform: {platform.platform()}")
    if platform.system().lower().startswith("win"):
        di = get_windows_pcie_device_driver_versions()
        log.info(f"BIOS: {get_bios_version()}")
        log.info(f"NPU Driver: {check_windows_device_driver_version('AI Boost', di)}")
        log.info(f"GPU Driver: {check_windows_device_driver_version('Arc', di)}")
    elif platform.system().lower() == "linux":
        log.info(f"BIOS: {get_bios_version()}")
        try:
            with open("/sys/module/intel_vpu/version", "r", encoding="utf-8") as f:
                log.info(f"NPU Driver: {f.readline().strip()}")
        except Exception:
            log.info("NPU Driver: <unknown>")
        log.info("GPU Driver: <unsupported>")


# ----------------------- Engine selection -----------------------

def initialize_engine(model_name: str, model_path: str, device_list: List[str]):
    """Route to correct engine implementation."""
    if model_name == "sd_1.5_square_int8":
        return stable_diffusion_engine.StableDiffusionEngineAdvanced(model=model_path, device=device_list)
    if model_name == "sd_3.0_square":
        return stable_diffusion_3.StableDiffusionThreeEngine(model=model_path, device=["GPU"])
    if model_name == "sd_1.5_inpainting":
        return stable_diffusion_engine_inpainting_genai.StableDiffusionEngineInpaintingGenai(model=model_path, device=device_list[0])
    if model_name in ("sd_1.5_square_lcm", "sdxl_base_1.0_square", "sdxl_turbo_square", "sd_3.0_med_diffuser_square", "sd_3.5_med_turbo_square"):
        return stable_diffusion_engine_genai.StableDiffusionEngineGenai(model=model_path, model_name=model_name, device=device_list)
    if model_name == "sd_1.5_inpainting_int8":
        return stable_diffusion_engine_inpainting_advanced.StableDiffusionEngineInpaintingAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_openpose_int8":
        return controlnet_openpose_advanced.ControlNetOpenPoseAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny_int8":
        return controlnet_cannyedge_advanced.ControlNetCannyEdgeAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_scribble_int8":
        return controlnet_scribble.ControlNetScribbleAdvanced(model=model_path, device=device_list)
    if model_name == "controlnet_canny":
        return controlnet_canny_edge.ControlNetCannyEdge(model=model_path, device=device_list)
    if model_name == "controlnet_scribble":
        return controlnet_scribble.ControlNetScribble(model=model_path, device=device_list)
    if model_name == "controlnet_openpose":
        return controlnet_openpose.ControlNetOpenPose(model=model_path, device=device_list)
    if model_name == "controlnet_referenceonly":
        return stable_diffusion_engine.StableDiffusionEngineReferenceOnly(model=model_path, device=device_list)
    if model_name in fast_sd_models:
        return stable_diffusion_engine_fastsd.StableDiffusionEngineFastSD(model=model_path, device=device_list, model_name=model_name)
    return stable_diffusion_engine.StableDiffusionEngine(model=model_path, device=device_list, model_name=model_name)


# ----------------------- CLI & validation -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    g = p.add_argument_group("Options")
    g.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    g.add_argument("-l", "--list", action="store_true", help="Show list of models currently installed.")
    g.add_argument("-bp", "--model_base_path", type=str, default=None, help="Absolute base path to model weights.")
    g.add_argument("-m", "--model_name", type=str, default="sd_1.5_square_int8", help="Model name key.")
    g.add_argument("-d", "--device", type=str, default=None, help="Global device for all submodels (CPU/GPU/NPU).")
    g.add_argument("-td", "--text_device", type=str, default=None, help="Device for Text encoder.")
    g.add_argument("-ud", "--unet_device", type=str, default=None, help="Device for UNet.")
    g.add_argument("-und", "--unet_neg_device", type=str, default=None, help="Device for UNet Negative.")
    g.add_argument("-vd", "--vae_device", type=str, default=None, help="Device for VAE.")
    g.add_argument("-seed", "--seed", type=int, default=None, help="Seed for first image; others random if not set.")
    g.add_argument("-niter", "--iterations", type=int, default=20, help="Num inference steps.")
    g.add_argument("-si", "--save_image", action="store_true", help="Save output image(s).")
    g.add_argument("-n", "--num_images", type=int, default=1, help="Number of images to generate.")
    g.add_argument("-g", "--guidance_scale", type=float, default=7.5, help="Guidance scale.")
    g.add_argument("-pm", "--power_mode", type=str, default="best performance", help="Power mode key in config.json.")
    g.add_argument("-pp", "--prompt", type=str, default="a portrait of an old coal miner in 19th century, beautiful painting with highly detailed face by greg rutkowski and magali villanueve", help="Positive prompt.")
    g.add_argument("-np", "--neg_prompt", type=str, default=None, help="Negative prompt.")
    g.add_argument("--init-image", type=str, default=None, help="Path to init image (controlnet/reference/inpainting).")
    g.add_argument("--mask-image", type=str, default=None, help="Path to mask image (inpainting).")
    return p.parse_args()


def validate_model_paths(base_path: str, model_paths: Dict[str, List[str]]) -> Dict[str, str]:
    """Return {model_name: full_path} for available models."""
    results: Dict[str, str] = {}
    for model_name, relative_parts in model_paths.items():
        full_path = os.path.join(base_path, *relative_parts)
        if not os.path.isdir(full_path):
            continue
        if "int8a16" in model_name:
            if os.path.isfile(os.path.join(full_path, "unet_int8a16.xml")):
                results[model_name] = full_path
        elif "fp8" in model_name:
            if os.path.isfile(os.path.join(full_path, "unet_fp8.xml")):
                results[model_name] = full_path
        else:
            results[model_name] = full_path
    return results


# ----------------------- Main -----------------------

def main() -> int:
    args = parse_args()

    weight_path = args.model_base_path or get_weight_path()
    if not os.path.isdir(weight_path):
        log.error(f"Model base path not found: {weight_path}")
        return 2

    model_name_key = (args.model_name or "").lower()

    if args.model_base_path:
        weight_path = args.model_base_path
    else:
        weight_path = get_weight_path()
    
    # Check if the directory path exists
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"The directory path {weight_path} does not exist.")
    
    available_devices = Core().get_available_devices()
    execution_devices = ["GPU"]*5 if "GPU" in available_devices else ["CPU"]*5
    
    model_paths = {
        "sd_1.4": ["stable-diffusion-ov", "stable-diffusion-1.4"],
        "sd_1.5_square_lcm": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_lcm"],
        "sdxl_base_1.0_square": ["stable-diffusion-ov", "stable-diffusion-xl", "square_base"],
        "sdxl_turbo_square": ["stable-diffusion-ov", "stable-diffusion-xl", "square_turbo"],
        "sd_1.5_portrait": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait"],
        "sd_1.5_square": ["stable-diffusion-ov", "stable-diffusion-1.5", "square"],
        "sd_1.5_square_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],
        "sd_1.5_square_int8a16": ["stable-diffusion-ov", "stable-diffusion-1.5", "square_int8"],
        "sd_3.0_med_diffuser_square": ["stable-diffusion-ov", "stable-diffusion-3.0-medium", "square_diffusers"],
        "sd_3.5_med_turbo_square": ["stable-diffusion-ov", "stable-diffusion-3.5-medium", "square_turbo"],
        "sd_1.5_landscape": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape"],
        "sd_1.5_portrait_512x768": ["stable-diffusion-ov", "stable-diffusion-1.5", "portrait_512x768"],
        "sd_1.5_landscape_768x512": ["stable-diffusion-ov", "stable-diffusion-1.5", "landscape_768x512"],
        "sd_1.5_inpainting": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting"],
        "sd_1.5_inpainting_int8": ["stable-diffusion-ov", "stable-diffusion-1.5", "inpainting_int8"],
        "sd_2.1_square_base": ["stable-diffusion-ov", "stable-diffusion-2.1", "square_base"],
        "sd_2.1_square": ["stable-diffusion-ov", "stable-diffusion-2.1", "square"],
        "sd_3.0_square": ["stable-diffusion-ov", "stable-diffusion-3.0"],
        "controlnet_referenceonly": ["stable-diffusion-ov", "controlnet-referenceonly"],
        "controlnet_openpose": ["stable-diffusion-ov", "controlnet-openpose"],
        "controlnet_canny": ["stable-diffusion-ov", "controlnet-canny"],
        "controlnet_scribble": ["stable-diffusion-ov", "controlnet-scribble"],
        "controlnet_openpose_int8": ["stable-diffusion-ov", "controlnet-openpose-int8"],
        "controlnet_canny_int8": ["stable-diffusion-ov", "controlnet-canny-int8"],
        "controlnet_scribble_int8": ["stable-diffusion-ov", "controlnet-scribble-int8"],
    }

    # --list: show available + FastSD catalog
    if args.list:
        print("\nPreinstalled models:")
        for key in validate_model_paths(weight_path, model_paths).keys():
            print(f"  {key}")
        print("\nFastSD models (installed at runtime):")
        if fast_sd_models:
            for fsd_model in fast_sd_models:
                print(f"  {fsd_model}")
        else:
            print("  <none configured>")
        return 0

    # Determine path vs FastSD
    use_fastsd = model_name_key in fast_sd_models_up
    if not use_fastsd and model_name_key not in model_paths:
        log.error(f"Unknown model name: {args.model_name}")
        return 2

    if not use_fastsd:
        rel = model_paths[model_name_key]
        model_path = os.path.join(weight_path, *rel)
        if not os.path.isdir(model_path):
            log.error(f"Model directory missing: {model_path}")
            return 2
        model_config_file_name = os.path.join(model_path, "config.json")
    else:
        # FastSD models are pulled at runtime in the engine; no local path.
        model_path = ""

    # Devices: 5 entries for regular, 1 for FastSD
    execution_devices: List[str] = (["GPU"] * 5) if not use_fastsd else ["CPU"]

    # Optional power-mode config (non-FastSD)
    if not use_fastsd and args.power_mode and os.path.exists(model_config_file_name):
        try:
            with open(model_config_file_name, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if str(cfg.get("power modes supported", "no")).lower() == "yes":
                key = args.power_mode.lower()
                if key in cfg:
                    execution_devices = list(cfg[key])
                else:
                    execution_devices = list(cfg.get("best performance", execution_devices))
            else:
                execution_devices = list(cfg.get("best performance", execution_devices))
        except Exception as e:
            log.warning(f"Power-mode config ignored: {e}")

    # CLI overrides (bounds-checked)
    if args.device:
        execution_devices = [args.device] if use_fastsd else [args.device] * 5
    if not use_fastsd:
        if args.text_device:
            execution_devices[0] = args.text_device
        if args.unet_device:
            execution_devices[1] = args.unet_device
        if args.unet_neg_device:
            execution_devices[2] = args.unet_neg_device
        if args.vae_device:
            idx = 3 if "lcm" not in model_name_key else 2
            if 0 <= idx < len(execution_devices):
                execution_devices[idx] = args.vae_device
    else:
        # FastSD takes a single device slot
        if args.text_device or args.unet_device or args.unet_neg_device or args.vae_device:
            log.info("Note: per-submodel device flags are ignored for FastSD; using single device.")

    print_system_info()

    # OpenVINO versions (best-effort)
    try:
        log.info("")
        log.info("Device : Version")
        core = Core()
        for dev in core.available_devices:
            ver = core.get_versions(dev)[dev].build_number
            log.info(f"  {dev} : {ver}")
        log.info("")
    except Exception as e:
        log.info(f"OpenVINO device info unavailable: {e}")

    if not use_fastsd:
        log.info("Initializing Inference Engine...")
        log.info("Model Path: %s", model_path)
        if "turbo" in model_name_key and args.guidance_scale > 1.0:
            log.warning(f"Max guidance scale for {model_name_key} is 1.0; clamping to 1.0")
            args.guidance_scale = 1.0
    else:
        log.info("Initializing FastSD engine...")

    prompt = args.prompt
    negative_prompt = args.neg_prompt
    init_image_path = args.init_image
    mask_image_path = args.mask_image
    num_infer_steps = args.iterations
    guidance_scale = args.guidance_scale
    strength = 1.0  # kept for compat

    # Default scheduler; some models override below
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # Engine init
    if not use_fastsd:
        engine = initialize_engine(model_name=model_name_key, model_path=model_path, device_list=execution_devices)
    else:
        # Use catalog name with exact case; model path unused for FastSD
        engine = initialize_engine(model_name=fast_sd_models_map[model_name_key], model_path="", device_list=execution_devices)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    results = []
    generation_times: List[float] = []

    for i in range(args.num_images):
        log.info("Starting inference...")
        log.info("Prompt: %s", prompt)
        log.info("negative_prompt: %s", negative_prompt)
        log.info("num_inference_steps: %s", num_infer_steps)
        log.info("guidance_scale: %s", guidance_scale)
        log.info("strength: %s", strength)
        log.info("init_image: %s", init_image_path)

        # Seed policy: honor --seed for the first image only
        if args.seed is not None and i == 0:
            ran_seed = int(args.seed)
        else:
            ran_seed = random.randrange(2**32 - 2)
        np.random.seed(ran_seed)
        log.info("Random Seed: %s", ran_seed)

        start_time = time.time()

        # Dispatch by model
        if use_fastsd:
            # FastSD currently fixed to 512x512
            output = engine(
                prompt=prompt,
                negative_prompt=None,
                height=512,
                width=512,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                seed=ran_seed,
            )
        elif model_name_key in ("sd_1.5_inpainting", "sd_1.5_inpainting_int8"):
            # Require both paths
            if not mask_image_path or not os.path.exists(mask_image_path):
                raise FileNotFoundError("--mask-image is required for inpainting and must exist.")
            if not init_image_path or not os.path.exists(init_image_path):
                raise FileNotFoundError("--init-image is required for inpainting and must exist.")
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_path=init_image_path,
                mask_path=mask_image_path,
                scheduler=scheduler,
                strength=strength,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                callback=None,
                callback_userdata=None,
            )
        elif model_name_key == "controlnet_referenceonly":
            if not init_image_path or not os.path.exists(init_image_path):
                raise FileNotFoundError("--init-image is required for controlnet_referenceonly.")
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.open(init_image_path),
                scheduler=scheduler,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=False,
                model=model_path,
                callback=None,
                callback_userdata=None,
            )
        elif "controlnet" in model_name_key:
            if not init_image_path or not os.path.exists(init_image_path):
                raise FileNotFoundError("--init-image is required for controlnet models.")
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=Image.open(init_image_path),
                scheduler=scheduler,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=False,
                model=model_path,
                callback=None,
                callback_userdata=None,
            )
        elif model_name_key == "sd_1.5_square_lcm":
            output = engine(
                prompt=prompt,
                negative_prompt=None,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                seed=ran_seed,
                callback=None,
                callback_userdata=None,
            )
        elif "sdxl" in model_name_key:
            output = engine(
                prompt=prompt,
                negative_prompt=None,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                seed=ran_seed,
                callback=None,
                callback_userdata=None,
            )
        elif "sd_3.0_med" in model_name_key or "sd_3.5_med" in model_name_key:
            if model_name_key == "sd_3.5_med_turbo_square":
                negative_prompt = None
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                seed=ran_seed,
                callback=None,
                callback_userdata=None,
            )
        else:
            # sd_2.1 tweaks
            sched = scheduler
            if model_name_key == "sd_2.1_square":
                sched = EulerDiscreteScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", prediction_type="v_prediction"
                )
            mdl = model_path if "sd_2.1" not in model_name_key else model_name_key
            output = engine(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=None,
                scheduler=sched,
                strength=strength,
                num_inference_steps=num_infer_steps,
                guidance_scale=guidance_scale,
                eta=0.0,
                create_gif=False,
                model=mdl,
                callback=None,
                callback_userdata=None,
            )

        gen_time = time.time() - start_time
        print(f"Image Generation Time: {round(gen_time, 2)} seconds")

        # Build safe stem
        stem_model = (fast_sd_models_map[model_name_key] if use_fastsd else model_name_key).replace("/", "_")
        stem = (
            f"{stem_model}_{timestamp}_{'_'.join(map(str, execution_devices))}_{ran_seed}_{num_infer_steps}_steps"
        )

        results.append((output, stem, gen_time))
        generation_times.append(gen_time)

    if args.num_images > 1:
        print(f"Average Image Generation Time: {round(mean(generation_times), 2)} seconds")

    if args.save_image:
        for idx, (result, stem, _) in enumerate(results, start=1):
            out_path = f"{stem}_{idx}.jpg"
            # Save by type
            if isinstance(result, np.ndarray):
                cv2.imwrite(out_path, result)
            elif hasattr(result, "save"):  # PIL Image or similar
                result.save(out_path)
            else:
                # Last resort: try to unwrap {images: [PIL]}
                img = None
                if isinstance(result, dict):
                    imgs = result.get("images") or result.get("image")
                    if isinstance(imgs, list) and imgs:
                        img = imgs[0]
                if img is not None and hasattr(img, "save"):
                    img.save(out_path)
                else:
                    log.warning(f"Unknown output type; cannot save: {type(result)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

