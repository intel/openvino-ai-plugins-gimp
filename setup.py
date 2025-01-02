from setuptools import setup, find_packages
import os
from pathlib import Path
import re
import subprocess
from gimpopenvino import install_utils 

this_dir     = Path(__file__).resolve().parent
weights_dir  = this_dir.joinpath("weights")
readme       = this_dir.joinpath("README.md")

with open(readme, "r", encoding="utf8") as fh:
    long_description = fh.read()

plugin_version = install_utils.get_plugin_version(this_dir) #  get_plugin_version(here)

setup(
    name="gimpopenvino",  # Required
    version=plugin_version,  # Required
    description="OpenVINO™ AI Plugins for GIMP",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="http://github.com/intel/openvino-ai-plugins-gimp",  # Optional
    author="Arisha Kumar",  # Optional
    author_email="",  # Optional

    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.7",  # Specify supported versions
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="openvino gimp ai plugins",
    packages=find_packages(),
    python_requires=">=3.7",  # Update Python requirement
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "typing",
        "gdown",
        "requests",
        "opencv-python>=4.8.1.78",
        "scikit-image",
        "timm==0.4.5",
        "transformers>=4.37.0",
        "diffusers",
        "controlnet-aux>=0.0.6",
        "openvino",
        "psutil",
        "matplotlib",
    ],

)

install_utils.complete_install(repo_weights_dir=weights_dir)
