from setuptools import setup, find_packages
import os
from pathlib import Path
import re
import subprocess

this_dir     = Path(__file__).resolve().parent
weights_dir  = this_dir.joinpath("weights")
readme       = this_dir.joinpath("README.md")

with open(readme, "r", encoding="utf8") as fh:
    long_description = fh.read()


def get_plugin_version(file_dir=None):
    """
    Retrieves the plugin version via git tags if available, ensuring
    the command is run from the directory where this Python file resides.

    Returns:
        str: Plugin version from git or "0.0.0dev0" if git is unavailable.

    Why use git describe for this? Because generates a human-readable string to 
    identify a particular commit in a Git repository, using the closest (most recent) 
    annotated tag reachable from that commit. Typically, it looks like:
    <tag>[-<number_of_commits_since_tag>-g<abbreviated_commit_hash>]
    
    For example, if your commit is exactly tagged 1.0.0, running 
    git describe might simply return 1.0.0. If there have been 10 
    commits since the v1.0.0 tag, git describe might return something like:
    1.0.0-10-g3ab12ef
    where:

    1.0.0 is the closest tag in the commit history.
    10 is how many commits you are ahead of that tag.
    g3ab12ef is the abbreviated hash of the current commit.

    we can then turn this into a PEP440 compliant string
    """
    try:
        raw_version = subprocess.check_output(
            ["git", "describe", "--tags"],
            cwd=file_dir,
            encoding="utf-8"
        ).strip()
        
        # Normalize the git version to PEP 440
        match = re.match(r"v?(\d+\.\d+\.\d+)(?:-(\d+)-g[0-9a-f]+)?", raw_version)

        if match:
            version, dev_count = match.groups()
            if dev_count:
                return f"{version}.dev{dev_count}"  # PEP 440 dev version
            return version
        else:
            raise ValueError(f"Invalid version format: {raw_version}")
    except Exception as e:
        print(f"Error obtaining version: {e}")
        return "0.0.0"  # Fallback version    

plugin_version = get_plugin_version(this_dir)

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
        "gdown",
        "requests",
        "opencv-python>=4.8.1.78",
        "scikit-image",
        "timm==0.4.5",
        "transformers>=4.37.0",
        "diffusers==0.33.0",
        "controlnet-aux>=0.0.6",
        "openvino",
        "psutil",
        "matplotlib",
        "sentencepiece"
    ],
)

