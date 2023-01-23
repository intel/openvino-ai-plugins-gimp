from setuptools import setup, find_packages
import os

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="gimpopenvino",  # Required
    version="0.0.1",  # Required
    description="OpenVINO™ AI Plugins for GIMP",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="",  # Optional
    author="Arisha Kumar",  # Optional
    author_email="",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.8",
        # 'Programming Language :: Python :: 2.7 :: Only',
    ],
    keywords="sample, setuptools, development",  # Optional
    packages=find_packages(),
    python_requires=">=2.7",
    include_package_data=True,  
    install_requires=[
        "numpy",
        'future; python_version <= "2.7"',
        "scipy",
        "typing",
        "gdown",
        'enum; python_version <= "2.7"',
        "requests",
        "opencv-python<=4.3",
        "scikit-image",
        "timm==0.4.5",
        "transformers",
        "diffusers",
        "openvino==2022.3.0"
        
        
    ]
)

