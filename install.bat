:<<BATCH
    @echo off
    echo **** GIMP-ML Setup started **** I AMM HERE ***************
    python -m pip install virtualenv
	python -m virtualenv gimpenv3
	if "%1"=="gpu" (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html) else (gimpenv3\Scripts\python.exe -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html)
	gimpenv3\Scripts\python.exe -m pip install numpy opencv-python==4.5.5.64 transformers==4.16.2 diffusers==0.2.4 tqdm==4.64.0 openvino==2022.1.0 huggingface_hub==0.9.0 scipy streamlit==1.12.0 watchdog==2.1.9 ftfy==6.1.1
    gimpenv3\Scripts\python.exe -m pip install GIMP-ML\.
    gimpenv3\Scripts\python.exe -c "import gimpml; gimpml.setup_python_weights()"
	echo **** GIMP-ML Setup Ended ****
    exit /b
BATCH


