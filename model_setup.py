import sys
import os
import traceback
import shutil
from pathlib import Path

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "gimpopenvino", "plugins", "openvino_utils", "tools")])
from model_manager import ModelManager

base_model_dir = os.path.join(os.path.expanduser("~"), "openvino-ai-plugins-gimp", "weights")
install_location = os.path.join(base_model_dir, "stable-diffusion-ov")
src_dir = os.path.join(os.path.dirname(__file__), "weights")
test_path = os.path.join(install_location, "superresolution-ov")

def install_base_models():
    for folder in os.scandir(src_dir):
        model = os.path.basename(folder)
        model_path = os.path.join(base_model_dir, model)
        if not os.path.isdir(model_path):
            print("Copying {} to {}".format(model, base_model_dir))
            shutil.copytree(Path(folder), model_path)

    print("Setup done for superresolution and semantic-segmentation")

def main():

    try:
        install_base_models()

        weight_path = base_model_dir

        #Move to a temporary working directory in a known place.
        # This is where we'll be downloading stuff to, etc.
        tmp_working_dir=os.path.join(weight_path, '..', 'mms_tmp')

        #if this dir doesn't exist, create it.
        if not os.path.isdir(tmp_working_dir):
            os.mkdir(tmp_working_dir)

        # go there.
        os.chdir(tmp_working_dir)

        model_manager = ModelManager(weight_path)

        # we want to display progress bars in the running terminal.
        model_manager.show_hf_download_tqdm = True

        while True:
            installed_models, installable_model_details = model_manager.get_all_model_details()

            user_choice_to_model_id = {}
            for i in range(0, len(installable_model_details)):
                install_details = installable_model_details[i]
                user_choice_to_model_id[str(i + 1)] = install_details

            print("=========Choose Stable Diffusion models to download=========")
            for user_choice_val, install_details in user_choice_to_model_id.items():
                install_detail_full_name = install_details["name"]
                install_status = install_details["install_status"]
                if install_status == "installed":
                    install_status = "(Installed)"
                else:
                    install_status = ""
                print(f"{user_choice_val}  - {install_detail_full_name} {install_status}")

            print("0  - Exit Stable Diffusion Model setup")
            choice = input("Enter the number for the model you want to download.\nSpecify multiple options using spaces: ")

            choices = choice.split(" ")

            for ch in choices:
                if ch == "0":
                    print("Exiting Model setup...")
                    return

                if ch in user_choice_to_model_id:
                    install_details = user_choice_to_model_id[ch]
                    model_manager.install_model(install_details["id"])
            else:
                print(f"Invalid choice: {ch.strip()}")
    except Exception as e:
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
