from os import path
import platform


def show_system_info():
    try:
        print(f"Running on {platform.system()} platform")
        print(f"OS: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
    except Exception as ex:
        print(f"Error occurred while getting system information {ex}")


def get_image_file_extension(image_format: str) -> str:
    if image_format == "JPEG":
        return ".jpg"
    elif image_format == "PNG":
        return ".png"
