
import subprocess
import os


sd_python_path=r'model_conv\Scripts\python.exe'
chose_model=r'openvino-ai-plugins-gimp\sd_model_conversion.py'

print("=========Chose SD-1.5 models to download and convert=========")
print("1 - Square (512x512 output image) ")
print("2 - Landscape (640x360 output image, 16:9) ")	
print("3 - Portrait (360x640 output image, 9:16) ")
print("4 - Portrait_512x768 (512x768 output image), will take time since model is large ")
print("5 - Landscape_768x512 (768x512 output image),  will take time since model is large ")
print("6 - ALL the above SD-1.5 models ")
print("7 - Only Square, Landscape, Portrait")
print("8 - Skip All SD-1.5 Model setup ")   
    
choice = input("Enter the Number for the model you want to download & convert: ")


if choice=="6":
    for i in range(1,6):
        command = sd_python_path + ' ' + chose_model + ' ' + str(i)
        os.system(command)

elif choice=="7":
    for i in range(1,4):
        command = sd_python_path + ' ' + chose_model + ' ' + str(i)
        os.system(command)

elif choice=="8":
    print("Exiting SD-1.5 Model setup.........")

else:
    try:
        command = sd_python_path + ' ' + chose_model + ' ' + choice
        os.system(command)
    except:
        print("Wrong option selected. Exiting...")
    


    
    
    