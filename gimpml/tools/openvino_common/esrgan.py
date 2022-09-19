import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from openvino.inference_engine import IECore

def img_preprocess(input_image, w, h):
	img = cv2.imread(input_image)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img,(w, h))
	ir_img = img / 255.0
	ir_img = np.array(ir_img).astype('float32')
	# ir_img = np.array(img).astype('int')
	ir_img = np.transpose(ir_img, axes=[2, 0, 1])
	return img, ir_img

print("test start")
input_image = r"D:\test_images\R.png"
input_nodes = ['input']

img = cv2.imread(input_image)
img, ir_img = img_preprocess(input_image, 64, 64)
cv2.imwrite("test.png", img)
inp = np.expand_dims(ir_img, axis=0)
print("image resized shape", inp.shape)

ie = IECore()
net = ie.read_network(
	model=r'D:\optimized\realesrgan.xml', weights=r'D:\optimized\realesrgan.bin'
    # model="ONNX/IR/optimized/realesrgan.xml", weights="ONNX/IR/optimized/realesrgan.bin"
)
exec_net = ie.load_network(network=net, device_name="CPU")

input_key = next(iter(exec_net.input_info))
output_key = next(iter(exec_net.outputs.keys()))

result = exec_net.infer(inputs={input_key: ir_img})[output_key]
print(np.array(result).shape)
# result = (np.array(result)[0] * 255).astype(int)
result = np.array(result)[0]

plt.figure(figsize=(25, 10))
grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
plt.subplot(grid_spec[0])
#img1 = cv2.imread(input_image)
bicubic = cv2.resize(img,(256, 256), interpolation=cv2.INTER_CUBIC)
plt.imshow(bicubic)
plt.axis('off')
plt.title('bicubic interpolation')
print(bicubic)
#cv2.imshow(img)

output = np.transpose(result, axes=[1, 2, 0])
print("output is ready")
print(output.shape, output.dtype)
plt.subplot(grid_spec[1])
# output = np.clip(output, 0, 1)
# output = output.astype('int')
print(output)
plt.imshow(output)
plt.axis('off')
plt.title('super-resolution')
plt.show()