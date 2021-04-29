'''
visualize the filters 
'''
import torch
import matplotlib.pyplot as plt
import torchvision
from model import SelfDrivingModel


driver = SelfDrivingModel()


checkpoint = torch.load('models\model-5gen-5epoch.h5')['state_dict']
# print(checkpoint)
driver.load_state_dict(checkpoint)

kernels = driver.conv_layers[0].weight.detach().clone()
print(kernels.size())

# normalize to (0,1) range so that matplotlib
# can plot them
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
# change ordering since matplotlib requires images to 
# be (H, W, C)
plt.imshow(filter_img.permute(1, 2, 0))
# img = save_image(kernels, 'encoder_conv1_filters.png' ,nrow = 12)
plt.show()