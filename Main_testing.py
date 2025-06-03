import time, math, glob
import torch
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from NETWORK.Demosaiced_RNAN import RNAN
from NETWORK.Proposed_Network import ExRNet_FusionModule_v4
from NETWORK.Network_subimages import Feature_Selection, Fusion_Net

from TOOLS.ulti import Convert2LogDomain_norm, Convert2Luma, Convert2Rad_from_LogDomain, mat2tensor
from TOOLS.ulti_v3 import Generate_HDR_image_with_GaussianWeightv3, Torchtensor2Array

#################################################### KALANTARI DATASET
num_of_images = 1
image_list = sorted(glob.glob("Kalantari/input" + "/*.*"))

image_savename = ['01']

softmask_list = sorted(glob.glob("Kalantari/softmask" + "/*.*"))

###################################################
device = "cpu"

########################################################## Proposed
############### 32x32
net_path_ExRNet = 'WEIGHTS_ECCV2022/Proposed/ExRNet/net_84.pth'
net_path_DINet = 'WEIGHTS_ECCV2022/Proposed/DINet/net_84.pth'
net_path_FusionNet = 'WEIGHTS_ECCV2022/Proposed/FusionNet/net_84.pth'
net_path = 'WEIGHTS_ECCV2022/Proposed/Color/net_84.pth'

############################################# 
save_path = 'test_results/Kalantari/Proposed/'

with torch.no_grad():
    ######## Proposed
    ExRNet = ExRNet_FusionModule_v4()
    state_dict = torch.load(net_path_ExRNet, map_location = lambda s, l: s)
    ExRNet.load_state_dict(state_dict)
    ExRNet.eval()
    ExRNet.to(device)

    DINet = Feature_Selection()
    state_dict = torch.load(net_path_DINet, map_location = lambda s, l: s)
    DINet.load_state_dict(state_dict)
    DINet.eval()
    DINet.to(device)

    FusionNet = Fusion_Net()
    state_dict = torch.load(net_path_FusionNet, map_location = lambda s, l: s)
    FusionNet.load_state_dict(state_dict)
    FusionNet.eval()
    FusionNet.to(device)

    HDR_net = RNAN() # Zhang 2020
    state_dict = torch.load(net_path, map_location = lambda s, l: s)
    HDR_net.load_state_dict(state_dict)
    HDR_net.eval()
    HDR_net.to(device)

    count = 0
    for image_name in image_list:
        print('Image: %d' %(count + 1))

        # image_name = image_list[i]
        input = mat2tensor(image_name, 'E_hat', channel=1)
        input = input.float()

        softmask_name = softmask_list[count]
        softmask = mat2tensor(softmask_name, 'SoftMask', channel = 1)
        softmask = softmask.float()

        ### Proposed
        input_log = Convert2LogDomain_norm(input)
        # 32 x 32
        HDR = Generate_HDR_image_with_GaussianWeightv3(input_log, softmask, ExRNet, DINet, FusionNet, HDR_net, device, size_patch=32, stride=16)

        HDR = Torchtensor2Array(HDR) # {Proposed}

        sdr_image = input_log[0, 0, :, :].cpu().numpy()
        sdr_image = (sdr_image - sdr_image.min()) / (sdr_image.max() - sdr_image.min())

        hdr_image = HDR
        hdr_image = (hdr_image - hdr_image.min()) / (hdr_image.max() - hdr_image.min())

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(sdr_image, cmap='gray')
        plt.title(f'SDR (log) — Image {count + 1}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(hdr_image, cmap='gray')
        plt.title(f'HDR Result — Image {count + 1}')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_path}comparison_{count + 1}.png')
        plt.close()
        name = str(count + 1).zfill(6)
        # name = image_savename[count]
        scipy.io.savemat(save_path + name + '.mat', mdict={'HDR': HDR}) # {Proposed}

        count = count + 1

# net_path = '/media/vgan/00a91feb-14ee-4d63-89c0-1bb1b7e57b8a/LOCAL/CVPR_2022_Networks/Proposed/BJDD/64/Color/net_89.pth'
# save_path = 'test_data/HDR-Eye/MergeNet_v2/BJDD_64/'




import cv2
import numpy as np
from scipy.io import loadmat

save_path = 'test_results/Kalantari/Proposed/'

data = loadmat('Kalantari/softmask/01.mat')
print(data.keys()) 
SoftMask = data['SoftMask']
print("SoftMask shape:", SoftMask.shape)

img = data['SoftMask']

if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)

SoftMask_log = np.log1p(img)
normalized = cv2.normalize(SoftMask_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(save_path + 'SoftMask.png', normalized)

data = loadmat('Kalantari/gt/01.mat')
print(data.keys())
gt = data['N']
print("gt shape:", gt.shape)

data = loadmat('Kalantari/gt/01.mat')
gt   = data['N']

gt_min, gt_max = gt.min(), gt.max()
gt_norm = (gt - gt_min) / (gt_max - gt_min + 1e-8)   # → [0,1]
gt_uint8 = (gt_norm * 255).astype(np.uint8)          # → uint8

cv2.imwrite(save_path + 'gt.png', cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR))




data = loadmat('Kalantari/input/01.mat')
print(data.keys())
E_hat = data['E_hat']
print("E_hat shape:", E_hat.shape)

img = data['E_hat']

if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)

E_hat_log = np.log1p(img)
normalized = cv2.normalize(E_hat_log, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(save_path + 'input_tonned.png', normalized)




data = loadmat("test_results/Kalantari/Proposed/000001.mat")
hdr = data['HDR']

if hdr.ndim == 2:
    hdr = np.stack([hdr] * 3, axis=-1)

# Понизим экспозицию (например, делим на 4)
hdr_toned = hdr / 50.0
# hdr_toned = np.clip(hdr_toned, 0, 1)

cv2.imwrite(save_path + "image_hdr_exposure.hdr", hdr_toned.astype(np.float32))