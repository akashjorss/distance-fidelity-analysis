from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import sys
import os
from tqdm import tqdm
from load_llff import load_llff_data
from img2vec_pytorch import Img2Vec

sam = sam_model_registry["default"](checkpoint="model/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.80)

imgs, poses, bds, render_poses, i_test = load_llff_data("data/nerf_llff_data/trex", factor=1) #
hwf = poses[0,:3,-1]
H, W, focal = hwf[0], hwf[1], hwf[2]
poses = poses[:,:3,:4]
hwf.shape, poses.shape # (3,) (N, 3, 4)
images = imgs
#convert images to uint8 ranging from 0 to 255
images = (images * 255).astype(np.uint8)

masks = []
for image in tqdm(images):
    mask = mask_generator.generate(image)
    masks.append(mask)

#Crop the images by their masks and store them in a N x M array, where N is the number of images and M is the number of maximum masks detected in the images. Add padding where there is no more masks.
cropped_images = []
cropped_images_to_images = {}
for i in tqdm(range(len(images)), desc="Cropping images"):
    for j in range(len(masks[i])):
        mask = np.dstack([masks[i][j]['segmentation']]*3)
        cropped_images.append(Image.fromarray(images[i]*mask))
        cropped_images_to_images[len(cropped_images)-1] = i

#downsample the cropped images by 4
downsampled_cropped_images = []
for cropped_image in tqdm(cropped_images, desc="Downsampling images"):
    downsampled_cropped_images.append(cropped_image.resize((cropped_image.width//4, cropped_image.height//4)))

#find mask embeddings
img2vec = Img2Vec(cuda=True, model='efficientnet_b0')
mask_vectors = img2vec.get_vec(downsampled_cropped_images)

target_dir = "./mask_vectors"
os.makedirs(target_dir, exist_ok=True)

#save mask_vectors in the directory
np.save(os.path.join(target_dir, "trex.npy"), mask_vectors)

#save the dictionary that maps cropped images to their original images as json
import json
with open(os.path.join(target_dir, "cropped_images_to_images.json"), "w") as f:
    json.dump(cropped_images_to_images, f)


