import os
import cv2
import numpy as np
from tqdm import tqdm


def sp_division(img):
    slic = cv2.ximgproc.createSuperpixelSLIC(img)
    slic.iterate(10)
    labels = slic.getLabels()
    # mask_slic = slic.getLabelContourMask() #获取Mask，超像素边缘Mask==1
    # number_slic = slic.getNumberOfSuperpixels()  #获取超像素数目
    # mask_inv_slic = cv2.bitwise_not(mask_slic)  
    # img_slic = cv2.bitwise_and(img,img,mask =  mask_inv_slic) #在原图上绘制超像素边界
    # cv2.imwrite('3.png', img_slic)
    return labels


def run_sp(image_dir, output_root, kf_list=None):
    sp_dir = os.path.join(output_root, "sp")
    os.makedirs(sp_dir, exist_ok=True)
    try:
        image_names = sorted(os.listdir(image_dir), key=lambda x: int(x.split('/')[-1][:-4]))
    except:
        image_names = sorted(os.listdir(image_dir))
    image_names = [image_names[k] for k in kf_list]
    for name in tqdm(image_names):
        if os.path.exists(os.path.join(sp_dir, name[:-4] + '.npy')):
            continue
        img = cv2.imread(os.path.join(image_dir, name))
        labels = sp_division(img)
        np.save(os.path.join(sp_dir, name[:-4]), labels)


if __name__ == '__main__':
    img = cv2.imread('/home/hyz/git-plane/PixelPlane/data/scene0000_00/images/36.jpg')
    sp_division(img)