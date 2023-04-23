import os
import copy

import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# if not os.path.exists(BASE_DIR + f'method_{method_}'):
#     os.makedirs(BASE_DIR + f'/method_{method_}', exist_ok=True)

# for root, dirs, files in os.walk("."):
#     for filename in files:
#         print(filename)

resize_w = 320
resize_h = 320
folder_name = 'jaguar_train'
path_folder = f'{BASE_DIR}/{folder_name}'
path_folder_resize = f'{BASE_DIR}/{resize_w}_{resize_h}_{folder_name}'
os_listdir = os.listdir(path_folder)

augmentation = True
img_aug = []
seq = iaa.Sequential([
    iaa.Affine(rotate=(-25, 25)),
    iaa.AdditiveGaussianNoise(scale=(10, 45)),
    # iaa.AddToHueAndSaturation((-60, 60)),  # change their color
    # iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
])

# print(len(os_listdir))


for filename in os_listdir:
    if filename[filename.rfind(".") + 1:] in ['jpg', 'jpeg', 'png']:
        print('обрабатываем')
        raw_img = Image.open(f'{path_folder}/{filename}').convert('RGB')
        print(raw_img.size)
        resize_image = raw_img.resize((resize_w, resize_h))
        if not os.path.exists(path_folder_resize):
            os.makedirs(path_folder_resize, exist_ok=True)
        if augmentation:
            img_np = np.asarray(resize_image)
            img_aug = [copy.copy(img_np) for el in range(2)]
            img_aug = seq(images=img_aug)
            # PIL.Image.fromarray(numpy.uint8(I))
            img_aug = [Image.fromarray(np.uint8(el)) for el in img_aug]
            for el in img_aug:
                el.save(f'{path_folder_resize}/{resize_w}_{resize_h}_{img_aug.index(el)}_{filename}')
        resize_image.save(f'{path_folder_resize}/{resize_w}_{resize_h}_{filename}')
        print(resize_image.size)
    else:
        print(f'{filename}===========================NOOOO===========================')
