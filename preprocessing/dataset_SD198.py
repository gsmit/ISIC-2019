import os
import time
import glob
import shutil
import pandas as pd
from pathlib import Path

# data and project paths
data_path = Path('D:/Data/ISIC-2019/')
project_path = Path('C:/Users/GijsSmit/PycharmProjects/ISIC-2019')
classes = sorted(['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])

# dataset paths
dataset_path = {
    '7-point': data_path / 'external/7-point',
    'mednode': data_path / 'external/mednode',
    'PH2Dataset': data_path / 'external/PH2Dataset',
    'SD-128': data_path / 'external/SD-128',
    'SD-198': data_path / 'external/SD-198',
    'SD-260': data_path / 'external/SD-260',
    'SKINL2_v1': data_path / 'external/SKINL2_v1',
    'SKINL2_v2': data_path / 'external/SKINL2_v1'}

# class dictionary
sd_class_dict = {
    'Actinic_solar_Damage(Actinic_Keratosis)': 'AK',
    'Basal_Cell_Carcinoma': 'BCC',
    'Dermatofibroma': 'DF',
    "Becker's_Nevus": 'NV',
    'Blue_Nevus': 'NV',
    'Congenital_Nevus': 'NV',
    'Benign_Keratosis': 'BKL',
    'Seborrheic_Keratosis': 'BKL',
    'Solar_Lentigo': 'BKL',
    'Lichen_Planus': 'BKL',
    'Malignant_Melanoma': 'MEL',
    'Metastatic_Carcinoma': 'MEL',
    'Lentigo_Maligna_Melanoma': 'MEL'}

sd198_images = [str(p) for p in dataset_path['SD-198'].rglob('*.jpg')]
sd260_images = [str(p) for p in dataset_path['SD-260'].rglob('*.jpg')]

# make labels (198)
sd198_labels = []
for img in sd198_images:
    label = 'UNK'
    for key, value in sd_class_dict.items():
        if key in img:
            label = value
            break
    sd198_labels.append(label)

# combine data
sd198_data = pd.DataFrame(sd198_images, columns=['path'])
sd198_data['class'] = sd198_labels
dataset1 = 'SD198'

# make labels (260)
sd260_labels = []
for img in sd260_images:
    label = 'UNK'
    for key, value in sd_class_dict.items():
        if key in img:
            label = value
            break
    sd260_labels.append(label)

# combine data
sd260_data = pd.DataFrame(sd260_images, columns=['path'])
sd260_data['class'] = sd260_labels
dataset2 = 'SD260'

#%%
sd198_data['class'].value_counts()

#%%
sd260_data['class'].value_counts()

#%%
# move images to folder
for skin_class in ['UNK']:
    start = time.time()
    try:
        os.mkdir(f'D:/Data/ISIC-2019/images_train_external_SD101/{skin_class}')
    except FileExistsError:
        pass
    subset = sd_data.loc[sd_data['class'] == skin_class].copy()
    for img_path in subset['path'].tolist():
        img_name = img_path.split('\\')[-1]
        img_name = 'SD198_' + img_name
        shutil.copyfile(img_path, f'D:/Data/ISIC-2019/images_train_external/{skin_class}/{dataset}_{img_name}')
        print(f'Finished processing class {skin_class} in {round(time.time() - start, 1)}s')
