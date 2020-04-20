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

# set correct paths V1
skinl2_v1_BCC = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Basal-cell Carcinoma'
skinl2_v1_MEL = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Melanoma'
skinl2_v1_BKL = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Seborrheic Keratosis'
skinl2_v1_NV = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Nevus'
skinl2_v1_VASC = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Hemangioma'
skinl2_v1_UNK1 = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Other'
skinl2_v1_UNK2 = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Dermatofibroma'
skinl2_v1_UNK3 = dataset_path['SKINL2_v1'] / 'Dermatoscopic' / 'Psoriasis'

# set correct paths V2
skinl2_v2_BCC = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Basal-cell Carcinoma'
skinl2_v2_MEL = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Melanoma'
skinl2_v2_BKL = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Seborrheic Keratosis'
skinl2_v2_NV = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Nevus'
skinl2_v2_VASC = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Hemangioma'
skinl2_v2_UNK = dataset_path['SKINL2_v2'] / 'Dermatoscopic' / 'Other'

# get image paths
skinl2_BCC = set([str(p) for p in skinl2_v1_BCC.rglob('*.jpg')] + [str(p) for p in skinl2_v2_BCC.rglob('*.jpg')])
skinl2_MEL = set([str(p) for p in skinl2_v1_MEL.rglob('*.jpg')] + [str(p) for p in skinl2_v2_MEL.rglob('*.jpg')])
skinl2_BKL = set([str(p) for p in skinl2_v1_BKL.rglob('*.jpg')] + [str(p) for p in skinl2_v2_BKL.rglob('*.jpg')])
skinl2_NV = set([str(p) for p in skinl2_v1_NV.rglob('*.jpg')] + [str(p) for p in skinl2_v2_NV.rglob('*.jpg')])
skinl2_VASC = set([str(p) for p in skinl2_v1_VASC.rglob('*.jpg')] + [str(p) for p in skinl2_v2_VASC.rglob('*.jpg')])
skinl2_UNK = set([str(p) for p in skinl2_v1_UNK1.rglob('*.jpg')] + [str(p) for p in skinl2_v1_UNK2.rglob('*.jpg')] + \
                 [str(p) for p in skinl2_v1_UNK3.rglob('*.jpg')] + [str(p) for p in skinl2_v2_UNK.rglob('*.jpg')])

# make tables
skinl2_BCC = pd.DataFrame(skinl2_BCC, columns=['path'])
skinl2_BCC['class'] = 'BCC'
skinl2_MEL = pd.DataFrame(skinl2_MEL, columns=['path'])
skinl2_MEL['class'] = 'MEL'
skinl2_BKL = pd.DataFrame(skinl2_BKL, columns=['path'])
skinl2_BKL['class'] = 'BKL'
skinl2_NV = pd.DataFrame(skinl2_NV, columns=['path'])
skinl2_NV['class'] = 'NV'
skinl2_VASC = pd.DataFrame(skinl2_VASC, columns=['path'])
skinl2_VASC['class'] = 'VASC'
skinl2_UNK = pd.DataFrame(skinl2_UNK, columns=['path'])
skinl2_UNK['class'] = 'UNK'
skinl2_data = pd.concat([skinl2_BCC, skinl2_MEL, skinl2_BKL, skinl2_NV, skinl2_VASC, skinl2_UNK], ignore_index=True)

# format table
skinl2_data = skinl2_data[['path', 'class']]
dataset = 'SKINL2'

# move images to folder
for skin_class in classes:
    start = time.time()
    try:
        os.mkdir(f'D:/Data/ISIC-2019/images_train_external/{skin_class}')
    except FileExistsError:
        pass
    subset = skinl2_data.loc[skinl2_data['class'] == skin_class].copy()
    for img_path in subset['path'].tolist():
        img_name = img_path.split('\\')[-1]
        shutil.copyfile(img_path, f'D:/Data/ISIC-2019/images_train_external/{skin_class}/{dataset}_{img_name}')
    print(f'Finished processing class {skin_class} in {round(time.time() - start, 1)}s')
