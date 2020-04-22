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

# move external images to target folder
folder = Path('D:/Data/ISIC-2019/external/complete_mednode_dataset/naevus')
images = glob.glob(str(folder / '*.jpg'))

# create table for NV
mednode_NV = dataset_path['mednode'] / 'naevus'
mednode_NV_images = [str(p) for p in mednode_NV.rglob('*.jpg')]
mednode_NV_data = pd.DataFrame(mednode_NV_images, columns=['path'])
mednode_NV_data['class'] = 'NV'

# create table for MEL
mednode_MEL = dataset_path['mednode'] / 'melanoma'
mednode_MEL_images = [str(p) for p in mednode_MEL.rglob('*.jpg')]
mednode_MEL_data = pd.DataFrame(mednode_MEL_images, columns=['path'])
mednode_MEL_data['class'] = 'MEL'

# combine tables
mednode_data = pd.concat([mednode_NV_data, mednode_MEL_data])
dataset = 'mednode'

# move images to folder
for skin_class in classes:
    start = time.time()
    try:
        os.mkdir(f'D:/Data/ISIC-2019/images_train_external/{skin_class}')
    except FileExistsError:
        pass
    subset = mednode_data.loc[mednode_data['class'] == skin_class].copy()
    for img_path in subset['path'].tolist():
        img_name = img_path.split('\\')[-1]
        shutil.copyfile(img_path, f'D:/Data/ISIC-2019/images_train_external/{skin_class}/{dataset}_{img_name}')
    print(f'Finished processing class {skin_class} in {round(time.time() - start, 1)}s')
