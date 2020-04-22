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

# load meta data
df = pd.read_csv(dataset_path['PH2Dataset'] / 'PH2_dataset.txt',
                 sep="\|\|", skipfooter=25, engine='python', usecols=[1, 3])
df.rename(columns={'   Name ': 'path', ' Clinical Diagnosis ': 'class'}, inplace=True)

# assign labels to images
df.loc[df['class'] == 0, 'class'] = 'NV'
df.loc[df['class'] == 1, 'class'] = 'NV'
df.loc[df['class'] == 2, 'class'] = 'MEL'
df['path'] = df['path'].apply(lambda x: x.strip())
df['path'] = 'D:/Data/ISIC-2019/external/PH2Dataset/PH2 Dataset images/' + df['path'] +\
             '/' + df['path'] + '_Dermoscopic_Image/' + df['path'] + '.bmp'

# format table
df = df[['path', 'class']]
dataset = 'PH2'

# move images to folder
for skin_class in classes:
    start = time.time()
    try:
        os.mkdir(f'D:/Data/ISIC-2019/images_train_external/{skin_class}')
    except FileExistsError:
        pass
    subset = df.loc[df['class'] == skin_class].copy()
    for img_path in subset['path'].tolist():
        img_name = img_path.split('/')[-1]
        shutil.copyfile(img_path, f'D:/Data/ISIC-2019/images_train_external/{skin_class}/{dataset}_{img_name}')
    print(f'Finished processing class {skin_class} in {round(time.time() - start, 1)}s')
