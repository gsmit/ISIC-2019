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
df = pd.read_csv(dataset_path['7-point'] / 'meta/meta.csv', usecols=['derm', 'diagnosis'])
df = df.rename(columns={'derm': 'path', 'diagnosis': 'class'})
df['path'] = 'D:/Data/ISIC-2019/external/7-point/images/' + df['path']

# assign labels to images
df.loc[df['class'] == 'basal cell carcinoma', 'class'] = 'BCC'
df.loc[df['class'] == 'lentigo', 'class'] = 'BKL'
df.loc[df['class'] == 'seborrheic keratosis', 'class'] = 'BKL'
df.loc[df['class'] == 'dermatofibroma', 'class'] = 'DF'
df.loc[df['class'] == 'vascular lesion', 'class'] = 'VASC'
df.loc[df['class'] == 'melanosis', 'class'] = 'BKL'
df.loc[df['class'] == 'reed or spitz nevus', 'class'] = 'NV'
df.loc[df['class'] == 'blue nevus', 'class'] = 'NV'
df.loc[df['class'] == 'clark nevus', 'class'] = 'NV'
df.loc[df['class'] == 'combined nevus', 'class'] = 'NV'
df.loc[df['class'] == 'congenital nevus', 'class'] = 'NV'
df.loc[df['class'] == 'dermal nevus', 'class'] = 'NV'
df.loc[df['class'] == 'melanoma', 'class'] = 'MEL'
df.loc[df['class'] == 'melanoma metastasis', 'class'] = 'MEL'
df.loc[df['class'] == 'melanoma (in situ)', 'class'] = 'MEL'
df.loc[df['class'] == 'melanoma (less than 0.76 mm)', 'class'] = 'MEL'
df.loc[df['class'] == 'melanoma (0.76 to 1.5 mm)', 'class'] = 'MEL'
df.loc[df['class'] == 'melanoma (more than 1.5 mm)', 'class'] = 'MEL'
df.loc[df['class'] == 'miscellaneous', 'class'] = 'UNK'
df.loc[df['class'] == 'recurrent nevus', 'class'] = 'UNK'

# format table
df = df[['path', 'class']]
dataset = '7point'

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
