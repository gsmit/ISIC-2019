import os
import time
import glob
import shutil
import pandas as pd
from os import listdir
from os.path import isfile, join
from pathlib import Path

# skin lesion categories
classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
num_classes = 8  # excluding the unknown class

# data folders
data_path = Path('D:/Data/ISIC-2019')
model_path = data_path / 'models'  # model output folder
submission_path = data_path / 'submissions'  # submission output folder
train_labels = data_path / 'ISIC_2019_Training_GroundTruth.csv'
train_images = data_path / 'ISIC_2019_Training_Input'
test_images = data_path / 'ISIC_2019_Test_Input'

train_labels = pd.read_csv(train_labels)
train_labels['class'] = train_labels[classes].idxmax(axis=1)
train_labels['path'] = str(train_images) + '\\' + train_labels['image'] + '.jpg'

# copy and sort original images
for skin_class in classes:
    start = time.time()

    try:
        os.mkdir(f'D:/Data/ISIC-2019/images_train_external/{skin_class}')
        os.mkdir(f'D:/Data/ISIC-2019/images_test/{skin_class}')
    except FileExistsError:
        pass

    subset = train_labels.loc[train_labels['class'] == skin_class].copy()
    for img_path in subset['path'].tolist():
        img_name = img_path.split('\\')[-1]
        shutil.copyfile(img_path, f'D:/Data/ISIC-2019/images_train_external/{skin_class}/{img_name}')

    print(f'Finished processing class {skin_class} in {round(time.time() - start, 1)}s')
