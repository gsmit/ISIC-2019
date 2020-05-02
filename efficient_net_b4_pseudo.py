import math
import numpy as np
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import seaborn as sns

# cropping utils
from random_crop import load_and_crop_image
import keras_preprocessing.image
import tensorflow.keras.preprocessing.image

# augmentations
from augment import solarize, posterize, contrast, color, brightness, sharpness, cutout
from PIL import Image

from os import listdir
from os.path import isfile, join
from pathlib import Path
from metric import balanced_accuracy, categorical_focal_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.client import device_lib
from ImageDataAugmentor.image_data_augmentor import *

print(device_lib.list_local_devices())

# settings
model_name = 'b4'
version = 'v10-pseudo'

# plotting settings
sns.set_style('whitegrid')
sns.set_context('notebook')

# skin lesion categories
classes = sorted(['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'])
num_classes = len(classes)  # excluding the unknown class

# data folders
data_path = Path('D:/Data/ISIC-2019')
model_path = data_path / 'models'  # model output folder
submission_path = data_path / 'submissions'  # submission output folder
train_images = data_path / 'images_train_external_450_color'
# test_images = data_path / 'ISIC_2019_Test_Input'
test_images = data_path / 'images_test_external_450'

# load previous predictions
ensemble = pd.read_csv('D:/Data/ISIC-2019/submissions/ensemble-v5-augment-543.csv')

# format training labels
tables = []
for skin_class in classes:
    sub_folder = train_images / skin_class
    sub_images = sorted([str(p) for p in sub_folder.rglob('*.jpg')] + [str(p) for p in sub_folder.rglob('*.bmp')])
    data_frame = pd.DataFrame(sub_images, columns=['path'])
    data_frame['class'] = skin_class

    # take subset
    down_sampling = True
    num_samples = 6000
    if down_sampling:
        if len(data_frame) > num_samples:
            data_frame = data_frame.sample(n=num_samples, random_state=123)

    tables.append(data_frame)

    if skin_class != 'NV':
        pseudo = ensemble.loc[ensemble[skin_class] >= 0.65]
        pseudo = pseudo[['image']]
        pseudo['path'] = str(test_images) + '\\' + pseudo['image']
        pseudo = pseudo[['path']]
        pseudo['class'] = skin_class
        tables.append(pseudo)
        print(f'Added {len(pseudo)} pseudo-labeled images to the training set for class {skin_class}')

train_labels = pd.concat(tables, ignore_index=True)
print(train_labels['class'].value_counts())

class_dummies = pd.get_dummies(train_labels['class'], dtype=int)
train_labels = pd.concat([train_labels, class_dummies.reindex(train_labels.index)], axis=1)

# validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    train_labels[['path', 'class']], train_labels[classes],
    stratify=train_labels['class'].values, shuffle=True, test_size=0.2, random_state=123)
df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
df_valid = pd.concat([X_valid, y_valid], axis=1).reset_index(drop=True)
y_train, y_valid = y_train.values, y_valid.values
y_train_int, y_valid_int = np.argmax(y_train, axis=1), np.argmax(y_valid, axis=1)

# calculate class weights
class_weights_list = class_weight.compute_class_weight(
    'balanced', [i for i in range(num_classes)], y_train_int)
class_weights = {v: k for v, k in enumerate(class_weights_list)}

# specify input
img_size = 380  # 380
channels = 3
batch_size = 4
train_steps = math.ceil(len(df_train) / batch_size)
valid_steps = math.ceil(len(df_valid) / batch_size)

# monkey patch
keras_preprocessing.image.iterator.load_img = load_and_crop_image
tensorflow.keras.preprocessing.image.load_img = load_and_crop_image

# learning rate
learning_rate = 0.5e-5  # smaller learning rate

# loading pre-trained EfficientNetB3
base = efn.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_size, img_size, channels))
base.trainable = True
base.summary()

model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D(name='gap'))
model.add(Dropout(0.4, name='dropout_out'))
model.add(Dense(num_classes, activation='softmax'))
model.trainable = True


# callbacks
def drop_decay(epoch):
    drop = 0.5
    epochs_drop = 7.0
    return learning_rate * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))


lr_scheduler = LearningRateScheduler(drop_decay, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

# compilation
model.compile(
    loss=categorical_focal_loss(gamma=1.5, alpha=0.5),
    optimizer=Adam(lr=learning_rate),
    metrics=[balanced_accuracy(num_classes), 'accuracy'])


def augment_image(img):
    """Custom augmentations using PIL."""

    # convert numpy array to Image object
    img = Image.fromarray(img.astype(np.uint8))

    # contrast
    if np.random.rand() >= 0.6:
        img = contrast(img, np.random.randint(0, 10))

    # color
    if np.random.rand() >= 0.6:
        img = color(img, np.random.randint(0, 10))

    # sharpen
    if np.random.rand() >= 0.6:
        img = sharpness(img, np.random.randint(0, 10))

    # cutout
    if np.random.rand() >= 0.6:
        img = cutout(img, np.random.randint(0, 10))

    # convert Image object to numpy array
    img = np.array(img)
    img = img / 1.

    return img


# define train augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    vertical_flip=True,
    horizontal_flip=True,
    preprocessing_function=augment_image,
    fill_mode='reflect')

# copy train datagen
valid_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    vertical_flip=True,
    horizontal_flip=True,
    preprocessing_function=augment_image,
    fill_mode='reflect')

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=None,
    x_col='path',
    y_col='class',
    classes=classes,
    target_size=(img_size, img_size),
    interpolation='bicubic:fixed',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    validate_filenames=False)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory=None,
    x_col='path',
    y_col='class',
    classes=classes,
    target_size=(img_size, img_size),
    interpolation='bicubic:fixed',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    validate_filenames=False)

# show some images
experiment_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory=None,
    x_col='path',
    y_col='class',
    classes=classes,
    target_size=(img_size, img_size),
    interpolation='bicubic:fixed',
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    validate_filenames=False)

sns.set_style('white')
for i in range(20):
    batch = experiment_generator.next()
    image = np.copy(batch[0].squeeze())
    image = image * 255.
    image = image.astype('uint8')
    plt.imshow(image, vmin=0, vmax=255)
    plt.show()

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=35,
    shuffle=True,
    callbacks=[lr_scheduler],
    class_weight=None,
    validation_data=valid_generator,
    validation_steps=valid_steps,
    verbose=1)

# save model
model_out = model_path / f'efficient-net-{model_name}-baseline-{version}.h5'
model.save(str(model_out))
print(f'Saved model to: {str(model_out)}')

# plot metric scores
sns.set_style('whitegrid')
plt.plot(history.history['balanced_accuracy'])
plt.plot(history.history['val_balanced_accuracy'])
plt.title('Model Metric')
plt.ylabel('Balanced Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# plot loss scores
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Categorical Cross-entropy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


# prediction on test set
file_list = sorted([f for f in listdir(test_images) if (isfile(join(test_images, f)) and (f.endswith('.jpg')))])
df_test = pd.DataFrame(columns=['image', 'path'])
df_test['image'] = file_list
df_test['path'] = str(test_images) + '\\' + df_test['image']
df_test['class'] = ['N.A.' for i in range(len(df_test))]

test_generator = train_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="path",
    y_col='class',
    target_size=(img_size, img_size),
    interpolation='bicubic:fixed',
    batch_size=batch_size,
    class_mode='categorical')

# make submission
label_map = valid_generator.class_indices
file_names = test_generator.filenames
num_samples = len(file_names)
num_steps = int(np.ceil(num_samples / batch_size))
y_pred_list = []

# make multiple predictions
num_iterations = 25
for i, pred in enumerate(range(num_iterations)):
    y_pred = model.predict_generator(test_generator, steps=num_steps)
    y_pred_list.append(y_pred)
    print(f'Finished predicting {str(i + 1).zfill(2)}/{str(num_iterations).zfill(2)}')

y_pred_mean = np.sum(y_pred_list, axis=0)
y_pred_mean = y_pred_mean / len(y_pred_list)
predictions = pd.DataFrame(y_pred_mean, columns=label_map.keys())
predictions = predictions[classes]

images = [f.rstrip('.jpg') for f in file_list]
df_image = pd.DataFrame(file_list, columns=['image'])
df_preds = pd.DataFrame(predictions, columns=classes)
submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
submission.to_csv(str(submission_path) + f'\\efficient-net-{model_name}-baseline-{version}-450.csv', index=False)


# prediction on test set
file_list = sorted([f for f in listdir(test_images) if (isfile(join(test_images, f)) and (f.endswith('.jpg')))])
df_test = pd.DataFrame(columns=['image', 'path'])
df_test['image'] = file_list
df_test['path'] = str(test_images) + '\\' + df_test['image']
df_test['class'] = ['N.A.' for i in range(len(df_test))]

no_aug_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=True,
    horizontal_flip=True)

no_aug_generator = no_aug_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="path",
    y_col='class',
    target_size=(img_size, img_size),
    interpolation='bicubic:fixed',
    shuffle=False,
    class_mode='categorical',
    batch_size=batch_size)

# make submission
label_map = valid_generator.class_indices
file_names = no_aug_generator.filenames
num_samples = len(file_names)
num_steps = int(np.ceil(num_samples / batch_size))
y_pred_list = []

# make multiple predictions
num_iterations = 5
for i, pred in enumerate(range(num_iterations)):
    y_pred = model.predict_generator(test_generator, steps=num_steps)
    y_pred_list.append(y_pred)
    print(f'Finished predicting {str(i + 1).zfill(2)}/{str(num_iterations).zfill(2)}')

y_pred_mean = np.sum(y_pred_list, axis=0)
y_pred_mean = y_pred_mean / len(y_pred_list)
predictions = pd.DataFrame(y_pred_mean, columns=label_map.keys())
predictions = predictions[classes]

images = [f.rstrip('.jpg') for f in file_list]
df_image = pd.DataFrame(file_list, columns=['image'])
df_preds = pd.DataFrame(predictions, columns=classes)
submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
submission.to_csv(str(submission_path) + f'\\efficient-net-{model_name}-baseline-{version}-no-augment-450-crop.csv', index=False)