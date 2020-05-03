import math
import numpy as np
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

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
version = 'v1'

# plotting settings
sns.set_style('whitegrid')
sns.set_context('notebook')

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

# format training labels
train_labels = pd.read_csv(train_labels)
train_labels['class'] = train_labels[classes].idxmax(axis=1)
train_labels['path'] = str(train_images) + '\\' + train_labels['image'] + '.jpg'

# validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    train_labels[['image', 'path', 'class']], train_labels[classes[:-1]],
    stratify=train_labels['class'].values, shuffle=True, test_size=0.2, random_state=42)
df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
df_valid = pd.concat([X_valid, y_valid], axis=1).reset_index(drop=True)
y_train, y_valid = y_train.values, y_valid.values
y_train_int, y_valid_int = np.argmax(y_train, axis=1), np.argmax(y_valid, axis=1)

# calculate class weights
class_weights_list = class_weight.compute_class_weight(
    'balanced', [i for i in range(num_classes)], y_train_int)
class_weights = {v: k for v, k in enumerate(class_weights_list)}

# specify input
img_size = 380
channels = 3
batch_size = 4
train_steps = math.ceil(len(df_train) / batch_size)
valid_steps = math.ceil(len(df_valid) / batch_size)

# learning rate
learning_rate = 1e-4

# loading pre-trained EfficientNetB4
base = efn.EfficientNetB4(weights="imagenet", include_top=False, input_shape=(img_size, img_size, channels))
model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D(name="gap"))
model.add(Dropout(0.3, name="dropout_out"))
model.add(Dense(num_classes, activation="softmax"))
base.trainable = True


# callbacks
def drop_decay(epoch):
    drop = 0.5
    epochs_drop = 5.0
    return learning_rate * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))


def exp_decay(epoch):
    if epoch < 10:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(0.1 * (10 - epoch))


lr_scheduler = LearningRateScheduler(drop_decay, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

# compilation
model.compile(
    loss=categorical_focal_loss(gamma=2.0, alpha=0.25),
    optimizer=Adam(lr=learning_rate),
    metrics=[balanced_accuracy(num_classes), 'accuracy'])

# augmentations
augmentations_light = A.Compose([
    # color adjustment
    A.RandomBrightnessContrast(p=0.4),

    # rotations and flips
    A.RandomRotate90(p=0.6),
    A.VerticalFlip(p=0.6),

    # miscellaneous
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, approximate=False, p=0.2),
])

augmentations_heavy = A.Compose([
    # resize for constant augmentations
    A.Resize(height=600, width=600, p=1.0),

    # color adjustment
    A.RandomBrightnessContrast(p=0.25),
    A.RandomGamma(p=0.25),
    A.CLAHE(p=0.25),

    # rotations and flips
    A.RandomRotate90(p=0.75),
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
    ], p=1.0),

    # miscellaneous
    A.RandomCrop(height=500, width=500, p=0.2),
    A.CoarseDropout(max_holes=25, max_height=10, max_width=10, min_holes=5, min_height=5, min_width=5, p=0.2),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, approximate=False, p=0.2),

    # resize the model input size
    A.Resize(height=img_size, width=img_size, p=1.0),
])

train_datagen = ImageDataAugmentor(
    rescale=1.0/255,
    augment=augmentations_light,
    preprocess_input=None)



valid_datagen = ImageDataAugmentor(rescale=1.0/255)

test_datagen = ImageDataAugmentor(
    rescale=1.0/255,
    augment=augmentations_light,
    preprocess_input=None)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=None,
    x_col="path",
    y_col='class',
    classes=classes[:-1],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    validate_filenames=True)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=df_valid,
    directory=None,
    x_col="path",
    y_col='class',
    classes=classes[:-1],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    validate_filenames=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=30,
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

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="path",
    y_col='class',
    target_size=(img_size, img_size),
    shuffle=False,
    class_mode='categorical',
    batch_size=1)

# make submission
label_map = valid_generator.class_indices
file_names = test_generator.filenames
num_samples = len(file_names)
y_pred_list = []

# make multiple predictions
num_iterations = 10
for i, pred in enumerate(range(num_iterations)):
    y_pred = model.predict_generator(test_generator, steps=num_samples)
    y_pred_list.append(y_pred)
    print(f'Finished predicting {str(i).zfill(2)}/{str(num_iterations).zfill(2)}')

y_pred_mean = np.sum(y_pred_list, axis=0)
y_pred_mean = y_pred_mean / len(y_pred_list)
predictions = pd.DataFrame(y_pred_mean, columns=label_map.keys())
predictions = predictions[classes[:-1]]

images = [f.rstrip('.jpg') for f in file_list]
df_image = pd.DataFrame(file_list, columns=['image'])
df_preds = pd.DataFrame(predictions, columns=classes[:-1])
submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
submission['UNK'] = 0.0
submission['UNK'] = submission['UNK'].astype(float)
submission.to_csv(str(submission_path) + f'\\efficient-net-{model_name}-baseline-{version}.csv', index=False)
