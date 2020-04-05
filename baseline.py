import math
import numpy as np
import pandas as pd
import tensorflow as tf
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir
from os.path import isfile, join
from pathlib import Path
from metric import balanced_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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

class_weights_list = class_weight.compute_class_weight(
    'balanced', [i for i in range(num_classes)], y_train_int)
class_weights = {v: k for v, k in enumerate(class_weights_list)}

# specify input
img_size = 240
channels = 3
batch_size = 16
train_steps = math.ceil(len(df_train) / batch_size)
valid_steps = math.ceil(len(df_valid) / batch_size)

# learning rate
learning_rate = 1e-4

# loading pre-trained conv base model
base = efn.EfficientNetB1(weights="imagenet", include_top=False, input_shape=(img_size, img_size, channels))
dropout = 0.2
model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D(name="gap"))
model.add(Dropout(dropout, name="dropout_out"))
model.add(Dense(num_classes, activation="softmax"))
base.trainable = True


# callbacks
def scheduler(epoch):
    if epoch < 2:
        return learning_rate
    else:
        return learning_rate * 0.2


lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

# training
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=learning_rate),
    metrics=[balanced_accuracy(num_classes), 'accuracy'])

train_augment = ImageDataGenerator(
    rescale=1./255,
    rotation_range=360,
    vertical_flip=True,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=120.0,
    fill_mode='nearest')

valid_augment = ImageDataGenerator(
    rescale=1./255)

train_generator = train_augment.flow_from_dataframe(
    dataframe=df_train,
    directory=None,
    x_col="path",
    y_col='class',
    classes=classes[:-1],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    validate_filenames=True)

valid_generator = valid_augment.flow_from_dataframe(
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
    epochs=16,
    callbacks=[early_stopping],
    class_weight=class_weights,
    validation_data=valid_generator,
    validation_steps=valid_steps,
    verbose=2)

# save model
model_out = model_path / 'efficient-net-b1-baseline.h5'
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

test_generator = valid_augment.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="path",
    y_col='class',
    classes=classes[:-1],
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode='categorical',
    validate_filenames=True)

file_names = test_generator.filenames
num_samples = len(file_names)
predictions = model.predict(test_generator, steps=num_samples)

images = [f.rstrip('.jpg') for f in file_list]
df_image = pd.DataFrame(file_list, columns=['image'])
df_preds = pd.DataFrame(predictions, columns=classes[:-1])
submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
submission['UNK'] = 0.0
submission['UNK'] = submission['UNK'].astype(float)

# class_prob = submission[classes].values
# class_max = np.max(class_prob, axis=1)
# class_unk = np.where(class_max < 0.5, 1.0, 0.0)
# submission['UNK'] = class_unk
# submission['MEL'] = np.where(submission['UNK'] == 1.0, 0.0, submission['MEL'])
# submission['NV'] = np.where(submission['UNK'] == 1.0, 0.0, submission['NV'])
# submission['BCC'] = np.where(submission['UNK'] == 1.0, 0.0, submission['BCC'])
# submission['AK'] = np.where(submission['UNK'] == 1.0, 0.0, submission['AK'])
# submission['BKL'] = np.where(submission['UNK'] == 1.0, 0.0, submission['BKL'])
# submission['DF'] = np.where(submission['UNK'] == 1.0, 0.0, submission['DF'])
# submission['VASC'] = np.where(submission['UNK'] == 1.0, 0.0, submission['VASC'])
# submission['SCC'] = np.where(submission['UNK'] == 1.0, 0.0, submission['SCC'])

submission.to_csv(str(submission_path) + '\\efficient-net-b1-baseline-2.csv', index=False)
