import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# cropping utils
from random_crop import load_and_crop_image
import keras_preprocessing.image
import tensorflow.keras.preprocessing.image

# augmentations
from archive.augment import contrast, color, sharpness, cutout
from PIL import Image

from os import listdir
from os.path import isfile, join
from pathlib import Path
from metric import balanced_accuracy, categorical_focal_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from ImageDataAugmentor.image_data_augmentor import *

print(device_lib.list_local_devices())



# models = [
#     'D:/Data/ISIC-2019/models/efficient-net-b3-baseline-v2-augment.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b3-baseline-v3-augment.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b3-baseline-v4-subset.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b3-baseline-v5-subset.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b4-baseline-v1-augment.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b4-baseline-v1-subset.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b4-baseline-v2-augment.h5',
#     'D:/Data/ISIC-2019/models/efficient-net-b4-baseline-v10-pseudo.h5']

models = [
    'D:/Data/ISIC-2019/models/efficient-net-b3-baseline-v1-augment.h5']

# input_sizes = [300, 300, 300, 300, 380, 380, 380, 380]
input_sizes = [300]

for model_path, img_size in zip(models, input_sizes):

    print(f'Predicting using model {model_path}')
    print(f'Image size is {img_size}')

    # settings
    model_name = model_path.split('/')[-1].split('-')[2]
    version = model_path.split('/')[-1].split('-')[4] + '-' + model_path.split('/')[-1].split('-')[5]
    version = version.split('.')[0]

    print(model_name, version)

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

    # specify input
    # img_size = 380  # 380
    channels = 3
    batch_size = 4

    # monkey patch
    keras_preprocessing.image.iterator.load_img = load_and_crop_image
    tensorflow.keras.preprocessing.image.load_img = load_and_crop_image

    # learning rate
    learning_rate = 1.0e-5  # smaller learning rate

    # # loading pre-trained EfficientNetB3
    # base = efn.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_size, img_size, channels))
    # base.trainable = True
    # base.summary()
    #
    # model = Sequential()
    # model.add(base)
    # model.add(GlobalAveragePooling2D(name='gap'))
    # model.add(Dropout(0.4, name='dropout_out'))
    # model.add(Dense(num_classes, activation='softmax'))
    # model.trainable = True


    # callbacks
    def drop_decay(epoch):
        drop = 0.5
        epochs_drop = 7.0
        return learning_rate * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))


    lr_scheduler = LearningRateScheduler(drop_decay, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1)

    model_weights = 'D:/Data/ISIC-2019/models/efficient-net-b4-baseline-v10-pseudo.h5'
    model = tensorflow.keras.models.load_model(model_weights, compile=False)

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

    # prediction on test set
    test_dir_450 = 'D:/Data/ISIC-2019/images_test_external_450'
    file_list = sorted([f for f in listdir(test_dir_450) if (isfile(join(test_dir_450, f)) and (f.endswith('.jpg')))])
    df_test = pd.DataFrame(columns=['image', 'path'])
    df_test['image'] = file_list
    df_test['path'] = str(test_dir_450) + '/' + df_test['image']
    df_test['class'] = ['N.A.' for i in range(len(df_test))]

    # show some test images
    test_generator = train_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col='path',
        y_col='class',
        classes=['N.A.'],
        target_size=(img_size, img_size),
        interpolation='bicubic:fixed',
        batch_size=1,
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False)

    sns.set_style('white')
    for i in range(5):
        batch = test_generator.next()
        image = np.copy(batch[0].squeeze())
        image = image * 255.
        image = image.astype('uint8')
        plt.imshow(image, vmin=0, vmax=255)
        plt.show()

    # show some test images
    test_generator = train_datagen.flow_from_dataframe(
        dataframe=df_test,
        directory=None,
        x_col='path',
        y_col='class',
        classes=['N.A.'],
        target_size=(img_size, img_size),
        interpolation='bicubic:fixed',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False)

    # # make submission
    # # label_map = valid_generator.class_indices
    # file_names = test_generator.filenames
    # num_samples = len(file_names)
    # num_steps = int(np.ceil(num_samples / batch_size))
    # y_pred_list = []

    # # make multiple predictions
    # num_iterations = 3
    # for i, pred in enumerate(range(num_iterations)):
    #     y_pred = model.predict_generator(test_generator, steps=num_steps)
    #     y_pred_list.append(y_pred)
    #     print(f'Finished predicting {str(i + 1).zfill(2)}/{str(num_iterations).zfill(2)}')
    #
    # y_pred_mean = np.sum(y_pred_list, axis=0)
    # y_pred_mean = y_pred_mean / len(y_pred_list)
    # # predictions = pd.DataFrame(y_pred_mean, columns=label_map.keys())
    # predictions = pd.DataFrame(y_pred_mean, columns=classes)
    # predictions = predictions[classes]

    # images = [f.rstrip('.jpg') for f in file_list]
    # df_image = pd.DataFrame(file_list, columns=['image'])
    # df_preds = pd.DataFrame(predictions, columns=classes)
    # submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
    # submission.to_csv(str(submission_path) + f'//efficient-net-{model_name}-baseline-{version}-450.csv', index=False)

    # prediction on test set
    file_list = sorted([f for f in listdir(test_dir_450) if (isfile(join(test_dir_450, f)) and (f.endswith('.jpg')))])
    df_test = pd.DataFrame(columns=['image', 'path'])
    df_test['image'] = file_list
    df_test['path'] = str(test_dir_450) + '//' + df_test['image']
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
        classes=['N.A.'],
        target_size=(img_size, img_size),
        interpolation='bicubic:fixed',
        shuffle=False,
        class_mode='categorical',
        batch_size=batch_size)

    # make submission
    # label_map = valid_generator.class_indices
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
    # predictions = pd.DataFrame(y_pred_mean, columns=label_map.keys())
    predictions = pd.DataFrame(y_pred_mean, columns=classes)
    predictions = predictions[classes]

    images = [f.rstrip('.jpg') for f in file_list]
    df_image = pd.DataFrame(file_list, columns=['image'])
    df_preds = pd.DataFrame(predictions, columns=classes)
    submission = pd.concat([df_image, df_preds], axis=1).reset_index(drop=True)
    submission.to_csv(str(submission_path) + f'//efficient-net-{model_name}-baseline-{version}-no-augment-450.csv', index=False)

