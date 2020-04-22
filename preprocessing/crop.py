import cv2
import glob
import numpy as np
from pathlib import Path

#%%
crop_dir = 'D://Data//ISIC-2019//cropped//'
images = glob.glob(crop_dir + '*.jpg')


def crop_image_from_gray(img, tolerance=10, threshold=0.1, ratio=0.9):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """
    false_count = 0

    if img.ndim == 2:
        mask = img > tolerance
        return img[np.ix_(mask.any(1), mask.any(0))], false_count

    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tolerance
        false_count = (np.size(mask) - np.count_nonzero(mask)) / mask.size
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if check_shape == 0:
            return img, false_count
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)

            if false_count >= threshold:
                height, width, depth = img.shape
                h = int(np.ceil(height - (height * ratio) / 2))
                w = int(np.ceil(width - (width * ratio) / 2))
                # img = img[h-h_dist:h+h_dist, w-width:w+w_dist]
                img = img[h:-h, w:-w]

        return img, false_count

    else:
        raise ValueError('Number dimensions is not correct.')


def crop_image(img, tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """

    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


# skin lesion categories
classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
for label in ['MEL', 'NV', 'BCC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']:
    data_path = Path('D:/Data/ISIC-2019/images_train_external/')
    sub_dir = data_path / label
    sub_images = [str(p) for p in sub_dir.rglob('*.jpg')]
    for image in sub_images:
        img = cv2.imread(image)
        height, width, depth = img.shape
        crop = crop_image(img, tol=30)
        cv2.imwrite(image, crop)

#%%
# skin lesion categories
data_path = Path('D:/Data/ISIC-2019/images_test_external/')
sub_images = [str(p) for p in data_path.rglob('*.jpg')]
for image in sub_images:
    img = cv2.imread(image)
    height, width, depth = img.shape
    crop = crop_image(img, tol=30)
    cv2.imwrite(image, crop)


#%%
def center_crop(path, ratio=0.8):
    img = cv2.imread(path)
    height, width, depth = img.shape

    # center of the image
    h = int(np.floor(height / 2))
    w = int(np.floor(width / 2))

    # shift to border
    y = int(np.floor(h * ratio))
    x = int(np.floor(w * ratio))

    # crop the image
    crop = img[h-y:h+y, w-x:w+x]

    return crop


# crop unknown images
data_path = Path('D:/Data/ISIC-2019/images_train_external_SD198/UNK_selected_cropped_40')
sub_images = [str(p) for p in data_path.rglob('*.jpg')]

for image in sub_images:
    crop = center_crop(image, ratio=0.4)
    cv2.imwrite(image, crop)


#%%
def resize_with_aspect_ratio(path, min_size=450):
    img = cv2.imread(path)
    height, width, depth = img.shape

    if height < width:
        ratio = width / height
        new_height = min_size
        new_width = int(round(min_size * ratio, 0))
    else:
        ratio = height / width
        new_height = int(round(min_size * ratio, 0))
        new_width = min_size

    if min(height, width) > min_size:
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized


# crop unknown images
data_path = Path('D:/Data/ISIC-2019/images_train_external_600')
sub_images = sorted([str(p) for p in data_path.rglob('*.jpg')] + [str(p) for p in data_path.rglob('*.bmp')])

#%%
for e, image in enumerate(sub_images):
    resized = resize_with_aspect_ratio(image, min_size=450)
    cv2.imwrite(image, resized)

    if ((e + 1) % 1000 == 0) or (e + 1 == len(sub_images)):
        print(f'Resized {e + 1} of {len(sub_images)} images')


#%%

# crop unknown images
data_path = Path('D:/Data/ISIC-2019/images_test_external_450')
sub_images = sorted([str(p) for p in data_path.rglob('*.jpg')] + [str(p) for p in data_path.rglob('*.bmp')])

for e, image in enumerate(sub_images):
    resized = resize_with_aspect_ratio(image, min_size=450)
    cv2.imwrite(image, resized)

    if ((e + 1) % 1000 == 0) or (e + 1 == len(sub_images)):
        print(f'Resized {e + 1} of {len(sub_images)} images')
