import cv2
import glob
import numpy as np
from pathlib import Path

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


# for idx in range(len(images)):
#     img = cv2.imread(images[idx])
#     img = cv2.resize(img, (600, 600))
#     # crop, count = crop_image_from_gray(img, tolerance=10, threshold=0.9, ratio=0.9)
#     crop = crop_image(img, tol=30)
#     crop = cv2.resize(crop, (600, 600))
#     plt.imshow(np.hstack((img, crop)))
#     plt.title(f'Image {idx+1}')
#     plt.show()

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
