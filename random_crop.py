import random
import numpy as np
import keras_preprocessing.image


def load_and_crop_image(path, grayscale=False, color_mode='rgb', target_size=None, interpolation='bilinear'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of 'grayscale', 'rgb', 'rgba'. Default: 'rgb'.
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ':' where first part is interpolation and second is crop
            e.g. 'lanczos:random'.
            Supported interpolation methods are 'nearest', 'bilinear', 'bicubic', 'lanczos',
            'box', 'hamming' By default, 'nearest' is used.
            Supported crop methods are 'none', 'center', 'random', and 'fixed.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(':') if ':' in interpolation else (interpolation, 'none')

    if crop == 'none':
        return keras_preprocessing.image.utils.load_img(
            path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation)

    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(
        path, grayscale=grayscale, color_mode=color_mode, target_size=None,interpolation=interpolation)

    # Crop fraction of total image
    crop_fraction = 0.875
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:
        if img.size != (target_width, target_height):

            # get image dimensions
            width, height = img.size

            if crop not in ['center', 'random', 'fixed']:
                raise ValueError(f'Invalid crop method {crop} specified.')

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError('Invalid interpolation method specified.')

            if crop == 'center':
                left_corner = int(round(width / 2)) - int(round(target_width / 2))
                top_corner = int(round(height / 2)) - int(round(target_height / 2))
                img = img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))

            elif crop == 'random':
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                img = img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

            elif crop == 'fixed':
                crop_size = 405  # max size is 450
                x, y = img.size
                w_range = x - crop_size
                h_range = y - crop_size
                w = np.random.randint(low=0, high=w_range)
                h = np.random.randint(low=0, high=h_range)
                img = img.crop((w, h, w + crop_size, h + crop_size))

            else:
                pass

            # resize to target size
            resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize((target_width, target_height), resample=resample)
            return img

    return img

