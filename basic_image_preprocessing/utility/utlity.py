import os
import cv2
import numpy
from basic_image_preprocessing.exception.custom_exception import CustomException
import matplotlib.pyplot as plt
from typing import List


def load_image(image_path: str, cmap: str) -> numpy.ndarray:
    """
    load_image -> The definition will be used to load the image from the
    file path and return the loaded image to the calling definition.
    This is a generic definition which will be used across the project
    for loading the image from a file path

    :param image_path:
    :param cmap: Allowed value for cmap are 'gray' and 'rgb'
    :return numpy.ndarray:
    """
    if os.path.exists(image_path):
        if cmap.lower() == 'gray':
            image = cv2.imread(f"{image_path}", 0)

        elif cmap.lower() == 'rgb':
            image = cv2.imread(f"{image_path}", cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            raise CustomException("Invalid cmap value specified. cmap value can only be either gray or rgb")

        if image is None:
            raise CustomException("Unable to read the image. Please check the image path")
        return image
    else:
        raise FileNotFoundError(f"File not found: {image_path}")


def is_color_image(image) -> bool:
    """
    is_color_image -> Definition used to check whether the loaded image is
    a color image or gray scale image. Based on the return value the calling
    definition will apply the transformation on the single channel for the gray scale
    image and on all the 3 channels for a color image

    :param image:
    :return bool | str:
    """
    if len(image.shape) == 2:
        return False

    if len(image.shape) == 3:
        return True

    if len(image.shape) > 3:
        raise CustomException("The image has unexpected number of color channels. Please use only Color or gray scale "
                              "image")


def plot_graph(original_image, processed_image, is_color_image_flag, pre_processing_method) -> None:
    """
    plot_graph -> A generic definition to plot the original image and the processed image side
    by side for comparison purposes

    :param original_image:
    :param processed_image:
    :param is_color_image_flag:
    :param pre_processing_method:
    :return:
    """
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    if is_color_image_flag:
        plt.imshow(original_image)
    else:
        plt.imshow(original_image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title(f"Image post applying the {pre_processing_method} transformation")
    if is_color_image_flag:
        plt.imshow(processed_image)
    else:
        plt.imshow(processed_image, cmap="gray")


def validate_channel_param(channel: List[int]) -> bool:
    """
    Definition to validate if the given channel param satisfy all the required conditions
    :param channel:
    :return:
    """
    if len(channel) > 3:
        raise CustomException(f"channel parameter excepts 1 - 3 values; but received {len(channel)}")

    if any(x > 2 for x in channel):
        raise CustomException(f"channel parameter can have values as 0, 1, 2. No other values are allowed")

    if any(type(x) != int for x in channel):
        raise CustomException(f"channel parameter can take only integer values")

    return True
