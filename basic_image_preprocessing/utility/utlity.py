import os
import cv2
import numpy
from basic_image_preprocessing.exception.custom_exception import CustomException
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_image(image_path: str, cmap: str) -> Tuple[numpy.ndarray, bool]:
    """
    load_image -> The definition will be used to load the image from the
    file path and return the loaded image to the calling definition.
    This is a generic definition which will be used across the project
    for loading the image from a file path

    :param image_path:
    :param cmap: Allowed value for cmap are 'gray' and 'rgb'
    :return numpy.ndarray:
    """
    is_color_image = False
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

        if cmap.lower() == 'rgb':
            is_color_image = True

        return image, is_color_image
    else:
        raise FileNotFoundError(f"File not found: {image_path}")


def plot_graph(original_image, processed_image, is_color_image_flag, pre_processing_method) -> None:
    """
    plot_graph -> A generic definition to plot the original image and the processed image side
    by side for comparison purposes

    :param original_image:
    :param processed_image:
    :param is_color_image_flag:
    :param pre_processing_method:
    :return None:
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
    :return bool:
    """
    if any(type(x) != int for x in channel):
        raise CustomException(f"channel parameter can take only integer values")

    if len(channel) > 3:
        raise CustomException(f"channel parameter excepts 1 - 3 values; but received {len(channel)}")

    if any(x > 2 for x in channel):
        raise CustomException(f"channel parameter can have values as 0, 1, 2. No other values are allowed")

    return True

def validate_non_linear(type : str):
    types_on_input = ['exp','log','power','negative']

    if type not in types_on_input:
        raise CustomException(f"input parameter can have value as  exp,log,power")
    if type in types_on_input:
        return True