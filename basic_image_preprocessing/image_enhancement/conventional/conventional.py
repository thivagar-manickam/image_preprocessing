import numpy as np
import cv2
from warnings import filterwarnings
from basic_image_preprocessing.exception.custom_exception import CustomException
from basic_image_preprocessing.utility.utlity import load_image, plot_graph, validate_channel_param, \
    validate_param_list_value
from typing import List, Union

filterwarnings('ignore')

class ConventionalImageEnhancement:
    def __init__(self, image_path: str, cmap: str):
        try:
            self.image_path = image_path
            self.is_color_image = False

            self.image, self.is_color_image = load_image(self.image_path, cmap)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.image_path}")

        except CustomException as ex:
            raise ex

    def Equalization_histogram(self, cmap: str = None,
                            plot_output: bool = True,
                            channel: List[int] = None) -> np.ndarray:
        try:

            is_hsv = True if cmap is not None and cmap.lower() == 'hsv' else False
            is_lab = True if cmap is not None and cmap.lower() == 'lab' else False


            if type(plot_output) is not bool:
                raise ValueError(
                    f"plot_output parameter takes only True or False boolean value. No other values allowed")

            if not self.is_color_image:
                image = self.image.astype('float')

                image = cv2.equalizeHist(self.image)
                return image

            else:
                image = self.image.copy()

                if is_hsv:
                    if channel is not None:
                        raise CustomException("Non - Linear equation can be applied only on the Value channel for a HSV"
                                              " type image. Remove the channel parameter")

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                elif is_lab:
                    if channel is not None:
                        raise CustomException("Non - Linear equation can be applied only of the Lightness channel for a"
                                              " LAB type image. Remove the channel parameter")

                    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                if channel is None:
                    if is_hsv:
                        image[:, :, 2] = np.clip(cv2.equalizeHist(image[:, :, 2]), 0, 255)
                    elif is_lab:
                        image[:, :, 0] = np.clip(cv2.equalizeHist(image[:, :, 0]), 0, 255)
                    else:
                        image[:, :, 0] = np.clip(cv2.equalizeHist(image[:, :, 0]), 0, 255)
                        image[:, :, 1] = np.clip(cv2.equalizeHist(image[:, :, 1]), 0, 255)
                        image[:, :, 2] = np.clip(cv2.equalizeHist(image[:, :, 2]), 0, 255)
                        eq_image = image

                else:
                 #If the channel validation passes, then apply the transformation on the specified planes
                    if validate_channel_param(channel=channel):
                        for x in channel:
                            image[:, :, x] = np.clip(cv2.equalizeHist(image[:, :, x]), 0, 255)

                if is_hsv:
                    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                if is_lab:
                    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

                image = image.astype(np.uint8)

                if plot_output:
                    plot_graph(self.image, image, self.is_color_image, f'Equalization Histogram')

            return image

        except ValueError as ex:
            raise ValueError(ex)

        except CustomException as ex:
            raise CustomException(ex)

        except Exception as ex:
            raise Exception(f"An error occurred while trying to apply the Equalization Histogram on the given image - {ex}")
