import numpy as np
import cv2
from warnings import filterwarnings
from basic_image_preprocessing.exception.custom_exception import CustomException
from basic_image_preprocessing.utility.utlity import load_image, is_color_image, plot_graph, validate_channel_param
from typing import List

filterwarnings('ignore')


class TraditionalImageEnhancement:
    """
    This class contains the linear transformation methods
    that can be applied on the image.

    Input for Constructor:
    image_path -> image file path to read the image from
    cmap -> to know whether the image is a gray scale image or a color image.
        Accepted Values -> gray and rgb

    List of Linear Transformation methods available:
    1. linear_equation(slope: int | float, constant: int | float) -> np.ndarray | str
    """
    def __init__(self, image_path: str, cmap: str):
        try:
            self.image_path = image_path
            self.is_color_image = False

            self.image = load_image(self.image_path, cmap)
            self.is_color_image = is_color_image(self.image)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.image_path}")

        except CustomException as ex:
            raise ex

    def linear_equation(self, slope: int | float, constant: int | float, cmap: str = None,
                        channel: List[int] = None) -> np.ndarray:
        """
        This definition will apply the linear equation formula on the given image
        with the slope and constant value given in the parameters

        Input:
            slope -> slope value to be used for the linear equation
            constant -> constant value to be used for the linear equation
            cmap -> This value will denote on which plane the transformation needs to be applied on provided the cmap
                during object creation was mentioned as rgb
                Accepted value:
                    'gray' -> will apply the transformation on the gray scale image
                    'rgb' -> will apply the transformation on the channels given in 'channels' list. By default, apply
                            transformation on all the three channels
                    'hsv' -> will apply the transformation on the channels given in 'channels' list. By default, apply
                            transformation on the Value channel.
                    'lab' -> will apply the transformation on the channels given in 'channels' list. By default, apply
                            transformation on the Lightness channel.

                Default value -> None. Will default to the cmap value specified during the object creation

            channel -> Specify the channel index on which the transformation to be applied.
                Default value -> None
                Accepted value -> [0, 1, 2]
        Output:
            numpy.ndarray -> image post applying the linear equation formula on the given image
        """
        try:
            is_hsv = True if cmap.lower() == 'hsv' else False
            is_lab = True if cmap.lower() == 'lab' else False

            if not self.is_color_image:
                image = self.image.astype('float')
                image = np.clip(slope * image + constant, 0, 255).astype(np.uint8)
            else:
                image = self.image

                # Convert the rgb image to the required cmap color scale
                if is_hsv:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                if is_lab:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                # Apply the transformation on the default planes when the channel value is None
                if channel is None:
                    if is_hsv:
                        image[:, :, 2] = np.clip(slope * image[:, :, 2] + constant, 0, 255)

                    elif is_lab:
                        image[:, :, 0] = np.clip(slope * image[:, :, 0] + constant, 0, 255)

                    else:
                        image = np.clip(slope * image + constant, 0, 255)
                else:
                    # If the channel validation passes, then apply the transformation on the specified planes
                    if validate_channel_param(channel=channel):
                        for x in channel:
                            image[:, :, x] = np.clip(slope * image[:, :, x] + constant, 0, 255)

                if is_hsv:
                    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                if is_lab:
                    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

                image = image.astype(np.unit8)

            plot_graph(self.image, image, self.is_color_image, 'Linear Equation')
            return image

        except TypeError as ex:
            print(f"Type Error encountered in the method - {ex}. Please validate if proper value given for slope and "
                  f"constant")

        except ValueError as ex:
            print(ex)

        except CustomException as ex:
            print(ex)

        except:
            print("An error occurred while trying to apply the linear equation on the given image")
