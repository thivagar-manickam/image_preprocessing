import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt
from warnings import filterwarnings
import image_frequency_analysis as ifa
from basic_image_preprocessing.exception.custom_exception import CustomException
from basic_image_preprocessing.utility.utlity import validate_param_type, validate_wavelet_type

filterwarnings('ignore')


class FrequencyNoiseFiltering:
    def __init__(self, image_path: str):
        try:
            self.image_path = image_path

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.image_path}")

        except CustomException as ex:
            raise ex

    def fourier_transform(self, filter_radius: int, high_pass_filter: bool = False,
                          low_pass_filter: bool = False) -> np.ndarray:
        """
        Definition to perform the Fourier Transform algorithm on the image.
        In this method even though a color image path is mentioned, it will
        be read as a gray scale image, as the fourier transform can be done
        only on a gray scale image
        :param filter_radius: This defines the amount of the filtering to be done
        :param high_pass_filter: Boolean flag to enable the high frequency to be allowed
        :param low_pass_filter: Boolean flag to enable the low frequency to be allowed
        :return: numpy.ndarray -> image post applying the required fourier transform on the given image
        """
        try:
            validate_param_type('filter_radius', filter_radius, type(filter_radius), int)
            validate_param_type('high_pass_filter', high_pass_filter, type(high_pass_filter), bool)
            validate_param_type('low_pass_filter', low_pass_filter, type(low_pass_filter), bool)

            obj = ifa.FrequencyAnalysis(self.image_path, filter_radius, high_pass_filter,
                                        low_pass_filter)
            processed_image = obj.perform_image_frequency_analysis()

            return processed_image
        except ValueError as ex:
            raise ValueError(ex)

        except CustomException as ex:
            raise CustomException(ex)

        except Exception as ex:
            raise Exception(f"An error occurred while trying to apply the Fourier Transformation on "
                            f"the given image - {ex}")

    def wavelet_transform(self, wavelet_name: str = 'haar', level: int = 1, plot_output: bool = True):
        """
        This definition performs the wavelet transformation on the given image.
        Though a color image path is given, the image is read as a gray scale and
        then the transformation is performed on the gray scale image.
        :param wavelet_name: This the family name of the wavelet transformation type to be
            performed.
            Accepted Family name - ['db', 'sym', 'coif', 'bior', 'haar']
        :param level: This is an integer value which defines the level of decomposition in the image
        :param plot_output:This is a boolean value which will instruct the program whether to display the
                            images post pre-processing or not. Will throw value error if value other than the accepted
                            value passed.
                                Accepted values - True , False
        :return: numpy.ndarray -> image post applying the required wavelet transform on the given image
        """
        try:
            validate_param_type('plot_output', plot_output, type(plot_output), bool)
            validate_param_type('level', level, type(level), int)
            validate_param_type('wavelet_name', wavelet_name, type(wavelet_name), str)
            validate_wavelet_type(wavelet_name)

            image = cv2.imread(self.image_path, 0)

            titles = ['Approximation', ' Horizontal detail',
                      'Vertical detail', 'Diagonal detail']

            coeffs = pywt.dwt2(image, wavelet_name, mode='symmetric')
            cA, (cH, cV, cD) = coeffs

            fig = plt.figure(figsize=(12, 3))

            # Plot the original image and all the wavelet coefficients
            ax = fig.add_subplot(1, 5, 1)
            ax.imshow(image, interpolation="nearest", cmap='gray')
            ax.set_title('Original Image', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            for i, a in enumerate([cA, cH, cV, cD]):
                ax = fig.add_subplot(1, 5, i + 2)
                ax.imshow(a, interpolation="nearest", cmap='gray')
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()

        except ValueError as ex:
            raise ValueError(ex)

        except CustomException as ex:
            raise CustomException(ex)

        except Exception as ex:
            raise Exception(f"An error occurred while trying to apply the Wavelet Transformation on "
                            f"the given image - {ex}")
