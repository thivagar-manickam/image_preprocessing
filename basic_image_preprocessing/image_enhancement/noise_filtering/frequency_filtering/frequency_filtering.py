import numpy as np
from warnings import filterwarnings
import image_frequency_analysis as ifa
from basic_image_preprocessing.exception.custom_exception import CustomException
from basic_image_preprocessing.utility.utlity import validate_param_type

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
