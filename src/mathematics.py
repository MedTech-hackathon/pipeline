import numpy as np
import math
from typing import Tuple, Callable


class Mathematics:

    ######################################################################################
    @staticmethod
    def shift_to_positive(
                          arr_xd):
        
        # Find the minimum element in the array
        min_val = np.min(arr_xd)
        
        # Calculate the shift value to make all elements positive
        shift_val = abs(min_val) if min_val < 0 else 0
        
        # Shift all elements by the shift value
        shifted_arr = arr_xd + shift_val
        
        return shifted_arr
    
    ######################################################################################
    @staticmethod
    def get_derivative(
                       function: Callable,  # The function to differentiate (Callable)
                       dt: float,           # Time step (float)
                       order: int           # Order of the derivative (int)
                       ) -> Callable:       # Return type: Callable
        """
        Compute the derivative of a function.

        Args:
        - function (Callable): The function to differentiate.
        - dt (float): Time step.
        - order (int): Order of the derivative.

        Returns:
        - Callable: The derivative function.
        """
        # Compute the derivative by applying numpy.gradient 'order' times
        for _ in range(order):
            function = np.gradient(function, dt)  # Compute derivative using numpy.gradient
            
        return function  # Return the derivative function

    ###################################################################################
    @staticmethod
    def min_max_mean_2d_array(
                              array: np.ndarray                 # Input 2D array (np.ndarray)
                              ) -> tuple[float, float, float]:  # Tuple of floats
        """
        Compute the minimum, maximum, and mean of a 2D array.

        Args:
        - array (np.ndarray): Input 2D array.

        Returns:
        - tuple[float, float, float]: Minimum, maximum, and mean values.
        """
        # Compute minimum, maximum, and mean
        minimum = np.min(np.apply_along_axis(np.min, axis=0, arr=array))
        maximum = np.max(np.apply_along_axis(np.max, axis=0, arr=array))
        mean = np.mean(np.apply_along_axis(np.mean, axis=0, arr=array))

        return minimum, maximum, mean  # Return tuple of minimum, maximum, and mean
    
    ###################################################################################
    @staticmethod
    def scale_down_array(
                         arr: np.ndarray   # Input numpy array to be scaled down (np.ndarray)
                         ) -> np.ndarray:  # Return type: Scaled down numpy array (np.ndarray)
        """
        Scales down the input numpy array by dividing it with the maximum absolute value
        among its elements.

        Parameters:
        - arr (np.ndarray): Input numpy array to be scaled down.

        Returns:
        - scaled_array (np.ndarray): Scaled down numpy array.
        """

        # Calculating the scaling factor as the maximum absolute value among the maximum and minimum values of the input array.
        scaling_factor = np.maximum(np.abs(np.max(arr)), np.abs(np.min(arr)))

        # Scaling down the input array by dividing it with the scaling factor.
        scaled_array = arr / scaling_factor

        return scaled_array

    ###################################################################################
    @staticmethod
    def normalize_with_min_max(
                   original_array: np.ndarray,   # Input numpy array to be normalized (np.ndarray)
                   new_min: int,                 # The minimum value of the new range (int)
                   new_max: int) -> np.ndarray:  # The maximum value of the new range (int)
        """
        Normalizes the values of the input numpy array to a new range defined by new_min and new_max.

        Parameters:
        - original_array (np.ndarray): Input numpy array to be normalized.
        - new_min (int): The minimum value of the new range.
        - new_max (int): The maximum value of the new range.

        Returns:
        - normalized_array (np.ndarray): Numpy array with values normalized to the new range.
        """

        # Finding the minimum and maximum values of the original array.
        min_value = np.min(original_array)
        max_value = np.max(original_array)

        # Normalizing the values of the original array to the new range.
        normalized_array = ((original_array - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min

        return normalized_array
    
    ###################################################################################
    @staticmethod
    def normalize_to_0_1(
                   original_array: np.ndarray,   # Input numpy array to be normalized (np.ndarray)
                   ) -> np.ndarray:  # The maximum value of the new range (int)

        # Normalizing the values of the original array to the new range.
        normalized_array = original_array / np.max(original_array)

        return normalized_array
    
    ###################################################################################
    @staticmethod
    def normalize_array(
                        minimum: float,  # Minimum value of the new range for normalization
                        maximum: float,  # Maximum value of the new range for normalization
                        arr: np.ndarray) -> np.ndarray:
        """
        Normalize the values of the input numpy array to a new range defined by minimum and maximum.

        Parameters:
        - minimum (float): The minimum value of the new range for normalization.
        - maximum (float): The maximum value of the new range for normalization.
        - arr (np.ndarray): Input numpy array to be normalized.

        Returns:
        - normalized_array (np.ndarray): Numpy array with values normalized to the new range.
        """

        # Finding the minimum and maximum values of the input array.
        arr_min = np.min(arr)
        arr_max = np.max(arr)

        # Defining the new minimum and maximum values for normalization.
        new_min = minimum
        new_max = maximum

        # Normalizing the values of the input array to the new range.
        normalized_array = (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

        return normalized_array
    
    ###################################################################################
    @classmethod
    def normalize_convolved_signal_2d(cls,
                                      original_signal_2d: np.ndarray,  # The original 2D signal before normalization (np.ndarray)
                                      convolved_signal_2d: np.ndarray  # The convolved 2D signal (np.ndarray)
                                      ) -> np.ndarray:  
        """
        Normalize the convolved 2D signal based on the normalization of each row individually.

        Parameters:
        - original_signal_2d (np.ndarray): The original 2D signal before normalization.
        - convolved_signal_2d (np.ndarray): The convolved 2D signal.

        Returns:
        - normalized_convolved_signal_2d (np.ndarray): Normalized convolved 2D signal.
        """

        # Loop through each row of the convolved signal to normalize them individually
        for i in range(convolved_signal_2d.shape[0]):
            convolved_signal_2d[i, :] = cls.normalize_convolved_signal_1d(original_signal_2d[i, :],
                                                                           convolved_signal_2d[i, :])

        normalized_convolved_signal_2d = convolved_signal_2d
        
        return normalized_convolved_signal_2d

    ###################################################################################
    @staticmethod
    def normalize_convolved_signal_1d(
                                      original_signal_1d: np.ndarray,                  # The original 1D signal (np.ndarray)
                                      convolved_signal_1d: np.ndarray) -> np.ndarray:  # The convolved 1D signal (np.ndarray)
        """
        Normalize the convolved 1D signal based on the range of the original 1D signal.

        Parameters:
        - original_signal_1d (np.ndarray): The original 1D signal.
        - convolved_signal_1d (np.ndarray): The convolved 1D signal to be normalized.

        Returns:
        - normalized_array (np.ndarray): Normalized convolved 1D signal.
        """
        # Finding the minimum and maximum values of the convolved signal
        arr_min = np.min(convolved_signal_1d)
        arr_max = np.max(convolved_signal_1d)

        # Finding the minimum and maximum values of the original signal
        new_min = np.min(original_signal_1d)
        new_max = np.max(original_signal_1d)

        # Normalizing the convolved signal based on the range of the original signal
        normalized_array = (convolved_signal_1d - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

        return normalized_array
        
    ###################################################################################
    @staticmethod
    def gaussian(
                 time: np.ndarray,             # Time values (np.ndarray)
                 mu: float,                    # Mean of the Gaussian distribution (float)
                 sigma: float) -> np.ndarray:  # Standard deviation of the Gaussian distribution (float)
        """
        Generate a Gaussian function.

        Parameters:
        - time (np.ndarray): Time values.
        - mu (float): Mean of the Gaussian distribution.
        - sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
        - gaussian (np.ndarray): Gaussian function.
        """
        gaussian = np.exp(-1 * ((time - mu)/sigma)**2)  # Calculate the Gaussian function
        
        return gaussian  # Return the Gaussian function

    ###################################################################################
    @staticmethod
    def hermite_poly(
                     order: int,                       # Order of the Hermite polynomial (int)
                     time: np.ndarray) -> np.ndarray:  # Time values (np.ndarray)
        """
        Generate a Hermite polynomial of a given order.

        Parameters:
        - order (int): Order of the Hermite polynomial.
        - time (np.ndarray): Time values.

        Returns:
        - hermite_poly (np.ndarray): Hermite polynomial evaluated at the given time values.
        """
        # Generate the Hermite polynomial of the given order
        hermite_poly = np.polynomial.hermite.Hermite.basis(order)
        
        # Evaluate the Hermite polynomial at the given time values
        return hermite_poly(time)
    
    ###################################################################################
    @classmethod
    def hermite_poly_eq(cls,
                        order: int,                   # Order of the Hermite polynomial (int)
                        time: np.ndarray,             # Time values (np.ndarray)
                        mu: float,                    # Mean of the Gaussian distribution (float)
                        sigma: float) -> np.ndarray:  # Standard deviation of the Gaussian distribution (float)
        """
        Generate the Hermite polynomial equation of a given order.

        Parameters:
        - order (int): Order of the Hermite polynomial.
        - time (np.ndarray): Time values.
        - mu (float): Mean of the Gaussian distribution.
        - sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
        - hermite_poly (np.ndarray): Hermite polynomial equation evaluated at the given time values.
        """
        # Calculate the Hermite polynomial equation using the formula
        hermite_poly = (-1)**order * np.exp(((time - mu)/sigma)**2) * \
                       cls.get_derivative(np.exp(-1 * ((time - mu)/sigma)**2), time[1] - time[0], order)
        
        return hermite_poly
    
    ###################################################################################
    # working without numerical problem
    @classmethod
    def gaussian_hermite(cls,
                         order: int,               # Order of the Hermite polynomial (int)
                         time: np.ndarray,         # Time values (np.ndarray)
                         sigma: float,             # Standard deviation of the Gaussian distribution (float)
                         mu: float,                # Mean of the Gaussian distribution (float)
                         normalization: bool = False) -> np.ndarray:  # Whether to normalize the result (bool)
        """
        Generate a Gaussian-Hermite function.

        Parameters:
        - order (int): Order of the Hermite polynomial.
        - time (np.ndarray): Time values.
        - sigma (float): Standard deviation of the Gaussian distribution.
        - mu (float): Mean of the Gaussian distribution.
        - normalization (bool, optional): Whether to normalize the result. Defaults to False.

        Returns:
        - gaussian_hermite (np.ndarray): Gaussian-Hermite function.
        """
        if normalization:
            # Compute the normalization factor
            normalization_factor = (2**order * math.factorial(order) * np.sqrt(np.pi)) ** -0.5 
            
            # Compute the normalized Gaussian-Hermite function
            gaussian_hermite = normalization_factor * cls.gaussian(time, mu, sigma) * cls.hermite_poly(order, (time - mu)/sigma)
        else:
            # Compute the Gaussian-Hermite function without normalization
            gaussian_hermite = cls.gaussian(time, mu, sigma) * cls.hermite_poly(order, time)
        
        return gaussian_hermite

    ###################################################################################
    # numerical error with this method
    @classmethod
    def gaussian_hermite_eq(cls,
                            order: int,               # Order of the Hermite polynomial (int)
                            time: np.ndarray,         # Time values (np.ndarray)
                            mu: float,                # Mean of the Gaussian distribution (float)
                            sigma: float,             # Standard deviation of the Gaussian distribution (float)
                            normalization = False) -> np.ndarray:  # Whether to normalize the result (bool)
        """
        Generate the equation for a Gaussian-Hermite function.

        Parameters:
        - order (int): Order of the Hermite polynomial.
        - time (np.ndarray): Time values.
        - mu (float): Mean of the Gaussian distribution.
        - sigma (float): Standard deviation of the Gaussian distribution.
        - normalization (bool, optional): Whether to normalize the result. Defaults to False.

        Returns:
        - gaussian_hermite_eq (np.ndarray): Equation for the Gaussian-Hermite function.
        """
        if normalization:
            # Compute the normalization factor
            normalization_factor = (2**order * math.factorial(order) * np.sqrt(np.pi))**-0.5 
            
            # Compute the normalized equation for the Gaussian-Hermite function
            gaussian_hermite_eq = normalization_factor * cls.gaussian(time, mu=mu, sigma=sigma) * cls.hermite_poly_eq(order, time, mu, sigma)
        else:
            # Compute the equation for the Gaussian-Hermite function without normalization
            gaussian_hermite_eq = cls.gaussian(time, mu=mu, sigma=sigma) * cls.hermite_poly_eq(order, time, mu, sigma)
        
        return gaussian_hermite_eq
        
    ###################################################################################
    @classmethod
    def gaussian_derivative(cls,
                            order: int,               # Order of the derivative (int)
                            time: np.ndarray,         # Time values (np.ndarray)
                            mu: float,                # Mean of the Gaussian distribution (float)
                            sigma: float,             # Standard deviation of the Gaussian distribution (float)
                            normalization = False) -> np.ndarray:  # Whether to normalize the result (bool)
        """
        Generate the derivative of a Gaussian function.

        Parameters:
        - order (int): Order of the derivative.
        - time (np.ndarray): Time values.
        - mu (float): Mean of the Gaussian distribution.
        - sigma (float): Standard deviation of the Gaussian distribution.
        - normalization (bool, optional): Whether to normalize the result. Defaults to False.

        Returns:
        - gaussian_derivative (np.ndarray): Derivative of the Gaussian function.
        """
        # Compute the Gaussian function
        gaussian_function = cls.gaussian(time, mu, sigma)
        
        # Compute the derivative of the Gaussian function
        gaussian_derivative = cls.get_derivative(gaussian_function, time[1] - time[0], order)
        
        if normalization:
            # Compute the normalization factor
            normalization_factor = (2**order * math.factorial(order) * np.sqrt(np.pi))**-0.5 
            
            # Compute the normalized derivative of the Gaussian function
            return normalization_factor * gaussian_derivative
        else:
            # Return the derivative of the Gaussian function without normalization
            return gaussian_derivative
    
    ###################################################################################
    @staticmethod
    def linear_function(
                        x_1: float,     # X-coordinate of the first point (float)
                        y_1: float,     # Y-coordinate of the first point (float)
                        x_2: float,     # X-coordinate of the second point (float)
                        y_2: float,     # Y-coordinate of the second point (float)
                        input_x: float) -> float:  # X-coordinate for interpolation (float)
        """
        Compute linear interpolation between two points.

        Parameters:
        - x_1 (float): X-coordinate of the first point.
        - y_1 (float): Y-coordinate of the first point.
        - x_2 (float): X-coordinate of the second point.
        - y_2 (float): Y-coordinate of the second point.
        - input_x (float): X-coordinate for interpolation.

        Returns:
        - output_y (float): Interpolated Y-coordinate.
        """
        # Calculate the slope between the two points
        slope = (y_2 - y_1) / (x_2 - x_1)
        
        # Use the slope to interpolate the Y-coordinate for the given X-coordinate
        output_y = y_1 + slope * (input_x - x_1)
        
        return output_y
    
    ###################################################################################
    @staticmethod
    def rotate_flip(
                    two_dimension_array: np.ndarray) -> np.ndarray:
        """
        Rotate a two-dimensional array counterclockwise by 90 degrees and then flip it horizontally.

        Parameters:
        - two_dimension_array (np.ndarray): Input two-dimensional array to be rotated and flipped.

        Returns:
        - rotated_flipped_array (np.ndarray): Rotated and flipped two-dimensional array.
        """
        # Rotate the input array counterclockwise by 90 degrees
        rotated_array = np.rot90(two_dimension_array, k=-1)
        
        # Flip the rotated array horizontally
        rotated_flipped_array = np.flip(rotated_array, axis=1)
        
        return rotated_flipped_array

    ###################################################################################
    @staticmethod
    def mask_3d(
             array_3d: np.ndarray,   # Input two-dimensional array (np.ndarray)
             v1: int,                # Top vertical coordinate of the mask (int)
             v2: int,                # Bottom vertical coordinate of the mask (int)
             h1: int,                # Left horizontal coordinate of the mask (int)
             h2: int,                # Right horizontal coordinate of the mask (int)
             inverse: bool = False) -> np.ndarray:  # Whether to apply inverse masking (bool)
        """
        Apply a rectangular mask to a two-dimensional array.

        Parameters:
        - array_2d (np.ndarray): Input two-dimensional array.
        - v1 (int): Top vertical coordinate of the mask.
        - v2 (int): Bottom vertical coordinate of the mask.
        - h1 (int): Left horizontal coordinate of the mask.
        - h2 (int): Right horizontal coordinate of the mask.
        - inverse (bool, optional): Whether to apply inverse masking. Defaults to False.

        Returns:
        - masked_data (np.ma.MaskedArray): Masked array.
        """
        # Create a mask with zeros, indicating areas to be masked
        mask = np.zeros_like(array_3d)
        
        # Set the mask to 1 within the specified rectangular region
        mask[v1:v2, h1:h2, :] = 1 
        
        # Apply the mask to the input array
        if inverse:
            # Mask where the mask is not 0 (inverse masking)
            masked_data = np.ma.masked_where(mask != 0, array_3d)
        else:
            # Mask where the mask is 1
            masked_data = np.ma.masked_where(mask == 1, array_3d)
            
        return masked_data
    
    ###################################################################################
    @staticmethod
    def mask_2d(
             array_2d: np.ndarray,   # Input two-dimensional array (np.ndarray)
             v1: int,                # Top vertical coordinate of the mask (int)
             v2: int,                # Bottom vertical coordinate of the mask (int)
             h1: int,                # Left horizontal coordinate of the mask (int)
             h2: int,                # Right horizontal coordinate of the mask (int)
             inverse: bool = False) -> np.ndarray:  # Whether to apply inverse masking (bool)
        """
        Apply a rectangular mask to a two-dimensional array.

        Parameters:
        - array_2d (np.ndarray): Input two-dimensional array.
        - v1 (int): Top vertical coordinate of the mask.
        - v2 (int): Bottom vertical coordinate of the mask.
        - h1 (int): Left horizontal coordinate of the mask.
        - h2 (int): Right horizontal coordinate of the mask.
        - inverse (bool, optional): Whether to apply inverse masking. Defaults to False.

        Returns:
        - masked_data (np.ma.MaskedArray): Masked array.
        """
        # Create a mask with zeros, indicating areas to be masked
        mask = np.zeros_like(array_2d)
        
        # Set the mask to 1 within the specified rectangular region
        mask[v1:v2, h1:h2] = 1 
        
        # Apply the mask to the input array
        if inverse:
            # Mask where the mask is not 0 (inverse masking)
            masked_data = np.ma.masked_where(mask != 0, array_2d)
        else:
            # Mask where the mask is 1
            masked_data = np.ma.masked_where(mask == 1, array_2d)
            
        return masked_data
        
    ###################################################################################
    @staticmethod
    def positive_negative_separator_xd(
                                    arr_xd: np.ndarray,
                                    limit=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the given array into two arrays containing only positive and negative values, respectively.

        Parameters:
        - arr (np.ndarray): Input array to be split.

        Returns:
        - positive_array (np.ndarray): Array containing only positive values.
        - negative_array (np.ndarray): Array containing only negative values.
        """
        # Initialize arrays to store positive and negative values
        positive_array = np.zeros_like(arr_xd)
        negative_array = np.zeros_like(arr_xd)
              
        # 2D array
        if len(arr_xd.shape) == 2:
            
            # Iterate over each element in the input array
            for i in range(arr_xd.shape[0]):
                for j in range(arr_xd.shape[1]):
                    if arr_xd[i, j] > limit:
                        # Assign positive values to the positive array
                        positive_array[i, j] = arr_xd[i, j]
                    if arr_xd[i, j] < -limit:
                        # Assign negative values to the negative array
                        negative_array[i, j] = arr_xd[i, j]
                        
        # 3D array
        if len(arr_xd.shape) == 3:
            
            for k in range(arr_xd.shape[2]):
                for i in range(arr_xd.shape[0]):
                    for j in range(arr_xd.shape[1]):
                        if arr_xd[i, j, k] > limit:
                            # Assign positive values to the positive array
                            positive_array[i, j, k] = arr_xd[i, j, k]
                        if arr_xd[i, j, k] < -limit:
                            # Assign negative values to the negative array
                            negative_array[i, j, k] = arr_xd[i, j, k]
                                      
                    
        return positive_array, negative_array
    
    ###################################################################################
    @staticmethod
    def range_float(
                    start, stop, step):
        
        if start > stop:
            num_iterations = int((start - stop) / step)
            values = [start - i * step for i in range(num_iterations + 1)]
        elif stop > start:
            num_iterations = int((stop - start) / step)
            values = [start + i * step for i in range(num_iterations + 1)]
            
        return values

    ###################################################################################
    @staticmethod
    def multiply_3darray(
                        array,
                        coeff,
                        axis):
        
        dim_0_length = array.shape[0]
        dim_1_length = array.shape[1]
        dim_2_length = array.shape[2]
        
        if axis == 0:
            for index_1 in range(dim_1_length):
                for index_2 in range(dim_2_length): 
                    array[:, index_1, index_2] *= coeff
                    
        if axis == 1:
            for index_0 in range(dim_0_length):
                for index_2 in range(dim_2_length): 
                    array[index_0, :, index_2] *= coeff
                    
        if axis == 2:
            for index_1 in range(dim_1_length):
                for index_2 in range(dim_2_length): 
                    array[:, index_1, index_2] *= coeff
        
        return array

    ###################################################################################