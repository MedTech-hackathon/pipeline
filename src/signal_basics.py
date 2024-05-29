from src.mathematics import Mathematics
from scipy.signal import correlate2d

import numpy as np


class SignalBasics:
          
    C3_SPEED_OF_SOUND = 1536.8852459016393      
    L15_SPEED_OF_SOUND = 1542.9831006612785     
          
    ###################################################################################
    @staticmethod
    def create_time(
                    signal_1d_size: np.ndarray,
                    sampling_rate_Hz,
                    delay_samples
                    ) -> np.ndarray:

        # Get the sampling rate
        delta_time_second = 1 / sampling_rate_Hz
        
        # Get the number of samples
        number_of_samples = signal_1d_size + delay_samples
        
        # Calculate the total time of recording for each frame
        time_of_recording_for_each_frame = delta_time_second * number_of_samples
        
        # Create a time array
        time_second_1d = np.linspace(0, time_of_recording_for_each_frame, number_of_samples)
        
        return time_second_1d    
    
    ###################################################################################
    @staticmethod
    def create_depth(
                    signal_1d_size: int,
                    delay_samples, 
                    imaging_depth_mm
                    ) -> np.ndarray:
 
        # Get the number of samples
        number_of_samples = signal_1d_size + delay_samples
    
        depth_mm_1d = np.linspace(0, imaging_depth_mm, number_of_samples)
        
        return depth_mm_1d
    
    ###################################################################################
    @classmethod
    def create_attenuation_linear(cls,
                    signal_1d_size: int,
                    imaging_depth_mm: float, 
                    depth_ac_list: list[float],
                    attenuation_ac_list: list[float]
                    ) -> list[float]:
 
        
        depth_array_1d = cls.create_depth(signal_1d_size    = signal_1d_size,
                                           delay_samples    = 0,
                                           imaging_depth_mm = imaging_depth_mm) 
        
        attenuation_array_1d = [0] * len(depth_array_1d)


        for index_1 in range(len(depth_array_1d)):
            
            for index_2 in range(len(depth_ac_list)):
                
                # print("index_1 = ", index_1)
                # print("index_2 = ", index_2)
                # print("depth_array_1d[index_1] = ", depth_array_1d[index_1])
                # print("depth_ac_list[index_2] = ", depth_ac_list[index_2])

                if  depth_array_1d[index_1] < depth_ac_list[index_2] :
                    
                    if index_2 == 0:
                        
                        x_1 = 0
                        y_1 = 0
                        x_2 = depth_ac_list[index_2]
                        y_2 = attenuation_ac_list[index_2]
                        
                    elif  0 < index_2 < len(depth_ac_list) - 2:
                        
                        x_1 = depth_ac_list[index_2]
                        y_1 = attenuation_ac_list[index_2]
                        x_2 = depth_ac_list[index_2 + 1]
                        y_2 = attenuation_ac_list[index_2 + 1]
                            
                    elif index_2 == len(depth_ac_list) - 1:
                        
                        x_1 = depth_ac_list[index_2 - 1]
                        y_1 = attenuation_ac_list[index_2 - 1]
                        x_2 = imaging_depth_mm
                        y_2 = attenuation_ac_list[index_2]
                             

                    attenuation_array_1d[index_1] = Mathematics.linear_function(
                                                        x_1 = x_1,
                                                        y_1 = y_1,
                                                        x_2 = x_2,
                                                        y_2 = y_2,
                                                        input_x = depth_array_1d[index_1])
                    
                    break
                
            else:
                # If depth_array_1d[index_2] is not less than any depth in depth_ac_list
                # Set attenuation to the last attenuation value in attenuation_ac_list
                attenuation_array_1d[index_1] = attenuation_ac_list[-1]
        
        # print(depth_ac_list)
        # print(attenuation_ac_list)
        # print(attenuation_array_1d)
        
        # import matplotlib.pyplot as plt
        # plt.plot(depth_array_1d, attenuation_array_1d)
        # plt.xlabel('depth [mm]')
        # plt.ylabel('ratio')
        # plt.grid()
        
        return attenuation_array_1d     
    
    ###################################################################################
    @classmethod
    def create_attenuation_linear_2(cls,
                    signal_1d_size: int,
                    imaging_depth_mm: float, 
                    depth_ac_list: list[float],
                    attenuation_ac_list: list[float]
                    ) -> list[float]:
 
        
        depth_array_1d = cls.create_depth(signal_1d_size    = signal_1d_size,
                                           delay_samples    = 0,
                                           imaging_depth_mm = imaging_depth_mm) 
        
        attenuation_array_1d = [0] * len(depth_array_1d)

        # depth from yaml file # [0 .... 150]
        for index_1 in range(len(depth_array_1d)):
            
            # depth from created array # [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5, 112.5, 127.5, 142.5]
            for index_2 in range(len(depth_ac_list)):
                
                # if depth from created array is less that elements from array
                if depth_array_1d[index_1] < depth_ac_list[index_2]:
                    
                    if index_2 == 0:             
                        
                        attenuation_array_1d[index_1] = attenuation_ac_list[index_2]
                        break
                                            
                    elif  0 < index_2 < len(depth_ac_list):
                        
                        x_1 = depth_ac_list[index_2 - 1]
                        y_1 = attenuation_ac_list[index_2 - 1]
                        x_2 = depth_ac_list[index_2 ]
                        y_2 = attenuation_ac_list[index_2]
                        
                        
                    # elif index_2 == len(depth_ac_list) - 2:
                        
                    #     x_1 = depth_ac_list[index_2 - 1]
                    #     y_1 = attenuation_ac_list[index_2 - 1]
                    #     x_2 = imaging_depth_mm
                    #     y_2 = attenuation_ac_list[index_2]
                             

                        attenuation_array_1d[index_1] = Mathematics.linear_function(
                                                            x_1 = x_1,
                                                            y_1 = y_1,
                                                            x_2 = x_2,
                                                            y_2 = y_2,
                                                            input_x = depth_array_1d[index_1])
                    
                    break
                
            else:
                # If depth_array_1d[index_2] is not less than any depth in depth_ac_list
                # Set attenuation to the last attenuation value in attenuation_ac_list
                attenuation_array_1d[index_1] = attenuation_ac_list[-1]
        
        # print(depth_ac_list)
        # print(attenuation_ac_list)
        #print(attenuation_array_1d)
        
        # import matplotlib.pyplot as plt
        # plt.plot(depth_array_1d, attenuation_array_1d)
        # plt.xlabel('depth [mm]')
        # plt.ylabel('ac [dB]')
        # plt.grid()
                
        return attenuation_array_1d     
    
    ###################################################################################
    @classmethod
    def create_attenuation_constant(cls,
                    signal_1d_size: np.ndarray,
                    imaging_depth_mm, 
                    depth_ac_list,
                    attenuation_ac_list
                    ) -> np.ndarray:
 
        depth_array_1d = cls.create_depth(signal_1d_size    = signal_1d_size,
                                           delay_samples    = 0,
                                           imaging_depth_mm = imaging_depth_mm) 
        
        attenuation_array_1d = [0] * len(depth_array_1d)

        for index_1 in range(len(depth_array_1d)):
            
            for index_2 in range(len(depth_ac_list)):
                
                if depth_array_1d[index_1] < depth_ac_list[index_2]:
                    attenuation_array_1d[index_1] = attenuation_ac_list[index_2]
                    break
            else:
                # If depth_array_1d[index_2] is not less than any depth in depth_ac_list
                # Set attenuation to the last attenuation value in attenuation_ac_list
                attenuation_array_1d[index_1] = attenuation_ac_list[-1]

        # import matplotlib.pyplot as plt
        # plt.plot(depth_array_1d, attenuation_array_1d)
        # plt.xlabel('depth [mm]')
        # plt.ylabel('ac [dB]')
        # plt.grid()
        
        return attenuation_array_1d     
          
    ###################################################################################
    @staticmethod
    def calculate_sampling_rate(
                                time_second: np.ndarray) -> float:
        """
        Calculate the sampling rate based on the time array.

        Parameters:
        - time_array (np.ndarray): Time array containing the time points of the signal.

        Returns:
        - sampling_rate (float): The calculated sampling rate.
        """
        # Calculate the sampling rate based on the time array
        sampling_rate_Hz = 1 / (time_second[1] - time_second[0])
        
        return sampling_rate_Hz

    ###################################################################################
    @staticmethod
    def get_maximum_frequency(
                              freq_fft_Hz: np.ndarray,            # Frequency axis of the FFT spectrum (np.ndarray)
                              amp_fft : np.ndarray) -> float:  # Amplitude axis of the FFT spectrum (np.ndarray)
        """
        Calculate the maximum frequency value from the FFT spectrum.

        Parameters:
        - fft_frequency_axis (np.ndarray): Frequency axis of the FFT spectrum.
        - fft_amplitude_axis (np.ndarray): Amplitude axis of the FFT spectrum.

        Returns:
        - max_frequency_value (float): Maximum frequency value.
        """
        # Set zero frequency component to zero
        amp_fft[0] = 0
        
        # Find the index of the maximum amplitude in the FFT spectrum
        max_amplitude_index = np.argmax(np.abs(amp_fft))
        
        # Retrieve the frequency value corresponding to the maximum amplitude
        max_frequency_value_Hz = freq_fft_Hz[max_amplitude_index] 
        
        return max_frequency_value_Hz
    
    ###################################################################################
    @staticmethod
    def put_limit_on_envelope(
                              envelope: np.ndarray,             # Input envelope array (np.ndarray)
                              min_limit: float,                 # Minimum limit for envelope values (float)
                              max_limit: float) -> np.ndarray:  # Maximum limit for envelope values (float)
        """
        Limit the envelope values between the specified minimum and maximum limits.

        Parameters:
        - envelope (np.ndarray): Input envelope array.
        - min_limit (float): Minimum limit for envelope values.
        - max_limit (float): Maximum limit for envelope values.

        Returns:
        - envelope (np.ndarray): Envelope array with values limited between the specified limits.
        """
        # Limit the envelope values
        for i in range(0, envelope.shape[0]):
            for j in range(0, envelope.shape[1]):
                if envelope[i, j] < min_limit or envelope[i, j] > max_limit:
                    envelope[i, j] = 0
                    
        return envelope

    ###################################################################################
    @classmethod
    def calulate_speed_of_sound(cls,
                                signal_1d_size,
                                sampling_rate_Hz,
                                delay_samples,
                                imaging_depth_mm
                                ):
        
        time_1d = cls.create_time(signal_1d_size, sampling_rate_Hz, delay_samples)
        depth_1d = cls.create_depth(signal_1d_size, delay_samples, imaging_depth_mm) 
        speed_of_sound = depth_1d[len(depth_1d) - 1] / time_1d[len(time_1d) - 1]
        
        return speed_of_sound

    ###################################################################################
    @classmethod
    def find_signal_position_2d(cls,
                                main_signal_2d,
                                sub_signal_2d,
                                ):
        
        # Compute cross-correlation
        correlation = correlate2d(main_signal_2d, sub_signal_2d, mode='valid')
        
        # Find the positions of maximum correlation (exact)
        max_corr_indices = np.argwhere(correlation == np.max(correlation))
        result = max_corr_indices[0][0]

        # Find the position with the highest correlation
        #max_corr_index = np.unravel_index(np.argmax(correlation), correlation.shape)
        #result = max_corr_index[0]
              
        return result

    ###################################################################################




