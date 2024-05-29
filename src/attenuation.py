from src.mathematics import Mathematics
from src.signal_basics import SignalBasics
from src.transformation import Transformation
import math
import numpy as np
from scipy.signal import stft, istft, hilbert
import matplotlib.pyplot as plt
    
class AttenuationCorrection:

    speed_of_sound: float = 1540              # [m/s] 
    attenuation_correction_alpha: float = 0.1 # [dB/cm.MHz]    # C3

    # attenuation_correction_alpha: float = 0.275/2  # [dB/cm.MHz]   # L15 

    ######################################################################################    
    @classmethod
    def deattenuation_function(cls, 
                               time_1d: np.ndarray,          
                               frequency_MHz: float = None,
                               ) -> np.ndarray:

        # Calculate the exponential attenuation function
        exp_function = np.exp(cls.attenuation_correction_alpha * frequency_MHz * time_1d * cls.speed_of_sound *100 / (20 * np.log10(math.e)))

        return exp_function
    
    ######################################################################################
    @classmethod
    def deattenuation_function_with_changing_alpha(cls, 
                            time: np.ndarray,              # 1D array representing time
                            alpha_at_1MHz,
                            desirable_frequency_MHz: float  # Desired frequency in Hz
                            ) -> np.ndarray:

        # Estimate the attenuation coefficient at the desired frequency
        alpha = cls.attenuation_alpha_estimator(alpha_at_1MHz, desirable_frequency_MHz)
        
        # Calculate the exponential attenuation function
        exp_function = np.exp(alpha * desirable_frequency_MHz * time * cls.speed_of_sound * 100 / (20 * np.log10(math.e)))
        
        return exp_function

    ######################################################################################
    @staticmethod
    def attenuation_alpha_estimator( 
                                    alpha_at_1MHz: float,     # Attenuation coefficient at 1 MHz
                                    alpha_at_xMHz: float) -> float:

        # Use a linear function to estimate the attenuation coefficient at the specified frequency
        output_alpha = Mathematics.linear_function(1, alpha_at_1MHz, 2, alpha_at_1MHz * 2, alpha_at_xMHz)
        
        return output_alpha

    ######################################################################################
    @staticmethod
    def signal_frequency_eliminator( 
                                    frequency_point_to_eliminate_0d: int,
                                    fft_freq,
                                    fft_amp) -> np.ndarray:

            
        indices_positive = np.where(fft_freq ==  frequency_point_to_eliminate_0d)
        indices_negative = np.where(fft_freq == -frequency_point_to_eliminate_0d)
        
        frequency_to_eliminate_positive_index = indices_positive[0][0]
        frequency_to_eliminate_negative_index = indices_negative[0][0]
        
        # print(frequency_to_eliminate_positive_index)
        # print(frequency_to_eliminate_negative_index)
        # print(fft_amp[frequency_to_eliminate_positive_index])
        # print(fft_amp[frequency_to_eliminate_negative_index])

        fft_amp[frequency_to_eliminate_positive_index] = 0 + 0j 
        fft_amp[frequency_to_eliminate_negative_index] = 0 + 0j 
            
        # Inverse Fourier Transform to obtain the signal in time domain
        remained_signal = np.fft.ifft(fft_amp)
        
        return remained_signal
    
    ######################################################################################
    @classmethod
    def generate_eliminated_signal(cls, 
                                    signal_1d: np.ndarray,      # 1D array representing signal
                                    frequency_point_to_eliminate_0d: int,
                                    fft_freq,
                                    fft_amp) -> np.ndarray:
   
   
        # Obtain the signal with the specific frequency component eliminated
        remained_signal = cls.signal_frequency_eliminator(frequency_point_to_eliminate_0d, fft_freq, fft_amp)
        
        # Generate the eliminated signal by subtracting the eliminated component from the original signal
        
        eliminated_signal = signal_1d - remained_signal
        
        return eliminated_signal
       
    ######################################################################################
    @classmethod
    def attenuation_corrector_1D_freqeuency_based(cls, 
                                                  signal_1d: np.ndarray,       # 1D array representing the original signal
                                                  time_1d
                                                  ) -> np.ndarray:
       

        # get frequnecy spectrum of signal
        fft_freq, fft_amp = Transformation.fourier_transform(signal_1d  = signal_1d,
                                                             time_1d    = time_1d,
                                                             correction = False)
    
        # Extract magnitudes and phases
        magnitudes = np.abs(fft_amp)
        phases = np.angle(fft_amp)

        final_signal_1d = np.zeros_like(signal_1d, dtype=np.complex128)
        
        for freq, mag, phase in zip(fft_freq, magnitudes, phases):
            
            deattenuation_array_1d = cls.deattenuation_function(time_1d=time_1d,
                                                                frequency_MHz = freq / 1e6)
                    
            signal_for_correction_1d = mag * np.exp(1j * phase) * np.exp(2j * np.pi * freq * time_1d)
            corrected_signal = deattenuation_array_1d * signal_for_correction_1d / len(signal_1d) 
            final_signal_1d += corrected_signal
                
        final_signal_1d = np.real(final_signal_1d)      
                
        return final_signal_1d

    ######################################################################################
    @classmethod
    def attenuation_corrector_2D_freqeuency_based(cls, 
                                                signal_2d: np.ndarray,       # 2D array representing the original signal
                                                time_1d) -> np.ndarray:

        # Apply attenuation correction along the specified axis
        signal_2d = np.apply_along_axis(func1d = cls.attenuation_corrector_1D_freqeuency_based,
                                        arr    = signal_2d,
                                        axis   = 1,
                                        time_1d = time_1d)

        return signal_2d
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_3D_freqeuency_based(cls, 
                                                signal_3d: np.ndarray,       # 2D array representing the original signal
                                                time_1d
                                                ) -> np.ndarray:

        final_signal_3d = np.zeros_like(signal_3d)

        for frame in range(signal_3d.shape[2]):
        
            signal_2d = signal_3d[:, :, frame]
            signal_2d = cls.attenuation_corrector_2D_freqeuency_based(signal_2d = signal_2d,
                                                                      time_1d = time_1d)
            
            final_signal_3d[:, :, frame] = signal_2d
        
        return final_signal_3d    
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_1D_stFFT_based(cls, 
                                        signal: np.ndarray,            # 1D array representing the original signal
                                        alpha_at_1MHz: float,          # Attenuation coefficient at 1 MHz
                                        n_time_points: int) -> np.ndarray:
        """
        Corrects the signal for attenuation based on short-time Fourier transform (SFT).

        Args:
            signal (np.ndarray): The original signal.
            alpha_at_1MHz (float): Attenuation coefficient at 1 MHz.
            n_time_points (int): Number of time points per segment.

        Returns:
            np.ndarray: The corrected signal.
        """
        # Generate the time vector
        #frequencies, times, Zxx = stft(signal, fs= self.sampling_frequency_rate, nperseg = 80, window='box')
        time = SignalBasics.create_time(signal)
        
        # Calculate the number of time segments
        n_time_segments = len(time) // n_time_points
        
        # Initialize an array to store the corrected signal
        main_corrected_signal = np.zeros(len(signal))

        # Iterate over each time segment
        for i in range(0, n_time_segments):
            # Extract the current time segment and corresponding signal segment
            time_segment = time[n_time_points*i : n_time_points*(i+1) - 1]
            signal_segment = signal[n_time_points*i : n_time_points*(i+1) - 1]
            
            # Perform Fourier transform to get frequency components
            fft_freq, fft_result = Transformation.fourier_transform(time_segment, signal_segment, plot_mode="full", correction="off")
            
            # Get the maximum frequency component
            max_freq_Hz = SignalBasics.get_maximum_frequency(fft_freq, fft_result)
            
            # Calculate the attenuation function based on the maximum frequency component
            attenuation_function = cls.deattenuation_function(time_segment, alpha_at_1MHz, max_freq_Hz)
            
            # Apply attenuation correction to the signal segment
            corrected_signal = attenuation_function * signal_segment
            
            # Store the corrected signal segment in the main corrected signal array
            main_corrected_signal[n_time_points*i : n_time_points*(i+1) - 1] = corrected_signal

        # Return the main corrected signal
        return main_corrected_signal
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_2D_stFFT_based(cls, 
                                            signal: np.ndarray,           # 2D array representing the original signal
                                            alpha_at_1MHz: float,         # Attenuation coefficient at 1 MHz
                                            n_time_points: int,           # Number of time points per segment
                                            axis: int) -> np.ndarray:     # Axis along which to apply the correction
        """
        Corrects the 2D signal for attenuation based on short-time Fourier transform (SFT).

        Args:
            signal (np.ndarray): The original 2D signal.
            alpha_at_1MHz (float): Attenuation coefficient at 1 MHz.
            n_time_points (int): Number of time points per segment.
            axis (int): Axis along which to apply the correction.

        Returns:
            np.ndarray: The corrected 2D signal.
        """
        # Apply the attenuation corrector 1D SFT-based function along the specified axis
        signal_2D = np.apply_along_axis(cls.attenuation_corrector_1D_stFFT_based, axis=axis, arr=signal, 
                                        alpha_at_1MHz=alpha_at_1MHz, n_time_points=n_time_points)
        
        # Return the corrected 2D signal
        return signal_2D
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_1D_wavelet_based(cls, 
                                                signal: np.ndarray,       # 1D array representing the original signal
                                                alpha_at_1MHz: float) -> np.ndarray:
        """
        Corrects the signal for attenuation based on frequency.

        Args:
            signal (np.ndarray): The original signal.
            alpha_at_1MHz (float): Attenuation coefficient at 1 MHz.

        Returns:
            np.ndarray: The corrected signal.
        """
        # Create time vector corresponding to the signal
        time = SignalBasics.create_time(signal)
        
        # Perform Fourier transform to obtain frequency domain information
        fft_freq, _ = Transformation.fourier_transform(time, signal, correction="off")
        
        # Initialize an array to store the corrected signal
        main_corrected_signal = np.zeros(len(signal), dtype=complex)

        # Iterate over frequency points to apply correction
        for frequency_point in range(1, int(len(fft_freq)/2) - 1):
            # Generate signal with the specified frequency eliminated
            signal_for_correction = cls.eliminated_signal_generator(time, signal, frequency_point) 
            
            # Compute attenuation function based on frequency
            attenuation_function = cls.deattenuation_function(time, alpha_at_1MHz, fft_freq[frequency_point])
            
            # Apply attenuation correction to the signal
            corrected_signal = attenuation_function * signal_for_correction
            
            # Accumulate corrected signal
            main_corrected_signal += corrected_signal
            
        # Return the corrected signal
        return main_corrected_signal
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_1D_with_alpha_estimator(cls,
                                                      signal_1d,
                                                      depth_cm,
                                                      fs,
                                                      window,
                                                      nperseg,
                                                      noverlap,
                                                      plot):
        
        # Perform STFT
        f_stft, t_stft, Zxx = stft(signal_1d, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        
        Zxx_corrected = np.zeros_like(Zxx)
        signal_corrected = np.zeros_like(signal_1d)
        
        for i in range(len(f_stft)):
            # Extract the STFT of the current frequency component
            Zxx_single_freq = np.zeros_like(Zxx)
            Zxx_single_freq[i, :] = Zxx[i, :]
            
            # Reconstruct the time-domain signal
            _, x_single_freq = istft(Zxx_single_freq, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            
            # Calculate the envelope of the time-domain signal using the Hilbert transform
            envelope = np.abs(hilbert(x_single_freq))
            
            # Adjust the length of the depth array and envelope
            min_length = min(len(depth_cm), len(envelope))
            depth_cm_adjusted = depth_cm[:min_length]
            envelope_adjusted = envelope[:min_length]
            
            # Calculate the logarithm of the envelope
            log_envelope = 20 * np.log10(1 + np.abs(envelope_adjusted))
            
            # Perform linear regression to estimate the slope of the line
            slope, intercept = np.polyfit(depth_cm_adjusted, log_envelope, deg=1)
            
            # Calculate the y-values of the slope line
            slope_line = slope * depth_cm_adjusted + intercept
            
            # Calculate the exponential function for attenuation correction
            exp_function = np.exp(-slope * depth_cm_adjusted / (20 * np.log10(math.e)))
            
            # Apply the exponential function to correct the attenuation
            x_single_freq_corrected = x_single_freq[:min_length] * exp_function
            
            # Pad or truncate the corrected signal to match the length of the original signal
            if len(x_single_freq_corrected) < len(signal_1d):
                signal_corrected[:len(x_single_freq_corrected)] += x_single_freq_corrected
            else:
                signal_corrected += x_single_freq_corrected[:len(signal_1d)]
            
            # Reconstruct the corrected STFT for this frequency component
            _, _, Zxx_corrected_single = stft(x_single_freq_corrected, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
            Zxx_corrected[i, :] = Zxx_corrected_single[i, :]
            
            if plot:
                # Calculate the envelope of the corrected time-domain signal using the Hilbert transform
                envelope_corrected = np.abs(hilbert(x_single_freq_corrected))
                
                # Calculate the logarithm of the corrected envelope
                log_envelope_corrected = 20 * np.log10(1 + np.abs(envelope_corrected))
                
                # Perform linear regression on the corrected envelope
                slope_corrected, intercept_corrected = np.polyfit(depth_cm_adjusted, log_envelope_corrected, deg=1)
                
                # Calculate the y-values of the corrected slope line
                slope_line_corrected = slope_corrected * depth_cm_adjusted + intercept_corrected
                
                # Plotting
                plt.figure(figsize=(10, 12))
                
                plt.subplot(5, 1, 1)
                plt.plot(depth_cm_adjusted, x_single_freq[:min_length], label=f'{f_stft[i]:.1f} Hz')
                plt.xlabel('Depth [cm]')
                plt.ylabel('Amplitude')
                plt.legend(loc='upper right')
                plt.title(f'Signal Corresponding to {f_stft[i]:.1f} Hz Frequency Component')
                
                plt.subplot(5, 1, 2)
                plt.plot(depth_cm_adjusted, log_envelope, color='red', label='Log Envelope')
                plt.plot(depth_cm_adjusted, slope_line, '--', color='green', label='Slope Line')
                plt.xlabel('Depth [cm]')
                plt.ylabel('Log Envelope [dB]')
                plt.legend(loc='upper right')
                plt.title(f'Logarithm of Envelope of Signal Corresponding to {f_stft[i]:.1f} Hz Frequency Component')
                plt.text(0.5, 0.95, f'Slope: {slope:.2f} dB/cm', transform=plt.gca().transAxes, ha='center', va='top', color='blue')
                
                plt.subplot(5, 1, 3)
                plt.plot(depth_cm_adjusted, exp_function)
                plt.xlabel('Depth [cm]')
                plt.ylabel('Exponential Function')
                plt.title('Exponential Function vs. Distance')
                plt.grid(True)
                
                plt.subplot(5, 1, 4)
                plt.plot(depth_cm_adjusted, x_single_freq_corrected, label=f'{f_stft[i]:.1f} Hz')
                plt.xlabel('Depth [cm]')
                plt.ylabel('Amplitude')
                plt.legend(loc='upper right')
                plt.title(f'Corrected Signal Corresponding to {f_stft[i]:.1f} Hz Frequency Component')
                
                plt.subplot(5, 1, 5)
                plt.plot(depth_cm_adjusted, log_envelope_corrected, color='red', label='Log Envelope')
                plt.plot(depth_cm_adjusted, slope_line_corrected, '--', color='green', label='Slope Line')
                plt.xlabel('Depth [cm]')
                plt.ylabel('Log Envelope [dB]')
                plt.legend(loc='upper right')
                plt.title(f'Logarithm of Corrected Envelope of Signal Corresponding to {f_stft[i]:.1f} Hz Frequency Component')
                plt.text(0.5, 0.95, f'Slope: {slope_corrected:.2f} dB/cm', transform=plt.gca().transAxes, ha='center', va='top', color='blue')
                
                plt.tight_layout()
                plt.show()
        
        return signal_corrected
    
    ######################################################################################
    @classmethod
    def attenuation_corrector_2D_with_alpha_estimator(cls, 
                                                signal_2d: np.ndarray,       # 2D array representing the original signal
                                                depth_cm,                    # 1D array representing depth
                                                fs,                          # Sampling frequency
                                                window,                      # Windowing function
                                                nperseg,                     # Length of each segment for STFT
                                                noverlap,                    # Overlap between segments for STFT
                                                plot):                 # Boolean indicating whether to plot


        # Apply the 1D attenuation correction method along the appropriate axis
        corrected_signal_2d = np.apply_along_axis(cls.attenuation_corrector_1D_with_alpha_estimator,
                                                    arr=signal_2d,
                                                    axis=1,
                                                    depth_cm=depth_cm,
                                                    fs=fs,
                                                    window=window,
                                                    nperseg=nperseg,
                                                    noverlap=noverlap,
                                                    plot=plot)

        return corrected_signal_2d

    ######################################################################################
    @classmethod
    def attenuation_corrector_3D_with_alpha_estimator(cls, 
                                                signal_3d: np.ndarray,       # 3D array representing the original signal
                                                depth_cm,                    # 1D array representing depth
                                                fs,                          # Sampling frequency
                                                window,                      # Windowing function
                                                nperseg,                     # Length of each segment for STFT
                                                noverlap,                    # Overlap between segments for STFT
                                                plot=False):                 # Boolean indicating whether to plot

        final_signal_3d = np.zeros_like(signal_3d)

        for frame in range(signal_3d.shape[2]):
        
            signal_2d = signal_3d[:, :, frame]
            corrected_signal_2d  = cls.attenuation_corrector_2D_with_alpha_estimator(signal_2d=signal_2d,
                                                                    depth_cm=depth_cm,
                                                                    fs=fs,
                                                                    window=window,
                                                                    nperseg=nperseg,
                                                                    noverlap=noverlap,
                                                                    plot=plot)
            
            final_signal_3d[:, :, frame] = corrected_signal_2d 
        
        return final_signal_3d    
        
    ######################################################################################
