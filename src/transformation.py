from scipy.signal import hilbert, stft
from typing import Tuple
import numpy as np


class Transformation:
    
    ###################################################################################
    @staticmethod
    def fourier_transform(
                          signal_1d: np.ndarray,      # Input 1D signal (np.ndarray)
                          time_1d: np.ndarray,  # Time array of the signal (np.ndarray)
                          correction: bool = False    # Whether to apply frequency correction (bool)
                          ) -> Tuple[np.ndarray, np.ndarray]:  

        # Compute the Fourier transform of the input signal
        fft_amplitude = np.fft.fft(signal_1d)
        fft_frequency = np.fft.fftfreq(len(signal_1d), time_1d[1] - time_1d[0])

        # Keep only positive frequencies if correction is enabled
        if correction:
            positive_freq_mask = fft_frequency >= 0
            positive_freq = fft_frequency[positive_freq_mask]
            positive_amplitudes = np.abs(fft_amplitude[positive_freq_mask])
            fft_frequency, fft_amplitude = positive_freq, positive_amplitudes 

        return fft_frequency, fft_amplitude
    
    ###################################################################################
    @classmethod
    def fourier_transform_along_axis(cls,
                          signal_xd: np.ndarray,      # Input 1D signal (np.ndarray)
                          time_1d: np.ndarray,  # Time array of the signal (np.ndarray)
                          axis: int,
                          correction: bool = False,    # Whether to apply frequency correction (bool)
                          ) -> Tuple[np.ndarray, np.ndarray]:  

        results = np.apply_along_axis(cls.fourier_transform, axis=axis, arr=signal_xd,
                                                time_1d=time_1d, correction=correction)
        
        fft_frequency = results[:, 0, :, :]
        fft_amplitude = results[:, 1, :, :]

        return fft_frequency, fft_amplitude

    ###################################################################################
    @staticmethod
    def short_fourier_transform(
                                signal_2d: np.ndarray,  # Input 2D signal (np.ndarray)
                                sampling_rate_Hz,
                                nperseg: int            # Number of samples per segment (int)
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  
        """
        Compute the short-time Fourier transform (STFT) of a 2D signal.

        Parameters:
        - signal_2d (np.ndarray): Input 2D signal.
        - nperseg (int): Number of samples per segment for the STFT computation.

        Returns:
        - frequencies (np.ndarray): Array of Fourier frequencies.
        - times (np.ndarray): Array of time points.
        - Zxx (np.ndarray): Short-time Fourier transform of the input signal.
        """
        # Compute the short-time Fourier transform (STFT) of the input signal
        frequencies, times, Zxx = stft(signal_2d,
                                    fs=sampling_rate_Hz,
                                    nperseg=nperseg,
                                    window='hann')
        
        return frequencies, times, Zxx

    ###################################################################################
    @staticmethod
    def hilbert_transform_along_axis(
                          signal_xd: np.ndarray,  # Input 1D signal (np.ndarray)
                          axis: int) -> Tuple[complex, float, float]:  # Axis along which to apply the transform (int)
        """
        Apply the Hilbert transform to the input signal.

        Parameters:
        - signal_1d (np.ndarray): Input 1D signal.
        - axis (int): Axis along which to apply the transform.

        Returns:
        - analytic_signal (complex): Analytic signal obtained from the Hilbert transform.
        - envelope (float): Envelope of the analytic signal.
        - phase (float): Phase of the analytic signal.
        """
        # Apply the Hilbert transform to the input signal
        analytic_signal = np.apply_along_axis(hilbert, arr=signal_xd, axis=axis)
        
        # Compute the envelope and phase of the analytic signal
        envelope = np.abs(analytic_signal)
        #phase = np.angle(analytic_signal)
        
        return envelope
           
    ###################################################################################
    @staticmethod
    def convolve_two_signals_1d(
                                signal_1_1d: np.ndarray,   # First input 1D signal (np.ndarray)
                                signal_2_1d: np.ndarray,   # Second input 1D signal (np.ndarray)
                                mode: str) -> np.ndarray:  # Convolution mode (str)
        """
        Convolve two 1D signals.

        Parameters:
        - signal_1_1d (np.ndarray): First input 1D signal.
        - signal_2_1d (np.ndarray): Second input 1D signal.
        - mode (str): Convolution mode.

        Returns:
        - convolved_signal (np.ndarray): Convolution result.
        """
        # Apply convolution to the input signals
        return np.convolve(signal_1_1d, signal_2_1d, mode=mode)
    
    ###################################################################################
    @classmethod
    def convolve_signal_along_axis(cls,
                                        signal_xd,
                                        wavelet_1d,
                                        axis: int):
        
        # Convolve wavelet with trimmed signal using GH2 and GH8 wavelets
        signal_2d_c_wavelet_1d = np.apply_along_axis(cls.convolve_two_signals_1d, axis=axis, arr=signal_xd,
                                                    signal_2_1d=wavelet_1d, mode="same")

        return signal_2d_c_wavelet_1d

    ###################################################################################

