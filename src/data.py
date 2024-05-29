from src.basics import Basics
from src.lzo_file import LzoFile
from src.raw_file import RawFile
from src.yaml_file import YamlFile
from src.no_tgc import noTGC
from src.tar_file import TarFile
from src.transformation import Transformation
from src.attenuation import AttenuationCorrection
from src.signal_basics import SignalBasics
from src.mathematics import Mathematics

import os
import numpy as np

class Data:
    
    def __init__(self,
                 tar_folder_path: str,
                 device: str,
                 size: int,
                 signal_type: str,
                 ac_method: str,
                 v1: int = None,
                 v2: int = None,
                 h1: int = None,
                 h2: int = None) -> None:  
        
        # the main folder of tar files 
        self.tar_folder_path = tar_folder_path
        self.device = device
        self.size = size
        self.signal_type = signal_type
        self.ac_method = ac_method
        self.v1 = v1
        self.v2 = v2
        self.h1 = h1
        self.h2 = h2
        
        # extracted folders
        self.extracted_folder_path: str = None
        self.extracted_folder_files_path_list: list[str] 
        self.numpy_files_path_list_filtered:   list[str]
        
        # sample related
        self.sample_name: str = None
        
        # availability    
        self.L15_large_tgc_found    = False
        self.L15_large_no_tgc_found = False
        self.L15_small_tgc_found    = False
        self.L15_small_no_tgc_found = False
        
        self.C3_large_tgc_found    = False
        self.C3_large_no_tgc_found = False
        self.C3_small_tgc_found    = False
        self.C3_small_no_tgc_found = False

        # data
        self.L15_large_tgc_dict:    dict = {}
        self.L15_large_no_tgc_dict: dict = {}
        self.L15_large_yaml_dict:   dict = {}
    
        self.L15_small_tgc_dict:    dict = {}
        self.L15_small_no_tgc_dict: dict = {}
        self.L15_small_yaml_dict:   dict = {}
        
        self.C3_large_tgc_dict:     dict = {}
        self.C3_large_no_tgc_dict:  dict = {}
        self.C3_large_yaml_dict:    dict = {}
    
        self.C3_small_tgc_dict:     dict = {}
        self.C3_small_no_tgc_dict:  dict = {}
        self.C3_small_yaml_dict:    dict = {}
        
        
        # selected data dictionary
        self.signal_data_dict: dict = {}
        self.yaml_obj_dict:    dict = {}
        
        
        # main signal 2d 
        self.signal_3d          = None
        self.signal_envelope_3d = None
        self.signal_2d          = None
        self.signal_envelope_2d = None  
        self.signal_1d          = None
        self.signal_envelope_1d = None  


        # ROI location and frame, ROI with delay
        self.frame = 0
        self.ROI_analysis_vline_0d = None           
        self.v1_with_delay = None
        self.v2_with_delay = None
        self.h1_with_delay = None
        self.h2_with_delay = None
        
        self.hilbert_transform_axis = 2
        
        # trimmed signal and time
        self.trimmed_signal_3d             = None
        self.trimmed_signal_envelope_3d    = None
        self.trimmed_signal_fft_freq_3d    = None
        self.trimmed_signal_fft_amp_3d     = None
        self.trimmed_signal_fft_amp_abs_3d = None
        
        self.trimmed_signal_2d             = None
        self.trimmed_signal_envelope_2d    = None
        self.trimmed_signal_fft_freq_2d    = None
        self.trimmed_signal_fft_amp_2d     = None
        self.trimmed_signal_fft_amp_abs_2d = None
        
        self.trimmed_signal_1d             = None
        self.trimmed_signal_envelope_1d    = None
        self.trimmed_signal_fft_freq_1d    = None
        self.trimmed_signal_fft_amp_1d     = None
        self.trimmed_signal_fft_amp_abs_1d = None
        
        self.total_time_1d       = None
        self.trimmed_time_1d     = None
        self.trimmed_depth_mm_1d = None
        self.total_depth_mm_1d   = None
        self.speed_of_sound      = None
        
        
        # initialize
        self.__run()
                   
    ######################################################################################  

    def __run(self):
               
        self.set_sample_name()       
               
        # delete hidden files
        if self.check_hidden_file_availability():
            self.delete_hidden_files_inside_tar_folder()
                    
        # prepare extracted folder
        if not self.check_extracted_folders_availability():
            self.prepare_extracted_folder()
        
        # get files inside extracted folder    
        self.set_extracted_folders_path()
        
        # search and set required numpies inside extracted folders
        for extracted_folder_path in self.extracted_folders_path_list:
            
            self.extracted_folder_path = extracted_folder_path
            
            self.set_files_path_inside_extracted_folder()
            yaml_notgc_files_available = self.check_yaml_notgc_files_availability()
            
            if(not yaml_notgc_files_available):
                continue

            self.get_numpy_files_path()
            self.set_data()                     
            
            
        # set signal
        self.set_selected_data()
        self.set_signal_3d()
        #self.limit_frames(1)
        #self.apply_window()
        self.set_signal_envelope_3d()
        
        self.set_signal_2d()
        self.set_signal_envelope_2d()
        
        self.set_signal_1d()
        self.set_signal_envelope_1d()      
         
        # ROI 
        self.set_ROI()
        self.set_delay_ROI()
        self.set_ROI_analyze_vline_0d()
        
        # get trimmed signals
        self.set_total_time_1d()
        self.set_trimmed_time_1d()
        self.set_trimmed_depth_1d()
        self.set_speed_of_sound()
        
        self.set_trimmed_signal_3d()
        #self.trimmed_signal_3d = self.shift_signal_xd(self.trimmed_signal_3d)
        self.set_trimmed_signal_envelope_3d()
        self.set_trimmed_signal_fft_3d()
        
        self.set_trimmed_signal_2d()
        self.set_trimmed_signal_envelope_2d()
        self.set_trimmed_signal_fft_2d()
        
        self.set_trimmed_signal_1d()
        self.set_trimmed_signal_envelope_1d()
        self.set_trimmed_signal_fft_1d()
        
        self.do_attenuation_correction(mode="3d")
        self.do_after_attenuation_correction_updates(mode="3d")
      
    ###################################################################################### 
       
    def set_sample_name(self):
        
        self.sample_name = Basics.get_directory_after_backslashes_from_end(path = self.tar_folder_path,
                                                                           num_backslashes = 1)     
                         
    ###################################################################################### 
    
    def check_hidden_file_availability(self):
        
        return Basics.check_hidden_file_availability(folder_path = self.tar_folder_path)
         
    ######################################################################################  
    
    def delete_hidden_files_inside_tar_folder(self):
        
        Basics.delete_hidden_files_inside_folder(folder_path = self.tar_folder_path)
        
    ######################################################################################
    
    def check_extracted_folders_availability(self):
        
        # Get list of subdirectories in the folder
        subdirectories = [d for d in os.listdir(self.tar_folder_path) if os.path.isdir(os.path.join(self.tar_folder_path, d))]
        
        # Iterate through the subdirectories
        for directory in subdirectories:
            # Check if the directory name includes "raw"
            if "raw" in directory.lower():
                return True
        
        # If no directory with "raw" in its name is found
        return False

    ######################################################################################  
     
    def prepare_extracted_folder(self):
        
        # Rename files, replacing spaces with underscores
        Basics.replace_space_with_underscore(self.tar_folder_path)
        
        # Delete already existing extracted folders
        Basics.delete_folders(Basics.get_subfolders_path(self.tar_folder_path), including="extracted")

        # Extract all TAR files found inside the folder 
        TarFile.extract_tar_files(self.tar_folder_path)

        # Get paths of extracted folders
        extracted_folders_path_list = Basics.get_subfolders_path(self.tar_folder_path)
        
        # Get paths of .lzo files from the extracted folders
        lzo_files_path_list = Basics.get_files_path_inside_folders(folders_path_list = extracted_folders_path_list,
                                                                    mode             = "main_folder",
                                                                    extension        = '.lzo')
        
        # Read .lzo files
        LzoFile.read_lzo_files(lzo_files_path_list)
        
        # Get paths of .raw files inside extracted folders
        raw_files_path_list = Basics.get_files_path_inside_folders(folders_path_list = extracted_folders_path_list,
                                                                    mode             = "main_folder",
                                                                    extension        = '.raw')
        
        
        # Read all .raw files and save them as numpy arrays inside their folders  
        RawFile.read_raw_files(raw_files_path_list) 
        
        
        for extracted_folder in extracted_folders_path_list:
            
            # create no tgc from tgc
            noTGC(folder_path = extracted_folder)
            
            # break
                                            
    ######################################################################################  
    
    def set_extracted_folders_path(self):
        
        self.extracted_folders_path_list = Basics.get_folders_path_inside_folder(folder_path = self.tar_folder_path)
    
    ######################################################################################
    
    def set_files_path_inside_extracted_folder(self):
        
        self.extracted_folder_files_path_list = Basics.get_files_path_inside_folder(folder_path = self.extracted_folder_path,
                                                                            mode="main_folder",
                                                                            extension="all")
        
    ######################################################################################

    def check_yaml_notgc_files_availability(self):
        
        """
        Check if required files (_rf.yaml and _rf.raw.rf.no_tgc.npy) are available.
        
        Returns:
            bool: True if both files are found, False otherwise.
        """
        
        # Check if the file list is empty or None
        if not self.extracted_folder_files_path_list:
            raise ValueError("The file list is empty or None.")

        # Initialize flags
        yaml_file_found = False
        no_tgc_file_found = False

        # Iterate through file paths
        for file_path in self.extracted_folder_files_path_list:
            # Check if YAML file is found
            if "_rf.yaml" in file_path:
                yaml_file_found = True
            # Check if no_tgc file is found
            elif "_rf.raw.rf.no_tgc.npy" in file_path:
                no_tgc_file_found = True

        # Return True only if both files are found
        return yaml_file_found and no_tgc_file_found

    ######################################################################################

    def get_numpy_files_path(self):
               
        self.numpy_files_path_list_filtered = Basics.filter_files_path_with_include_exclude(
                                                        files_path_list = self.extracted_folder_files_path_list,
                                                        include_list    = None,
                                                        exclude_list    = ["_env.raw.rf.npy"],
                                                        extension_list  = [".npy"])
    
    ######################################################################################

    def set_data(self):
        
        for numpy_file_path in self.numpy_files_path_list_filtered:
            
            if "L15_large_rf.raw.rf.npy" in numpy_file_path and not self.L15_large_tgc_found:
                self.L15_large_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.L15_large_yaml_dict[self.extracted_folder_path] = YamlFile(extracted_folder_path=self.extracted_folder_path, mode="renamed_files")
                self.L15_large_tgc_found = True
                
            elif "L15_large_rf.raw.rf.no_tgc.npy" in numpy_file_path and not self.L15_large_no_tgc_found:
                self.L15_large_no_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.L15_large_no_tgc_found = True
                
            elif "L15_small_rf.raw.rf.npy" in numpy_file_path and not self.L15_small_tgc_found:
                self.L15_small_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.L15_small_yaml_dict[self.extracted_folder_path] = YamlFile(extracted_folder_path=self.extracted_folder_path, mode="renamed_files")
                self.L15_small_tgc_found = True

            elif "L15_small_rf.raw.rf.no_tgc.npy" in numpy_file_path and not self.L15_small_no_tgc_found:
                self.L15_small_no_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.L15_small_no_tgc_found = True

            elif "C3_large_rf.raw.rf.npy" in numpy_file_path and not self.C3_large_tgc_found:
                self.C3_large_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.C3_large_yaml_dict[self.extracted_folder_path] = YamlFile(extracted_folder_path=self.extracted_folder_path, mode="renamed_files")
                self.C3_large_tgc_found = True

            elif "C3_large_rf.raw.rf.no_tgc.npy" in numpy_file_path and not self.C3_large_no_tgc_found:
                self.C3_large_no_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.C3_large_no_tgc_found = True

            elif "C3_small_rf.raw.rf.npy" in numpy_file_path and not self.C3_small_tgc_found:
                self.C3_small_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.C3_small_yaml_dict[self.extracted_folder_path] = YamlFile(extracted_folder_path=self.extracted_folder_path, mode="renamed_files")
                self.C3_small_tgc_found = True

            elif "C3_small_rf.raw.rf.no_tgc.npy" in numpy_file_path and not self.C3_small_no_tgc_found:
                self.C3_small_no_tgc_dict = Basics.read_numpy_and_create_dict(file_path=numpy_file_path)
                self.C3_small_no_tgc_found = True

                    
        #Basics.print_list_in_new_lines(list = self.numpy_files_path_list_filtered, additional_name="npy")  
        #Error.check_array_size(array=self.numpy_files_path_list_filtered)
                    
    ######################################################################################





    # main signal
    ###################################################################################

    def set_selected_data(self): 
        
        if  self.device == "L15" and self.size == "large" and self.signal_type == "tgc":
            
            self.signal_data_dict = self.L15_large_tgc_dict
            self.yaml_obj_dict    = self.L15_large_yaml_dict

        elif self.device == "L15" and self.size == "large" and self.signal_type == "no_tgc":
            
            self.signal_data_dict = self.L15_large_no_tgc_dict
            self.yaml_obj_dict    = self.L15_large_yaml_dict
            
        elif self.device == "L15" and self.size == "small" and self.signal_type == "tgc":
            
            self.signal_data_dict = self.L15_small_tgc_dict
            self.yaml_obj_dict    = self.L15_small_yaml_dict
            
        elif self.device == "L15" and self.size == "small" and self.signal_type == "no_tgc":
            
            self.signal_data_dict = self.L15_small_no_tgc_dict
            self.yaml_obj_dict    = self.L15_small_yaml_dict

        elif self.device == "C3" and self.size == "large" and self.signal_type == "tgc":
            
            self.signal_data_dict = self.C3_large_tgc_dict
            self.yaml_obj_dict    = self.C3_large_yaml_dict

        elif self.device == "C3" and self.size == "large" and self.signal_type == "no_tgc":
            
            self.signal_data_dict = self.C3_large_no_tgc_dict
            self.yaml_obj_dict   = self.C3_large_yaml_dict
            
        elif self.device == "C3" and self.size == "small" and self.signal_type == "tgc":
            
            self.signal_data_dict = self.C3_small_tgc_dict
            self.yaml_obj_dict    = self.C3_small_yaml_dict
            
        elif self.device == "C3" and self.size == "small" and self.signal_type == "no_tgc":
            
            self.signal_data_dict = self.C3_small_no_tgc_dict
            self.yaml_obj_dict    = self.C3_small_yaml_dict

        first_key = next(iter(self.yaml_obj_dict))  
        self.yaml_obj = self.yaml_obj_dict[first_key]   
      
    ###################################################################################

    def set_signal_3d(self): 
               
        first_key = next(iter(self.signal_data_dict))  
        self.signal_3d = self.signal_data_dict[first_key] 
        
    ###################################################################################

    def limit_frames(self, max_frame): 
        
        self.signal_3d = self.signal_3d[:, :, 0:max_frame]
               
    ###################################################################################

    def apply_window(self): 
        
        n = 1
        
        for index_1 in range(self.signal_3d.shape[0]):
            for index_3 in range(self.signal_3d.shape[2]):
                
                signal_1d = self.signal_3d[index_1, :, index_3]
                
                # Set first n elements to zero
                for i in range(n):
                    signal_1d[i] = 0
                
                # Set last n elements to zero
                for i in range(-n, 0):
                    signal_1d[i] = 0
                
                self.signal_3d[index_1, :, index_3] = signal_1d  # Assign modified signal back to original array

    ###################################################################################
        
    def set_signal_envelope_3d(self): 
        
        # axis 2 applied on frames
        self.signal_envelope_3d = Transformation.hilbert_transform_along_axis(
                                                        signal_xd = self.signal_3d,
                                                        axis      = self.hilbert_transform_axis) 
                  
    ###################################################################################

    def set_signal_2d(self): 
        
        self.signal_2d = self.signal_3d[:, : , self.frame]
    
    ###################################################################################

    def set_signal_envelope_2d(self): 
        
        self.signal_envelope_2d = self.signal_envelope_3d[:, : , self.frame]

    ###################################################################################

    def set_signal_1d(self): 
        
        self.signal_1d = self.signal_2d[self.ROI_analysis_vline_0d, :]
        
    ###################################################################################

    def set_signal_envelope_1d(self): 
        
        self.signal_envelope_1d = self.signal_envelope_2d[self.ROI_analysis_vline_0d, :]

    ###################################################################################
    
    
    
    
    
    
    # ROI                          
    ###################################################################################

    def set_ROI(self): 

        # get full array if size is not defined
        if(self.v1) is None: self.v1 = 0         
        if(self.v2) is None: self.v2 = self.signal_2d.shape[0]
        if(self.h1) is None: self.h1 = 0              
        if(self.h2) is None: self.h2 = self.signal_2d.shape[1]
                                                                    
    ###################################################################################

    def set_delay_ROI(self): 
        
        self.v1_with_delay = None
        self.v2_with_delay = None
        self.h1_with_delay = self.yaml_obj.delay_samples + self.h1
        self.h2_with_delay = self.yaml_obj.delay_samples + self.h2   
        
    ###################################################################################

    def set_ROI_analyze_vline_0d(self): 

        # set it on the center line of ROI    
        self.ROI_analysis_vline_0d =  (self.v2 - self.v1) // 2
        
    ###################################################################################
    
    
    
    
        
    # trimmed signal  
    ###################################################################################

    def set_trimmed_signal_3d(self):
                                             
        #trim signal and time array
        self.trimmed_signal_3d = self.signal_3d[self.v1 : self.v2,
                                                self.h1 : self.h2,
                                                :]
        
    ###################################################################################

    def set_trimmed_signal_envelope_3d(self):
                            
        self.trimmed_signal_envelope_3d = self.signal_envelope_3d[self.v1 : self.v2,
                                                                  self.h1 : self.h2,
                                                                    :]
    
    ###################################################################################

    def calculate_trimmed_signal_envelope_3d(self):       
        
        self.trimmed_signal_envelope_3d = Transformation.hilbert_transform_along_axis(
                                                        signal_xd = self.trimmed_signal_3d,
                                                        axis      = self.hilbert_transform_axis)
        
    ###################################################################################

    def set_trimmed_signal_fft_3d(self):
                                                        
        self.trimmed_signal_fft_freq_3d, self.trimmed_signal_fft_amp_3d = Transformation.fourier_transform_along_axis(
                                            signal_xd  = self.trimmed_signal_3d,
                                            time_1d    = self.trimmed_time_1d,
                                            axis       = 1,
                                            correction = True)
        
        self.trimmed_signal_fft_amp_abs_3d = np.abs(self.trimmed_signal_fft_amp_3d)    
        
    ###################################################################################

    def set_trimmed_signal_2d(self):
                                   
        #trim signal and time array
        self.trimmed_signal_2d = self.trimmed_signal_3d[:, : , self.frame]
    
    ###################################################################################

    def set_trimmed_signal_2d_temporary(self):
                            
        #trim signal and time array
        self.trimmed_signal_2d = self.signal_2d[self.v1 : self.v2,
                                                self.h1 : self.h2]
        
    ###################################################################################

    def set_trimmed_signal_envelope_2d(self):
                            
        # Convolve wavelet with trimmed signal using GH2 and GH8 wavelets
        self.trimmed_signal_envelope_2d = self.trimmed_signal_envelope_3d[:, : , self.frame]
        
    ###################################################################################

    def set_trimmed_signal_fft_2d(self):
                            
        self.trimmed_signal_fft_freq_2d    = self.trimmed_signal_fft_freq_3d[:, : , self.frame]
        self.trimmed_signal_fft_amp_2d     = self.trimmed_signal_fft_amp_3d[:, : , self.frame]
        self.trimmed_signal_fft_amp_abs_2d = self.trimmed_signal_fft_amp_abs_3d[:, : , self.frame]

    ###################################################################################

    def set_trimmed_signal_1d(self):
                            
        self.trimmed_signal_1d = self.trimmed_signal_2d[self.ROI_analysis_vline_0d, :]
            
    ###################################################################################

    def set_trimmed_signal_envelope_1d(self):
        
        self.trimmed_signal_envelope_1d = self.trimmed_signal_envelope_2d[self.ROI_analysis_vline_0d, :]

    ###################################################################################

    def set_trimmed_signal_fft_1d(self):
                                                            
        self.trimmed_signal_fft_freq_1d    = self.trimmed_signal_fft_freq_2d[self.ROI_analysis_vline_0d, :]
        self.trimmed_signal_fft_amp_1d     = self.trimmed_signal_fft_amp_2d[self.ROI_analysis_vline_0d, :]
        self.trimmed_signal_fft_amp_abs_1d = self.trimmed_signal_fft_amp_abs_2d[self.ROI_analysis_vline_0d, :]
                
    ###################################################################################

    def set_total_time_1d(self):
                                        
        # create time array
        self.total_time_1d = SignalBasics.create_time(signal_1d_size   = self.signal_2d[self.ROI_analysis_vline_0d, :].size,
                                                    sampling_rate_Hz = self.yaml_obj.sampling_rate_MHz * 1e6,
                                                    delay_samples    = self.yaml_obj.delay_samples)
      
    ###################################################################################

    def set_trimmed_time_1d(self):
                                               
        #trim signal and time array
        self.trimmed_time_1d = self.total_time_1d[self.h1_with_delay : self.h2_with_delay]  
            
    ###################################################################################

    def set_trimmed_depth_1d(self):
        
        if self.yaml_obj.delay_samples == 0:
            
            # create time array
            total_depth_1d = SignalBasics.create_depth(
                signal_1d_size   = self.signal_2d[self.ROI_analysis_vline_0d, :].size,
                delay_samples    = self.yaml_obj.delay_samples,
                imaging_depth_mm = self.yaml_obj.imaging_depth_mm)
            
            #trim depth
            self.total_depth_mm_1d = total_depth_1d  
            self.trimmed_depth_mm_1d = total_depth_1d[self.h1_with_delay : self.h2_with_delay]     
                            
        else:

            last_element_of_time = self.total_time_1d[self.total_time_1d.shape[0] - 1]
            imaging_depth_mm = last_element_of_time * 1540 * 1e3 / 2
                                        
            # create time array
            total_depth_1d = SignalBasics.create_depth(
                signal_1d_size   = self.signal_2d[self.ROI_analysis_vline_0d, :].size + self.yaml_obj.delay_samples,
                delay_samples    = 0,
                imaging_depth_mm = imaging_depth_mm)
            
            #trim depth
            self.total_depth_mm_1d = total_depth_1d  
            self.trimmed_depth_mm_1d = total_depth_1d[self.h1_with_delay : self.h2_with_delay]     
            
    ###################################################################################
    
    def set_speed_of_sound(self):

        if self.yaml_obj.delay_samples == 0:
            self.speed_of_sound = (
                ((self.trimmed_depth_mm_1d[self.trimmed_depth_mm_1d.shape[0] - 1]) * 1e-3) / self.trimmed_time_1d[self.trimmed_time_1d.shape[0] - 1]
                ) * 2    

    ###################################################################################

    
    
        
    # correction
    ###################################################################################
    @staticmethod
    def shift_signal_xd(signal):
        
        # shift signal by mean
        signal = signal - np.mean(signal)
        
        return signal
    
    ###################################################################################

    def do_attenuation_correction(self, mode):       
                        
        if self.ac_method == 'off':           
            pass
        
        # constant frequency depth
        elif self.ac_method == 'afd_base':           
            
            if mode == "3d":
                self.trimmed_signal_3d = AttenuationCorrection.attenuation_corrector_3D_with_alpha_estimator(
                                    signal_3d = self.trimmed_signal_3d,
                                    depth_cm = self.total_time_1d * 1540 * 100,                    
                                    fs = self.yaml_obj.sampling_rate_MHz * 1e6,                          
                                    window = 'hann',                      
                                    nperseg = 64,                     
                                    noverlap = 32,                    
                                    plot=False)              
                
        # constant frequency depth
        elif self.ac_method == 'fd_base':           
            
            if mode == "3d":
                self.trimmed_signal_3d = AttenuationCorrection.attenuation_corrector_3D_freqeuency_based(
                                    time_1d      = self.trimmed_time_1d,
                                    signal_3d = self.trimmed_signal_3d)
           
        
        elif self.ac_method == 'd_base':          
            
            if mode == "3d":

                deattenuation_array_1d = AttenuationCorrection.deattenuation_function(
                                            time_1d      = self.trimmed_time_1d,
                                            frequency_MHz = self.yaml_obj.sampling_rate_MHz / 6)
                
                self.signal_3d = Mathematics.multiply_3darray(array = self.signal_3d,
                                                        coeff = deattenuation_array_1d,
                                                        axis  = 1)
                        

        elif self.ac_method == 'w_base':          
             
            # only on central frequency of wavelet, which is not totally correct

            self.ac_array_1_1d = AttenuationCorrection.deattenuation_function(
                        time_1d     = self.trimmed_time_1d,
                        frequency_MHz   = self.wavelet_1.central_freq_Hz / 1e6, 
                        )

            self.ac_array_2_1d = AttenuationCorrection.deattenuation_function(
                        time_1d     = self.trimmed_time_1d,
                        frequency_MHz   = self.wavelet_2.central_freq_Hz / 1e6,
                        )

            self.raw_convolved_with_ghx_1_3d = Mathematics.multiply_3darray(array = self.raw_convolved_with_ghx_1_3d,
                                                                coeff = self.ac_array_1_1d,
                                                                axis  = 1)

            self.raw_convolved_with_ghx_2_3d = Mathematics.multiply_3darray(array = self.raw_convolved_with_ghx_2_3d,
                                                                coeff = self.ac_array_2_1d,
                                                                axis  = 1)
            
    ###################################################################################

    def do_after_attenuation_correction_updates(self, mode):
        
        if self.ac_method == 'off':           
            pass
            # self.ac_array_1_1d = np.ones_like(self.trimmed_time_1d)
            # self.ac_array_2_1d = np.ones_like(self.trimmed_time_1d)
        
        elif self.ac_method == 'fd_base' or self.ac_method == 'd_base' or self.ac_method == 'afd_base':           

            if mode == "3d":  
                       
                self.trimmed_signal_3d = self.shift_signal_xd(self.trimmed_signal_3d)
                #self.set_trimmed_signal_envelope_3d()
                self.calculate_trimmed_signal_envelope_3d()                
                self.set_trimmed_signal_fft_3d()
                
                self.set_trimmed_signal_2d()
                self.set_trimmed_signal_envelope_2d()
                self.set_trimmed_signal_fft_2d()
                
                self.set_trimmed_signal_1d()
                self.set_trimmed_signal_envelope_1d()
                self.set_trimmed_signal_fft_1d()
                   
        elif self.ac_method == 'w_base':           

            self.get_envelope_of_convolved_signals_3d()
            self.get_fft_of_convolved_signals_3d()
            self.convolve_signal_with_wavelets_2d()
            self.get_envelope_of_convolved_signals_2d()
            self.get_fft_of_convolved_signals_2d()
            self.convolve_signal_with_wavelets_1d()
            self.get_envelope_of_convolved_signals_1d()
            self.get_fft_of_convolved_signals_1d()
            
    ###################################################################################
