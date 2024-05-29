from src.basics import Basics
from src.signal_basics import SignalBasics
from src.yaml_file import YamlFile

import numpy as np


class noTGC:
    
    def __init__(self,
                 folder_path,
                 ) -> None:
        
        self.folder_path = folder_path
        self.numpy_files_path: list[str] = None
        self.env_raw_rf: str = None
        self.rf_raw_rf: str = None
        self.no_tgc_values: list[float, float] = None
        
        self.yaml_obj = YamlFile(self.folder_path, mode="raw_files")

        self.set_numpy_files_path()
        self.create_noTGC_numpy()       
    
    ######################################################################################

    def set_numpy_files_path(self):
        
        # get numpy files path
        self.numpy_files_path = Basics.get_files_path_inside_folder(folder_path = self.folder_path,
                                            mode = "main_folder",
                                            extension= ".npy")
        
        for file_path in self.numpy_files_path:
            
            if "_env.raw.rf" in file_path:
                self.env_raw_rf = file_path
                
            elif "_rf.raw.rf" in file_path:
                self.rf_raw_rf = file_path
    
    ######################################################################################

    def create_noTGC_numpy(self):
        
        if self.yaml_obj.clean_ac_parameters is not None:     
            
            numpy_file_path = self.rf_raw_rf
                        
            signal_3d = Basics.read_numpy(numpy_file_path)
            signal_3d = np.array(signal_3d, dtype=np.float64)
            
            for frame in range(signal_3d.shape[2]):
                for line in range(signal_3d.shape[0]):
                                            
                    signal_3d[line, :, frame] /= self.create_noTGC_ratio(frame, signal_3d.shape[1])
                    
                    break
                break
                                              
            file_name = Basics.get_file_name_from_file_path(numpy_file_path)
            folder_path = Basics.get_folder_path_from_file_path(numpy_file_path)
            save_dir = folder_path + "\\" +  file_name + ".no_tgc"
            
            signal_3d = np.array(signal_3d, dtype=np.float16)

            np.save(save_dir, signal_3d)            
            
        else: print("yaml_obj.clean_ac_parameters is None")

    ######################################################################################
    
    def create_noTGC_ratio(self, frame, signal_1d_size):
        
        if self.yaml_obj.clean_ac_parameters is not None:     
                  
            try: 
                value = self.yaml_obj.clean_ac_parameters[f"timestamp_{frame}"]
            except KeyError:
                value = self.yaml_obj.clean_ac_parameters[f"timestamp_{frame - 1}"]

            depth_mm_list = list(value.keys())
            attenuation_dB_list = list(value.values())
            
            
            # total_time_1d = SignalBasics.create_time(signal_1d_size   = signal_1d_size,
            #                                 delay_samples    = self.yaml_obj.delay_samples,
            #                                 sampling_rate_Hz = self.yaml_obj.sampling_rate_MHz * 1e6)
            
            # last_element_of_time = total_time_1d[total_time_1d.shape[0] - 1]
            # imaging_depth_mm_1d = total_time_1d * 1540 * 1e3 /2
                          
            attenuation_dB_1d = SignalBasics.create_attenuation_linear_2(
                                        signal_1d_size = signal_1d_size + self.yaml_obj.delay_samples,
                                        imaging_depth_mm = self.yaml_obj.imaging_depth_mm,
                                        depth_ac_list       = depth_mm_list,
                                        attenuation_ac_list = attenuation_dB_list)
                                                       
            attenuation_dB_1d_trimmed = attenuation_dB_1d[self.yaml_obj.delay_samples : self.yaml_obj.delay_samples + signal_1d_size] 
            
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(8, 6))  # Width: 8 inches, Height: 6 inches
            # plt.plot(attenuation_dB_1d_trimmed)
            # plt.xlabel('depth [mm]')
            # plt.ylabel('ac [dB]')
            # plt.grid()
                       
            attenuation_dB_1d_trimmed = np.array(attenuation_dB_1d_trimmed, dtype=np.float64)           
            attenuation_value_1d_trimmed = 10**(attenuation_dB_1d_trimmed/20)     
            
            return attenuation_value_1d_trimmed
            
    ######################################################################################

        