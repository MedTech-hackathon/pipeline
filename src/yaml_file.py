from src.basics import Basics
from typing import Union
import yaml
import os
import re

    
class YamlFile:
    
    # Constructor method (initializer)
    def __init__(self,
                 extracted_folder_path:str,
                 mode:str = None):
        
        #self.predefined_software_versions: list[str] = ['9.3.0-411', '10.1.1-454']
        self.folder_path = extracted_folder_path
        self.yaml_files_path_list: list[str] = []
        
        # data
        self.yaml_files_data:     dict = {}
        self.env_tgc_yml:         dict = None
        self.env_yml:             dict = None
        self.rf_yml:              dict = None
        self.clean_ac_parameters: dict = None
        
        self.image_size:        str = None
        self.device_name:       str = None 
        self.sampling_rate_MHz: int = None 
        self.delay_samples:     int = None
        self.imaging_depth_mm:  float = None
        self.sample_per_line:   int = None
        self.software_version:  str = None
        self.spead_of_sound:    int = None
        self.first_element:     int = None
        self.last_element:      int = None

        
        if   mode == "raw_files":     self.__run_raw()
        elif mode == "renamed_files": self.__run_renamed()

    ###################################################################################

    def __str__(self):
        
        class_name = self.__class__.__name__
        attributes = "\n".join(f"{attr_name} = {getattr(self, attr_name)}" for attr_name in self.__dict__)
        return f"{100*"*"}\n{class_name}(\n{attributes}\n){100*"*"}"
    
    ###################################################################################
    
    def __run_raw(self):
        
        self.delete_hidden_files()
        self.get_yaml_files_path()  
        self.correct_wrong_naming()
        self.modify_extension(initial='.yml', final='.yaml')
        self.get_yaml_files_path()
        self.comment_tgc_in_yamls()
        self.append_timestamp_counter_in_env_tgc_yaml()
        self.read_yaml_files()
        self.set_yaml_data()   

        # set values
        self.set_sampling_rate(self.rf_yml)   
        self.set_image_size(self.rf_yml)
        self.set_delay_samples(self.rf_yml)
        self.set_elements(self.rf_yml)
        self.set_imaging_depth(self.rf_yml)
        self.set_sample_per_line(self.rf_yml)
        self.set_device_name(self.rf_yml)
        self.set_software_version(self.rf_yml)
        self.set_speed_of_sound(self.rf_yml)

        self.get_clean_ac_parameters()

        self.rename_files_in_folder()    
                   
    ###################################################################################
    
    def __run_renamed(self):
        
        self.get_yaml_files_path()  
        self.read_yaml_files()
        self.set_yaml_data()   

        self.set_sampling_rate(self.rf_yml)   
        self.set_image_size(self.rf_yml)
        self.set_delay_samples(self.rf_yml)
        self.set_elements(self.rf_yml)
        self.set_imaging_depth(self.rf_yml)
        self.set_sample_per_line(self.rf_yml)
        self.set_device_name(self.rf_yml)
        self.set_software_version(self.rf_yml)
        self.set_speed_of_sound(self.rf_yml)

        self.get_clean_ac_parameters()

    ###################################################################################

    def get_clean_ac_parameters(self):
        
        #if self.software_version in self.predefined_software_versions:
        if self.env_tgc_yml is not None:            
            
            data_dict = {}
            for key, value in self.env_tgc_yml.items():
                if "timestamp" in key:
                    data_dict[key] = self.clean_ac_string(value)
                    
            self.clean_ac_parameters = data_dict
                
    ###################################################################################

    def clean_ac_string(self, data_string):

        # Regular expression pattern to extract data points
        pattern = r'{\s*([\d.]+)mm,\s*([\d.]+)dB\s*}'

        # Extract data points using regular expression
        matches = re.findall(pattern, data_string)

        # Initialize an empty dictionary to store data
        data_dict = {}

        # Organize extracted data into dictionary
        for match in matches:
            distance = float(match[0])
            attenuation = float(match[1])
            data_dict[distance] = attenuation

        return data_dict
                        
    ###################################################################################

    def set_software_version(self, yaml_data):
        
        if yaml_data['software version']:
            
            self.software_version = yaml_data['software version']
            
    ###################################################################################

    def set_speed_of_sound(self, yaml_data):
        
        self.speed_of_sound = 1540
        return
        
        if self.device_name == "L15":
            if yaml_data['vsound']:
                self.speed_of_sound = Basics.get_int_in_str(yaml_data['vsound'])
                
        if self.device_name == "C3":
            self.speed_of_sound = 1450
                
    ###################################################################################
    
    def append_timestamp_counter_in_env_tgc_yaml(self):
        """
        Open a YAML file, find keys containing the word "timestamp",
        and append an underscore with an increasing number to them.

        Parameters:
        - file_path (str): The path to the YAML file.
        """

        for file_path in self.yaml_files_path_list:
            
            if "_env.tgc.yaml" in file_path:
                
                # Read the content of the YAML file
                with open(file_path, 'r') as file:
                    content = file.readlines()

                # Track the count of keys modified
                count = 0

                # Modify the content
                for i, line in enumerate(content):
                    
                    if 'timestamp' in line:
                        
                        if 'timestamp_' not in line:
                            
                            # Find the position of the key
                            key_index = line.index('timestamp') + len('timestamp')

                            # Insert the underscore and count
                            modified_line = line[:key_index] + '_' + str(count) + line[key_index:]

                            # Replace the line with the modified one
                            content[i] = modified_line

                            # Increment the count
                            count += 1

                # Write the modified content back to the file
                with open(file_path, 'w') as file:
                    file.writelines(content)

    ###################################################################################
    
    def modify_extension(self, initial, final):
        
        # Get the list of files in the folder
        files = os.listdir(self.folder_path)
        
        # Iterate over each file
        for file_name in files:
            
            # Check if the file has the .yml extension
            if file_name.endswith(initial):
                
                # Construct the full path to the file
                file_path = os.path.join(self.folder_path, file_name)
                
                # Construct the new file name with the .yaml extension
                new_file_name = os.path.splitext(file_name)[0] + final
                new_file_path = os.path.join(self.folder_path, new_file_name)
                
                try:
                    # Rename the file to .yaml extension
                    os.rename(file_path, new_file_path)
                    print(f"File {file_name} renamed to {new_file_name}")
                    
                except Exception as e:
                    # Print an error message if renaming fails
                    print(f"Failed to rename {file_name}: {e}")

    ###################################################################################

    def comment_tgc_in_yamls(self):
        """
        Open a YAML file, check if the "size" key exists,
        comment out the line if it does, and save the modified content back to the file.

        Parameters:
        - file_path (str): The path to the YAML file.
        """
        
        for file_path in self.yaml_files_path_list:
            
            if "_env.yaml" or "_rf.yaml" in file_path:
        
                # Read the content of the YAML file
                with open(file_path, 'r') as file:
                    content = file.readlines()

                # Check if the "tgc" key exists and comment out the line if it does
                for i, line in enumerate(content):
                    if 'tgc: {' in line:
                        if '# tgc: {' not in line:
                            content[i] = '# ' + line  # Comment out the line

                # Write the modified content back to the file
                with open(file_path, 'w') as file:
                    file.writelines(content)

    ###################################################################################

    def delete_hidden_files(self):
        
        """
        Delete files in the given folder that start with the specified prefix.

        Parameters:
        - folder_path (str): The path to the folder containing the files.
        - prefix (str): The prefix that files should start with to be deleted.
        """

        # Get the list of files in the folder
        files = os.listdir(self.folder_path)
        
        # Iterate over each file
        for file in files:
            
            # Check if the file starts with the specified prefix and is hidden
            if file.startswith('.'):
                            
                # Construct the full path to the file
                file_path = os.path.join(self.folder_path, file)
                
                try:
                    # Attempt to remove the file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    
                except Exception as e:
                    # Print an error message if deletion fails
                    print(f"Failed to delete {file_path}: {e}")
    
    ###################################################################################

    def rename_medium_to_small(self):     
        
        # get file path inside folder path
        files_path = Basics.get_files_path_inside_folder(folder_path = self.folder_path,
                                                         mode = "main_folder",
                                                         extension = "all")
        
        # check if name has medium and then rename it to small
        for file_path in files_path:
            if "L15_medium" in file_path:
                new_path = file_path.replace("medium", "small")
                os.rename(file_path, new_path)      
        
    ###################################################################################

    def get_yaml_files_path(self):
        
        # Get paths of yaml and yml files inside the folder    
        self.yaml_files_path_list = Basics.get_files_path_inside_folder(self.folder_path, mode="main_folder", extension='.yml')  + \
                                    Basics.get_files_path_inside_folder(self.folder_path, mode="main_folder", extension='.yaml') + \
                                    Basics.get_files_path_inside_folder(self.folder_path, mode="main_folder", extension='.tgc')
                
    ###################################################################################

    def set_yaml_data(self):
        
        for key, value in self.yaml_files_data.items():
                        
            if "_env.tgc" in key:
                self.env_tgc_yml = value
                
            if "_env" in key:
                self.env_yml = value
                
            if "_rf" in key:
                self.rf_yml = value

    ###################################################################################

    def correct_wrong_naming(self):
        
        for file_path in self.yaml_files_path_list:
            
            # get extension of the file from the file path
            extension = Basics.get_file_extension_from_file_path(file_path)
            
            # if extension is tgc corrrect the file format to yml
            if extension == ".tgc":
                
                # rename the file endling to .yml
                Basics.append_extension_to_file(file_path, additional_extension=".yaml")
    
    ###################################################################################

    def read_yaml_files(self,
                        ) -> dict:

        data = {}
        
        for file in self.yaml_files_path_list:
                        
            data[file] = self.read_yaml_file(file)
        
        self.yaml_files_data = data
        
    ###################################################################################

    def read_yaml_file(self,
                        file_path) -> dict:
        """
        Read a YAML file and return its contents as a dictionary.
        
        Args:
        file_path (str): Path to the YAML file.
        
        Returns:
        dict: Contents of the YAML file as a dictionary.
        """
        # Open the YAML file in read mode
        with open(file_path, 'r') as yaml_file:  
            
            # Load the YAML content into a dictionary
            yaml_data = yaml.safe_load(yaml_file)  
            
        # Return the dictionary containing the YAML data
        return yaml_data  
        
    ###################################################################################

    def set_image_size(self,
                       yaml_data,
                       threshold: int = 100) -> str:

        # Determine the sampling size based on the number of lines
        if yaml_data['size']['number of lines'] > threshold:
            self.image_size = "large"
            
        elif yaml_data['size']['number of lines'] < threshold:
            self.image_size = "small"
                    
    ###################################################################################  

    def set_sampling_rate(self,
                          yaml_data) -> Union[int, None]:

        self.sampling_rate_MHz = yaml_data['sampling rate']
        self.sampling_rate_MHz = Basics.get_int_in_str(self.sampling_rate_MHz)
            
    ###################################################################################  

    def set_sample_per_line(self,
                          yaml_data) -> Union[int, None]:

        self.sample_per_line = yaml_data['size']['samples per line']
                        
    ###################################################################################  

    def set_imaging_depth(self,
                          yaml_data) -> None:
      
        self.imaging_depth_mm = yaml_data['imaging depth']
        self.imaging_depth_mm = Basics.get_float_in_str(self.imaging_depth_mm)
                           
    ###################################################################################  

    def set_delay_samples(self,
                        yaml_data) -> None:
        
        self.delay_samples = yaml_data['delay samples']
    
    ###################################################################################  

    def set_elements(self,
                        yaml_data) -> None:
        
        self.first_element = yaml_data['lines'][0]['rx element']
        self.last_element  = yaml_data['lines'][len(yaml_data['lines']) - 1]['rx element']
           
    ###################################################################################  

    def set_device_name(self,
                        yaml_data) -> None:
        
        try:
            if yaml_data['probe']['radius']:
                self.device_name = "C3"
                
        except KeyError:
            self.device_name = "L15"
                         
    ###################################################################################      

    def rename_files_in_folder(self,
                                ) -> None:
        """
        Renames files in a folder based on sampling rate and size.

        Args:
            extraced_folder_path (str): Path to the folder containing the files to be renamed.
            yaml_files_data (list[dict]): List of YAML files data used to determine sampling rate and size.
        """
        key_name = "undefined"

        # Set file name based on sampling rate and size   
        if   self.image_size == "large":  key_name = f"{self.device_name}_large"          
        elif self.image_size == "small":  key_name = f"{self.device_name}_small"              
        
        # List all files in the folder
        files = Basics.get_files_name_inside_folder(folder_path = self.folder_path)
        
        # Iterate through each file
        for file_name in files:
            
            if ("large" or "small") not in file_name:  

                # Construct the old and new file paths
                old_file_path = os.path.join(self.folder_path, file_name)
                
                new_file_name = key_name + "_" + \
                Basics.get_part_of_string_after(
                    string = Basics.get_file_name_from_file_path(old_file_path, with_extension=False),
                    after  = "_" ) + \
                Basics.get_file_extension_from_file_path(old_file_path)
                                
                new_file_path = os.path.join(self.folder_path, new_file_name) 
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")
    
    ###################################################################################      
    
    
