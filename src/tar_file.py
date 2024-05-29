from src.basics import Basics

import tarfile
import os


class TarFile:
                        
    ######################################################################################  
    @staticmethod
    def extract_tar_files(folder_path):
        """
        Extracts tar files found in a specified folder.

        Args:
            folder_path (str): Path to the folder containing the tar files.
        """
        # Iterate over files in the folder
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            
            # Check if the item is a file
            if os.path.isfile(item_path):
                
                # Check if the file is a tar file
                if tarfile.is_tarfile(item_path):
                    
                    # Get the file name
                    file_name = Basics.get_part_of_string_after(item_path, after='/')
                    #print(file_name)
                                            
                    # Create a new folder for extracted files
                    extracted_folder = os.path.join(folder_path, f'{file_name}_extracted')
                    os.makedirs(extracted_folder, exist_ok=True)
                    
                    # Extract the tar file into the new folder
                    with tarfile.open(item_path, 'r') as tar:
                        tar.extractall(path=extracted_folder)
                        print(f"\nExtracted \n'{file_name}' \ninto \n'{extracted_folder}'\n")  
    
    ###################################################################################
    

