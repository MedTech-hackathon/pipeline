from typing import Union
import numpy as np
import os
import shutil
import re
import pandas as pd

# Set display options
#pd.set_option('display.max_rows', None)  # Display all rows
#pd.set_option('display.max_columns', None)  # Display all columns
#pd.set_option('display.max_colwidth', None)  # This will display the entire content of each cell without truncation

# reset display
#pd.reset_option('^display')
    
      
class Basics:
    
    index = 0
    
    ######################################################################################
    @classmethod
    def filter_files_path_with_include_exclude(cls,
                                               files_path_list: str,
                                               include_list: list[str] = None,
                                               exclude_list: list[str] = None,
                                               extension_list: list[str] = None,
                                               ) -> list[str]:

        filtered_paths = []
        
        for file_path in files_path_list:
            file_name = os.path.basename(file_path)

            # Check if file should be included based on include list
            if include_list and not any(name in file_name for name in include_list):
                continue

            # Check if file should be excluded based on exclude list
            if exclude_list and any(name in file_name for name in exclude_list):
                continue

            # Check if file has the required extension
            if extension_list and not file_name.endswith(tuple(extension_list)):
                continue

            filtered_paths.append(file_path)

        return filtered_paths
                            
    ######################################################################################
    @classmethod
    def read_numpys_and_create_dict(cls,
                                    files_path_list: list[str]) -> dict:
  
        dictionary = {}  # Initialize an empty dictionary
        
        for file_path in files_path_list:  # Iterate through each file path
            
            dictionary.update(cls.read_numpy_and_create_dict(file_path))

        return dictionary
    
    ######################################################################################
    @classmethod
    def read_numpy_and_create_dict(cls,
                                    file_path: str) -> dict:

        dictionary = {}  # Initialize an empty dictionary
                
        # Read the numpy file and add it to the dictionary with its file path as the key
        dictionary[file_path] = cls.read_numpy(file_path)

        return dictionary
    
    ######################################################################################
    @staticmethod
    def read_numpy(
                   file_path: str) -> np.ndarray:
        """
        Read a numpy file and return the data.

        Args:
            file_path (str): Path to the numpy file.

        Returns:
            np.ndarray: Numpy array containing the data.
        """
       
        # Load the data from the numpy file
        data = np.load(file_path)  

        return data
             
    ###################################################################################
    @staticmethod
    def get_files_name_inside_folder(
                    folder_path: str,
                    ) -> list[str]:
        
        files = []
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Iterate over each item in the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                # Check if it's a file
                if os.path.isfile(item_path):
                    files.append(item)
        else:
            print("Folder does not exist.")
        return files
        
        # files = os.listdir(folder_path)
        # files = filter(lambda f: os.path.isfile(os.path.join(folder_path, f)), files)
        # return [*files]
        
    ###################################################################################  
    @staticmethod
    def get_file_name_from_file_path(
                                     file_path: str,
                                     with_extension: bool = None) -> str:

        # Get the base name of the file path
        file_name = os.path.basename(file_path)  
        
        if not with_extension:
            # Split the base name to get the file name without extension
            file_name = os.path.splitext(file_name)[0]  
        
        return file_name
                
    ###################################################################################  
    @staticmethod
    def get_file_extension_from_file_path(
                                          file_path: str) -> str:
        """
        Extracts the file extension from a given file path.

        Args:
            file_path (str): The path of the file.

        Returns:
            str: The file extension.
        """
        # Split the file path into its base name and extension
        file_extension = os.path.splitext(file_path)[1]
        
        # Return the file extension
        return file_extension

    ######################################################################################
    @classmethod
    def get_files_path_inside_folder(cls,
                    folder_path: str,
                    mode: str,
                    extension: str) -> list:

        # Initialize an empty list to store the file paths
        file_paths = []        
        file_paths_filtered = []        

        if mode == "all_files":
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))

        elif mode == "main_folder":
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
                    
        elif mode == "sub_folders":
            for root, dirs, files in os.walk(folder_path):
                if root != folder_path:  # Exclude files in the main folder
                    for file in files:
                        file_paths.append(os.path.join(root, file))
                            
        if extension == "all":
            return file_paths
            
        else:
            # filter the file paths
            for file_path in file_paths:
                if extension == cls.get_file_extension_from_file_path(file_path):
                    file_paths_filtered.append(file_path)
                    
            return file_paths_filtered
        
    ###################################################################################
    @classmethod
    def get_files_path_inside_folders(cls,
                                     folders_path_list: list[str],
                                     mode,
                                     extension) -> list[str]:
        """
        Retrieves the file paths of all files inside a list of folders, optionally filtered by extension.

        Args:
            folders_path_list (list): A list of paths to folders where files will be searched.
            extension (str, optional): The file extension to filter by. Defaults to None.

        Returns:
            list: A list of lists containing file paths found inside each folder in the input list,
            optionally filtered by extension.
        """

        files_path_list = []  # List to store the paths of matching files

        # Iterate over each folder path in the provided list
        for folder_path in folders_path_list:
            
            # Call get_files_path_inside_folder method to retrieve file paths inside each folder
            for path in cls.get_files_path_inside_folder(folder_path, mode, extension):
                files_path_list.append(path)
            
        # Return the list of lists containing file paths for each folder
        return files_path_list

    ###################################################################################
    @staticmethod
    def get_folder_path_from_file_path(
                                    file_path: str) -> str:
        """
        Retrieves the folder path from a given file path.

        Args:
            file_path (str): The path to a file.

        Returns:
            str: The path to the folder containing the file.
        """
        folder_path = os.path.dirname(file_path)  # Get the directory containing the file
        
        return folder_path  # Return the folder path

    ###################################################################################
    @staticmethod
    def get_folders_path_inside_folder(
                                    folder_path: str) -> list[str]:

        folder_paths = []
        
        # Check if the provided path is a directory
        if os.path.isdir(folder_path):
            # Walk through the directory tree
            for root, dirs, files in os.walk(folder_path):
                # Iterate over directories
                for directory in dirs:
                    folder_paths.append(os.path.join(root, directory))
        else:
            print("Error: The provided path is not a directory.")

        return folder_paths

    ###################################################################################  
    @staticmethod
    def get_subfolders_path(
                        parent_folder):
        """
        Get folders path inside a given folder.
        
        Args:
        - parent_folder (str): Path to the parent folder.
        
        Returns:
        - folder_paths (list): List of folder paths inside the parent folder.
        """
        folders_path = []
        
        # Iterate through all items (files and folders) in the parent folder
        for item in os.listdir(parent_folder):
            
            #print(Basics.add_char_between_strings(f"Sample {item}", "found"))  

            item_path = os.path.join(parent_folder, item)
            
            # Check if the item is a folder
            if os.path.isdir(item_path):
                folders_path.append(item_path)
                
        return folders_path
    
    ###################################################################################
    @staticmethod
    def get_int_in_str(
                        input_string: str) -> Union[int, None]:
        """
        Extracts an integer from a given string.

        Args:
            input_string (str): The input string containing numbers.

        Returns:
            Union[int, None]: The extracted integer if found, otherwise None.
        """
        # Define the regular expression pattern to match numbers
        pattern = r'\d+'
        
        # Find all matches of the pattern in the input string
        matches = re.findall(pattern, input_string)
        
        # If matches are found, convert the first match to an integer
        if matches:
            return int(matches[0])
        else:
            return None
    
    ###################################################################################
    @staticmethod
    def get_float_in_str(
                        input_string: str) -> Union[int, None]:
        
        floats = re.search(r"[-+]?\d*\.\d+|\d+", input_string)
        
        if floats:
            return float(floats.group())
        else:
            return None

    ###################################################################################
    @staticmethod
    def get_part_of_string_after(
                                string: str,
                                after: str = None):
        """
        Get the last part of a string after the specified key.

        Args:
            string (str): The input string.
            key (str): The delimiter to split the string. Defaults to '/'.

        Returns:
            str: The last part of the string after the key.
        """
        return string.rsplit(after, 1)[-1]                                                       

    ###################################################################################
    @staticmethod
    def get_directory_after_backslashes_from_end(
                                                 path,
                                                 num_backslashes):
        
        parts = path.split(os.path.sep)
        backslashes_count = 0
        directory = ''
        
        for part in reversed(parts):  
            if part == '':
                continue  # skip empty parts
            directory = os.path.join(part, directory)
            backslashes_count += 1
            if backslashes_count == num_backslashes:
                return directory.rstrip(os.path.sep)
            
        return directory

    ###################################################################################  
    @staticmethod
    def delete_folders(
                    folder_paths: list[str],
                    including: str = None) -> None:   
        """
        Deletes folders based on certain conditions.

        Args:
            folder_paths (list): List of paths to folders to be deleted.
            including (str): If specified, only folders containing this substring in their name will be deleted. Defaults to None.
        """
        for folder_path in folder_paths:
            # Check if the folder name contains the specified substring
            if including in os.path.basename(folder_path):
                # Attempt to delete the folder and all its contents
                shutil.rmtree(folder_path)
                 
    ###################################################################################
    @staticmethod
    def path_of_working_space(
                               ) -> str:
        """
        Get the path of the current working directory.

        Returns:
            str: Path of the current working directory.
        """
        return os.getcwd()
        
    ###################################################################################
    @classmethod
    def replace_space_with_underscore(cls, 
                                    folder_path: str,
                                    type: str = "files_and_folders") -> None:
        """
        Replace spaces with underscores in file and folder names within a specified directory.

        Parameters:
            folder_path (str): The path of the directory containing the files and folders.
            type (str, optional): Specifies whether to rename files, folders, or both. 
                Possible values are "file", "folder", or "both" (default). 
                If "file", only files will be renamed. 
                If "folder", only folders will be renamed. 
                If "both", both files and folders will be renamed. 

        Returns:
            None
        """
        # Check if the provided path exists and is a directory
        if not os.path.isdir(folder_path):
            print("Error: The provided path is not a directory.")
            return
        
        # Iterate over all items (files and folders) in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            if type == "file" and os.path.isfile(item_path):
                # Check if the item is a file (not a directory)
                if ' ' in item:
                    # Rename the file by replacing spaces with underscores
                    new_item_name = item.replace(' ', '_')
                    new_item_path = os.path.join(folder_path, new_item_name)
                    os.rename(item_path, new_item_path)
                    print(f"Renamed file '{item}' to '{new_item_name}'.")
            
            elif type == "folder" and os.path.isdir(item_path):
                # Check if the item is a folder (not a file)
                if ' ' in item:
                    # Rename the folder by replacing spaces with underscores
                    new_item_name = item.replace(' ', '_')
                    new_item_path = os.path.join(folder_path, new_item_name)
                    os.rename(item_path, new_item_path)
                    print(f"Renamed folder '{item}' to '{new_item_name}'.")
            
            elif type == "files_and_folders":
                if ' ' in item:
                    # Rename the file or folder by replacing spaces with underscores
                    new_item_name = item.replace(' ', '_')
                    new_item_path = os.path.join(folder_path, new_item_name)
                    os.rename(item_path, new_item_path)
                    if os.path.isfile(new_item_path):
                        print(f"Renamed file '{item}' to '{new_item_name}'.")
                    elif os.path.isdir(new_item_path):
                        print(f"Renamed folder '{item}' to '{new_item_name}'.") 
    
    ###################################################################################     
    @classmethod
    def add_char_between_strings(cls,
                  string_1: str,           # The first string.
                  string_2: str,           # The second string.
                  max_char: int = 100,
                  char: str = ".") -> str:
        """
        Concatenates two strings with a specified maximum character length by inserting dots between them to fill the space.

        Args:
            string_1 (str): The first string.
            string_2 (str): The second string.
            max_char (int, optional): The maximum character length for the concatenated string. Defaults to 100.

        Returns:
            str: The concatenated string with added spaces.
        """
        # Calculate the lengths of string_1 and string_2
        len_1, len_2 = len(string_1), len(string_2)
        
        # Calculate the number of spaces needed between the two strings
        len_space = max_char - len_1 - len_2
        
        if len_space > 0:
            # Concatenate string_1, the calculated number of spaces (represented by dots), and string_2
            new_string = string_1 + len_space * char + string_2
            
            # Return the new string with added spaces
            return new_string
        
        else:
            # If the available space is not enough to insert dots, concatenate strings with a single space between them
            new_string = string_1 + " " + string_2

            return new_string
        
    ###################################################################################  
    @staticmethod
    def append_extension_to_file(
                                 file_path,
                                 additional_extension = None):
        
        # Construct the new path with the new name
        new_path = file_path + additional_extension

        # Rename the file
        os.rename(file_path, new_path)
            
    ###################################################################################
    @staticmethod
    def drop_nan_in_dataframe(
                  df: pd.DataFrame) -> None:

        df = df.dropna()
        
        return df
                    
    ###################################################################################
    @classmethod
    def print_list_in_new_lines(cls,
                                list,
                                additional_name):
    
        for item in list:
            print(additional_name, cls.index, " = " , item)
        
        cls.index += 1
            
        # if len(list) == 1:
        #     print(Basics.add_char_between_strings(str(len(list)) + " item", "found"))
        # else:
        #     print(Basics.add_char_between_strings(str(len(list)) + " items", "found"))

    ###################################################################################
    @staticmethod
    def add_underscore_to_strings(*strings):
        
        return '_'.join(strings)
    
    ###################################################################################
    @staticmethod
    def is_file_in_folder(folder_path, file_name):
        """
        Check if a given file exists within a specified folder.

        Parameters:
        - folder_path: The path to the folder where the file is supposed to be.
        - file_name: The name of the file to check for.

        Returns:
        - True if the file exists in the folder, False otherwise.
        """
        # Construct the full path to the file
        file_path = os.path.join(folder_path, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The file '{file_name}' exists in the folder '{folder_path}'.")
            return True
        else:
            print(f"The file '{file_name}' does not exist in the folder '{folder_path}'.")
            return False
        
    ###################################################################################
    @staticmethod
    def delete_hidden_files_inside_folder(folder_path):
        
        # Get list of files in the folder
        files = os.listdir(folder_path)
        
        # Iterate through the files
        for file_name in files:
            # Check if the file is hidden (starts with a dot)
            if file_name.startswith('.'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Attempt to delete the file
                    os.remove(file_path)
                    print(f"Deleted hidden file: {file_path}")
                except Exception as e:
                    # If deletion fails, print error message
                    print(f"Failed to delete file: {file_path}, Error: {e}")

    ###################################################################################
    @staticmethod
    def check_hidden_file_availability(folder_path):

        # Get the list of files in the directory
        files = os.listdir(folder_path)
        
        # Check if any file is hidden
        for file in files:
            if file.startswith('.'):
                return True
        
        # If no hidden files found
        return False

    ###################################################################################