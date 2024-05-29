from src.basics import Basics

import os
import subprocess

    
class LzoFile:
    
    lzo_exe_file_path = "data/source_files/lzop103w"

    ###################################################################################
    @classmethod
    def read_lzo_file(cls,
                    lzo_file_path: str) -> None:
        """
        Reads an LZO file by decompressing it using lzop.exe.

        Args:
            lzo_file_path (str): Path to the LZO file.
        """
        # Path to the directory containing lzop.exe
        lzop_exe_path = cls.get_lzo_exe_file_path()
        
        # Run the lzop.exe command to decompress the LZO file
        subprocess.run(f'{lzop_exe_path}/lzop.exe' + ' ' + '-d' + ' ' + lzo_file_path)

    ###################################################################################
    @classmethod
    def read_lzo_files(cls,
                    lzo_files_path_list: list[str]) -> None:
        """
        Reads and decompresses a list of LZO files using lzop.exe.

        Args:
            lzo_files_path_list (list): A list of file paths to LZO files.

        Note:
            This method assumes that lzop.exe is located in a directory named 'lzop103w'
            within the current working directory or its parent directory.
        """
        # Loop through each path in the list of LZO files
        for lzo_file_path in lzo_files_path_list:
            
            # Call read_lzo_file method to decompress each LZO file
            cls.read_lzo_file(lzo_file_path)

    ###################################################################################
    @classmethod
    def get_lzo_exe_file_path(cls):
        """
        Get the path to the directory containing lzop.exe.

        Returns:
            str: Path to the directory containing lzop.exe.
        """
        # Get the path of the current working directory
        working_space_path = Basics.path_of_working_space()
        
        # Construct the path to the directory containing lzop.exe
        path_of_lzo_exe_file = os.path.join(working_space_path, cls.lzo_exe_file_path)

        return path_of_lzo_exe_file
    
    ######################################################################################

