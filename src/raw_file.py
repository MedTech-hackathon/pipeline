import numpy as np


class RawFile:
    
    ###################################################################################
    @staticmethod
    def read_raw_file(
        file_path: str) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        """
        Read a raw file and extract header information, timestamps, and data.

        Args:
            file_path (str): Path to the raw file.

        Returns:
            tuple: A tuple containing header information (dict), timestamps (numpy.ndarray), and data (numpy.ndarray).
        """
        # Define header information fields
        hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')
        
        # Initialize dictionaries and arrays to store header, timestamps, and data
        hdr, timestamps, data = {}, None, None
        
        # Open the raw file in binary mode
        with open(file_path, 'rb') as raw_bytes:
            
            # Read header information (4 bytes each)
            for info in hdr_info:
                hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')
            
            # read timestamps and data
            timestamps = np.zeros(hdr['frames'], dtype='int64')
                            
            # Calculate the size of each frame
            sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
            
            # Initialize data array based on file type
            if "_rf.raw" in file_path:
                data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='int16')
            if "_env.raw" in file_path:
                data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='int8')

            # Loop over frames
            for frame in range(hdr['frames']):
                
                # Read timestamp for each frame (8 bytes)
                timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
                
                # Read frame data and reshape it to match dimensions specified in the header
                if "_rf.raw" in file_path:
                    data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='int16').reshape([hdr['lines'], hdr['samples']])
                if "_env.raw" in file_path:
                    data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='uint8').reshape([hdr['lines'], hdr['samples']])

        # Print message indicating the number of frames loaded and their size
        print('Loaded {d[2]} raw frames of size, {d[0]} x {d[1]} (lines x samples)'.format(d=data.shape))
            
        # Return header, timestamps, and data
        return hdr, timestamps, data
        
    ##################################################################################  
    @classmethod
    def read_raw_files(cls,
                        raw_files_path_list: list[str]) -> None:
        """
        Function to read raw files from a list of file paths, convert them to numpy arrays, 
        and save them as '.npy' files.

        Args:
            raw_files_path_list (list[str]): List of file paths to raw files.

        Returns:
            None
        """
        for raw_file_path in raw_files_path_list:
            
            # Read raw files and convert them to numpy arrays
            hdr, timestamps, data = cls.read_raw_file(raw_file_path) 
            
            # Save data as numpy array
            np.save(raw_file_path + ".rf", data)
    
    ###################################################################################
    
      
 