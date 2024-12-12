import os
import shutil

def find_files(directory, prefix='ind_', letters=['D', 'H'], count=100):
    # Dictionary to store the found files for each letter
    found_files = {letter: [] for letter in letters}
    
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            # Remove the prefix
            new_filename = filename[len(prefix):]
            # Check the first character of the new filename
            first_char = new_filename[0]
            if first_char in found_files and len(found_files[first_char]) < count:
                found_files[first_char].append(filename)
        
        # Stop if we have found enough files for both letters
        if all(len(files) >= count for files in found_files.values()):
            break
    
    return found_files

def copy_files(file_dict, source_directory, target_directory):
    # Create target directories for each letter if they don't exist
    for letter, files in file_dict.items():
        letter_dir = os.path.join(target_directory, f'{letter}_files')
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
        
        for file in files:
            shutil.copy(os.path.join(source_directory, file), letter_dir)

# Specify the source directory and target directory
source_directory = 'D:/Aqoustics/UMAP'
target_directory = 'D:/Aqoustics/UMAP/Sorted/'

# Find the files
found_files = find_files(source_directory)

# Copy the found files to their respective folders in the target directory
copy_files(found_files, source_directory, target_directory)

print(f"Found and copied {len(found_files['D'])} files starting with 'D' and {len(found_files['H'])} files starting with 'H'.")
