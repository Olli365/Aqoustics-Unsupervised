import os
import pandas as pd
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
from maad.features import centroid_features
import numpy as np

def process_audio(file_path):
    print(f"Processing file: {file_path}")
    start_time = time.time()

    try:
        # Load the audio file
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return 0

    s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and smoothing
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.25)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    if df_rois.empty:
        print(f"No ROIs found in file: {file_path}")
        return 0

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Calculate centroid features
    df_centroid = centroid_features(Sxx_db, df_rois)

    # Get median frequency and normalize
    median_freq = fn[np.round(df_centroid.centroid_y).astype(int)]
    df_centroid['centroid_freq'] = median_freq / fn[-1]

    # Filter ROIs for those with centroid frequency below 2000Hz
    low_freq_rois = df_rois[df_centroid['centroid_freq'] * fn[-1] < 2000]

    if low_freq_rois.empty:
        print(f"No low frequency ROIs found in file: {file_path}")
        return 0

    # Return the number of low-frequency ROIs (tags)
    num_tags = len(low_freq_rois)
    print(f"Number of tags found in file: {file_path}: {num_tags}")
    
    return num_tags

def process_folder(input_folder):
    total_tags = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            num_tags = process_audio(file_path)
            total_tags += num_tags

    print(f"Total number of tags found in folder: {total_tags}")
    return total_tags

# Example usage
input_folder = 'D:/Aqoustics/UMAP/Sorted/D_files/'
start_time = time.time()
total_tags = process_folder(input_folder)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
