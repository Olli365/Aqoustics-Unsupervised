import os
import pandas as pd
import soundfile as sf
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
import numpy as np
from scipy.signal.windows import gaussian



def calculate_spectral_energy(clip):
    """Calculate the spectral energy of an audio clip."""
    return np.sum(clip**2)

def process_audio(file_path, output_folder):
    # Parameters for spectrogram and ROI selection
    std = 1.2
    bin_std = 2 
    bin_per = 0.1
    nperseg = 2048  
    noverlap = 1024

    print(f"Processing file: {file_path}")

    try:
        # Load the audio file
        #print(f"Loading audio file: {file_path}")
        s, fs = sound.load(file_path)
        original_audio = s.copy()  # Keep the original audio unmodified for saving snippets
        #print(f"Audio file loaded successfully: {file_path}, Sample rate: {fs}, Length: {len(s)} samples")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

    # Generate spectrogram
    try:
        #print(f"Generating spectrogram for file: {file_path}")
        db_max = 70
        Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = power2dB(Sxx, db_range=db_max) + db_max
        #print(f"Spectrogram generated successfully for file: {file_path}, Shape: {Sxx.shape}")
    except Exception as e:
        print(f"Error generating spectrogram for file {file_path}: {e}")
        return pd.DataFrame()

    # Remove background and smooth spectrogram
    try:
        #print(f"Removing background and smoothing spectrogram for file: {file_path}")
        Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
        Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=std)
        #print(f"Background removed and spectrogram smoothed for file: {file_path}")
    except Exception as e:
        print(f"Error processing spectrogram for file {file_path}: {e}")
        return pd.DataFrame()

    # Create mask and select ROIs
    try:
        #print(f"Creating mask and selecting ROIs for file: {file_path}")
        im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=bin_std, bin_per=bin_per)
        im_rois, df_rois = select_rois(im_mask, min_roi=100, max_roi=None)
        print(f"ROIs selected for file: {file_path}, Number of ROIs: {len(df_rois)}")
    except Exception as e:
        print(f"Error selecting ROIs for file {file_path}: {e}")
        return pd.DataFrame()

    # Filter ROIs by frequency and duration
    if not df_rois.empty:
        df_rois = format_features(df_rois, tn, fn)
        low_freq_rois = df_rois[(df_rois['max_f'] <= 2000) & 
                                ((df_rois['max_t'] - df_rois['min_t']) >= 0.03)]
        #print(f"Filtered low-frequency ROIs for file: {file_path}, Number of low-frequency ROIs: {len(low_freq_rois)}")
    else:
        print(f"No ROIs found for file: {file_path}")
        return pd.DataFrame()

    if low_freq_rois.empty:
        print(f"No low-frequency ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Extract and save 5-second audio clips from the original audio for each ROI
    clip_duration = 5  # target clip duration in seconds
    pre_roi_time = 1   # time in seconds added before ROI
    audio_clips = []

    for i, (min_t, max_t) in enumerate(low_freq_rois[['min_t', 'max_t']].itertuples(index=False)):
        # Calculate the start and end of the clip
        clip_start = max(0, min_t - pre_roi_time)  # Ensure clip_start is not negative
        clip_end = clip_start + clip_duration

        # Check if there's enough audio to make a full 5-second clip
        if clip_end > len(original_audio) / fs:
            print(f"Skipping ROI {i} in file {file_path} as it does not reach 5 seconds.")
            continue

        # Extract the snippet from the original (un-normalized) audio
        start_sample = int(clip_start * fs)
        end_sample = int(clip_end * fs)
        audio_clip = original_audio[start_sample:end_sample]

        # Extract the segment from 1 second to 3 seconds for energy calculation
        energy_start_sample = start_sample + int(1 * fs)  # Start of energy calculation (1 second)
        energy_end_sample = start_sample + int(3 * fs)  # End of energy calculation (3 seconds)

        # Ensure indices are within bounds
        if energy_end_sample > len(original_audio):
            energy_end_sample = len(original_audio)

        energy_clip = original_audio[energy_start_sample:energy_end_sample]
        spectral_energy = calculate_spectral_energy(energy_clip)

        # Save the audio clip with spectral energy in the filename
        clip_filename = f"clip_{os.path.basename(file_path).split('.')[0]}_{i}_energy_{spectral_energy:.2f}.wav"
        clip_path = os.path.join(output_folder, clip_filename)
        #print(f"Saving clip {i} for file {file_path} to {clip_path}")
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append((clip_start, clip_filename))

    print(f"Finished processing file: {file_path}")
    return pd.DataFrame(audio_clips, columns=['start_time', 'audio_clip'])


def process_folder(input_folder, output_folder, prefix=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through all subfolders and files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if prefix and not filename.startswith(prefix):
                continue

            if filename.lower().endswith('.wav'):
                file_path = os.path.join(root, filename)
                
                # Determine subfolder structure relative to the input folder
                relative_path = os.path.relpath(root, input_folder)
                
                # Ensure the output folder retains the subfolder structure
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                #print(f"Processing file: {file_path} in subfolder: {relative_path}")
                process_audio(file_path, output_subfolder)

# Example usage
input_folder = '/mnt/f/mars_global_acoustic_study/maldives_acoustics/'
output_folder = "/mnt/d/Aqoustics/BEN/Maldives/Maldives_ROI"
prefix = ""
start_time = time.time()
print("Starting processing of folder.")
process_folder(input_folder, output_folder, prefix)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
