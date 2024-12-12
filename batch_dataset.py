import os
import pandas as pd
import soundfile as sf
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
from maad.features import centroid_features
import numpy as np

def process_audio(file_path, output_folder):
    print(f"Processing file: {file_path}")
    start_time = time.time()

    try:
        # Load the audio file
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

    try:
        # Attempt to filter the signal
        s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')
    except ValueError as e:
        print(f"Skipping file {file_path} due to filtering error: {e}")
        return pd.DataFrame()

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and smoothing
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.1)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    if df_rois.empty:
        print(f"No ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Filter ROIs for those with centroid frequency below 2000Hz
    low_freq_rois = df_rois[df_rois['max_f'] <= 2000]
    print(low_freq_rois)

    if low_freq_rois.empty:
        print(f"No low frequency ROIs found in file: {file_path}")
        return pd.DataFrame()

    # **Change**: If more than 30 ROIs, subsample until the number is 30
    while len(low_freq_rois) > 30:
        low_freq_rois = low_freq_rois.iloc[::2]  # Keep every 2nd ROI until there are 30 or fewer

    # Print how many ROIs were made for this file
    print(f"Number of ROIs made for file {file_path}: {len(low_freq_rois)}")

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    # Generate 5-second audio clips with detected region in the middle
    audio_clips = []
    clip_duration = 2  # seconds

    for i, (start, end) in enumerate(low_freq_timestamps.itertuples(index=False)):
        mid_point = (start + end) / 2
        clip_start = max(0, mid_point - clip_duration / 2)
        clip_end = clip_start + clip_duration

        # Check if the clip is in the first 5 seconds or last 5 seconds of the audio
        if clip_start < 0:
            clip_start = 0
            clip_end = clip_duration
        elif clip_end > len(s) / fs:
            clip_end = len(s) / fs
            clip_start = clip_end - clip_duration

        start_sample = int(clip_start * fs)
        end_sample = int(clip_end * fs)
        audio_clip = s[start_sample:end_sample]
        clip_filename = f'clip_{os.path.basename(file_path).split(".")[0]}_{i}.wav'
        clip_path = os.path.join(output_folder, clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append((clip_start, clip_filename))

        # Add zero padding to make the clip 5 seconds long
        target_duration = 5  # target duration in seconds
        target_samples = int(target_duration * fs)  # target duration in samples
        current_samples = len(audio_clip)  # current clip length in samples

        if current_samples < target_samples:
            # Calculate the amount of zero padding needed
            padding_needed = target_samples - current_samples
            # Add padding equally to the start and end
            pad_before = padding_needed // 2
            pad_after = padding_needed - pad_before
            # Apply padding
            padded_clip = np.pad(audio_clip, (pad_before, pad_after), 'constant')
        else:
            # If the clip is already the target duration, no padding is needed
            padded_clip = audio_clip

        # Save the padded clip to the same file
        sf.write(clip_path, padded_clip, fs)

    # Create DataFrame for the audio clips
    df_audio_clips = pd.DataFrame(audio_clips, columns=['start_time', 'audio_clip'])

    end_time = time.time()
    print(f"Finished processing file: {file_path}")

    return df_audio_clips


def process_folder(input_folder, output_folder, prefix=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_timestamps = []
    
    # Traverse through all subfolders and files in the input folder
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            # **Change 2**: Process files only with the given prefix (if any)
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

                timestamps = process_audio(file_path, output_subfolder)
                if not timestamps.empty:
                    timestamps['file'] = filename
                    timestamps['folder'] = relative_path
                    all_timestamps.append(timestamps)

    if all_timestamps:
        # Concatenate all timestamps and save to Excel
        all_timestamps_df = pd.concat(all_timestamps, ignore_index=True)
        excel_path = os.path.join(output_folder, 'timestamps.xlsx')
        all_timestamps_df.to_excel(excel_path, index=False)
        return all_timestamps_df
    else:
        print("No audio clips generated from any files.")
        return pd.DataFrame()


# Example usage
input_folder = '/mnt/f/mars_global_acoustic_study/maldives_acoustics/'
output_folder = "/mnt/d/Aqoustics/BEN/Maldives/Maldives_ROI/"
prefix = "ind_R"  
start_time = time.time()
all_timestamps = process_folder(input_folder, output_folder, prefix)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
