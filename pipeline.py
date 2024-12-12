import os
import pandas as pd
import soundfile as sf
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
from maad.features import centroid_features
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
import gc

# Remove saving functionality and create in-memory audio segment list
def process_audio(file_path, model, target_sr=32000, required_length=160000, features_df_path=None):
    print(f"Processing file: {file_path}")
    start_time = time.time()

    try:
        # Load the audio file in chunks to save memory
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    try:
        # Attempt to filter the signal
        s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')
    except ValueError as e:
        print(f"Skipping file {file_path} due to filtering error: {e}")
        return

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
        return

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Filter ROIs for those with centroid frequency below 2000Hz
    low_freq_rois = df_rois[df_rois['max_f'] <= 2000]
    print(low_freq_rois)

    if low_freq_rois.empty:
        print(f"No low frequency ROIs found in file: {file_path}")
        return

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    # Process each ROI
    for i, (start, end) in enumerate(low_freq_timestamps.itertuples(index=False)):
        mid_point = (start + end) / 2
        clip_duration = 2  # seconds
        clip_start = max(0, mid_point - clip_duration / 2)
        clip_end = clip_start + clip_duration

        if clip_start < 0:
            clip_start = 0
            clip_end = clip_duration
        elif clip_end > len(s) / fs:
            clip_end = len(s) / fs
            clip_start = clip_end - clip_duration

        start_sample = int(clip_start * fs)
        end_sample = int(clip_end * fs)
        audio_clip = s[start_sample:end_sample]

        # Resample the audio clip
        segment_resampled = librosa.resample(audio_clip, orig_sr=fs, target_sr=target_sr)

        # Pad or truncate the segment to match the required input length
        if len(segment_resampled) < required_length:
            # Pad with zeros to match the required length
            padding = required_length - len(segment_resampled)
            segment_resampled = np.pad(segment_resampled, (0, padding), mode='constant')
        else:
            # Truncate the segment if it's longer than required
            segment_resampled = segment_resampled[:required_length]

        # Model expects batch dimension, so use np.newaxis to add it
        segment_resampled = segment_resampled[np.newaxis, :]

        # Infer the segment using the model
        try:
            output = model.infer_tf(segment_resampled)

            # Extract the embedding tensor
            embeddings = output['embedding']

            # Convert to numpy array
            embedding = embeddings.numpy()[0]

            # Create a row data dictionary, starting with the clip name
            row_data = {'clip_name': f"{file_path}_segment_{i + 1}"}
            
            # Add the embedding features to the row data
            for j, feature in enumerate(embedding):
                row_data[f'feature_{j}'] = feature

            # Append the data to CSV immediately to avoid storing in memory
            features_df = pd.DataFrame([row_data])
            features_df.to_csv(features_df_path, mode='a', header=not os.path.exists(features_df_path), index=False)

        except Exception as e:
            print(f"Error processing segment {i} in file {file_path}: {e}")

    # Clear memory for the current file
    del s, s_filt, Sxx, Sxx_db, Sxx_db_rmbg, Sxx_db_smooth, im_mask, im_rois, df_rois, low_freq_rois
    gc.collect()

    end_time = time.time()
    print(f"Finished processing file: {file_path}")


# Process folder, one file at a time, writing results after each file
def process_folder(input_folder, model, features_df_path, target_sr=32000, required_length=160000):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]

    for filename in tqdm(files, desc="Processing folder"):
        file_path = os.path.join(input_folder, filename)
        process_audio(file_path, model, target_sr=target_sr, required_length=required_length, features_df_path=features_df_path)


# Main script
input_folder = 'F:/mars_global_acoustic_study/indonesia_acoustics/raw_audio'
model_path = 'D:/Aqoustics/Aqoustics-Unsupervised/kaggle'

# Load pre-trained TensorFlow model
model = tf.saved_model.load(model_path)

# Path to save the CSV
results_path = 'D:/Aqoustics/Aqoustics-Unsupervised/data/embeddings'
if not os.path.exists(results_path):
    os.mkdir(results_path)

features_df_path = os.path.join(results_path, 'IND_Dataset.csv')

# Process the input folder one file at a time
process_folder(input_folder, model, features_df_path)
