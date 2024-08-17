import os
import pandas as pd
import soundfile as sf
import time
import numpy as np
import librosa
import tensorflow as tf
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
from maad.features import centroid_features

def process_audio(file_path, output_folder):
    print(f"Processing file: {file_path}")
    start_time = time.time()

    try:
        # Load the audio file
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

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
        return pd.DataFrame()

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
        return pd.DataFrame()

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps['start_time'] = low_freq_timestamps['min_t']
    low_freq_timestamps['end_time'] = low_freq_timestamps['max_t']
    
    # Generate 5-second clips based on timestamps
    clips = []
    for _, row in low_freq_timestamps.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        clips.append((start_time, end_time))
    
    if len(clips) == 0:
        return pd.DataFrame()
    
    # Ensure clips are at least 5 seconds
    clips[-1] = (clips[-1][0], max(clips[-1][0] + 5, clips[-1][1]))
    
    # Generate audio clips and embeddings
    audio_clips = []
    for i, (start, end) in enumerate(clips):
        start_sample = int(max(0, start * fs))
        end_sample = int(min(len(s), end * fs))
        audio_clip = s[start_sample:end_sample]
        clip_filename = f'clip_{os.path.basename(file_path).split(".")[0]}_{i}.wav'
        clip_path = os.path.join(output_folder, clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append((start, clip_filename))
        
        # Generate embedding
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_clip, sr=fs, n_mels=128, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        feature = {
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[clip_filename.encode()])),
            'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=mel_spectrogram_db.flatten())),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        serialized_example = example.SerializeToString()
        tfrecord_filename = os.path.join(output_folder, f'embedding_{i}.tfrecord')
        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(serialized_example)
    
    # Create DataFrame for the audio clips
    df_audio_clips = pd.DataFrame(audio_clips, columns=['start_time', 'audio_clip'])

    end_time = time.time()
    print(f"Finished processing file: {file_path}")

    return df_audio_clips

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_timestamps = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            timestamps = process_audio(file_path, output_folder)
            if not timestamps.empty:
                timestamps['file'] = filename
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


input_folder = 'D:/Aqoustics/Unsupervised/Perch Test/Audio'
output_folder = 'D:/Aqoustics/Unsupervised/Perch Test'
start_time = time.time()
all_timestamps = process_folder(input_folder, output_folder)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")


