import os
import pandas as pd
import soundfile as sf
from maad import sound
from maad.util import power2dB, overlay_rois, format_features
from maad.rois import create_mask, select_rois
from maad.features import shape_features, centroid_features
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np

def process_audio(file_path, output_folder):
    # Load the audio file
    s, fs = sound.load(file_path)
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

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Calculate shape and centroid features
    df_shape, params = shape_features(Sxx_db, resolution='low', rois=df_rois)
    df_centroid = centroid_features(Sxx_db, df_rois)

    # Get median frequency and normalize
    median_freq = fn[np.round(df_centroid.centroid_y).astype(int)]
    df_centroid['centroid_freq'] = median_freq / fn[-1]

    # t-SNE for dimensionality reduction
    X = df_shape.loc[:, df_shape.columns.str.startswith('shp')]
    X = X.join(df_centroid.centroid_freq)  # add column and normalize values

    tsne = TSNE(n_components=2, perplexity=12, init='pca', verbose=True)
    Y = tsne.fit_transform(X)

    # Clustering using DBSCAN
    cluster = DBSCAN(eps=5, min_samples=4).fit(Y)

    # Overlay bounding box on the original spectrogram
    df_rois['label'] = cluster.labels_.astype(str)
    overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':60, 'extent':ext})

    # Filter ROIs for those with centroid frequency below 1000Hz
    low_freq_rois = df_rois[df_centroid['centroid_freq'] * fn[-1] < 1000]

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    audio_clips = []
    for i, row in low_freq_timestamps.iterrows():
        # Add half a second (0.5 seconds) to the beginning and end of the clip
        start_sample = int(max(0, (row['begin'] - 0.5) * fs))
        end_sample = int(min(len(s), (row['end'] + 0.5) * fs))
        audio_clip = s[start_sample:end_sample]
        clip_filename = f'clip_{os.path.basename(file_path).split(".")[0]}_{i}.wav'
        clip_path = os.path.join(output_folder, clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append(clip_filename)

    low_freq_timestamps['audio_clip'] = audio_clips

    return low_freq_timestamps

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_timestamps = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            timestamps = process_audio(file_path, output_folder)
            all_timestamps.append(timestamps)

    return all_timestamps

# Example usage
input_folder = "D:/Aqoustics/Unsupervised/Mary Test/Data"
output_folder = 'D:/Aqoustics/Unsupervised/Mary Test'
all_timestamps = process_folder(input_folder, output_folder)

# Save the timestamps to a CSV file
for i, timestamps in enumerate(all_timestamps):
    timestamps.to_csv(os.path.join(output_folder, f'timestamps_{i}.csv'), index=False)
