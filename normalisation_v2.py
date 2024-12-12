import os
import pandas as pd
import soundfile as sf
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def process_audio(file_path):
    # Parameters for spectrogram and ROI selection
    std = 2
    bin_std = 3 
    bin_per = 0.25  # Retains only the top 25% of pixel intensities
    nperseg = 2048  # FFT size to match Audacity
    noverlap = 1024  # 50% overlap

    print(f"Processing file: {file_path}")

    try:
        # Load the audio file
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()
    
    # Step 1: Remove DC offset
    s = s - np.mean(s)

    # Step 2: Normalize to a target peak level (e.g., -1 dB or 0.891 scale)
    # Target peak level in linear scale
    target_peak = 0.891
    current_peak = np.max(np.abs(s))
    if current_peak > 0:
        s = s * (target_peak / current_peak)

    print(f"Normalized audio range after DC removal: [{s.min()}, {s.max()}]")

    # Generate spectrogram
    db_max = 100  # dB range to match Audacity's scale
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Remove background and smooth spectrogram
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=std)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=bin_std, bin_per=bin_per)
    im_rois, df_rois = select_rois(im_mask, min_roi=100, max_roi=None)

    # Filter ROIs by frequency range
    if not df_rois.empty:
        df_rois = format_features(df_rois, tn, fn)
        low_freq_rois = df_rois[df_rois['max_f'] <= 2000]
        low_freq_rois = low_freq_rois[low_freq_rois['min_f'] >= 50]
        low_freq_rois = low_freq_rois[(low_freq_rois['max_t'] - low_freq_rois['min_t']) >= 0.03]
    else:
        low_freq_rois = pd.DataFrame()

    # Plot the spectrogram with ROIs
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(Sxx_db, aspect='auto', origin='lower', extent=ext, vmin=0, vmax=100, cmap='magma', interpolation='bilinear')

    # Overlay ROIs manually if there are any
    if not low_freq_rois.empty:
        for _, roi in low_freq_rois.iterrows():
            # Create a rectangle patch for each ROI
            rect = patches.Rectangle(
                (roi['min_t'], roi['min_f']),
                roi['max_t'] - roi['min_t'],
                roi['max_f'] - roi['min_f'],
                linewidth=1,
                edgecolor='yellow',
                facecolor='none'
            )
            ax.add_patch(rect)

    # Set the y-axis to log scale and set consistent limits and ticks
    ax.set_yscale('log')
    ax.set_ylim([100, 5000])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim([0, ext[1]])

    # Add labels and title
    ax.set_title(f"Spectrogram for {os.path.basename(file_path)}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # Add colorbar
    plt.colorbar(img, ax=ax, label="dB")
    
    # Display the plot
    plt.show()

# Specify the path to a single audio file
# Uncomment the appropriate file paths
# file_path = "/mnt/f/mars_global_acoustic_study/indonesia_acoustics/raw_audio/ind_D1_20220829_120000.WAV"

file_path = "/mnt/f/mars_global_acoustic_study/test/20211023_163600.WAV"
#file_path = "/mnt/f/mars_global_acoustic_study/test/20230206_212800.WAV"
#file_path = "/mnt/f/mars_global_acoustic_study/test/20230222_062800.WAV"
#file_path = "/mnt/f/mars_global_acoustic_study/test/20230528_213200.WAV"
file_path = "/mnt/f/mars_global_acoustic_study/test/ind_R4_20220919_062400.WAV"

process_audio(file_path)
