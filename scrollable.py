import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.io import wavfile
from scipy.signal import resample


# Load the audio file (replace 'your_audio_file.wav' with your file path)
sampling_rate, signal = wavfile.read("D:/Aqoustics/Audio/Megamix/n_CCC_106.20220628_144100.wav")

# If the audio has multiple channels (stereo), average them to convert to mono
if len(signal.shape) == 2:
    signal = np.mean(signal, axis=1)

# Downsample the audio if it's too large
desired_sampling_rate = 2000  # Adjust this value as needed
if sampling_rate > desired_sampling_rate:
    num_samples = int(len(signal) * (desired_sampling_rate / sampling_rate))
    signal = resample(signal, num_samples)
    sampling_rate = desired_sampling_rate

# Generate time vector
time = np.linspace(0, len(signal) / sampling_rate, num=len(signal))

# Create a spectrogram using matplotlib
fig, ax = plt.subplots()

# Increase NFFT or adjust overlap to reduce the resolution for better performance
Pxx, freqs, bins, im = ax.specgram(signal, NFFT=128, Fs=sampling_rate, noverlap=64)
plt.close(fig)  # Close the matplotlib figure

# Create a Plotly heatmap (scrollable spectrogram)
fig = go.Figure(data=go.Heatmap(z=10 * np.log10(Pxx), x=bins, y=freqs, colorscale='Viridis'))

# Adjust layout to make it scrollable
fig.update_layout(
    xaxis=dict(
        title='Time (s)',
        rangeslider=dict(visible=True),
    ),
    yaxis=dict(
        title='Frequency (Hz)',
        type='log'
    ),
    title='Scrollable Spectrogram'
)

# Display the plot
fig.show()