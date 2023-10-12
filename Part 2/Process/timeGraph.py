import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

def plot_wav_file(wav_file_path):
    # Read the .wav file
    sample_rate, data = wavfile.read(wav_file_path)

    # If stereo, take only one channel (assuming left channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Calculate time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

# Replace 'your_wav_file.wav' with the path to your .wav file
plot_wav_file('output_filtered_chunks/synthesized_signal.wav')
