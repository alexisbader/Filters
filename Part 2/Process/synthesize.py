import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

def apply_bandpass_filter(data, sample_rate, low_cutoff, high_cutoff):
    # Create the bandpass filter
    b, a = signal.butter(4, [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)], btype='band')
    # Apply the bandpass filter to the data
    return signal.lfilter(b, a, data)

def synthesize_chunk_rms(chunk_data, sample_rate, center_freq, rms_value):
    t = np.linspace(0, len(chunk_data) / sample_rate, len(chunk_data))
    return rms_value * np.sin(2 * np.pi * center_freq * t)

def split_wav_file(wav_file_path, chunk_duration_seconds, output_directory, bandwidth=20):
    # Read the .wav file
    sample_rate, data = wavfile.read(wav_file_path)

    # If stereo, take only one channel (assuming left channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Calculate time axis in seconds
    duration = len(data) / sample_rate

    # Calculate the number of samples in each chunk
    chunk_size = int(chunk_duration_seconds * sample_rate)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Calculate the number of chunks and initialize the RMS array
    num_chunks = int(np.ceil(len(data) / chunk_size))
    rms_array = np.zeros((num_chunks, 2))  # 2 columns for center frequency and RMS value

    # Initialize sum for the synthesized signal
    sum_signal = np.zeros(len(data))

    # Split the .wav file into chunks
    for i in range(0, len(data), chunk_size):
        chunk_data = data[i:i+chunk_size]

        # Compute the FFT of the chunk data
        fft_freq = np.fft.fftfreq(len(chunk_data), 1.0 / sample_rate)
        fft_data = np.fft.fft(chunk_data)

        # Only consider positive frequencies
        positive_mask = fft_freq >= 0
        fft_freq_pos = fft_freq[positive_mask]
        fft_data_pos = fft_data[positive_mask]

        # Find the center frequency (frequency with the highest amplitude)
        center_freq = np.abs(fft_freq_pos[np.argmax(np.abs(fft_data_pos))])

        # Calculate the low and high cutoff frequencies for the bandpass filter
        low_cutoff = center_freq - bandwidth / 2
        high_cutoff = center_freq + bandwidth / 2

        # Apply the bandpass filter to the chunk data
        filtered_chunk_data = apply_bandpass_filter(chunk_data, sample_rate, low_cutoff, high_cutoff)

        # Compute the RMS of the filtered chunk
        rms_value = np.sqrt(np.mean(filtered_chunk_data**2))

        # Save the filtered chunk as a new .wav file
        output_file_path = os.path.join(output_directory, f'chunk_{i//chunk_size + 1}.wav')
        wavfile.write(output_file_path, sample_rate, np.int16(filtered_chunk_data))

        # Store center frequency and RMS value in the rms_array
        rms_array[i//chunk_size] = [center_freq, rms_value]

        # Synthesize the chunk RMS by multiplying with sin function
        synthesized_chunk = synthesize_chunk_rms(chunk_data, sample_rate, center_freq, rms_value)

        # Add the synthesized chunk to the sum signal
        sum_signal[i:i+len(synthesized_chunk)] += synthesized_chunk

    # Normalize the sum signal to prevent clipping
    sum_signal /= np.max(np.abs(sum_signal))

    return rms_array, sum_signal

# Replace 'your_wav_file.wav' with the path to your .wav file
wav_file_path = 'Part 2/swamp.wav'
chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
output_directory = 'output_chunks'  # Output directory where the chunks will be saved
bandwidth = 20  # Set the bandwidth for the bandpass filter
sample_rate = 16000

RMS_and_center_freq, sum_signal = split_wav_file(wav_file_path, chunk_duration_seconds, output_directory, bandwidth)
print("RMS values and center frequencies for each chunk:")
print(RMS_and_center_freq)

# Plot each chunk separately
for i, (center_freq, rms_value) in enumerate(RMS_and_center_freq):
    t = np.linspace(0, chunk_duration_seconds, int(chunk_duration_seconds * sample_rate))
    chunk_waveform = synthesize_chunk_rms(np.zeros_like(t), sample_rate, center_freq, rms_value)
    plt.plot(t + i * chunk_duration_seconds, chunk_waveform)

plt.title('Synthesized Sine Waves for Each Chunk')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Plot the full synthesized signal
time_axis = np.linspace(0, len(sum_signal) / sample_rate, len(sum_signal))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, sum_signal)
plt.title('Full Synthesized Sine Wave')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

synthesized_output_path = 'synthesized_audio.wav'
wavfile.write(synthesized_output_path, sample_rate, np.int16(sum_signal))
