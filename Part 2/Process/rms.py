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

    return rms_array

# Replace 'your_wav_file.wav' with the path to your .wav file
wav_file_path = 'swamp.wav'
chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
output_directory = 'output_chunks'  # Output directory where the chunks will be saved
bandwidth = 20  # Set the bandwidth for the bandpass filter

RMS_and_center_freq = split_wav_file(wav_file_path, chunk_duration_seconds, output_directory, bandwidth)
print("RMS values and center frequencies for each chunk:")
print(RMS_and_center_freq)
