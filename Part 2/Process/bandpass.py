# import numpy as np
# import scipy.io.wavfile as wavfile
# import scipy.signal as signal
# import matplotlib.pyplot as plt
# import os

# def apply_bandpass_filter(data, sample_rate, low_cutoff, high_cutoff):
#     # Create the bandpass filter
#     b, a = signal.butter(4, [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)], btype='band')
#     # Apply the bandpass filter to the data
#     return signal.lfilter(b, a, data)

# def split_wav_file(wav_file_path, chunk_duration_seconds, output_directory, bandwidth=20):
#     # Read the .wav file
#     sample_rate, data = wavfile.read(wav_file_path)

#     # If stereo, take only one channel (assuming left channel)
#     if len(data.shape) > 1:
#         data = data[:, 0]

#     # Calculate time axis in seconds
#     duration = len(data) / sample_rate
#     time = np.linspace(0., duration, len(data))

#     # Calculate the number of samples in each chunk
#     chunk_size = int(chunk_duration_seconds * sample_rate)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     # Split the .wav file into chunks
#     for i in range(0, len(data), chunk_size):
#         chunk_data = data[i:i+chunk_size]

#         # Calculate time axis for the chunk
#         chunk_duration = len(chunk_data) / sample_rate
#         chunk_time = np.linspace(0., chunk_duration, len(chunk_data))

#         # Compute the FFT of the chunk data
#         fft_freq = np.fft.fftfreq(len(chunk_data), 1.0 / sample_rate)
#         fft_data = np.fft.fft(chunk_data)

#         # Only consider positive frequencies
#         positive_mask = fft_freq >= 0
#         fft_freq_pos = fft_freq[positive_mask]
#         fft_data_pos = fft_data[positive_mask]

#         # Find the center frequency (frequency with the highest amplitude)
#         center_freq = np.abs(fft_freq_pos[np.argmax(np.abs(fft_data_pos))])

#         # Calculate the low and high cutoff frequencies for the bandpass filter
#         low_cutoff = center_freq - bandwidth / 2
#         high_cutoff = center_freq + bandwidth / 2

#         # Apply the bandpass filter to the chunk data
#         filtered_chunk_data = apply_bandpass_filter(chunk_data, sample_rate, low_cutoff, high_cutoff)

#         # Compute the FFT of the filtered chunk data
#         fft_freq_filtered = np.fft.fftfreq(len(filtered_chunk_data), 1.0 / sample_rate)
#         fft_data_filtered = np.fft.fft(filtered_chunk_data)

#         # Plot the frequency domain for the filtered chunk
#         plt.figure(figsize=(10, 4))
#         plt.plot(fft_freq_filtered, np.abs(fft_data_filtered))
#         plt.title(f'Filtered Frequency Domain - Chunk {i//chunk_size + 1}')
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('Magnitude')
#         plt.tight_layout()
#         plt.show()

#         # Save the filtered chunk as a new .wav file
#         output_file_path = os.path.join(output_directory, f'chunk_{i//chunk_size + 1}.wav')
#         wavfile.write(output_file_path, sample_rate, np.int16(filtered_chunk_data))

# # Replace 'your_wav_file.wav' with the path to your .wav file
# wav_file_path = 'Part 2/swamp.wav'
# chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
# output_directory = 'output_chunks'  # Output directory where the chunks will be saved
# bandwidth = 20  # Set the bandwidth for the bandpass filter

# split_wav_file(wav_file_path, chunk_duration_seconds, output_directory, bandwidth)

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

def process_chunks(chunk_array, sample_rate, output_directory, bandwidth=20):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for i, chunk_data in enumerate(chunk_array):
        # Calculate time axis for the chunk
        chunk_duration = len(chunk_data) / sample_rate
        chunk_time = np.linspace(0., chunk_duration, len(chunk_data))

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

        # Compute the FFT of the filtered chunk data
        fft_freq_filtered = np.fft.fftfreq(len(filtered_chunk_data), 1.0 / sample_rate)
        fft_data_filtered = np.fft.fft(filtered_chunk_data)

        # Only consider positive frequencies in the filtered data
        fft_freq_filtered_pos = fft_freq_filtered[positive_mask]
        fft_data_filtered_pos = fft_data_filtered[positive_mask]

        # Plot the frequency domain for the filtered chunk (positive frequencies only)
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freq_filtered_pos, np.abs(fft_data_filtered_pos))
        plt.title(f'Filtered Frequency Domain - Chunk {i + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.show()

        # Save the filtered chunk as a new .wav file
        output_file_path = os.path.join(output_directory, f'filtered_chunk_{i + 1}.wav')
        wavfile.write(output_file_path, sample_rate, np.int16(filtered_chunk_data))

def split_wav_file(wav_file_path, chunk_duration_seconds, overlap_percentage, output_directory, bandwidth=20):
    # Read the .wav file
    sample_rate, data = wavfile.read(wav_file_path)

    # If stereo, take only one channel (assuming left channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Calculate the number of samples in each chunk
    chunk_size = int(chunk_duration_seconds * sample_rate)

    # Calculate the number of overlapping samples
    overlap_samples = int(chunk_size * overlap_percentage)

    # Split the .wav file into overlapping chunks
    chunk_array = []
    i = 0
    while i < len(data):
        chunk_data = data[i:i+chunk_size]
        chunk_array.append(chunk_data)
        i += chunk_size - overlap_samples

    # Call the process_chunks function to apply the bandpass filter to each chunk
    process_chunks(chunk_array, sample_rate, output_directory, bandwidth)

# Replace 'your_wav_file.wav' with the path to your .wav file
wav_file_path = 'Part 2/swamp.wav'
chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
overlap_percentage = 0.5      # Adjust this value to set the percentage of overlap between chunks
output_directory = 'output_filtered_chunks'  # Output directory where the filtered chunks will be saved
bandwidth = 20  # Set the bandwidth for the bandpass filter

split_wav_file(wav_file_path, chunk_duration_seconds, overlap_percentage, output_directory, bandwidth)
