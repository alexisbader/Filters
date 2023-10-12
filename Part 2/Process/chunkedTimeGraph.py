# import numpy as np
# import scipy.io.wavfile as wavfile
# import matplotlib.pyplot as plt
# import os

# def split_wav_file(wav_file_path, chunk_duration_seconds, output_directory):
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
#         chunk_time = time[i:i+chunk_size]

#         # Plot the waveform for the chunk
#         plt.figure(figsize=(10, 4))
#         plt.plot(chunk_time, chunk_data)
#         plt.title(f'Waveform - Chunk {i//chunk_size}')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Amplitude')
#         plt.tight_layout()

#         # Save the plot as an image in the output directory
#         output_file = os.path.join(output_directory, f"chunk_{i // chunk_size}.png")
#         plt.savefig(output_file)
#         plt.show()
#         plt.close()

# # Replace 'your_wav_file.wav' with the path to your .wav file
# wav_file_path = 'swamp.wav'
# chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
# output_directory = 'output_chunks'  # Output directory where the chunks will be saved

# split_wav_file(wav_file_path, chunk_duration_seconds, output_directory)


import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import os

def split_wav_file(wav_file_path, chunk_duration_seconds, output_directory):
    # Read the .wav file
    sample_rate, data = wavfile.read(wav_file_path)

    # If stereo, take only one channel (assuming left channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Calculate time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0., duration, len(data))

    # Calculate the number of samples in each chunk
    chunk_size = int(chunk_duration_seconds * sample_rate)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Split the .wav file into chunks
    for i in range(0, len(data), chunk_size):
        chunk_data = data[i:i+chunk_size]

        # Calculate time axis for the chunk
        chunk_duration = len(chunk_data) / sample_rate
        chunk_time = np.linspace(0., chunk_duration, len(chunk_data))

        # Compute the FFT of the chunk data
        fft_freq = np.fft.fftfreq(len(chunk_data), 1.0 / sample_rate)
        fft_data = np.fft.fft(chunk_data)

        # Plot the frequency domain for the chunk
        plt.figure(figsize=(10, 4))
        plt.plot(fft_freq, np.abs(fft_data))
        plt.title(f'Frequency Domain - Chunk {i//chunk_size + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.show()

# Replace 'your_wav_file.wav' with the path to your .wav file
wav_file_path = 'Part 2/swamp.wav'
chunk_duration_seconds = 0.2  # Adjust this value to set the duration of each chunk (in seconds)
output_directory = 'output_chunks'  # Output directory where the chunks will be saved

split_wav_file(wav_file_path, chunk_duration_seconds, output_directory)
