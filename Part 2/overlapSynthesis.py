# Import statements 
# Import statements
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt
import os


# Down sampling our audio file
def downSample(audio, sampling_rate, fs):
    down_sampled_audio = signal.resample(audio, len(audio) * fs // sampling_rate)
    return(down_sampled_audio)


# Split audio file into chunks
def splitWaveFile(wav_file_path, chunk_duration_seconds, overlap_percentage, sample_rate):
    
    # Read the .wav file
    fs, data = wavfile.read(wav_file_path)
    data = downSample(data, fs, fs=sample_rate)

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

    # Return array of chunks
    return chunk_array


# Apply bandpass filter to chunks
def apply_bandpass_filter(data, sample_rate, low_cutoff, high_cutoff):
    
    # Condition to account for lower cutoffs in chunks
    if low_cutoff <= 1:
        low_cutoff = 1

    # Create the bandpass filter
    b, a = signal.butter(4, [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)], btype='band')
    
    # Apply the bandpass filter to the data
    return signal.lfilter(b, a, data)


# Finding RMS values, returns array of RMS and center frequency for each chunk
def frequency_analysis(chunk_array, sample_rate, output_directory, bandwidth=100):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    chunk_info = []  # 2D array to store the RMS and center frequency of each chunk

    for i, chunk_data in enumerate(chunk_array):

        t = np.linspace(i * 0.3, (i + 1)*0.3, int(sample_rate * 0.3))
    
        center_freq = np.linspace(0, 7000, 140)

        sum_signal = []
        
        # Loop through the center frequencies in the
        for center_frequency in center_freq:
            
            # getting high and low cutoff values
            low_cutoff = center_frequency - bandwidth / 2
            high_cutoff = center_frequency + bandwidth / 2
            
            # Apply the bandpass filter function to the center frequency
            filtered_chunk_data = apply_bandpass_filter(chunk_data, sample_rate, low_cutoff, high_cutoff)

            # Time duration for each signal
            t = np.linspace(0., chunk_duration_seconds, int(sample_rate * 0.3))
    
            # Compute the RMS value of the filtered chunk data
            rms_value = np.sqrt(np.mean(filtered_chunk_data ** 2))

             # Create a sinusoidal signal for the current chunk
            chunk_signal = rms_value * np.sin(2 * np.pi * center_frequency * t)

            # Add the chunk signal to the sum signal
            sum_signal.append(chunk_signal)

        # Add the sum signals together
        value = np.sum(sum_signal, axis = 0)

        # Append signal into chunk_info array
        chunk_info.append(value)

    return np.concatenate(chunk_info, axis = 0)

# Plot
def plot_spectrum(data, sampling_rate):
    n = len(data)
    spectrum = np.abs(np.fft.fft(data)) / n
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    print("max spectrum", max(spectrum[:n//2]))
    print("min spectrum", min(spectrum[:n//2]))
    plt.plot(freq[:n//2], spectrum[:n//2])
    # plt.xlim(10_000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

# Main function to create graphs and output audio file
def main(wav_file_path, output_directory, chunk_array, sample_rate):
    
    # Call the frequency_analysis function to apply the bandpass filter to each chunk
    chunk_info = frequency_analysis(chunk_array, sample_rate, output_directory)
    
    # Plot the segemented audio wave
    sample_rate, data = wavfile.read(wav_file_path)
    fig, ax = plt.subplots(2, figsize=(10, 6), sharey=True)
    ax[0].plot(data, color='orange', label='audio recording')
    ax[1].plot(np.concatenate(chunk_array), color='blue', label='segmented')
    ax[1].set_xlabel('Samples (frames)')
    ax[0].legend()
    ax[1].legend()
    plt.suptitle('segmented audio:')
    plt.show()

    # Plot the synthesized audio wave
    sample_rate, data = wavfile.read(wav_file_path)
    fig, ax = plt.subplots(2, figsize=(10, 6), sharey=True)
    ax[0].plot(data, color='orange', label='audio recording')
    ax[1].plot(chunk_info, color='blue', label='syntheized')
    ax[1].set_xlabel('Samples (frames)')
    ax[0].legend()
    ax[1].legend()
    plt.suptitle('synthesized audio:')
    plt.show()

    # Plot
    plt.subplot(2, 2, 2)

    plt.title('Spectrum of Unfiltered EMG Contracted')
    #plt.show()
    plot_spectrum(data, sample_rate)
    plot_spectrum(chunk_info, sample_rate)
    plt.show()

    # Print RMS values and center frequencies for each chunk
    # for i, (rms_value, center_freq) in enumerate(chunk_info):
    #     print(f"Chunk {i + 1}: RMS = {rms_value:.2f}, Center Frequency = {center_freq:.2f} Hz")
    
    # Produce output audio file
    output_file_path = os.path.join(output_directory, 'synthesized_signal.wav')
    wavfile.write(output_file_path, sample_rate, np.int16(chunk_info))


# Constants
wav_file_path = 'Part_2/swamp.wav'
chunk_duration_seconds = 0.1
overlap_percentage = 0.5
output_directory = 'synthesized_signal'  # Output directory where the filtered chunks will be saved
sample_rate = 16000 # Rate of 16KHz
chunk_array = splitWaveFile(wav_file_path, chunk_duration_seconds, overlap_percentage, sample_rate)


# Call main file
main(wav_file_path, output_directory, chunk_array, sample_rate)
