import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

def butterworth_notch_filter(data, cutoff_freq, notch_width, sampling_rate, filter_order):
    # Calculate the notch frequencies
    f0 = cutoff_freq / (sampling_rate / 2)
    Q = cutoff_freq / notch_width

    # Design the notch filter
    b, a = signal.butter(N=filter_order, Wn=[(f0 - 1/Q), (f0 + 1/Q)], btype='bandstop')

    # Apply the notch filter to the data
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data

def plot_spectrum(data, sampling_rate):
    n = len(data)
    spectrum = np.abs(np.fft.fft(data)) / n
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    plt.plot(freq[:n//2], spectrum[:n//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

def main():
    # Replace 'your_data_file.csv' with the path to your CSV file
    data_file = 'Part 1/EMG_Datasets.csv'
    df = pd.read_csv(data_file)
    filter_order = 3

    # Set the sampling rate and notch filter parameters
    sampling_rate = 1000  # Replace with your actual sampling rate
    cutoff_freq = 60      # Frequency to be notched out (e.g., 60 Hz)
    notch_width = 1        # Width of the notch (can be adjusted based on your requirements)

    # Extract time and EMG data from the DataFrame
    time = df['Time (s)'].values
    relaxed_emg = df['EMG_Relaxed (mV)'].values
    contracted_emg = df['EMG_Contracted (mV)'].values

    # Apply the Butterworth notch filter to 'emg_relaxed' column
    filtered_relaxed_emg = butterworth_notch_filter(relaxed_emg, cutoff_freq, notch_width, sampling_rate, filter_order)

    # Apply the Butterworth notch filter to 'emg_contracted' column
    filtered_contracted_emg = butterworth_notch_filter(contracted_emg, cutoff_freq, notch_width, sampling_rate, filter_order)


    # Plot the spectrum of the unfiltered EMG data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Spectrum of Unfiltered EMG Relaxed')
    plot_spectrum(relaxed_emg, sampling_rate)

    plt.subplot(2, 2, 2)
    plt.title('Spectrum of Unfiltered EMG Contracted')
    plot_spectrum(contracted_emg, sampling_rate)

    # Plot the spectrum of the filtered EMG data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Spectrum of Filtered EMG Relaxed')
    plot_spectrum(filtered_relaxed_emg, sampling_rate)

    plt.subplot(2, 2, 2)
    plt.title('Spectrum of Filtered EMG Contracted')
    plot_spectrum(filtered_contracted_emg, sampling_rate)


    # Plot the unfiltered and filtered data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, relaxed_emg, label='EMG Relaxed (Unfiltered)', color='blue')
    plt.plot(time, filtered_relaxed_emg, label='Filtered EMG Relaxed', color='red')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Relaxed (Unfiltered) vs. Filtered EMG Relaxed')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, contracted_emg, label='EMG Contracted (Unfiltered)', color='green')
    plt.plot(time, filtered_contracted_emg, label='Filtered EMG Contracted', color='purple')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Contracted (Unfiltered) vs. Filtered EMG Contracted')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()