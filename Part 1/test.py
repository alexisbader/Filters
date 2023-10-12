import numpy as np
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt

def butterworth_notch_filter(data, cutoff_freq, notch_width, sampling_rate, *args):
    # Calculate the notch frequencies
    f0 = cutoff_freq / (sampling_rate / 2)
    Q = cutoff_freq / notch_width

    # Design the notch filter using butter function with specified filter order
    if args:
        filter_order = args[0]
    else:
        filter_order = 3  

    b, a = signal.butter(N=filter_order, Wn=[(f0 - 1/Q), (f0 + 1/Q)], btype='bandstop')

    # Apply the notch filter to the data
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data

def butterworth_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4):
    # Calculate the low and high frequencies
    low = lowcut / (sampling_rate / 2)
    high = highcut / (sampling_rate / 2)

    # Design the bandpass filter using butter function
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')

    # Apply the bandpass filter to the data
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

    # Set the sampling rate and filter parameters
    sampling_rate = 1000   # Replace with your actual sampling rate
    cutoff_freq = 60       # Frequency to be notched out (e.g., 60 Hz)
    notch_width = 1        # Width of the notch (can be adjusted based on your requirements)
    filter_order = 3       # Replace with your desired filter order

    # Extract time and EMG data from the DataFrame
    time = df['Time (s)'].values
    relaxed_emg = df['EMG_Relaxed (mV)'].values
    contracted_emg = df['EMG_Contracted (mV)'].values

    # RMS before applying the filters
    rms_relaxed_before_filter = np.sqrt(np.mean(relaxed_emg**2))
    rms_contracted_before_filter = np.sqrt(np.mean(contracted_emg**2))

    # Apply the Butterworth notch filter to 'emg_relaxed' column
    filtered_relaxed_emg = butterworth_notch_filter(relaxed_emg, cutoff_freq, notch_width, sampling_rate, filter_order)

    # Apply the Butterworth notch filter to 'emg_contracted' column
    filtered_contracted_emg = butterworth_notch_filter(contracted_emg, cutoff_freq, notch_width, sampling_rate, filter_order)

    # Apply the Butterworth bandpass filter to 'emg_relaxed' column
    bandpass_lowcut = 0.1
    bandpass_highcut = 450
    filtered_relaxed_emg_bandpass = butterworth_bandpass_filter(relaxed_emg, bandpass_lowcut, bandpass_highcut, sampling_rate)

    # Apply the Butterworth bandpass filter to 'emg_contracted' column
    filtered_contracted_emg_bandpass = butterworth_bandpass_filter(contracted_emg, bandpass_lowcut, bandpass_highcut, sampling_rate)

    # RMS after applying the filters
    rms_relaxed_after_filter = np.sqrt(np.mean(filtered_relaxed_emg**2))
    rms_contracted_after_filter = np.sqrt(np.mean(filtered_contracted_emg**2))

    # Plot the spectrum of the unfiltered and filtered EMG data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Spectrum of Unfiltered EMG Relaxed')
    plot_spectrum(relaxed_emg, sampling_rate)

    plt.subplot(2, 2, 2)
    plt.title('Spectrum of Unfiltered EMG Contracted')
    plot_spectrum(contracted_emg, sampling_rate)

    # plt.subplot(2, 2, 3)
    # plt.title('Spectrum of Filtered EMG Relaxed')
    # plot_spectrum(filtered_relaxed_emg, sampling_rate)

    # plt.subplot(2, 2, 4)
    # plt.title('Spectrum of Filtered EMG Contracted')
    # plot_spectrum(filtered_contracted_emg, sampling_rate)

    # Bandpass
    plt.subplot(2, 2, 3)
    plt.title('Spectrum of Filtered EMG Relaxed')
    plot_spectrum(filtered_relaxed_emg_bandpass, sampling_rate)

    plt.subplot(2, 2, 4)
    plt.title('Spectrum of Filtered EMG Contracted')
    plot_spectrum(filtered_contracted_emg_bandpass, sampling_rate)


    plt.tight_layout()
    plt.show()

    # Plot the unfiltered and filtered data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(time, relaxed_emg, label='EMG Relaxed (Unfiltered)', color='blue')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Relaxed (Unfiltered)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time, contracted_emg, label='EMG Contracted (Unfiltered)', color='green')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Contracted (Unfiltered)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(time, filtered_relaxed_emg, label='Filtered EMG Relaxed', color='red')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Relaxed (Filtered)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(time, filtered_contracted_emg, label='Filtered EMG Contracted', color='purple')
    plt.xlabel('Time')
    plt.ylabel('EMG')
    plt.title('EMG Contracted (Filtered)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Show the RMS values
    print("RMS of Relaxed EMG (before filter):", rms_relaxed_before_filter)
    print("RMS of Contracted EMG (before filter):", rms_contracted_before_filter)
    print("RMS of Relaxed EMG (after filter):", rms_relaxed_after_filter)
    print("RMS of Contracted EMG (after filter):", rms_contracted_after_filter)

if __name__ == "__main__":
    main()
