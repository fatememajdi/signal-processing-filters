import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import cheby1, filtfilt
import sounddevice as sd
import matplotlib.pyplot as plt

# Function to design a Chebyshev filter
def chebyshev_filter(data, cutoff, fs, btype='low', order=5, ripple=0.5):
    nyquist = 0.5 * fs
    if btype == 'band':
        normal_cutoff = [c / nyquist for c in cutoff]
        b, a = cheby1(order, ripple, normal_cutoff, btype='band', analog=False)
    else:
        normal_cutoff = cutoff / nyquist
        b, a = cheby1(order, ripple, normal_cutoff, btype=btype, analog=False)

    y = filtfilt(b, a, data)
    return y

# Function to play a .wav file
def play_wav_file(file_path):
    fs, data = wav.read(file_path)
    sd.play(data, fs)
    sd.wait()  # Wait until the sound has finished playing


# Read the .wav file
input_file = './test.wav'  # Replace with your input file
fs, data = wav.read(input_file)

# Check if the audio is stereo or mono
if len(data.shape) == 2:
    # If stereo, take only one channel (e.g., left channel)
    data = data[:, 0]

# Define filter parameters
low_cutoff = 1000  # Low-pass cutoff frequency in Hz
high_cutoff = 2000  # High-pass cutoff frequency in Hz
band_cutoff = [1000, 2000]  # Band-pass cutoff frequencies in Hz
order = 5  # Order of the filter
ripple = 0.5  # Ripple in dB for Chebyshev filter

# Apply the Chebyshev filters
low_passed_data = chebyshev_filter(
    data, low_cutoff, fs, btype='low', order=order, ripple=ripple)
high_passed_data = chebyshev_filter(
    data, high_cutoff, fs, btype='high', order=order, ripple=ripple)
band_passed_data = chebyshev_filter(
    data, band_cutoff, fs, btype='band', order=order, ripple=ripple)

# Write the filtered data to new .wav files
wav.write('chebyshev_output_low_pass.wav', fs, low_passed_data.astype(np.int16))
wav.write('chebyshev_output_high_pass.wav', fs, high_passed_data.astype(np.int16))
wav.write('chebyshev_output_band_pass.wav', fs, band_passed_data.astype(np.int16))

# Play the filtered audio files
print("Playing Low-Pass Filtered Audio...")
play_wav_file('chebyshev_output_low_pass.wav')

print("Playing High-Pass Filtered Audio...")
play_wav_file('chebyshev_output_high_pass.wav')

print("Playing Band-Pass Filtered Audio...")
play_wav_file('chebyshev_output_band_pass.wav')

# Optional: Plot the original and filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.title('Original Signal')
plt.plot(data)
plt.subplot(4, 1, 2)
plt.title('Low-Pass Filtered Signal')
plt.plot(low_passed_data)
plt.subplot(4, 1, 3)
plt.title('High-Pass Filtered Signal')
plt.plot(high_passed_data)
plt.subplot(4, 1, 4)
plt.title('Band-Pass Filtered Signal')
plt.plot(band_passed_data)
plt.tight_layout()
plt.show()
