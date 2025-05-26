import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import ellip, lfilter
import sounddevice as sd
import matplotlib.pyplot as plt

# Function to design an elliptic filter
def elliptic_filter(data, cutoff, fs, btype='low', order=5, ripple=0.5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Design the elliptic filter
    b, a = ellip(order, ripple, 40, normal_cutoff, btype=btype, analog=False)
    
    # Apply the filter
    filtered_data = lfilter(b, a, data)
    return filtered_data

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
ripple = 0.5  # Ripple in dB for elliptic filter

# Apply the elliptic filters
low_passed_data = elliptic_filter(data, low_cutoff, fs, btype='low', order=order, ripple=ripple)
high_passed_data = elliptic_filter(data, high_cutoff, fs, btype='high', order=order, ripple=ripple)

# For band-pass, we need to design the filter differently
nyquist = 0.5 * fs
band_normal_cutoff = [c / nyquist for c in band_cutoff]
b_band, a_band = ellip(order, ripple, 40, band_normal_cutoff, btype='band', analog=False)
band_passed_data = lfilter(b_band, a_band, data)

# Write the filtered data to new .wav files
wav.write('ellipitic_output_low_pass.wav', fs, low_passed_data.astype(np.int16))
wav.write('ellipitic_output_high_pass.wav', fs, high_passed_data.astype(np.int16))
wav.write('ellipitic_output_band_pass.wav', fs, band_passed_data.astype(np.int16))

# Play the filtered audio files
print("Playing Low-Pass Filtered Audio...")
play_wav_file('ellipitic_output_low_pass.wav')

print("Playing High-Pass Filtered Audio...")
play_wav_file('ellipitic_output_high_pass.wav')

print("Playing Band-Pass Filtered Audio...")
play_wav_file('ellipitic_output_band_pass.wav')

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