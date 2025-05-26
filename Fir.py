import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import firwin, lfilter
import sounddevice as sd
import matplotlib.pyplot as plt

def fir_filter(data, cutoff, fs, btype='low', numtaps=101):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Design the FIR filter
    if btype == 'low':
        taps = firwin(numtaps, normal_cutoff, pass_zero='lowpass')
    elif btype == 'high':
        taps = firwin(numtaps, normal_cutoff, pass_zero='highpass')
    else:
        raise ValueError("btype must be 'low' or 'high'")

    # Apply the filter
    filtered_data = lfilter(taps, 1.0, data)
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
numtaps = 101  # Number of filter taps (order + 1)

# Apply the FIR filters
low_passed_data = fir_filter(data, low_cutoff, fs, btype='low', numtaps=numtaps)
high_passed_data = fir_filter(data, high_cutoff, fs, btype='high', numtaps=numtaps)

# For band-pass, we need to design the filter differently
nyquist = 0.5 * fs
band_normal_cutoff = [c / nyquist for c in band_cutoff]
taps_band = firwin(numtaps, band_normal_cutoff, pass_zero=False)
band_passed_data = lfilter(taps_band, 1.0, data)

# Write the filtered data to new .wav files
wav.write('Fir_output_low_pass.wav', fs, low_passed_data.astype(np.int16))
wav.write('Fir_output_high_pass.wav', fs, high_passed_data.astype(np.int16))
wav.write('Fir_output_band_pass.wav', fs, band_passed_data.astype(np.int16))

# Play the filtered audio files
print("Playing Low-Pass Filtered Audio...")
play_wav_file('Fir_output_low_pass.wav')

print("Playing High-Pass Filtered Audio...")
play_wav_file('Fir_output_high_pass.wav')

print("Playing Band-Pass Filtered Audio...")
play_wav_file('Fir_output_band_pass.wav')

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