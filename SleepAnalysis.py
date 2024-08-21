# This is a script that reads data from a file and analyses it to understand the sleep patterns of a person.

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import SleepFunctions
from scipy import signal


# --------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------

# File parameters
filename = "YingResting.txt"  # Replace with your file name
start_line = 4 # Line number to start reading from

# Bitalino parameters
Fs_orig = 1000 # Orignal Sampling rate
Fs_target = 200 # Target sampling rate
# dT = 1000/Fs # Time between samples in ms
gain = 1 # This is the gain of the amplifier in volts output/ mV of brain signal

# EEG band Filters
Filter_delta_FLow = 0.5 # Low cutoff frequency
Filter_delta_FHigh = 3 # High cutoff frequency
Filter_theta_FLow = 4 # Low cutoff frequency
Filter_theta_FHigh = 7 # High cutoff frequency
Filter_alpha_FLow = 8 # Low cutoff frequency
Filter_alpha_FHigh = 12 # High cutoff frequency
Filter_beta_FLow = 12 # Low cutoff frequency
Filter_beta_FHigh = 30 # High cutoff frequency
Filter_gamma_FLow = 31 # Low cutoff frequency
Filter_gamma_FHigh = 90 # High cutoff frequency
Filter_Order = 4  # Filter order (adjust as needed)

# data filtering parameters
Filter_denoise_FLow = 1 # Low cutoff frequency
Filter_denoise_FHigh = 100 # High cutoff frequency
Filter_SpikeThresh = 3.75 # sets the gain of the median spike filter
window_median = 4000 # 60 seconds
F_power = 50.0  # frequency to be removed from the signal (Hz)
F_width = 2     # width of the powerlie notch (Hz)
order = 4       # powerline notch filter order

# figures
voltageRange = 0.4 # max voltae range (mv) to display in figures
medVoltageRange = 0.08 # max voltae range (mv) to display in median filtered figures
analysisLimit = 0.05

# --------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------

# Diagnostic to see the first n lines of the file to ensure the data is 
# the correct place
# filename = "SleepData.txt"  # Replace with your file name
# num_lines = 100
# read_first_n_lines(filename, num_lines)

# Count the number of lines in the file
numLines = SleepFunctions.count_lines(filename)
print(f"Number of lines in the file: {numLines}")

# Extract the data and insert into matrices. the key data should be in the
# 6th column (counting from 0 - 5 in python). The extracted data is in the
# form of a ADC digitized number.
end_line =  numLines
matrices = SleepFunctions.extract_data(filename, start_line, end_line)
ADCVal = matrices[5]

# convert the ADCVal matrix to voltage
voltage = ADCVal.astype(np.float32)  # Or any other desired floating point type
voltage = voltage - 512 # split the ADC values around 0
voltage = voltage * 3.3 / (gain * 1024) # convert to brain signal (mV)

# Downsample the data to 100Hz
data_downsample = SleepFunctions.downsample_average(voltage, int(Fs_orig/Fs_target))

# create a function to perform 45-55Hz notch filter to remove powerline noise 
# and harmonics
data_notched = SleepFunctions.notchFilter(data_downsample, F_power, F_width, order, Fs_target)

# Filter the data to remove spiking noise
data_filt, meanvals = SleepFunctions.artefact_filter(data_notched, window_median, Filter_SpikeThresh)

# attain the EEG bands
data_delta = SleepFunctions.dataFilter(data_filt, Filter_delta_FLow, Filter_delta_FHigh, Filter_Order, Fs_target)
data_theta = SleepFunctions.dataFilter(data_filt, Filter_theta_FLow, Filter_theta_FHigh, Filter_Order, Fs_target)
data_alpha = SleepFunctions.dataFilter(data_filt, Filter_alpha_FLow, Filter_alpha_FHigh, Filter_Order, Fs_target)
data_beta = SleepFunctions.dataFilter(data_filt, Filter_beta_FLow, Filter_beta_FHigh, Filter_Order, Fs_target)
data_gamma = SleepFunctions.dataFilter(data_filt, Filter_gamma_FLow, Filter_gamma_FHigh, Filter_Order, Fs_target)

# Attain absolute values of the EEG bands
data_delta_abs = np.abs(data_delta)
data_theta_abs = np.abs(data_theta)
data_alpha_abs = np.abs(data_alpha)
data_beta_abs = np.abs(data_beta)
data_gamma_abs = np.abs(data_gamma)

# Median filtering of the EEG data
median_delta = SleepFunctions.median_filter(data_delta_abs, window_median)
median_theta = SleepFunctions.median_filter(data_theta_abs, window_median)
median_alpha = SleepFunctions.median_filter(data_alpha_abs, window_median)
median_beta = SleepFunctions.median_filter(data_beta_abs, window_median)
median_gamma = SleepFunctions.median_filter(data_gamma_abs, window_median)

# get sleep estimates from the data by dividing gamma by delta
# Then doing some artefact fltering to tidy up the data
# sleep_estimates = data_delta_abs/data_gamma_abs 
# Filter_SpikeThresh = 2.5
# sleep_filt, meanvals = SleepFunctions.artefact_filter(sleep_estimates, window_median, Filter_SpikeThresh)

# get a sleep estimate by diving the median values of the delta and gamma bands
sleep_median = median_delta - median_gamma  

# downsample the sleep estimates to per second
# sleep_estimates = SleepFunctions.downsample_average(sleep_filt, int(Fs_target/1)) 
sleep_median = SleepFunctions.downsample_average(sleep_median, int(Fs_target/1))

# Create spectrogram
f_spec, t, data_Spec = signal.spectrogram(data_notched, Fs_target, nperseg=256)


# Create a time series for the data
dataPoints = np.linspace(0, len(data_downsample), num = len(data_downsample))
timeSeries = dataPoints.astype(np.float32)  # Or any other desired floating point type
timeSeries = timeSeries / Fs_target  # Convert to seconds
timeSeries = timeSeries /60 # Convert to minutes

# Create time series for slwwp analysis
dataPoints2 = np.linspace(0, len(sleep_median), num = len(sleep_median))
timeSeriesMins = dataPoints2.astype(np.float32)  # Or any other desired floating point type
timeSeriesMins = timeSeriesMins /60 # Convert to minutes

# Create spectrogram time series
dataPointsSpec = np.linspace(0, len(t), num = len(t))
timeSeriesSpec = dataPointsSpec.astype(np.float32)  # Or any other desired floating point type
timeSeriesSpec = timeSeriesSpec /60 # Convert to minutes


# --------------------------------------------------------------------
# Plot data
# --------------------------------------------------------------------


#---------------------------------------------
# Time domain analysis summary
#---------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 6))

# Plot data on each subplot
axs[0].plot(timeSeries, data_downsample)
axs[1].plot(timeSeries, data_notched)
axs[1].plot(timeSeries, data_filt)
axs[2].plot(timeSeriesMins, sleep_median)

# Set titles
axs[0].set_title('raw data', y=1.0, pad=-14)
axs[1].set_title('Filtered data', y=1.0, pad=-14)
axs[2].set_title('delta/gamma sleep estimate', y=1.0, pad=-14)

# hide the x-axis labels
for ax in axs:
    ax.label_outer()

# Set the axes labels
axs[0].set_ylabel('Voltage [mV]')
axs[1].set_ylabel('Voltage [mV]')
axs[2].set_ylabel('Voltage [mV]')
axs[2].set_xlabel('Time [min]')


#---------------------------------------------
# Plot the EEG analysis of the filtered data
#---------------------------------------------
fig, axs = plt.subplots(5, 1, figsize=(8, 6))

# Plot data on each subplot
axs[0].plot(timeSeries, data_delta)
axs[1].plot(timeSeries, data_theta)
axs[2].plot(timeSeries, data_alpha)
axs[3].plot(timeSeries, data_beta)
axs[4].plot(timeSeries, data_gamma)

# set maximum voltage range
axs[0].set_ylim([-voltageRange, voltageRange])
axs[1].set_ylim([-voltageRange, voltageRange])
axs[2].set_ylim([-voltageRange, voltageRange])
axs[3].set_ylim([-voltageRange, voltageRange])
axs[4].set_ylim([-voltageRange, voltageRange])

# Set titles
axs[0].set_title('delta', y=1.0, pad=-14)
axs[1].set_title('theta', y=1.0, pad=-14)
axs[2].set_title('alpha', y=1.0, pad=-14)
axs[3].set_title('beta', y=1.0, pad=-14)
axs[4].set_title('gamma', y=1.0, pad=-14)

# hide the x-axis labels
for ax in axs:
    ax.label_outer()

# Set the axes labels
axs[0].set_ylabel('Voltage [mV]')
axs[1].set_ylabel('Voltage [mV]')
axs[2].set_ylabel('Voltage [mV]')
axs[3].set_ylabel('Voltage [mV]')
axs[4].set_ylabel('Voltage [mV]')
axs[4].set_xlabel('Time [min]')

#---------------------------------------------
# Median Filter analysis of the EEG data
#---------------------------------------------
fig, axs = plt.subplots(5, 1, figsize=(8, 6))

# Plot data on each subplot
axs[0].plot(timeSeries, median_delta)
axs[1].plot(timeSeries, median_theta)
axs[2].plot(timeSeries, median_alpha)
axs[3].plot(timeSeries, median_beta)
axs[4].plot(timeSeries, median_gamma)

# set maximum voltage range
axs[0].set_ylim([0, medVoltageRange])
axs[1].set_ylim([0, medVoltageRange])
axs[2].set_ylim([0, medVoltageRange])
axs[3].set_ylim([0, medVoltageRange])
axs[4].set_ylim([0, medVoltageRange])

# Set titles
axs[0].set_title('delta', y=1.0, pad=-14)
axs[1].set_title('theta', y=1.0, pad=-14)
axs[2].set_title('alpha', y=1.0, pad=-14)
axs[3].set_title('beta', y=1.0, pad=-14)
axs[4].set_title('gamma', y=1.0, pad=-14)

# hide the x-axis labels
for ax in axs:
    ax.label_outer()

# Set the axes labels
axs[0].set_ylabel('Voltage [mV]')
axs[1].set_ylabel('Voltage [mV]')
axs[2].set_ylabel('Voltage [mV]')
axs[3].set_ylabel('Voltage [mV]')
axs[4].set_ylabel('Voltage [mV]')
axs[4].set_xlabel('Time [min]')

#---------------------------------------------
# Comparison
#---------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 6))

# Plot data on each subplot
axs[0].plot(timeSeries, data_filt)
axs[1].plot(timeSeries, median_delta)
axs[1].plot(timeSeries, median_theta)
axs[2].plot(timeSeriesMins, sleep_median)


# set maximum voltage range
# axs[0].set_ylim([0, medVoltageRange])
axs[1].set_ylim([0, 0.06])
axs[2].set_ylim([-analysisLimit, analysisLimit])
# axs[3].set_ylim([0, medVoltageRange])
# axs[4].set_ylim([0, medVoltageRange])

# Set titles
axs[0].set_title('time domain', y=1.0, pad=-14)
axs[1].set_title('delta, theta', y=1.0, pad=-14)
axs[2].set_title('sleep analysis', y=1.0, pad=-14)

# hide the x-axis labels
for ax in axs:
    ax.label_outer()

# Set the axes labels
axs[0].set_ylabel('Voltage [mV]')
axs[1].set_ylabel('Voltage [mV]')
axs[2].set_ylabel('Voltage [mV]')
axs[2].set_xlabel('Time [min]')

#---------------------------------------------
# Perform spectrogram analysis on the data
#---------------------------------------------
plt.figure()
plt.pcolormesh(timeSeriesSpec, f_spec, 10 * np.log10(data_Spec))
plt.yscale('log')
plt.ylim([0.5, 100])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [min]')
plt.title(' EEG Spectrogram (filtered data)')

# show all the images simultaneously
plt.show()
