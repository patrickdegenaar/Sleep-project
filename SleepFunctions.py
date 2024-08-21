# This is a script that reads data from a file and analyses it to understand the sleep patterns of a person.

import numpy as np
import os
from scipy import signal


# --------------------------------------------------------------------
# Function definitions
# --------------------------------------------------------------------

# This is a diagnostic function to see what the first few (n) lines of the 
# text file look like in order to extract the data properly. it will then
# print the first n lines to the console
# Inputs: 
# filename: filename of the data file
# n: number of lines to read
def read_first_n_lines(filename, n):
  
  # extract the current director filepath and add the filename
  # This allows us to read the file in the same director as the script
  file_path = os.path.join(os.path.dirname(__file__), filename)

  # Try to open the file, if not provide an error message
  try:
    # Open the file and read the first n lines - if the number of lines
    # in the file is less tan n, then break the loop 
    with open(file_path, 'r') as file:
      lines = []
      for i in range(n):
        line = file.readline()
        if not line:
          break
        lines.append(line.strip())
      for line in lines:
        print(line)
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")

# This function will count the number of lines in the text file
# Input : filename: filename of the data file
# Output: Number of lines in the file
def count_lines(filename):

  # extract the current director filepath and add the filename
  # This allows us to read the file in the same director as the script
  file_path = os.path.join(os.path.dirname(__file__), filename)

  # Open the file and count the number of lines 
  with open(file_path, 'r') as file:
    lines = file.readlines()
    return len(lines)

# This function will read the data from a text file and extract the data
# it assumes the data is in a tab separated columns.
# Inputs:
# filename: filename of the data file
# start_line: the line number to start reading from
# end_line: the line number to stop reading
# Output: A list of numpy arrays, each containing one column of data.
def extract_data(filename, start_line, end_line):
  
  # extract the current director filepath and add the filename
  # This allows us to read the file in the same director as the script
  file_path = os.path.join(os.path.dirname(__file__), filename)

  data = []
  with open(file_path, 'r') as file:
    for i, line in enumerate(file):
      if i + 1 < start_line:
        continue
      if i + 1 > end_line:
        break
      values = line.strip().split('\t')
      data.append(values)

  num_columns = len(data[0])
  matrices = [[] for _ in range(num_columns)]
  for row in data:
    for i, value in enumerate(row):
      matrices[i].append(float(value))

  return [np.array(matrix) for matrix in matrices]

# This function will downsample the data by averaging consecutive samples.
# Inputs:
# data: the data to be downsampled
# factor: the downsampling factor
# Output: the downsampled data
def downsample_average(data, factor):
  
  # Calculate the new size
  new_size = len(data) // factor
    
  # Truncate the array to be evenly divisible by factor
  truncated_data = data[:new_size * factor]
    
  # Reshape and calculate mean
  return np.mean(truncated_data.reshape(-1, factor), axis=1)


# This function will filter the data using a bandpass filter
# Inputs:
# data: the data to be filtered
# low: the low cutoff frequency
# high: the high cutoff frequency
# order: the order of the filter
# fs: the sampling frequency
# Output: the filtered data
def dataFilter(data, FLow, FHigh, order, fs):
 
  nyquist_freq = fs * 0.5  # Nyquist frequency
  low = FLow / nyquist_freq
  high = FHigh / nyquist_freq
  b, a = signal.butter(order, [low, high], btype='bandpass')
  filtered_data = signal.filtfilt(b, a, data)
  
  return filtered_data

# This function will filter the data using a notch filter
# To remove 50Hz powerline noise and harmonics
# Inputs:
# data_in: the data to be filtered
# F_power: the powerline frequency
# F_width: the width of the notch filter
# order: the order of the filter
# Fsample: the sampling frequency
# Output: the filtered data
def notchFilter(data_in, F_power, F_width, order, Fsample):

    # Define the band edges
    FLow = F_power - F_width / 2
    FHigh = F_power + F_width / 2

    # Normalise the frequency
    nyquist_freq = Fsample * 0.5  # Nyquist frequency
    low = FLow / nyquist_freq
    high = FHigh / nyquist_freq

    # Create the bandpass filter kernel
    # and peform the filter
    b, a = signal.butter(order, [low, high], btype='bandpass')
    notch = signal.filtfilt(b, a, data_in)

    # subtract the notch from the original data
    data_notched = data_in - notch

    return data_notched

# This function performs a median filter on the data to remove
# high frequency noise (spikes and artefacts)
# Inputs:
# data: the data to be filtered
# window_size: the size of the median filter window
# Output: the filtered data
def median_filter(data, window_size):
  
  filtered_data = np.zeros_like(data)
  half_window = window_size // 2

  for i in range(len(data)):

    # get the start/end positions of the window then attain the data for that window
    start = max(0, i - half_window)
    end = min(len(data), i + half_window + 1)
    window = data[start:end]

    # perform the median filter - attain the median value of the data within the window
    filtered_data[i] = np.median(window)

  return filtered_data

def artefact_filter(data, window_size, scale):
  
  filtered_data = np.zeros_like(data)
  meanvals = np.zeros_like(data)
  half_window = window_size // 2

  for i in range(len(data)):

    # get the start/end positions of the window then attain the data for that window
    start = max(0, i - half_window)
    end = min(len(data), i + half_window + 1)
    window = data[start:end]

    # attain the mean value within the recent window of values
    meanVal = np.median(np.abs(window))
    meanvals[i] = meanVal

    # insert the base data
    filtered_data[i] = data[i]

    # Key filter - if the data is greater than the mean value * scale, 
    # then set the data to the mean value * scale
    if np.abs(data[i]) > (meanVal*scale):
      if data[i] > 0: filtered_data[i] = (meanVal *scale)
      else: filtered_data[i] = -(meanVal *scale)
    
    # If the data is less than the mean value / scale, then set 
    # the data to the mean value / scale
    # if np.abs(data[i]) < (meanVal/scale):
    #   if data[i] > 0: filtered_data[i] = (meanVal/scale)
    #   else: filtered_data[i] = -(meanVal/scale)
    

  return filtered_data, meanvals

