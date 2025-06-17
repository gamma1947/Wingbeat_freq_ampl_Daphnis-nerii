# place this code in the same directory as the data files
# This code is used to process the data files and save the results in a new folder called 'results' 
# in the same directory as the data files 

# you need to rename the columns in the csv files to 'x_r', 'y_r', 'likelihood_r' for right wing
# and 'x_l', 'y_l', 'likelihood_l' for left wing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statistics as stat
import seaborn as sns
import openpyxl as oxl
import os
from scipy.signal import find_peaks
from scipy import stats as st

## Data stuff

q1 = input(str("Enter the wing of interest :"))

## appropriate assignments based on input

if q1 == 'right' :
    y = 'y_r'
    x = 'x_r' 
    l = 'likelihood_r'
    wing = 'right wing'
elif q1 == 'left' :
    y = 'y_l'
    x = 'x_l' 
    l = 'likelihood_l'
    wing = 'left wing'

## details to save the file

date = '5/7/2024'
culture = 'Wild Male'
file_name = f'{date}_trialx'
species = r'$\it{Daphnis\ nerii}$'
age = '2 days old'
title = f"{date} {culture} {age} {species} {wing}" 


### this function filters our data

def filter(csv_file,l):
    raw_data = pd.read_csv(csv_file)
    lh_r = raw_data[l].values
    #print(lh_r)
    index = np.where(lh_r>0.7)[0] # type: ignore
    return index

### this function is to plot the filtered data

def plot(csv_file, column, l) :
    raw_data = pd.read_csv(csv_file)
    the_good_ones = filter(csv_file, l)
    print(the_good_ones[0:13])
    b = raw_data[column].values
    #print(b[0:13])
    x = np.array([])
    for i in the_good_ones :
        x = np.append(x, b[i])
    print(x[0:13])
    data = {'coords': the_good_ones,
            column: x}
    df = pd.DataFrame(data)
    fig = px.line(df,y = column, x = 'coords')
    return fig.show()

### finding the angular amplitudes

def amplitudes(csv_file,column):
    v = filter(csv_file,column)
    data = pd.read_csv(csv_file)
    values = data[column].values
    time = data['coords'].values  
    peaks,_ = find_peaks(values, distance = 30)
    troughs,_ = find_peaks(-values, distance = 30) # type: ignore

    # Filter peaks and troughs to include only those in `v`
    peaks_f = [p for p in peaks if p in v]
    troughs_f = [t for t in troughs if t in v]
    # print(peaks[0:15])
    # Ensure the filtered peaks and troughs are sorted
    extremas = np.sort(np.concatenate((peaks_f, troughs_f)))
    # print(extremas[0:15])
    # Calculate amplitudes
    amplitudes = []
    for i in range(1, len(extremas)):
        start_idx = extremas[i - 1]
        end_idx = extremas[i]
        cycle_values = values[start_idx:end_idx + 1]
        max_value = cycle_values.max()
        min_value = cycle_values.min()
        amplitude = max_value - min_value
        amplitudes.append(amplitude*(15/16))
    # print([f"{x:.2f}" for x in amplitudes])
    return amplitudes

def angular_amplitudes(csv_file,x_i,y_i,l) :
    hypo = []
    u = filter(csv_file,x_i)
    x = amplitudes(csv_file, x_i)
    y = np.array(amplitudes(csv_file, y_i))
    # y = [f"{x:.2f}" for x in y]
    x_u = x[int(np.where == y.min())] - x[int(np.where == y.max())]
    # print(y)
    for i in y :
        i = float(i)
        h = np.sqrt(x_u**2 + i**2)#*(15/16) # pixel to mm conversion
        hypo.append(h)
    # print(len(hypo))

    angular_amplitudes = []
    r = 416  # Assumed radius of the wing
    ratio = []
    for a in hypo:
        b = a / (2 * r)
        ratio.append(b)
        theta = 2*np.arcsin(b)*(180/(np.pi))
        angular_amplitudes.append(theta)
    angular_amplitudes = np.array(angular_amplitudes)
    return angular_amplitudes

## Plotting the angular amplitudes

# histogram

def hist(data) :
    plt.xlabel('Angular Amplitude in Degrees')
    plt.ylabel('Counts')
    plt.title(title)
    hist = sns.histplot(data, bins = 100, kde = False)
    return hist

# KDE

def KDE(data) :
    ax = sns.kdeplot(data, color = 'red', cumulative= False)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    # print(x)
    peaks,_ = find_peaks(y)
    x_peaks = x[peaks] # type: ignore
    # print(x_peaks)
    y_peaks = y[peaks] # type: ignore
    # sorted_peaks = np.argsort(y[peaks]) # type: ignore
    # print(sorted_peaks)
    plt.plot(x[peaks], y[peaks], 'bo') # type: ignore
    plt.xlabel('Angular Amplitude in Degrees')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend(['KDE', 'Maxima'])
    # plt.show()
    return x_peaks.tolist() ,y_peaks.tolist() # type: ignore

# Function to process data with row range selection
def process_data(csv_file,column,start_row=0, end_row=None, sampling_rate=1000):
    # Step 1: Read the CSV file
    data = pd.read_csv(csv_file)

    # Verify the column names and select the correct column
    # print("Available columns:", data.columns)
    
    # Ensure the correct column name is used
    values = data[column].values

    # Select the range of rows
    if end_row is None:
        end_row = len(values)
    
    values_range = values[1:]

    # Remove the DC component by subtracting the mean
    values_detrended = values_range - np.mean(values_range)

    # Step 2: Apply FFT
    fft_result = np.fft.fft(values_detrended)
    fft_freq = np.fft.fftfreq(len(values_detrended), d=1/sampling_rate)

    # Step 3: Identify the dominant frequency
    fft_magnitude = np.abs(fft_result)

    # Only consider the positive frequencies for plotting
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]

    # Plotting the FFT result (only positive frequencies)
    # plt.figure(figsize=(12, 6))
    # plt.plot(positive_freqs, positive_magnitude)
    # plt.title('FFT of the Signal (DC Component Removed)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.grid(True)
    # plt.show()

    f = positive_freqs[np.argmax(positive_magnitude)]
    if f < 50 and f > 10 :
        peak_freq = f
    else :
        sorted_freq = np.argsort(positive_magnitude)
        peak_freq = positive_freqs[sorted_freq[-2]]
    # Find the peak in the magnitude (positive frequencies only)
    # print(f'The dominant frequency is {peak_freq} Hz')
    return peak_freq

## saving the results

directory = os.getcwd()
# print(directory)
os.listdir(directory)

d = {'file_name' : [], 'peak_1' : [], 'peak_2' : [], 'frequency' : []}

current_filepath = __file__
# print(current_filepath)
directory = os.path.dirname(current_filepath)
# print(os.listdir(directory))

for name in os.listdir(directory):
    os.chdir(directory)
    if not name.endswith('.csv'):   # skip non-csv files
        continue
    if not os.path.exists('results'):
        os.mkdir('results')
    # print(os.path.abspath('results'))
    # print(os.getcwd())
    if q1 == 'right' :
        if not os.path.exists('results\\right'):
            os.makedirs('results\\right')
        save_path = os.path.abspath('results\\right')   

    elif q1 == 'left' :
        if not os.path.exists('results\\left'):
            os.makedirs('results\\left')
        save_path = os.path.abspath('results\\left') 
    # print(save_path)
    file_path = os.path.join(directory, name)
    a = file_path
    # print(a)
    ang_amp = angular_amplitudes(a,x,y,l)
    # print(ang_amp[:5])
    plt.figure()
    _,b = KDE(ang_amp)
    # print(b)
    # plt.show()
    plt.figure()
    l,_ = KDE(ang_amp)
    # print(l)
    # print(l[aaaa1],b[aaaa1],l[aaaa2],b[aaaa2],'a')
    # plt.show()
    plt.savefig(os.path.join(save_path, f"{wing}_{name}.png"))
    plt.figure()
    hist(ang_amp)
    plt.savefig(os.path.join(save_path, f"{wing}_hist_{name}.png"))
    d['file_name'].append(name)
    if len(b) == 1 :
        d['peak_1'].append(l[0])
        d['peak_2'].append('-')
    else :
        # b.sort(reverse = True)
        aaaa=np.array(b).argsort()
        aaaa1=aaaa[-1]
        aaaa2=aaaa[-2]
        d['peak_1'].append(l[aaaa1]) 
        d['peak_2'].append(l[aaaa2])
    freq = float(process_data(a,y, start_row=1, end_row=20500, sampling_rate=1000))
    # print(type(freq))
    d['frequency'].append(freq)
    df = pd.DataFrame(data = d)
    df.to_excel(os.path.join(save_path, f"{wing}_results.xlsx"), index = False, sheet_name= f"{wing}")

print(d)
