# ----------------------------------------------
# standard Pan Tompkins QRS detection algorithm
# References from
# https://github.com/adityatripathiiit/Pan_Tompkins_QRS_Detection
# ----------------------------------------------

import math
import sys
import random
import mne
from mne.io import read_raw_edf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('rr_extraction')

def ECG_reverse_check(bandpass, sfreq):
    '''
    This function can be used to check whether the ECG signal 
    need to be reversed.
    '''
    numbers = [random.randint(1, 3000) for _ in range(15)]
    count = 0
    for x in numbers:
        maxm = max(bandpass[x*sfreq : (x+5)*sfreq])
        minm = min(bandpass[x*sfreq : (x+5)*sfreq])
        if abs(maxm) < abs(minm):
            count += 1
            
    if count >= 8:
        reverse_flag = True
    else:
        reverse_flag = False

    return reverse_flag

################### Pan-Tomkin preprcessing funcs ###########################
def band_pass_filter(signal):
    '''
    Band Pass Filter
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    Bandpass filter is used to attenuate the noise in the input signal.
    To acheive a passband of 5-15 Hz, the input signal is first passed 
    through a low pass filter having a cutoff frequency of 11 Hz and then
    through a high pass filter with a cutoff frequency of 5 Hz, thus
    achieving the required thresholds. 

    The low pass filter has the recursive equation:
        y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)

    The high pass filter has the recursive equation:
        y(nT) = 32x(nT - 16T) - y(nT - T) - x(nT) + x(nT - 32T)
    '''

    # Initialize result
    result = None

    # Create a copy of the input signal
    sig = signal.copy()

    # Apply the low pass filter using the equation given
    for index in range(len(signal)):
        sig[index] = signal[index]

        if (index >= 1):
            sig[index] += 2*sig[index-1]

        if (index >= 2):
            sig[index] -= sig[index-2]

        if (index >= 6):
            sig[index] -= 2*signal[index-6]

        if (index >= 12):
            sig[index] += signal[index-12] 

    # Copy the result of the low pass filter
    result = sig.copy()

    # Apply the high pass filter using the equation given
    for index in range(len(signal)):
        result[index] = -1*sig[index]

        if (index >= 1):
            result[index] -= result[index-1]

        if (index >= 16):
            result[index] += 32*sig[index-16]

        if (index >= 32):
            result[index] += sig[index-32]

    # Normalize the result from the high pass filter
    max_val = max(max(result),-min(result))
    result = result/max_val

    return result

def derivative(signal, sfreq):
    '''
    Derivative Filter 
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The derivative of the input signal is taken to obtain the
    information of the slope of the signal. Thus, the rate of change
    of input is obtain in this step of the algorithm.

    The derivative filter has the recursive equation:
        y(nT) = [-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]/(8T)
    '''

    # Initialize result
    result = signal.copy()

    # Apply the derivative filter using the equation given
    for index in range(len(signal)):
        result[index] = 0

        if (index >= 1):
            result[index] -= 2*signal[index-1]

        if (index >= 2):
            result[index] -= signal[index-2]

        if (index >= 2 and index <= len(signal)-2):
            result[index] += 2*signal[index+1]

        if (index >= 2 and index <= len(signal)-3):
            result[index] += signal[index+2]

        result[index] = (result[index]*sfreq)/8

    return result

def squaring(signal):
    '''
    Squaring the Signal
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The squaring process is used to intensify the slope of the
    frequency response curve obtained in the derivative step. This
    step helps in restricting false positives which may be caused
    by T waves in the input signal.

    The squaring filter has the recursive equation:
        y(nT) = [x(nT)]^2
    '''

    # Initialize result
    result = signal.copy()

    # Apply the squaring using the equation given
    for index in range(len(signal)):
        result[index] = signal[index]**2

    return result    

def moving_window_integration(signal, sfreq):
    '''
    Moving Window Integrator
    :param signal: input signal
    :return: prcoessed signal

    Methodology/Explaination:
    The moving window integration process is done to obtain
    information about both the slope and width of the QRS complex.
    A window size of 0.15*(sample frequency) is used for more
    accurate results.

    The moving window integration has the recursive equation:
        y(nT) = [y(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N

        where N is the number of samples in the width of integration
        window.
    '''

    # Initialize result and window size for integration
    result = signal.copy()
    win_size = round(0.150 * sfreq)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
        sum += signal[j]/win_size
        result[j] = sum

    # Apply the moving window integration using the equation given
    for index in range(win_size,len(signal)):  
        sum += signal[index]/win_size
        sum -= signal[index-win_size]/win_size
        result[index] = sum

    return result

    
def Pan_Tompkin_preprocessing(edf_path, chann_name):
    """ This function is preprocessing part of Pan-Tompkin. It includes
    channel transformation (bi-polar if need), polarity check, bandpass, 
    derivative, and moving intergration.
    Args:
        chann_name: (str or list) if it is a string that is a single channel
        like 'ECG', 'X1', 'X2', if it's a list, which transformed as a bipolar
        Montage, like ['X1', 'X2'] -> 'X1-X2'.
    Return:
        sfreq: (float) sampling frequency of this epoch. 
        ecg_arr: (np array) the ecg signal array.
        time_arr: (np array) the the time series of the ecg signal.
        b_pass: (np array) the badnpassed signal array
        derivatived: (np.array) the derivatived signal array.
        squared: (np.array) the squared signal array.
        m_win: (np.array) the moving intergarted signal array.
        channel_name: (str) the final channle name (might be transformed).
        reverse_flag: (bool) the ecg signal is reversed or not.
    """
    # a). Read edf file pick ECG channel
    try:
        raw = read_raw_edf(edf_path, preload=True, stim_channel='auto', verbose=False)
    except:
        raise Exception('No such a file!')

    sfreq = int(raw.info['sfreq'])
    if sfreq > 256:
        logger.info(f'raw file original sampling frequency: {sfreq}')
        filtraw_downsampled = raw.copy().resample(sfreq=256)
        sfreq = 256
    else:
        filtraw_downsampled = raw.copy()
        logger.info(f'raw file original sampling frequency: {sfreq}')

    # Get raw ECG data
    if isinstance(chann_name, list):
        raw_bip =  mne.set_bipolar_reference(filtraw_downsampled, anode=chann_name[0], cathode=chann_name[1], verbose=False)
        channel_name = f'{chann_name[0]}-{chann_name[1]}'
        # print(f'bipolar channel_name: {channel_name}')
        raw_ECG = raw_bip.pick(channel_name)
        
    elif isinstance(chann_name, str):
        raw_ECG = filtraw_downsampled.pick(chann_name)
        channel_name = chann_name
        
    data_and_time = raw_ECG[channel_name, :]
    ecg_arr = data_and_time[0].reshape(-1)
    time_arr = data_and_time[1]
    # print(f'shape of ECG: {ecg_arr.shape} \nshape of time:{time_arr.shape}')
    
    # b). Get bandpassed ECG 
    # b_pass_raw = raw_ECG.filter(4, 30, picks=channel_name, fir_design='firwin', verbose=False)
    # b_pass = b_pass_raw.get_data().reshape(-1)
    b_pass = band_pass_filter(ecg_arr)
    
    reverse_flag = ECG_reverse_check(b_pass, sfreq)
    if reverse_flag:
        b_pass = -b_pass # Reverse Bpass
        # ecg_arr = -ecg_arr  # reverse ECG
        if isinstance(chann_name, list):
            channel_name = f'{chann_name[1]}-{chann_name[0]}'
        logger.info(f'signal {channel_name} channel has been \033[1mreversed!\033[0m')
        
    logger.info(f'signal {channel_name} channel is being processed!')

    # c). Derivated ECG
    # derivatived = np.gradient(b_pass)
    derivatived = derivative(b_pass, sfreq)
    
    # d). Squared ECG 
    # squared = np.square(derivatived)
    squared = squaring(derivatived)
    
    # e). Moving integrated ECG
    m_win = moving_window_integration(squared, sfreq)

    return sfreq, ecg_arr, time_arr, b_pass, derivatived, squared, m_win, channel_name, reverse_flag


def detect_peaks(ecg_signal, band_pass_signal, integration_signal, fs): 
    # Initialization of variables

    possible_peaks = []
    signal_peaks = []
    r_peaks = []
    # running estimate of the signal peak
    SPKI = 0
    # running estimate of the signal peak
    SPKF = 0
    # running estimate of the noise peak
    NPKI = 0
    # running estimate of the noise peak
    NPKF = 0
    rr_avg_one = []
    # First Integrated result threshold
    THRESHOLDI1 = 0
    # First Filtered result threshold
    THRESHOLDF1 = 0
    rr_avg_two = []
    # Second Integrated result threshold
    THRESHOLDI2 = 0
    # Second Filtered result threshold
    THRESHOLDF2 = 0
    # T wave detection flag
    is_T_found = 0
    # A search window of samples corresponding to 0.15 seconds
    window = round(0.15 * fs)            

    # Stage I: Fudicial Mark possible_peaks on the integrated signal
    FM_peaks = []
    # Smoothening the integration signal
    integration_signal_smooth = np.convolve(integration_signal, np.ones((20,)) / 20, mode = 'same')    
    localDiff = np.diff(integration_signal_smooth)
    # finding local maxima using difference array and ignoring
    # possible_peaks before initialization step i.e before fs

    for i in range(1,len(localDiff)):
        if i-1 > 2*fs and localDiff[i-1] > 0 and localDiff[i] < 0 :
            FM_peaks.append(i-1)           

    # Find out the possbile peaks for all the local maximas
    for index in range(len(FM_peaks)):

        # Finding maximum value position in the current search window
        current_peak = FM_peaks[index]
        left_limit = max(current_peak-window, 0) 
        right_limit = min(current_peak+window+1, len(band_pass_signal))
        max_index = -1
        max_value = -sys.maxsize
        for i in range(left_limit, right_limit):
            if(band_pass_signal[i] > max_value):
                max_value = band_pass_signal[i]
                max_index = i
        if (max_index != -1):
            possible_peaks.append(max_index)

        if (index == 0 or index > len(possible_peaks)):
          # if first peak
          if (integration_signal[current_peak] >= THRESHOLDI1): 
              SPKI = 0.125 * integration_signal[current_peak]  + 0.875 * SPKI
              if possible_peaks[index] > THRESHOLDF1:                                            
                  SPKF = 0.125 * band_pass_signal[index] + 0.875 * SPKF 
                  signal_peaks.append(possible_peaks[index])                             
              else:
                  NPKF = 0.125 * band_pass_signal[index] + 0.875 * NPKF                                    
              
          elif ( (integration_signal[current_peak] > THRESHOLDI2 and integration_signal[current_peak] < THRESHOLDI1) or (integration_signal[current_peak] < THRESHOLDI2)):
              NPKI = 0.125 * integration_signal[current_peak]  + 0.875 * NPKI  
              NPKF = 0.125 * band_pass_signal[index] + 0.875 * NPKF

        else:
            RRAVERAGE1 = np.diff(FM_peaks[max(0,index-8):index + 1]) / fs
            rr_one_mean = np.mean(RRAVERAGE1)
            rr_avg_one.append(rr_one_mean) 
            limit_factor = rr_one_mean
              
            if (index >= 8):
                # calculate RR limits and rr_avg_two
                for RR in RRAVERAGE1:
                    if RR > RR_LOW_LIMIT and RR < RR_HIGH_LIMIT:                              
                        rr_avg_two.append(RR)
                        if (len(rr_avg_two) == 9):
                          rr_avg_two.pop(0)     
                          limit_factor = np.mean(rr_avg_two)
            # set the RR limits
            if (len(rr_avg_two) == 8 or index < 8):
                RR_LOW_LIMIT = 0.92 * limit_factor        
                RR_HIGH_LIMIT = 1.16 * limit_factor
                RR_MISSED_LIMIT = 1.66 * limit_factor

            # Decrease the thresholds to half, if irregular beats detected
            if rr_avg_one[-1] < RR_LOW_LIMIT or rr_avg_one[-1] > RR_MISSED_LIMIT: 
                THRESHOLDI1 = THRESHOLDI1/2
                THRESHOLDF1 = THRESHOLDF1/2
               
            # If current RR interval is greater than RR_MISSED_LIMIT perform search back
            curr_rr_interval = RRAVERAGE1[-1]
            search_back_window = round(curr_rr_interval * fs)
            if curr_rr_interval > RR_MISSED_LIMIT:
                left_limit = current_peak - search_back_window +1
                right_limit = current_peak + 1
                search_back_max_index = -1 
                max_value =  -sys.maxsize
                # local maximum in the search back interval
                for i in range(left_limit, right_limit):
                  if (integration_signal[i] > THRESHOLDI1 and integration_signal[i] > max_value ):
                    max_value = integration_signal[i]
                    search_back_max_index = i
              
                if (search_back_max_index != -1):   
                    SPKI = 0.25 * integration_signal[search_back_max_index] + 0.75 * SPKI                         
                    THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
                    THRESHOLDI2 = 0.5 * THRESHOLDI1               
                    # finding peak using search back of 0.15 seconds
                    left_limit = search_back_max_index - round(0.15 * fs)
                    right_limit = min(len(band_pass_signal), search_back_max_index)

                    search_back_max_index2 = -1 
                    max_value =  -sys.maxsize
                    # local maximum in the search back interval
                    for i in range(left_limit, right_limit):
                      if (band_pass_signal[i] > THRESHOLDF1 and band_pass_signal[i] > max_value ):
                        max_value = band_pass_signal[i]
                        search_back_max_index2 = i

                    # QRS complex detected
                    if band_pass_signal[search_back_max_index2] > THRESHOLDF2: 
                        SPKF = 0.25 * band_pass_signal[search_back_max_index2] + 0.75 * SPKF                            
                        THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
                        THRESHOLDF2 = 0.5 * THRESHOLDF1                            
                        signal_peaks.append(search_back_max_index2)                                                 
    
            # T-wave detection
            if (integration_signal[current_peak] >= THRESHOLDI1): 
                if (curr_rr_interval > 0.20 and curr_rr_interval < 0.36 and index > 0): 
                    # slope of current waveformm which is most probabaly a T-wave, using mean width of QRS complex 0.075
                    current_slope = max(np.diff(integration_signal[current_peak - round(fs * 0.075):current_peak + 1]))
                    # slope of the preceding waveform, which is mosty probabaly QRS complex
                    previous_slope = max(np.diff(integration_signal[FM_peaks[index - 1] - round(fs * 0.075): FM_peaks[index - 1] + 1]))
                    if (current_slope < 0.5 * previous_slope): 
                        NPKI = 0.125 * integration_signal[current_peak] + 0.875 * NPKI                                            
                        is_T_found = 1                              
                #  This is a signal peak
                if (not is_T_found):
                    SPKI = 0.125 * integration_signal[current_peak]  + 0.875 * SPKI
                    # check if it is present in the possible peaks otherwise it is a noise peak
                    if possible_peaks[index] > THRESHOLDF1:                                            
                        SPKF = 0.125 * band_pass_signal[index] + 0.875 * SPKF 
                        signal_peaks.append(possible_peaks[index])                             
                    else:
                        NPKF = 0.125 * band_pass_signal[index] + 0.875 * NPKF                   
                                        
            elif ((integration_signal[current_peak] > THRESHOLDI1 and integration_signal[current_peak] < THRESHOLDI2) or (integration_signal[current_peak] < THRESHOLDI1)):
                NPKI = 0.125 * integration_signal[current_peak]  + 0.875 * NPKI  
                NPKF = 0.125 * band_pass_signal[index] + 0.875 * NPKF
       
        THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLDI2 = 0.5 * THRESHOLDI1 
        THRESHOLDF2 = 0.5 * THRESHOLDF1
        is_T_found = 0  

    # searching in ECG signal to increase accuracy
    for i in np.unique(signal_peaks):
        i = int(i)
        window = round(0.2 * fs)
        left_limit = i-window
        right_limit = min(i+window+1, len(ecg_signal))
        max_value = -sys.maxsize
        max_index = -1
        for i in range(left_limit, right_limit):
            if (ecg_signal[i] > max_value):
                max_value = ecg_signal[i]
                max_index = i

        r_peaks.append(max_index)
        
    return r_peaks
                    



if __name__ == "__main__":
    print("Pan Tompkins QRS Detection Algorithm")
    print("This is a standard Pan Tompkins QRS detection algorithm")

    # r_peaks = detect_peaks(ecg_signal, fs)
    # heart_beat = np.average(np.diff(r_peaks))/ fs
    # print("Heart Rate: "+ str(60/heart_beat) + " BPM")


    # QRS_detector = Pan_Tompkins_QRS()

    # ecg = pd.DataFrame(np.array([list(range(len(record.adc()))),record.adc()[:,0]]).T,columns=['TimeStamp','ecg'])

    # ecg_signal = ecg.iloc[:,1].to_numpy()
    # time_stamp = ecg.TimeStamp

    # fs = annotation.fs
    # integration_signal, band_pass_signal, derivative_signal, square_signal  = QRS_detector.solve(ecg_signal.copy())
    