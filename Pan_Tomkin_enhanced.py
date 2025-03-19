# ----------------------------------------------
# enhanced Pan Tompkins QRS detection algorithm
# References from
# https://github.com/adityatripathiiit/Pan_Tompkins_QRS_Detection
# https://github.com/antimattercorrade/Pan_Tompkins_QRS_Detection/
# ----------------------------------------------


import random
import numpy as np
import sys
import copy
import logging

from scipy import signal

import mne
from mne.io import read_raw_edf

logger = logging.getLogger('rr_extraction')
# logger = logging.getLogger('channel_selection')

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
    
def Pan_Tompkin_preprocessing(ANSeR_num, file_id, chann_name, dataset):
    """ This function is preprocessing part of Pan-Tompkin. It includes
    channel transformation (bi-polar if need), polarity check, bandpass, 
    derivative, and moving intergration.
    Args:
        *ANSeR_num: (int or str) the number of the ANSeR dataset. 
        *file_id: (str) the file id of the procesed epoch. 
        *chann_name: (str or list) if it is a string that is a single channel
        like 'ECG', 'X1', 'X2', if it's a list, which transformed as a bipolar
        Montage, like ['X1', 'X2'] -> 'X1-X2'.
        *dataset: (str) 'ANSeR' or 'Delphi'.
    Return:
        *sfreq: (float) sampling frequency of this epoch. 
        *ecg_arr: (np array) the ecg signal array.
        *time_arr: (np array) the the time series of the ecg signal.
        *b_pass: (np array) the badnpassed signal array
        *derivatived: (np.array) the derivatived signal array.
        *squared: (np.array) the squared signal array.
        *m_win: (np.array) the moving intergarted signal array.
        *channel_name: (str) the final channle name (might be transformed).
        *reverse_flag: (bool) the ecg signal is reversed or not.
    """
    # a). Read edf file pick ECG channel
    home_dir = '/mnt/files'
    if dataset == 'ANSeR':
        fn = '%s/Datasets/One_Hour-Epochs_Raw_Data/ANSeR%s/%s.edf' %(home_dir, ANSeR_num, file_id)

    try:
        raw = read_raw_edf(fn, preload=True, stim_channel='auto', verbose=False)
    except:
        raise Exception('No such a file!')

    sfreq = int(raw.info['sfreq'])
    if sfreq > 256:
        logger.info(f'{file_id} original sampling frequency: {sfreq}')
        filtraw_downsampled = raw.copy().resample(sfreq=256)
        sfreq = 256

    # Get raw ECG data
    if isinstance(chann_name, list):
        raw_bip =  mne.set_bipolar_reference(raw, anode=chann_name[0], cathode=chann_name[1], verbose=False)
        channel_name = f'{chann_name[0]}-{chann_name[1]}'
        raw_ECG = raw_bip.pick(channel_name)
        
    elif isinstance(chann_name, str):
        raw_ECG = raw.pick(chann_name)
        channel_name = chann_name
        
    data_and_time = raw_ECG[channel_name, :]
    ecg_arr = data_and_time[0].reshape(-1)
    time_arr = data_and_time[1]
    
    # b). Get bandpassed ECG 
    b_pass_raw = raw_ECG.filter(4, 30, picks=channel_name, fir_design='firwin', verbose=False)
    b_pass = b_pass_raw.get_data().reshape(-1)
    
    reverse_flag = ECG_reverse_check(b_pass, sfreq)
    if reverse_flag:
        b_pass = -b_pass # Reverse Bpass
        ecg_arr = -ecg_arr  # reverse ECG
        if isinstance(chann_name, list):
            channel_name = f'{chann_name[1]}-{chann_name[0]}'
        logger.info(f'{file_id} {channel_name} channel has been \033[1mreversed!\033[0m')
        
    logger.info(f'{file_id} {channel_name} channel is being processed!')

    # c). Derivated ECG
    derivatived = np.gradient(b_pass)
    
    # d). Squared ECG 
    squared = np.square(derivatived)
    
    # e). Moving integrated ECG
    # Method 2: moving average using convolve 
    def MovingAverage(ECG,N):
        '''
        Compute moving average of signal ECG with a rectangular window of N.
        The moving window integration has the recursive equation:
          y(nT) = [x(nT - (N-1)T) + x(nT - (N-2)T) + ... + x(nT)]/N
        
        N is the window length
        '''
        window  = np.ones((1,N))/N
        ECG_ma  = np.convolve(np.squeeze(ECG),np.squeeze(window), mode='same')
        return ECG_ma
        
    win_size = round(0.150 * sfreq)
    m_win = MovingAverage(squared, win_size)

    return sfreq, ecg_arr, time_arr, b_pass, derivatived, squared, m_win, channel_name, reverse_flag



class QRS_detect():
    '''
    The main class for Pan-Tompkin QRS detection algorithm. Including dual-Threholds detection, 
    search back, and artifact correction.
    '''
    def __init__(self, ecg, time_series, fs, moving_integrated, bandpassed):
        '''
        Params:
            *ecg: (np array like) the ecg signal array.
            *time_series: (np array like) the the time series of the ecg signal.
            *fs: (float) sampling frequency of this epoch. 
            *moving_integrated: (np array like) the moving intergarted signal array.
            *bandpassed: (np array like) the badnpassed signal array
        '''
        # Initialized veriables
        self.FM_peaks, self.possible_peaks, self.signal_peaks, self.r_peaks, self.RR1, self.RR2 = ([] for i in range(6))
        self.SPKI, self.SPKF, self.NPKI, self.NPKF, self.THRESHOLDI1, self.THRESHOLDF1, self.THRESHOLDI2, self.THRESHOLDF2, self.is_T_wave = (0 for i in range(9))
        
        self.I1_lst, self.I2_lst, self.F1_lst, self.F2_lst, self.RR_Avg_1_lst, self.RR_Avg_2_lst = ([] for i in range(6))
        self.SPKI_lst, self.NPKI_lst, self.SPKF_lst, self.NPKF_lst = ([] for i in range(4))
        self.detected_R_Avg, self.R_invals = ([] for i in range(2))
        self.rr_intervals_corrected = np.array([])
        
        self.samp_freq = fs
        self.ecg_signal = ecg
        self.time_series = time_series
        self.m_win = moving_integrated
        self.bpass = bandpassed
        self.window_150ms = round(0.15 * self.samp_freq)
    
        self.RR_LOW_LIMIT = 0
        self.RR_HIGH_LIMIT = 0
        self.RR_MISSED_LIMIT = 0
        self.RR_Average1 = 0
        self.detected_R_mean = 0

    def FM_peaks_detect(self):
        '''
        Fudicial Mark (FM) peaks detection.
        Detect all possible peaks from the moving integrated signal. 
        The first posssible peak detected from 0.5 second.
        '''
        # FFT convolution get slopes
        smooth_mawin = signal.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')
        slopes = np.diff(smooth_mawin)
        
        # Finding approximate peak locations
        for i in range(round(0.5*self.samp_freq) + 1,len(slopes)-1):
            if (slopes[i-1] >= 0) and (slopes[i] < 0) and (smooth_mawin[i] > 1e-12):
                self.FM_peaks.append(i)

    def Reset_threshold(self, before_start, index=None):
        '''
        Initialize THRESHOLD I1 F1, I2, F2 with the first 2 seconds interval.
        '''
        if before_start == True:
            peak_amp_mwin_2s = self.m_win[0:2*self.samp_freq]
            peak_amp_bpass_2s = self.bpass[0:2*self.samp_freq]
        else:
            index = self.FM_peaks[index]
            peak_amp_mwin_2s = self.m_win[max(0, int(index-1*self.samp_freq)) : min(int(index+1*self.samp_freq), len(self.m_win))]
            peak_amp_bpass_2s = self.bpass[max(0, int(index-1*self.samp_freq)) : min(int(index+1*self.samp_freq), len(self.bpass))]

            
        MAXI = max(peak_amp_mwin_2s)
        MEANI = np.mean(peak_amp_mwin_2s)
        self.THRESHOLDI1 = MAXI/3
        self.THRESHOLDI2 = 0.5*MEANI
        
        self.SPKI = self.THRESHOLDI1
        self.NPKI = self.THRESHOLDI2

        # Initialize THREHOLD F1 F2
        MAXF = max(peak_amp_bpass_2s)
        MEANF = np.mean(peak_amp_bpass_2s)
        self.THRESHOLDF1 = MAXF / 3
        self.THRESHOLDF2 = 0.5*MEANF
        
        self.SPKF = self.THRESHOLDF1
        self.NPKF = self.THRESHOLDF2

        # print('\033[36;1mThresholds have been reset!\033[0m')

        
    def adjust_threshold(self, current_peak, index):
        '''
        Check the first potential peak (FM_peaks[0]) is signal peak or noise peak
        and update the thresholds.
        '''
        if (self.m_win[current_peak] >= self.THRESHOLDI1):
            self.SPKI = 0.125 * self.m_win[current_peak]  + 0.875 * self.SPKI
            
            ######### signal peak ##########
            peakf = self.bpass[self.possible_peaks[index]]
            if peakf > self.THRESHOLDF1:                                            
                self.SPKF = 0.125 * peakf + 0.875 * self.SPKF 
                self.signal_peaks.append(self.possible_peaks[index])                             
            else:
                self.NPKF = 0.125 * peakf + 0.875 * self.NPKF
              
        ######### noise peak ##########                      
        # Threshold I2 < mvi[peak] < Threshold I1 or mvi[peak] < Threshold I2: Noise peak
        elif ( (self.THRESHOLDI2 < self.m_win[current_peak] < self.THRESHOLDI1) or (self.m_win[current_peak] < self.THRESHOLDI2) ):
            self.NPKI = 0.125 * self.m_win[current_peak] + 0.875 * self.NPKI  
            self.NPKF = 0.125 * self.bpass[self.possible_peaks[index]] + 0.875 * self.NPKF
              
    def adjust_rr_interval(self, index):
        '''
        Update RR1, RR2, and related rr intervals limit.
        '''
        # Take the most recent 8 peaks as RR_Average_1
        self.RR1 = np.diff(self.FM_peaks[max(0,index-8):index + 1]) / self.samp_freq

        # Take the most recent detected RR interval and average
        signal_peaks = np.sort(self.signal_peaks)
        n_peaks_detected = len(signal_peaks)
        if n_peaks_detected > 2:
            self.R_invals = np.diff(signal_peaks[max(0,n_peaks_detected-8):])/self.samp_freq
            self.detected_R_mean = np.mean(self.R_invals)
            
        # Update RR Average 1&2
        self.RR_Average1 =  np.mean(self.RR1)
        RR_Average2 = self.RR_Average1
          
        if (index >= 8):
            # calculate RR limits and RR_Average2
            for RR in self.RR1:
                if RR > self.RR_LOW_LIMIT and RR < self.RR_HIGH_LIMIT:                              
                    self.RR2.append(RR)
                    if (len(self.RR2) == 9):
                        self.RR2.pop(0)     
                        RR_Average2 = np.mean(self.RR2)
                        
        # set the RR limits
        if (len(self.RR2) == 8 or index < 8):
            self.RR_LOW_LIMIT = 0.92 * RR_Average2        
            self.RR_HIGH_LIMIT = 1.16 * RR_Average2
            self.RR_MISSED_LIMIT = 1.66 * RR_Average2
            
        self.RR_Avg_1_lst.append(self.RR_Average1)
        self.RR_Avg_2_lst.append(RR_Average2)
        self.detected_R_Avg.append(self.detected_R_mean)
        
        # Decrease the thresholds to half, if irregular beats detected
        if self.RR_Average1 < self.RR_LOW_LIMIT or self.RR_Average1 > self.RR_MISSED_LIMIT: 
            self.THRESHOLDI1 = self.THRESHOLDI1/2
            self.THRESHOLDF1 = self.THRESHOLDF1/2

    def detect_T_wave(self, curr_peak, curr_rr_interval, index):
        '''
        When an RR interval is less than 360ms (it must be greater than the 200ms latency),
        a judgement is made to determine whether the current peak is a QRS complex or T-wave.

        Params:
            curr_peak: current peak (detected from FM_peaks).
            curr_rr_interval: current rr interval (RR1[-1]).
            index: index of current peak (from FM_peaks).
        '''
        # Potential signal peak
        if (self.m_win[curr_peak] >= self.THRESHOLDI1):
            
            # if current rr interval less than 360ms (must less than 200ms latency)
            # all FM peaks are 231ms apart from each other (200ms latency is no longer necessary)
            # if ((0.2 < curr_rr_interval < 0.36 or curr_rr_interval < self.RR_MISSED_LIMIT) and index > 0): 
            if ((0.2 < curr_rr_interval < 0.36) or curr_rr_interval < 0.5*self.RR_Average1 and index > 0): 
                
                # slope of current waveformm which is most probabaly a T-wave, using mean width of QRS complex 75ms
                current_slope = max(np.diff(self.m_win[curr_peak - round(self.samp_freq * 0.075):curr_peak + 1]))
                # slope of the preceding waveform, which is mosty probabaly QRS complex
                previous_slope = max(np.diff(self.m_win[self.FM_peaks[index - 1] - round(self.samp_freq * 0.075): self.FM_peaks[index - 1] + 1]))
                if (current_slope < 0.3 * previous_slope): 
                    self.NPKI = 0.125 * self.m_win[curr_peak] + 0.875 * self.NPKI                                            
                    self.is_T_wave = 1 
                    
            #  This is a signal peak. Update Thresholds params.
            if (not self.is_T_wave):
                self.SPKI = 0.125 * self.m_win[curr_peak] + 0.875 * self.SPKI
                # check if it is present in the possible peaks otherwise it is a noise peak
                peakf = self.bpass[self.possible_peaks[index]]
                if peakf > self.THRESHOLDF1:                                            
                    # self.SPKF = 0.125 * self.bpass[index] + 0.875 * self.SPKF 
                    self.SPKF = 0.125 * peakf + 0.875 * self.SPKF 
                    self.signal_peaks.append(self.possible_peaks[index])                             
                else:
                    self.NPKF = 0.125 * peakf + 0.875 * self.NPKF 
                        
        # Noise peak            
        elif (self.THRESHOLDI2 < self.m_win[curr_peak] < self.THRESHOLDI1) or (self.m_win[curr_peak] < self.THRESHOLDI2):
            self.NPKI = 0.125 * self.m_win[curr_peak]  + 0.875 * self.NPKI  
            self.NPKF = 0.125 * self.bpass[self.possible_peaks[index]] + 0.875 * self.NPKF

    def searchback(self, index, curr_peak, curr_rr_interval, searback_window):
        '''
        perform a search back program to avoid the missed peaks.
        Params:
            curr_peak: current peak location (detected from FM_peaks).
            curr_rr_interval: current rr interval (RR1[-1]).
            searback_window: length of searchback window (current rr interval).
        '''
        # Process No R-peak in a long period of time.
        # If a QRS complex is not found during a define interval (166% of RR2 Average),
        # or 1s, the searchback is conduct to find missing peaks. 

        if len(self.signal_peaks) > 0:
            signal_peaks = np.sort(self.signal_peaks)
            time_no_rr = (curr_peak - signal_peaks[-1])/self.samp_freq
        
            if time_no_rr >= 1.4:
                self.Reset_threshold(before_start=False, index=index)
                # print(f'time_no_rr: {time_no_rr}')
            
        if curr_rr_interval > self.RR_MISSED_LIMIT or (1.0 < curr_rr_interval < 1.4) or (len(self.signal_peaks) > 0 and 1.0 < time_no_rr < 1.4):

            # 1). Initialize a searchback window 
            # (from current peak back to the current rr interval)
            # searback_window = int(0.36*self.samp_freq)
            if len(self.signal_peaks)>0:
                left_limit = int(signal_peaks[-1] + 0.2*self.samp_freq)
            else:
                left_limit = curr_peak - searback_window + 1
                
            right_limit = int(curr_peak - 0.2*self.samp_freq)
            search_back_max_index = -1 
            max_value =  -sys.maxsize
            
            # 2). Check (moving integration signal) whether current peak is greater than noise threshold I2?
            # identify the local maximum in the search back interval
            # To be a signal peak, the peak must exceed THRESHOLD I2 as the signal is first analyzed
            # or THRESHOLD I2 is searchback is required to find the QRS.
            for i in range(left_limit, right_limit):
                if ( self.m_win[i] > self.THRESHOLDI2 and self.m_win[i] > max_value ):
                    max_value = self.m_win[i]
                    search_back_max_index = i
            
            # 3). Check (bandpass signal) whether current peak is greater than noise threshold F2
            # Initialized another search back window with a length of 150ms.
            # (window end at the local maximum of previous RR interval search back window)
            if (search_back_max_index != -1): 
                
                # a). if detected in searchback, using the second thresholds
                # self.SPKI = 0.75 * self.m_win[search_back_max_index] + 0.25 * self.SPKI                         
                self.SPKI = 0.25 * self.m_win[search_back_max_index] + 0.75 * self.SPKI                         
                self.THRESHOLDI1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.THRESHOLDI2 = 0.5 * self.THRESHOLDI1  
                
                # b). finding peak using search back window of 150ms in bandpassed signal
                left_limit = search_back_max_index - round(0.15 * self.samp_freq)
                right_limit = min(len(self.bpass), search_back_max_index)
            
                # c). Detect local maximum in this 150ms search back window
                search_back_max_index2 = -1 
                max_value =  -sys.maxsize
                for i in range(left_limit, right_limit):
                    if (self.bpass[i] > self.THRESHOLDF2 and self.bpass[i] > max_value ):
                        max_value = self.bpass[i]
                        search_back_max_index2 = i

                # d). QRS complex detected (if detected in searchback using the second thresholds)
                # Test the Threshold F2. If passed, update Thresholds and signal peaks
                # if self.bpass[search_back_max_index2] > self.THRESHOLDF2: 
                if (search_back_max_index2 != -1): 
                    # self.SPKF = 0.75 * self.bpass[search_back_max_index2] + 0.25 * self.SPKF                            
                    self.SPKF = 0.25 * self.bpass[search_back_max_index2] + 0.75 * self.SPKF                            
                    self.THRESHOLDF1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                    self.THRESHOLDF2 = 0.5 * self.THRESHOLDF1                            
                    self.signal_peaks.append(search_back_max_index2)

        
    def update_thresholds(self):
        '''
        Update Thresholds after each peak or noise identified.
        '''
        self.THRESHOLDI1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.THRESHOLDF1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.THRESHOLDI2 = 0.5 * self.THRESHOLDI1 
        self.THRESHOLDF2 = 0.5 * self.THRESHOLDF1
        self.is_T_wave = 0 
        
        self.I1_lst.append(self.THRESHOLDI1)
        self.I2_lst.append(self.THRESHOLDI2)
        self.F1_lst.append(self.THRESHOLDF1)
        self.F2_lst.append(self.THRESHOLDF2)

        self.SPKI_lst.append(self.SPKI)
        self.SPKF_lst.append(self.SPKF)
        self.NPKI_lst.append(self.NPKI)
        self.NPKF_lst.append(self.NPKF)

    def ecg_searchback(self):
        '''
        Using the detected peaks (from m_win and bpass) to check 
        real peak location in ECG signal. Process the delay from the 
        raw ECG signal.
        '''
        
        signal_peaks = np.unique(self.signal_peaks)
        for i in signal_peaks:
            i = int(i)
            window = round(0.2 * self.samp_freq)
            left_limit = i-window
            right_limit = min(i+window+1, len(self.bpass))
            max_value = -sys.maxsize
            max_index = -1
            for i in range(left_limit, right_limit):
                if (self.bpass[i] > max_value):
                    max_value = self.bpass[i]
                    max_index = i
    
            self.r_peaks.append(max_index)

        # remove the duplicate peaks    
        self.r_peaks = np.unique(self.r_peaks)

    def artifact_correction_rr(self):
        '''
        Correct some unusual rr intervals that might effected by artifacts.
        If rr interval greater than 1.5s or less than 0.1s, it would be replaced by
        the RR average 2 in that time.
        '''
        # 1). Get rr intervals
        RPeaks = list(self.r_peaks)
        r_peaks_corrected = copy.deepcopy(RPeaks)
        n_insert_peks = 0
        n_insert_rr = 0
        n_peak_segment = 0
        assert sorted(RPeaks) == RPeaks
        
        time_stamps = self.time_series[RPeaks]
        rr_intervals = np.diff(time_stamps)
        
        if len(RPeaks) != 0:
            assert len(RPeaks) == (len(rr_intervals) + 1), "Length of RPeaks {}, length of rr intervals {}".format(len(RPeaks), len(rr_intervals))

        # calculate rr mean and moving average
        rr_mean = np.mean(rr_intervals)
        N = 8
        rr_ma = signal.fftconvolve(rr_intervals, np.full((N,), 1) / N, mode='same')
        
        # 2). Check an correct each rr interval.
        # self.rr_intervals_corrected = np.zeros(len(rr_intervals))
        self.rr_intervals_corrected = []
        for i, rr in enumerate(rr_intervals):
            # for rr that might experience small artifact lost one or two beats
            if 2.05*rr_mean < rr <= 2 or rr < 0.2:
                self.rr_intervals_corrected.append(rr_ma[i]) 

            # for rr that experienced large artifacts
            elif 2 < rr < 10:
                # Setup a window with a length of (current rr interval-0.5s).
                # Take the possible peaks on bandpassed signal in this window 
                # as the interpolated peaks.               
                
                current_peak = RPeaks[i+1]
                peak_window_start = current_peak - (rr-0.5)*self.samp_freq
                peaks_idx = np.searchsorted(self.possible_peaks, [peak_window_start, current_peak])
                insert_peaks = self.possible_peaks[peaks_idx[0]:peaks_idx[1]]
                insert_peaks = np.unique(insert_peaks)
                
                # filter insert peaks that causes rr lower than 0.2s
                filter_threshold = 0.6*rr_mean
                previous_peak = RPeaks[i]
                peaks_added = [previous_peak] + list(insert_peaks) + [current_peak]
                rr_all = np.diff(peaks_added)/self.samp_freq
                rr_low = np.diff(peaks_added)/self.samp_freq < filter_threshold
                drop_i, = np.where(rr_low == True)
                
                while len(drop_i) > 0:
                    drop_idx = drop_i[0]
                    # if the final rr is lower than 0.2s, drop the final inserted peaks
                    # rather than the current peak
                    if drop_idx + 2 == len(peaks_added):
                        peaks_added.pop(drop_idx)
                    else:
                        peaks_added.pop(drop_idx+1)
                
                    rr_all = np.diff(peaks_added)/self.samp_freq
                    rr_low = rr_all < filter_threshold
                    drop_i, = np.where(rr_low == True)
                
                insert_peaks = peaks_added[1:-1]
                rr_inter = rr_all

                # Only process inserted peaks are not empty.
                if  len(insert_peaks) > 0:
                    
                    n_peak_segment += 1
                    
                    # Update the rr intervals
                    self.rr_intervals_corrected.extend(rr_inter)
                    n_insert_rr += len(rr_inter)
                    
                    # Update the insert peak location on R_peaks
                    # for new time stamps generation
                    # insert index should based on the precious r_peaks_corrected list  
                    idx_insert = i + 1 + n_insert_peks                
                    r_peaks_corrected[idx_insert:idx_insert] = list(insert_peaks)
                    n_insert_peks += len(insert_peaks)
                    assert len(r_peaks_corrected) == len(RPeaks) + n_insert_peks, f"length of r_peaks_corrected: {len(r_peaks_corrected)}, length of RPeaks: {len(RPeaks)}"
                    seg_start = idx_insert - 3
                    seg_end = seg_start + 3 + len(insert_peaks) + 3
                    
                    assert len(rr_inter) == (len(insert_peaks) + 1), "length of inserted rr {}, length of inserted peaks {}".format(len(rr_inter), len(insert_peaks))

                # do nothing if no peak is inserted
                else:
                    self.rr_intervals_corrected.append(rr)

            # rr intervals greater than 10s might caused by lost connection 
            elif rr >= 10:
                self.rr_intervals_corrected.append(rr)

            # for the normal rr 
            else:
                self.rr_intervals_corrected.append(rr) 
                
        assert sorted(r_peaks_corrected) == r_peaks_corrected
        time_stamps_corrected = self.time_series[r_peaks_corrected]
        
        if len(self.r_peaks) != 0:
            assert len(time_stamps_corrected) == (len(self.rr_intervals_corrected) + 1), 'length of corrected time stamp {}, length of corrected rr {}'.format(len(time_stamps_corrected), len(self.rr_intervals_corrected))        
        
        return time_stamps_corrected, r_peaks_corrected

    
    def detect_r_peaks(self):
        '''
        The main program for detect rr peaks.
        '''
        # 1). Find all approximate peaks (fiducial Mark) from moving integrated signal
        self.FM_peaks_detect()
        # Initialize THRESHOLD I1, F1 with the first 2 second interval of the signal
        self.Reset_threshold(before_start=True)
        
        for index in range(len(self.FM_peaks)):
            # 2). Detect possible peaks from b_passed signal
            # Initialized a 300ms search window cerntered in current peak  back and forward 150ms each.
            # Find the maximum of this window as the possible peaks.
            current_peak = self.FM_peaks[index]
            left_limit = max(current_peak-self.window_150ms, 0) 
            right_limit = min(current_peak+self.window_150ms+1, len(self.bpass))
            max_index = -1
            max_value = -sys.maxsize
            for i in range(left_limit, right_limit):
                if(self.bpass[i] > max_value):
                    max_value = self.bpass[i]
                    max_index = i
            if (max_index != -1):
                self.possible_peaks.append(max_index)

            # 3). Dual-thresholds Check
            # - For non-first potential peak check and threshold update
            if (index != 0 and index < len(self.possible_peaks)):

                # a). update RR intervals and RR limit
                self.adjust_rr_interval(index)
                
                current_rr_interval = self.RR1[-1]
                # b). if current RR interval is too close, R peak or T peak?
                self.detect_T_wave(current_peak, current_rr_interval, index)

                # c). if current rr interval is too high, it might loss some peaks.
                # search back if beyond the RR limit
                searchback_window = round(current_rr_interval*self.samp_freq)
                self.searchback(index, current_peak, current_rr_interval, searchback_window)

            else:
                # identify whether current peak is signal or noise peak and update related params
                self.adjust_threshold(current_peak, index)
                
            # 4). Update thresholds for next interation.
            self.update_thresholds()

        # Seachback in ECG/bandpassed signal
        self.ecg_searchback()
        
        # Correct some unusual RR due to the artifacts
        time_stamps_corrected, r_peaks_corrected = self.artifact_correction_rr()

        return self.r_peaks, self.rr_intervals_corrected, time_stamps_corrected
        