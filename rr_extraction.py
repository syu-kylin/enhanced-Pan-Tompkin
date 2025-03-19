

import os
import math
import copy
import json
import random
from datetime import date
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # for progress bar
import scipy.io as scio    # read .mat file
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.style.use(['science', 'no-latex'])

from Pan_Tomkin_enhanced import Pan_Tompkin_preprocessing as Pan_Tompkin_preprocessing_enhanced
from Pan_Tomkin_enhanced import QRS_detect

from Pan_Tompkin_stand import Pan_Tompkin_preprocessing as Pan_Tompkin_preprocessing_standard
from Pan_Tompkin_stand import detect_peaks


logger = logging.getLogger('rr_extraction')

def check_file_ids(ANSeR_num, baby_num, hours):
    '''
    This function can be used to get the original and new formated file ids 
    according to the provided ANSeR number and baby IDs.
    
    Args:
    ANSeR_num: The ANSeR number (int 1 or 2)
    baby_IDs: the list of baby IDs in the corresponding ANSeR dataset.
    hours: (string) "ANSeR", or (Delphi)"7-11" or "13-23", "25-35" or "37-47".
    
    Output:
    [file_ids_new_lst, file_ids_orig_lst]: the list of the new formated file ids and
    the list of the original formated file ids.
    file_ids_info: given how many epoches does each baby id has.
    baby_ids: a list of the baby ids for each corresponding epoch files. 
    '''
    file_ids_info = []
    file_ids_lst = []
    file_ids_new_lst = []
    baby_ids = []
    # epoch_time = [6,12,24,36,48]    # Define the epoch hour
    # epoch_time = range(13,24)    # Define the epoch hour
    home_dir = '/mnt/files'

    if hours == 'ANSeR':
        epoch_time = [6,12,24,36,48]
    elif '-' in hours:
        epoch_time = range(int(hours.split('-')[0]), int(hours.split('-')[1])+1)

    for baby_id in baby_num:
        n_epoch = 0
        for i in range(len(epoch_time)):
            file_id = f'ID%d_ANSeR%s_%d_HR' %(baby_id, ANSeR_num, epoch_time[i])
            file_id_new = f'ID%03d_ANSeR%s_%02d_HR' %(baby_id, ANSeR_num, epoch_time[i])
            if hours == 'ANSeR':
                fn = f'%s/Datasets/One_Hour-Epochs_Raw_Data/ANSeR%s/%s.edf' %(home_dir, ANSeR_num, file_id)   
            else:
                fn = '//central/cerebro/INFANT_Projects_2/Delphi/Shuwen_1_Hour_ANSeR{}_Files/{}.edf'.format(ANSeR_num, file_id)
            
            flag = os.path.isfile(fn) 
            
            if flag:
                n_epoch += 1
                file_ids_lst.append(file_id)
                file_ids_new_lst.append(file_id_new)
                baby_ids.append(baby_id)
                
        assert len(baby_ids) == len(file_ids_lst)  
        
        id_check = f'ID%d_ANSeR%s has %d epochs' %(baby_id, ANSeR_num, n_epoch)
        file_ids_info.append(id_check)
                
    return [file_ids_new_lst, file_ids_lst], file_ids_info, baby_ids

def channel_selection(file_id_raw, channel_df):
    '''
    This function aims to get the channel name of all epoch file
    by check from a dataframe (read from a xlsx file).
    '''

    try:
        channel_name = channel_df[channel_df['file_id'] == file_id_raw]['channel_name'].values[0]
    except:
        raise Exception("Channel name cannot be found!")
    # print(f'channel name: {channel_name}')

    # Process bipolar monatge channel name
    if channel_name in ("ECG1-ECG2", "ECG2-ECG1"):
        channel_name = ["ECG1", "ECG2"]
    elif channel_name in ("X1-X2", "X2-X1"):
        channel_name = ["X1", "X2"]
    
    return channel_name


def rr_interval_collection_baby_level(ANSeR_num, baby_num, hours, channel_df, invalid_epoch, version='enhanced'):
    '''
    This function will collect all rr (from Pan_Tompkin) from the listed 
    babies epoch files. The channel name of each epoch can be access from 
    "channel_selection" function (by reading an xlsx file).
    Input:
        *ANSeR_num (int): 1 or 2. 
        *baby_num (list): list of baby IDs. 

    Rerurn:
        # *rr_struct(numpy struct): the extracted rr and its related attributes (see the dtype).
        # *signal_struct(numpy struct): the bandpassed ECG and its attributes (see the dtype).
    '''
    # Get all epoch file ids
    home_dir = '/mnt/files'
    file_ids_lst = check_file_ids(ANSeR_num, baby_num, hours)[0]
    file_id_formated = file_ids_lst[0]
    file_id_raw = file_ids_lst[1]
    
    # Remove invalid epochs
    logger.debug(f'number of files before remove invalid epochs: {len(file_id_raw)}')
    file_ids_copy = copy.deepcopy(file_id_raw)
    invalid_epoch_idx = []
    for i, epoch_id in enumerate(file_ids_copy):
        if epoch_id in invalid_epoch:
            file_id_raw.remove(epoch_id)
            invalid_epoch_idx.append(i)
    # print(f'file_id_raw: {file_id_raw}')
    logger.debug(f'number of files after remove invalid epochs: {len(file_id_raw)}')
    valid_file_id_format = np.delete(np.array(file_id_formated), invalid_epoch_idx)
    assert len(file_id_raw) == len(valid_file_id_format)
    # print(f'file_id_raw: {file_id_raw}')
    # print(f'file_id_formated: {valid_file_id_format}')
    
    # Define a numpy object array to store rr information 
    num_epoch = len(file_id_raw)
    dt = np.dtype([('file_id', '<U25'), ('rr_interval', 'O'), ('rr_interval_corrected', 'O'), 
                   ('time_stamp', 'O'), ('time_stamp_corrected', 'O'), ('EEG_grade', 'i4'),
                  ('mean', 'f8'), ('std', 'f8'), ('std_corrected', 'f8')])
    rr_struct = np.zeros((num_epoch), dtype=dt)
    dt = np.dtype([('bandpass', 'O'), ('time_series', 'O'), ('sfreq', 'f8'), 
                   ('reverse_flag', '?'), ('chan_name', '<U12')])
    signal_struct = np.zeros((num_epoch), dtype=dt)

    # Get rr of each epoch
    for i, file_id in enumerate(tqdm(file_id_raw, leave=True, ncols=100)):
        # Get channel name and extract rr
        logger.info(' ')
        # logger.debug(f'file_id: {file_id}')
        chann_name = channel_selection(file_id, channel_df)
        # print(f'chann_name: {chann_name}')

        if version == 'enhanced':
            sfreq, ecg_arr, time_arr, b_pass, _, _, m_win, chann_name, reverse_flag = Pan_Tompkin_preprocessing_enhanced(ANSeR_num, file_id, chann_name, hours)
            hr = QRS_detect(ecg_arr, time_arr, sfreq, m_win, b_pass)
            
            r_peaks, rr_interval_corrected, time_stamp_corrected = hr.detect_r_peaks()
            std_corrected = np.std(rr_interval_corrected)
        else:
            sfreq, ecg_arr, time_arr, b_pass, _, _, ma_win, chann_name, reverse_flag = Pan_Tompkin_preprocessing_standard(ANSeR_num, file_id, chann_name, hours)
            r_peaks = detect_peaks(ecg_arr, b_pass, ma_win, sfreq)
            rr_interval_corrected, time_stamp_corrected = 0, 0
            std_corrected = 100

        rr_time_stamp = time_arr[r_peaks]
        rr_intervals = np.diff(rr_time_stamp)
        mean = np.mean(rr_intervals)
        std = np.std(rr_intervals)

        # Get EEG Grade
        # for ANSeR dataset
        if hours == 'ANSeR':
            anno_fn = f'%s/Datasets/One_Hour-Epochs_Raw_Data/ANSeR%s/AnSeR%s_annotation.csv' %(home_dir, ANSeR_num, ANSeR_num)
        # for Delphi dataset
        else:
            anno_fn = f"{home_dir}/Datasets/Delphi/ANSeR{ANSeR_num}/ANSeR{ANSeR_num}_annotation.csv"
        annodf = pd.read_csv(anno_fn)
        score = annodf[annodf.ID == valid_file_id_format[i]]['score'].values[0]
        # for epoch files without labels give EEG Grade as 100 (invalid)
        if math.isnan(score):
            score = 100

        # Save data into numpy struct
        rr_struct[i] = (str(valid_file_id_format[i]), rr_intervals, rr_interval_corrected, rr_time_stamp, 
                        time_stamp_corrected, score, mean, std, std_corrected)
        signal_struct[i] = (b_pass, time_arr, sfreq, reverse_flag, chann_name)
        logger.info(f'{valid_file_id_format[i]} QRS detection done!')

    logger.info(f'\033[34;1mANSeR{ANSeR_num} {hours}h all ({num_epoch}) epochs RR collection done!\033[0m')
    
    return rr_struct, signal_struct


def rr_interval_collection_epoch_level(ANSeR_num, file_ids_lst, hours, channel_df, invalid_epoch, version='enhanced'):
    '''
    This function will collect all rr (from Pan_Tompkin) from the listed 
    epoch files (Mainly for those files that don't have labels). 
    The channel name of each epoch can be access from 
    "channel_selection" function (by reading an xlsx file).
    Input:
        *ANSeR_num (int): 1 or 2. 
        *file_ids_lst (list): list of file ids with the uniform format. 

    Rerurn:
        # *rr_struct(numpy struct): the extracted rr and its related attributes (see the dtype).
        # *signal_struct(numpy struct): the bandpassed ECG and its attributes (see the dtype).
    '''
    
    logger.debug(f'length of file_ids_lst: {len(file_ids_lst)}')
    # Remove invalid epochs
    for i, epoch_id in enumerate(file_ids_lst):
        if epoch_id in invalid_epoch:
            file_ids_lst.remove(epoch_id)
    logger.debug(f'length of file_ids_lst after remove invalid epochs: {len(file_ids_lst)}')   

    # Define a numpy object array to store rr information 
    num_epoch = len(file_ids_lst)
    dt = np.dtype([('file_id', '<U25'), ('rr_interval', 'O'), ('rr_interval_corrected', 'O'), 
                   ('time_stamp', 'O'), ('time_stamp_corrected', 'O'), ('EEG_grade', 'i4'),
                  ('mean', 'f8'), ('std', 'f8'), ('std_corrected', 'f8')])
    rr_struct = np.zeros((num_epoch), dtype=dt)
    dt = np.dtype([('bandpass', 'O'), ('time_series', 'O'), ('sfreq', 'f8'), 
                   ('reverse_flag', '?'), ('chan_name', '<U12')])
    signal_struct = np.zeros((num_epoch), dtype=dt)

    # Get rr of each epoch
    for i, file_id in enumerate(tqdm(file_ids_lst, leave=True, ncols=100)):

        # Get raw file id
        baby_id = int(file_id[2:5])
        hour = int(file_id[-5:-3])
        file_id_raw = f'ID{baby_id}_ANSeR2_{hour}_HR'

        # Get channel name and extract rr
        # print(f'file_id: {file_id}')
        chann_name = channel_selection(file_id_raw, channel_df)
        # print(f'chann_name: {chann_name}')

        if version == 'enhanced':
            sfreq, ecg_arr, time_arr, b_pass, _, _, m_win, chann_name, reverse_flag = Pan_Tompkin_preprocessing_enhanced(ANSeR_num, file_id, chann_name, hours)
            hr = QRS_detect(ecg_arr, time_arr, sfreq, m_win, b_pass)
            
            r_peaks, rr_interval_corrected, time_stamp_corrected = hr.detect_r_peaks()
            std_corrected = np.std(rr_interval_corrected)
        else:
            sfreq, ecg_arr, time_arr, b_pass, _, _, ma_win, chann_name, reverse_flag = Pan_Tompkin_preprocessing_standard(ANSeR_num, file_id, chann_name, hours)
            r_peaks = detect_peaks(ecg_arr, b_pass, ma_win, sfreq)
            rr_interval_corrected, time_stamp_corrected = None, None
            std_corrected = 100

        rr_time_stamp = time_arr[r_peaks]
        rr_intervals = np.diff(rr_time_stamp)
        mean = np.mean(rr_intervals)
        std = np.std(rr_intervals)
        std_corrected = np.std(rr_interval_corrected)

        # Those files don't have labels, give EEG Grade as 100 (invalid)
        # keep the rr struct have the same shape.
        score = 100

        # Save data into numpy struct
        rr_struct[i] = (file_id, rr_intervals, rr_interval_corrected, rr_time_stamp, 
                        time_stamp_corrected, score, mean, std, std_corrected)
        signal_struct[i] = (b_pass, time_arr, sfreq, reverse_flag, chann_name)
        logger.info(f'{file_id} QRS detection done!')

    logger.info(f'\033[34;1mANSeR{ANSeR_num} {hours}h all ({num_epoch}) epochs QRS detection done!\033[0m')
    
    return rr_struct, signal_struct

    
def plot_rr(ANSeR_num, rr_struct, signal_struct, file_path):    
    '''This function aims to plot the extracted rr of all epoch file
    from the rr_struct, and the bandpassed signal from the signal_struct.
    '''
    
    T = random.randint(1, 3000)
    # group_name = group_name_select(chann_name)
    
    for i in tqdm(range(len(rr_struct)), ncols=100):
    
        file_id = rr_struct[i]['file_id'].replace('_', ' ')
        rr_interval = rr_struct[i]['rr_interval']
        rr_interval_corrected = rr_struct[i]['rr_interval_corrected']
        rr_time_stamp = rr_struct[i]['time_stamp']
        time_stamp_corrected = rr_struct[i]['time_stamp_corrected']
        mean = rr_struct[i]['mean']
        std = rr_struct[i]['std']
        std_corrected = rr_struct[i]['std_corrected']
        
        
        bandpass = signal_struct[i]['bandpass']
        time_series = signal_struct[i]['time_series']
        chan_name_selected = signal_struct[i]['chan_name']
        sfreq = signal_struct[i]['sfreq']
        reverse_flag = signal_struct[i]['reverse_flag']
        # file_id = file_id.replace('_', ' ')
        # print(f'sfreq:{sfreq}')
        # print(f'{T*sfreq}')
        
        indx_start = int(T*sfreq)
        indx_end = int((T+5)*sfreq)
        
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,9), constrained_layout=True)
        
        text0 = f"std: {std:.4f}\nchannel: {chan_name_selected}"
        axs[0].plot(rr_time_stamp[1:], rr_interval, label=f'std={std:.4f}')
        if min(rr_interval)<0.3:
            axs[0].axhline(0.2,color='r')
        axs[0].set_title(f'{file_id} rr intervals')
        axs[0].set_xlabel('time(s)')
        axs[0].set_ylabel(f'rr interval (s)')
        axs[0].text(1.01, 0.8, text0, transform=axs[0].transAxes)
        axs[0].grid()
        # axs[0].legend()
        
        text1 = f"std: {std_corrected:.4f}\nmean: {mean:.4f}\nchannel: {chan_name_selected}"
        axs[1].plot(time_stamp_corrected[1:], rr_interval_corrected, label=f'std={std_corrected:.4f}')
        if min(rr_interval_corrected) < 0.3:
            axs[1].axhline(0.2,color='r')
        axs[1].set_title(f'{file_id} rr intervals corrected')
        axs[1].set_xlabel('time(s)')
        axs[1].set_ylabel(f'rr interval (s)')
        axs[1].text(1.01, 0.8, text1, transform=axs[1].transAxes)
        axs[1].grid()
        # axs[1].legend()
    
        text2 = f"reverse: {reverse_flag}"
        axs[2].plot(time_series[indx_start:indx_end], bandpass[indx_start:indx_end])
        axs[2].set_title(f'{file_id} Bandpassed {chan_name_selected} from {T} to {T+5}s')
        axs[2].text(1.01, 0.8, text2, transform=axs[2].transAxes)
        axs[2].set_xlabel('time(s)')

        today = date.today()
        folder_name = f"{file_path}/{today}/"
        
        os.makedirs(folder_name, exist_ok=True)
        # plt.show()
        
        if reverse_flag:
            plt.savefig(f"{folder_name}/{file_id} rr interval (reversed).svg")
        else:
            plt.savefig(f"{folder_name}/{file_id} rr interval.svg")
            
        plt.close()

    logger.info('\033[34;1mAll epoch plot done!\033[0m')


def plot_stand_enhanced_rr(rr_struct_stand, rr_struct_enhanced, out_dir):
    '''This function aims to plot the bandpassed signal from the signal_struct.
    '''
    assert len(rr_struct_stand) == len(rr_struct_enhanced), "The length of the two rr struct should be the same!"

    for i in tqdm(range(len(rr_struct_stand)), ncols=100):

        file_id_enhanced = rr_struct_enhanced[i]['file_id'].replace('_', ' ')
        rr_interval_enhanced = rr_struct_enhanced[i]['rr_interval']
        rr_time_stamp_enhanced = rr_struct_enhanced[i]['time_stamp']
        rr_interval_corrected = rr_struct_enhanced[i]['rr_interval_corrected']
        time_stamp_corrected = rr_struct_enhanced[i]['time_stamp_corrected']
        mean_enhanced = rr_struct_enhanced[i]['mean']
        std_enhanced = rr_struct_enhanced[i]['std']
        std_corrected = rr_struct_enhanced[i]['std_corrected']

        file_id_stand = rr_struct_stand[i]['file_id'].replace('_', ' ')
        assert file_id_enhanced == file_id_stand, "The file id should be the same!"
        rr_interval_stand = rr_struct_stand[i]['rr_interval']
        rr_time_stamp_stand = rr_struct_stand[i]['time_stamp']
        mean_stand = rr_struct_stand[i]['mean']
        std_stand = rr_struct_stand[i]['std']

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,9), constrained_layout=True)

        text0 = f"mean: {mean_stand:.4f}\nstd: {std_stand:.4f}"
        axs[0].plot(rr_time_stamp_stand[1:], rr_interval_stand)
        if min(rr_interval_stand)<0.3:
            axs[0].axhline(0.2,color='r')
        axs[0].set_title(f'rr intervals of {file_id_stand} from standard Pan-Tompkin')
        axs[0].text(0.9, 0.83, text0, transform=axs[0].transAxes)
        axs[0].set_xlabel('time(s)')
        axs[0].grid()

        text1 = f"mean: {mean_enhanced:.4f}\nstd: {std_corrected:.4f}"
        axs[1].plot(time_stamp_corrected[1:], rr_interval_corrected)
        if min(rr_interval_corrected) < 0.3:
            axs[1].axhline(0.2,color='r')
        axs[1].set_title(f'corrected rr intervals of {file_id_enhanced} from enhanced Pan-Tompkin')
        axs[1].text(0.9, 0.83, text1, transform=axs[1].transAxes)
        axs[1].set_xlabel('time(s)')
        axs[1].grid()

        text2 = f"mean: {mean_enhanced:.4f}\nstd: {std_enhanced:.4f}"
        axs[2].plot(rr_time_stamp_enhanced[1:], rr_interval_enhanced)
        if min(rr_interval_enhanced) < 0.3:
            axs[2].axhline(0.2,color='r')
        axs[2].set_title(f'rr intervals of {file_id_enhanced} from enhanced Pan-Tompkin')
        axs[2].text(0.9, 0.83, text2, transform=axs[2].transAxes)
        axs[2].set_xlabel('time(s)')
        axs[2].grid()

        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/{file_id_enhanced} rr interval comparison.svg")
        plt.close()



if __name__ == "__main__":

    """ 
    Before the rr extraction, the "rr_channel_check.py" have to be
    run to produce the best ECG channel. 
    """
    
    # 0). Select baby ids and epoch period
    # --------------------- ANSeR 1 --------------------------
    # Baby one-hour epochs 6, 12, 24, 36, 48 hours (215/216 epochs)
    baby_ids_ANSeR1 = [54, 224, 231, 235, 236, 276, 279, 284, 313, 337, 347, 348, 
                       349, 351, 353, 366, 374, 393, 476, 493, 504, 512, 517, 558, 
                       559, 576, 631, 261, 32,  33,  48,  95, 102, 116, 142, 144, 314, 408, 411, 446, 
                       477, 529, 546, 554, 572, 590, 730, 734, 751, 601, 620, 639, 647, 710, 713, 722,
                       24, 53]
    hours = 'ANSeR'

    # --------------------- ANSeR 2 --------------------------
    # Baby one-hour epochs 6, 12, 24, 36, 48 hours (257/259 epochs)
    # baby_ids = [45, 95, 101, 113, 145, 185, 197, 234, 133, 170, 175, 201, 251, 252,
    #         4, 5, 19, 35, 40, 47, 50, 59, 62, 75, 77, 79, 82, 86, 87,
    #         88, 93, 94, 96, 98, 107, 108, 112, 119,120, 126, 127, 128, 132, 
    #         134, 135, 141, 147, 150, 151, 159, 160, 162, 168, 174, 178, 186, 
    #         188, 199, 204, 205, 213, 217, 227, 231, 232, 238, 241, 244, 245, 
    #         250, 256, 257, 264, 265, 63]
    # hours = 'ANSeR'

    baby_ids = baby_ids_ANSeR1
    # baby_ids = [45, 95, 101, 113, 145, 185, 197, 234, 133, 170, 175, 201, 251, 252,
    #         4, 5, 19, 35, 40, 47, 50, 59, 62, 75, 77, 79, 82, 86, 87,
    #         88, 93, 94, 96, 98, 107, 108, 112, 119,120, 126, 127, 128, 132, 
    #         134, 135, 141, 147, 150, 151, 159, 160, 162, 168, 174, 178, 186, 
    #         188, 199, 204, 205, 213, 217, 227, 231, 232, 238, 241, 244, 245, 
    #         250, 256, 257, 264, 265, 63]
    # hours = '37_47'

    # Test
    # baby_ids = [5, 35]
    # hours = '13_23'

    # ANSeR_num = 2
    ANSeR_num = 1
    # dataset = 'Delphi'
    dataset = 'ANSeR'

    home_dir = '/mnt/files'
    output_dir = f'./log/'
    out_data_dir = './RPeaks/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(out_data_dir, exist_ok=True)
    
    # for ANSeR1&2 dataset
    channel_fn = f"{home_dir}/Datasets/One_Hour-Epochs_Pro_Data/RPeaks/ANSeR{ANSeR_num}/Channel selected ANSeR{ANSeR_num}.csv"
    # file name for saving the extracted rr intervals
    outfile = f'{out_data_dir}/rr_intervals_ANSeR{ANSeR_num}.mat'
    # Invalid epoch files from ANSeR dataset
    invalid_epoch_file = ['ID224_ANSeR1_24_HR', 'ID235_ANSeR1_12_HR', 'ID348_ANSeR1_6_HR', 
                        'ID493_ANSeR1_36_HR','ID35_ANSeR2_24_HR', 'ID40_ANSeR2_24_HR', 
                        'ID141_ANSeR2_24_HR', 'ID151_ANSeR2_48_HR', 'ID160_ANSeR2_24_HR', 
                        'ID201_ANSeR2_36_HR', 'ID101_ANSeR2_12_HR', ]
    

    # set up logger
    logger = logging.getLogger('rr_extraction')
    logger.setLevel(logging.DEBUG)
    log_file_name = f'rr_extraction_stand_enhanced_PTm.log'
    # log_file_name = f'rr_extraction_epoch_no_label.log'

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(f'{output_dir}/{log_file_name}', mode='w')
    fileHandler.setLevel(logging.DEBUG)

    # formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    consoleHandler.setFormatter(formatter)              
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    # 1). Get the selected channel name dataframe
    channame_df = pd.read_csv(channel_fn)

    # 2). Extract RR intervals and plot figures
    logger.info(f'\033[34;1mStart to extract RR intervals with standard version of Pan-Tompkin...\033[0m')
    rr_struct_stand, _ = rr_interval_collection_baby_level(ANSeR_num, baby_ids, hours, channame_df, invalid_epoch_file, version='standard')
    logger.info(f'\033[34;1mStart to extract RR intervals with enhanced version of Pan-Tompkin...\033[0m')
    rr_struct_enhanced, _ = rr_interval_collection_baby_level(ANSeR_num, baby_ids, hours, channame_df, invalid_epoch_file, version='enhanced')
    plot_stand_enhanced_rr(rr_struct_stand, rr_struct_enhanced, f'{output_dir}/RR figures/ANSeR{ANSeR_num}/')


    # 3). Save the extracted rr as .mat file
    rr_peaks_ANSeR_dict = {'rr_intervals': rr_struct_stand}
    outfile_stand = f'{out_data_dir}/rr_intervals_ANSeR{ANSeR_num}_stand.mat'
    scio.savemat(outfile_stand, rr_peaks_ANSeR_dict)
    logger.info(f'\033[34;1mstandard version RR intervals of ANSeR{ANSeR_num} saved!\033[0m')

    rr_peaks_ANSeR_dict = {'rr_intervals': rr_struct_enhanced}
    outfile_enhanced = f'{out_data_dir}/rr_intervals_ANSeR{ANSeR_num}_enhanced.mat'
    scio.savemat(outfile_enhanced, rr_peaks_ANSeR_dict)
    logger.info(f'\033[34;1menhanced version RR intervals of ANSeR{ANSeR_num} saved!\033[0m')




