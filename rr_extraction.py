import os
import math
import copy
import random
from datetime import date
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # for progress bar
import scipy.io as scio    # read .mat file
import logging
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
import matplotlib_inline
# mpl.use('TkAgg')
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
plt.style.use(['science', 'no-latex'])

from Pan_Tomkin_enhanced import Pan_Tompkin_preprocessing as Pan_Tompkin_preprocessing_enhanced
from Pan_Tomkin_enhanced import QRS_detect

from Pan_Tompkin_stand import Pan_Tompkin_preprocessing as Pan_Tompkin_preprocessing_standard
from Pan_Tompkin_stand import detect_peaks


logger = logging.getLogger('rr_extraction')


def main(edf_fn, select_chann_name):
    """
    Main function to extract RR intervals from ECG data
    with standard and enhanced version of Pan-Tompkin and plot the 
    comparison of extracted RR.
    
    Parameters:
    - edf_fn: The edf file path and name.
    - chann_name: The name of the channel to process.
    - version: The version of the Pan-Tompkins algorithm to use ('enhanced' or 'standard').
    
    """

    # enhanced Pan-Tompkins
    logger.info("Starting enhanced Pan-Tompkins preprocessing...")
    sfreq, ecg_arr_enhanced, time_arr_enhanced, b_pass_enhanced, _, _, m_win_enhanced, chann_name, reverse_flag = Pan_Tompkin_preprocessing_enhanced(edf_fn, select_chann_name)
    hr = QRS_detect(ecg_arr_enhanced, time_arr_enhanced, sfreq, m_win_enhanced, b_pass_enhanced)
    
    r_peaks_enhanced, rr_interval_corrected, time_stamp_corrected = hr.detect_r_peaks()
    rr_time_stamp_enhanced = time_arr_enhanced[r_peaks_enhanced]
    rr_intervals_enhanced = np.diff(rr_time_stamp_enhanced)

    # standard Pan-Tompkins
    logger.info("Starting standard Pan-Tompkins preprocessing...")
    sfreq, ecg_arr_stand, time_arr_stand, b_pass_stand, _, _, ma_win_stand, chann_name, reverse_flag = Pan_Tompkin_preprocessing_standard(edf_fn, select_chann_name)
    r_peaks_stand = detect_peaks(ecg_arr_stand, b_pass_stand, ma_win_stand, sfreq)

    rr_time_stamp_stand = time_arr_stand[r_peaks_stand]
    rr_intervals_stand = np.diff(rr_time_stamp_stand)

    # Plotting rr intervals
    logger.info("Plotting RR intervals...")
    plot_rr(rr_time_stamp_stand, rr_intervals_stand, rr_time_stamp_enhanced, rr_intervals_enhanced, time_stamp_corrected, rr_interval_corrected)




def plot_rr(rr_time_stamp_stand, rr_interval_stand, rr_time_stamp_enhanced, rr_interval_enhanced, rr_time_stamp_corrected, rr_interval_corrected):
    """
    Plot RR intervals for standard and enhanced Pan-Tompkins methods.
    """
    mean_stand = np.mean(rr_interval_stand)
    std_stand = np.std(rr_interval_stand)
    mean_enhanced = np.mean(rr_interval_enhanced)
    std_enhanced = np.std(rr_interval_enhanced)
    std_corrected = np.std(rr_interval_corrected)

    # Plotting
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10,9), constrained_layout=True)

    text0 = f"mean: {mean_stand:.4f}\nstd: {std_stand:.4f}"
    axs[0].plot(rr_time_stamp_stand[1:], rr_interval_stand)
    if min(rr_interval_stand)<0.3:
        axs[0].axhline(0.2,color='r')
    axs[0].set_title(f'rr intervals from standard Pan-Tompkin')
    axs[0].text(0.9, 0.83, text0, transform=axs[0].transAxes)
    axs[0].set_xlabel('time(s)')
    axs[0].grid()

    text1 = f"mean: {mean_enhanced:.4f}\nstd: {std_corrected:.4f}"
    axs[1].plot(rr_time_stamp_corrected[1:], rr_interval_corrected)
    if min(rr_interval_corrected) < 0.3:
        axs[1].axhline(0.2,color='r')
    axs[1].set_title(f'corrected rr intervals from enhanced Pan-Tompkin')
    axs[1].text(0.9, 0.83, text1, transform=axs[1].transAxes)
    axs[1].set_xlabel('time(s)')
    axs[1].grid()

    text2 = f"mean: {mean_enhanced:.4f}\nstd: {std_enhanced:.4f}"
    axs[2].plot(rr_time_stamp_enhanced[1:], rr_interval_enhanced)
    if min(rr_interval_enhanced) < 0.3:
        axs[2].axhline(0.2,color='r')
    axs[2].set_title(f'rr intervals from enhanced Pan-Tompkin')
    axs[2].text(0.9, 0.83, text2, transform=axs[2].transAxes)
    axs[2].set_xlabel('time(s)')
    axs[2].grid()

    # Save the figure
    save_dir = './output'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/rr interval comparison.svg")
    plt.close()

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

if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        )
    logger.info("Starting RR extraction...")

    parser = argparse.ArgumentParser(description='RR extraction from ECG data')
    parser.add_argument('--edf_fn', type=str, help='Path to the EDF file')
    parser.add_argument('--chann_name', type=str, help='Name of the ECG channel to process')
    args = parser.parse_args()
    

    if args.edf_fn and args.chann_name:
        main(args.edf_fn, args.chann_name)
    else:
        # Input file id
        ANSeR_num = 1
        baby_id = 24
        hour_id = 12
        file_id = f'ID{baby_id}_ANSeR{ANSeR_num}_{hour_id}_HR'
        home_dir = '/mnt/files'
        edf_fn = f'%s/Datasets/One_Hour-Epochs_Raw_Data/ANSeR%s/%s.edf' %(home_dir, ANSeR_num, file_id)

        # Select channel name
        channel_fn = f"{home_dir}/Datasets/One_Hour-Epochs_Pro_Data/RPeaks/ANSeR{ANSeR_num}/Channel selected ANSeR{ANSeR_num}.csv"
        channel_df = pd.read_csv(channel_fn)
        chann_name = channel_selection(file_id, channel_df)
        # chann_name = 'ECG'  # Replace with your channel name
        logger.info(f"selected channel name: {chann_name}")

        main(edf_fn, chann_name)
