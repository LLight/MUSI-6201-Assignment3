##############################
#program name: ass3solution.py
#started 10/6/19
#MUSI 6201 Assignment 3: fundamental frequency detection/pitch tracking
#Implement pitch trackers based on maximum spectral peak, harmonic product spectrum
##############################

import numpy as np
import math
import matplotlib.pyplot as plt

#Block audio function
def  block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])

        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return (xb,t)

######################################################
#A. Maximum spectral peak based pitch tracker

#A1. compute_spectrogram computes the magnitude spectrum for each block of audio in xb
# inputs: xb=block, fs=sample rate
# outputs: X = magnitude spectrogram (dimensions blockSize/2+1 X numBlocks), fInHz = central frequency of each bin (dim blockSize/2+1,)
def compute_spectrogram(xb, fs):

    return (X, fInHz)

#A2. track_pitch_fftmax estimates the fundamental frequency f0 of the audio signal
# based on a block-wise maximum spectral peak finding approach
def track_pitch_fftmax(x, blockSize, hopSize, fs):

    return (f0,timeInSec)

########################################################
#B HPS (Harmonic Product Spectrum) based pitch tracker

#B1. get_f0_from_Hps computes the block-wise fundamental frequency and the sampling rate based on a HPS approach of specified order
# inputs: X=magnitude spectrogram, fs=sample rate, order
# output: f0=fundamental frequency for the block

def get_f0_from_Hps(X, fs, order):

    return f0

#B2. track_pitch_hps calls compute_spectrogram with order 4 to estimate the fundamental frequency f0 of the audio signal
# Inputs: x=block, blockSize = 1024, fs
# Outputs: f0=fundamental frequency of signal, timeInSec

def track_pitch_hps(x, blockSize, hopSize, fs):

    return (f0, timeInSec)

######################################################
#C. Voicing Detection

#C1. extract_rms calculates the RMS
# input: xb=block
# output: rmsDb = vector of RMS values in decibels
def extract_rms(xb):

    return rmsDb

#C2.create_voicing_mask
# inputs: rmsDb = vector of decibel values for the different blocks of audio, thresholdDb
# outputs: mask = binary mask (column vector of the same size as 'rmsDb' containing 0's and 1's).
#  mask= 0 if rmsDb < thresholdDb, mask = 1 if rmsDb >=threshold
def create_voicing_mask(rmsDb, thresholdDb):

    return mask

#C3. apply_voicing_mask applies the voicing mask, setting f0 of blocks with low energy to 0
# inputs: f0= previously computed fundamental frequency vector, mask = binary column vector
# outputs f0Adj, a vector of same dimensions as f0
def apply_voicing_mask(f0, mask):

    return f0Adj

######################################################
#D. Evaluation Metrics

#D1. eval_voiced_fp computes the percentage of false positives for the fundamental frequency estimation

def eval_voiced_fp(estimation, annotation):
#denominator = num blocks with annotation = 0
#numerator = num blocks in the denominator with fundamental freq not equal to 0
    return pfp

#D2. eval_voiced_fn computes the percentage of false negatives
def eval_voiced_fn(estimation, annotation):
#denominator = num blocks with non-zero fundamental frequency in the annotation.
#numerator = num blocks in denominator that were detected as zero
    return pfn

#D3. Modified version of eval_pitchtrack from Assignment 1
#input: estimation, annotation
#output: errCentRms, pfp, pfn
def eval_pitchtrack_v2(estimation, annotation):
######## Need to modify the eval_pitchtrack function from assignment 1 (copied below).
######## Update errCentRMS to take into account zeros in estimation, change variable names, incorporate pfp and pfn
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0

    # truncate longer vector
    if groundtruthInHz.size > estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0, groundtruthInHz.size)]
    elif estimateInHz.size > groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0, estimateInHz.size)]

    diffInCent = 100 * (convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))

    rms = np.sqrt(np.mean(diffInCent[groundtruthInHz != 0] ** 2))

    return (errCentRms, pfp, pfn)

