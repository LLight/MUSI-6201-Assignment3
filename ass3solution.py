##############################
#program name: ass3solution.py
#started 10/6/19
#MUSI 6201 Assignment 3: fundamental frequency detection/pitch tracking
#Implement pitch trackers based on maximum spectral peak, harmonic product spectrum
##############################

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram, medfilt

#Block audio function
def block_audio(x,blockSize,hopSize,fs):
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

#Hann window function
def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

######################################################
#A. Maximum spectral peak based pitch tracker

#A1. compute_spectrogram computes the magnitude spectrum for each block of audio in xb
# inputs: xb=block, fs=sample rate
# outputs: X = magnitude spectrogram (dimensions blockSize/2+1 X numBlocks), fInHz = central frequency of each bin (dim blockSize/2+1,)
def compute_spectrogram(xb, fs):
    (NumOfBlocks, blockSize) = xb.shape
    hann=compute_hann(iWindowLength=blockSize)
    fInHz, t, X = spectrogram(xb, fs, window=hann, nfft=blockSize)
##need to move the plt and print statements to main section at the end
    #plt.pcolormesh(t, fInHz, X)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    X = np.array(X)

    X = np.squeeze(X, axis=2)
    X = np.transpose(X)

    fInHz = np.array(fInHz)
    print(X.shape)
    print(fInHz.shape)
    return (X, fInHz)

# #A2. track_pitch_fftmax estimates the fundamental frequency f0 of the audio signal
# # based on a block-wise maximum spectral peak finding approach
# def track_pitch_fftmax(x, blockSize, hopSize, fs):

#     return (f0,timeInSec)

########################################################
#B HPS (Harmonic Product Spectrum) based pitch tracker

#B1. get_f0_from_Hps computes the block-wise fundamental frequency and the sampling rate based on a HPS approach of specified order
# inputs: X=magnitude spectrogram, fs=sample rate, order
# output: f0=fundamental frequency for the block

def get_f0_from_Hps(X, fs, order):

    # FFT point is the same as blockSize
    FFT_point = (X.shape[0] - 1) * 2
    f0 = []
    # block wise HPS
    for i in range(0,X.shape[1]):
        X1 = X[:,i]
        # print(X1.shape)
        X_hps = np.ones(X1.shape[0])
        for k in range(0,X_hps.shape[0]):
            for j in range(1,order+1):
                if j*k < X_hps.shape[0]:
                    X_hps[k] = X_hps[k] * math.pow(X1[j*k],2)
        # f0 at the peak of HPS
        freq_bin = np.argmax(X_hps)
        freq = freq_bin * fs / FFT_point
        f0.append(freq)
    f0 = np.array(f0)
    print(f0.shape)
    
    return f0

#B2. track_pitch_hps calls compute_spectrogram with order 4 to estimate the fundamental frequency f0 of the audio signal
# Inputs: x=block, blockSize = 1024, fs
# Outputs: f0=fundamental frequency of signal, timeInSec

def track_pitch_hps(x, blockSize, hopSize, fs):

    xb,t = block_audio(x,1024,hopSize,fs)
    X, fInHz = compute_spectrogram(xb, fs)
    f0 = get_f0_from_Hps(X, fs, 4)
    timeInSec = t
    return (f0, timeInSec)

######################################################
#C. Voicing Detection

#C1. extract_rms calculates the RMS
# input: xb=block
# output: rmsDb = vector of RMS values in decibels
def extract_rms(xb):
    rms_matrix = []
    for b in xb:
        rms = np.sqrt(np.mean(np.square(b)))
        if rms < 1e-5:
            rms= 1e-5
        rms = 20*np.log10(rms)
        rms_matrix.append(rms)
    # print(rms_matrix)
    rmsDb = np.array(rms_matrix) 
    # print(rmsDb.shape)

    return rmsDb

#C2.create_voicing_mask
# inputs: rmsDb = vector of decibel values for the different blocks of audio, thresholdDb
# outputs: mask = binary mask (column vector of the same size as 'rmsDb' containing 0's and 1's).
#  mask= 0 if rmsDb < thresholdDb, mask = 1 if rmsDb >=threshold
def create_voicing_mask(rmsDb, thresholdDb):
    mask = np.zeros(len(rmsDb))
    for i in range(0, len(rmsDb)):
        if rmsDb[i] < thresholdDb:
            mask[i] = 0
        else:
            mask[i] = 1
    # print(mask.shape)

    return mask

#C3. apply_voicing_mask applies the voicing mask, setting f0 of blocks with low energy to 0
# inputs: f0= previously computed fundamental frequency vector, mask = binary column vector
# outputs f0Adj, a vector of same dimensions as f0
def apply_voicing_mask(f0, mask):
    f0Adj = np.multiply(f0, mask)
    print(f0Adj.shape)

    return f0Adj

if __name__ == "__main__":
    fs, audio = read('C:/Users/bhxxl/OneDrive/GT/Computational Music Analysis/HW1/developmentSet/trainData/01-D_AMairena.wav')
    txt_file = 'C:/Users/bhxxl/OneDrive/GT/Computational Music Analysis/HW1/developmentSet/trainData/01-D_AMairena.f0.Corrected.txt'
    with open(txt_file) as f:
        annotations = f.readlines()
    for i in range(len(annotations)):
        annotations[i] = list(map(float, annotations[i][:-2].split('     ')))
    annotations = np.array(annotations)

    blockSize = 2048
    hopSize = 512
    xb, t = block_audio(audio,blockSize,hopSize,fs)
    X, fInHz = compute_spectrogram(xb, fs)
    order = 4
    f0 = get_f0_from_Hps(X, fs, order)

    trimmed_f0 = np.ones(f0.shape)
    trimmed_annotations = np.ones(f0.shape)
    print(f0.shape)
    for i in range(len(f0)):
        if annotations[i, 2] > 0:
            trimmed_f0[i] = f0[i]
            trimmed_annotations[i] = annotations[i, 2]
    plt.plot(trimmed_f0)
    plt.plot(trimmed_annotations)
    plt.legend(['f0','annotation'])
    plt.show()

    rmsDb = extract_rms(xb)
    thresholdDb = -20
    mask = create_voicing_mask(rmsDb, thresholdDb)
    f0Adj = apply_voicing_mask(f0, mask)