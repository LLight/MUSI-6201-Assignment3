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
# outputs: X = magnitude spectrogram (dimensions blockSize/2+1 X numBlocks), fInHz = central frequency of each bin (dim blockSize/2+1)
def compute_spectrogram(xb, fs):
    (NumOfBlocks, blockSize) = xb.shape
    FFT_point = blockSize
    hann = compute_hann(blockSize)

    fInHz, t, X = spectrogram(xb, fs, window=hann, nfft=FFT_point)

    X = np.array(X)
    X = np.squeeze(X, axis=2)
    X = np.transpose(X)

    fInHz = np.array(fInHz)

    return (X, fInHz)


#A2. track_pitch_fftmax estimates the fundamental frequency f0 of the audio signal
# based on a block-wise maximum spectral peak finding approach
def track_pitch_fftmax(x, blockSize, hopSize, fs):
    (xb, timeInSec) = block_audio(x, blockSize, hopSize, fs)
    (X, fInHz) = compute_spectrogram(xb, fs)

    # get index of max magnitude within each block and the corresponding frequency
    nBlocks = len(timeInSec)
    f0 = np.zeros(nBlocks)
    for block in range(0, nBlocks - 1):
        i = np.argmax(X[:, block])
        f0[block] = fInHz[i]
    return (f0, timeInSec)

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


######################################################
#D. Evaluation Metrics

#D1. eval_voiced_fp computes the percentage of false positives for the fundamental frequency estimation

def eval_voiced_fp(estimation, annotation):
    # denominator = num blocks with annotation = 0
    denom = (annotation == 0).sum()
    n = len(annotation)
    # create a vector to define which blocks in the denominator are also in the numerator (value=1 if it is in the numerator)
    numvector = np.zeros(n)
    for i in range(0, n - 1):
        if (annotation[i] == 0) & (estimation[i] != 0):
            numvector[i] = 1
    # numerator = num blocks in the denominator with fundamental freq not equal to 0
    num = (numvector == 1).sum()
    pfp = num / denom
    return pfp


#D2. eval_voiced_fn computes the percentage of false negatives
def eval_voiced_fn(estimation, annotation):
    # denominator = num blocks with non-zero fundamental frequency in the annotation.
    denom = (annotation != 0).sum()
    n = len(annotation)
    # create a vector to define which blocks in the denominator are also in the numerator (value=1 if in the numerator)
    numvector = np.zeros(n)
    for i in range(0, n - 1):
        if (annotation[i] != 0) & (estimation[i] == 0):
            numvector[i] = 1
    # numerator = num blocks in denominator that were detected as zero
    num = (numvector == 1).sum()
    pfn = num / denom
    return pfn

#D3. Modified version of eval_pitchtrack from Assignment 1
#input: estimation, annotation
#output: errCentRms, pfp, pfn
def eval_pitchtrack_v2(estimation, annotation):
    # truncate longer vector
    if annotation.size > annotation.size:
        estimateInHz = estimation[np.arange(0, annotation.size)]
    elif estimation.size > annotation.size:
        annotationInHz = annotation[np.arange(0, estimation.size)]

    diffInCent = 100 * (convert_freq2midi(estimateInHz) - convert_freq2midi(annotationInHz))

    errCentRms = np.sqrt(np.mean(diffInCent ** 2))

    eval_voiced_fn(estimateInHz,annotationInHz)

    eval_voiced_fp(estimateInHz,annotationInHz)

    return (errCentRms, pfp, pfn)

#read audio files
def audio_read(path):
    samplerate, x = read(path)
    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    # special case of unsigned format
    if x.dtype == 'uint8':
        audio = audio - 1.

    if audio.ndim > 1:
        audio = audio[:, 0]

    return samplerate, audio

#read labels from annotation files
def read_label(path, estimateTime):

        es_idx = 0
        pre = -1

        oup = []
        time = []
        f = open(path, "r")
        for x in f:
            time = float(x.split('     ')[0])
            if es_idx < len(estimateTime):
                while es_idx < len(estimateTime) and estimateTime[es_idx] < time and estimateTime[es_idx] > pre:
                    oup.append(x.split('     ')[2])
                    pre = estimateTime[es_idx]
                    es_idx+=1
        return oup

#E. Evaluation

#E1. generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec),

fs = 44100
timeA = np.linspace(start=0, stop=1, num=fs, endpoint=False)
timeB = np.linspace(start=1, stop=2, num=fs, endpoint=False)

# generate test signals at 441 Hz from 0 to 1 sec and 882 Hz from 1 to 2 sec
testsignalA = np.sin(2 * np.pi * 441 * timeA)
testsignalB = np.sin(2 * np.pi * 882 * timeB)

# append arrays to create a 2 sec test signal
testSignal = np.append(testsignalA, testsignalB)

# apply track_pitch_fftmax() to test signal (blockSize = 1024, hopSize = 512)
(f0TestMaxPeak,timeTestMaxPeak)=track_pitch_fftmax(x=testSignal,blockSize=1024,hopSize=512,fs=44100)

# apply track_pitch_hps() to the test signal with the same signal and parameters.
(f0TestHps,timeTestHps)=track_pitch_hps(x=testSignal,blockSize=1024,hopSize=512,fs=44100)

#calculate absolute error per block for the test signals
def errtest(f0):
    err=np.zeros(len(f0))
    for i in range(len(f0)):
        if 0 <= timeInSec[i] < 1:
            err[i] = abs(f0[i] - 441)
        elif timeInSec[i] >= 1:
            err[i] = abs(f0[i] - 882)
        np.append(err[i],err_nonzero)
    return err

errTestMaxPeak=errtest(f0=f0TestMaxPeak)
errTestHps=errtest(f0=f0testHps)

#E2. Repeat E1 using blockSize = 2048, hopSize = 512, only for the max spectra method.

(f0TestMaxPeak2,timeTestMaxPeak2)=track_pitch_fftmax(x=testSignal,blockSize=2048,hopSize=512,fs=44100)

errTestMaxPeak2=errtest(f0=f0TestMaxPeak)

#E3 (TBD). Evaluate track_pitch_fftmax() using the training set and the eval_pitchtrack_v2() method, calculate avg metrics over training set
#E4 (TBD). Evaluate track_pitch_hps() using the training set and the eval_pitchtrack_v2() method, calculate avg metrics over training set

#E5. Compute the fundamental frequency and apply the voicing mask based on the threshold parameter.
# input: x=audio signal, blockSize, hopSize, fs=sample rate, method='acf','max', or 'hps', voicingThres=threshold parameter
# output: f0Adj = vector of fundamental frequencies, timeInSec=time vector
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    return (f0Adj, timeInSec)

#E6 (TBD). Evaluate track_pitch() using the training set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512)
# over all 3 pitch trackers (acf, max and hps) with two values of threshold (threshold = -40, -20)


if __name__ == "__main__":
    #E1-E2: Plot the f0 curve and and absolute error per block for the sine wave test signals for FFT max and HPS method

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.title("Estimated Fundamental Frequency for Sine Wave Test Signal")
    ax.plot(timeTestMaxPeak, f0TestMaxPeak, color='tab:blue', label='Max Peak (block size=1024)')
    ax.plot(timeTestHps, f0TestHps, color='tab:orange', label='HPS (block size=1024)')
    ax.plot(timeTestMaxPeak2, f0MaxPeak2, color='tab:purple', label='Max Peak (blockSize=2048)')
    ax.legend()
    plt.show()

    fig2 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.title("Absolute Error in Fundamental Frequency for Sine Wave Test Signal")
    ax.plot(timeTestMaxPeak, errTestMaxPeak, color='tab:blue', label='Max Peak (block size=1024)')
    ax.plot(timeTestHps, errTestHps, color='tab:orange', label='HPS (block size=1024)')
    ax.plot(timeTestMaxPeak2, errMaxPeak2, color='tab:purple', label='Max Peak (blockSize=2048)')
    ax.legend()
    plt.show()
    
    fs, audio = read('C:/Users/bhxxl/OneDrive/GT/Computational Music Analysis/HW1/developmentSet/trainData/01-D_AMairena.wav')
    blockSize = 1024
    hopSize = 256
    xb, t = block_audio(audio,blockSize,hopSize,fs)
    # X, fInHz = compute_spectrogram(audio, fs)
    # order = 4
    # f0 = get_f0_from_Hps(X, fs, order)
    rmsDb = extract_rms(xb)
    thresholdDb = -20
    mask = create_voicing_mask(rmsDb, thresholdDb)

