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
import os
from scipy.io.wavfile import read as wavread

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
    # if annotation.size > annotation.size:
    #     estimateInHz = estimation[np.arange(0, annotation.size)]
    # elif estimation.size > annotation.size:
    #     annotationInHz = annotation[np.arange(0, estimation.size)]

    diffInCent = 100 * (convert_freq2midi(estimation) - convert_freq2midi(annotation))

    errCentRms = np.sqrt(np.mean(diffInCent ** 2))

    pfn = eval_voiced_fn(estimation,annotation)

    pfp = eval_voiced_fp(estimation,annotation)

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
# def make_sin(fs, freq, seconds):
#     time_in_sec = np.arange(fs*seconds)/fs
#     radians = time_in_sec * freq * 2 * np.pi
#     sin = np.sin(radians)
#     return sin


def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = wavread(cAudioFilePath)

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

    return (samplerate, audio)



def convert_freq2midi(fInHz, fA4InHz=440):
    def convert_freq2midi_scalar(f, fA4InHz):

        if f <= 0:
            return 0
        else:
            return (69 + 12 * np.log2(f / fA4InHz))
        fInHz = np.asarray(fInHz)
        if fInHz.ndim == 0:
            return convert_freq2midi_scalar(fInHz, fA4InHz)

    fInHz = np.asarray(fInHz)
    if fInHz.ndim == 0:
        return convert_freq2midi_scalar(fInHz, fA4InHz)

    midi = np.zeros(fInHz.shape)
    for k, f in enumerate(fInHz):
        midi[k] = convert_freq2midi_scalar(f, fA4InHz)

    return (midi)


def eval_pitchtrack(estimateInHz, groundtruthInHz):
    if np.abs(groundtruthInHz).sum() <= 0:
        return 0

    # truncate longer vector
    if groundtruthInHz.size > estimateInHz.size:
        estimateInHz = estimateInHz[np.arange(0, groundtruthInHz.size)]
    elif estimateInHz.size > groundtruthInHz.size:
        groundtruthInHz = groundtruthInHz[np.arange(0, estimateInHz.size)]

    diffInCent = 100 * (convert_freq2midi(estimateInHz) - convert_freq2midi(groundtruthInHz))

    rms = np.sqrt(np.mean(diffInCent[groundtruthInHz != 0] ** 2))
    return (rms)


def run_evaluation1(complete_path_to_data_folder):


    # init
    rmsAvg_1 = 0
    rmsAvg_2 = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0] + '.f0.Corrected.txt')
        else:
            continue

        # extract pitch
        [f0, t] = track_pitch_fftmax(afAudioData, 1024, 512, fs)
        [f0_2, t] = track_pitch_hps(afAudioData, 1024, 512, fs)
        # compute rms and accumulate

        rmsAvg_1 += eval_pitchtrack_v2(f0, refdata[:, 2])
        rmsAvg_2 += eval_pitchtrack_v2(f0_2, refdata[:, 2])
        rmsAvg = np.concatenate([rmsAvg_1, rmsAvg_2])



    if iNumOfFiles == 0:
        return -1

    return rmsAvg / iNumOfFiles


# Question E5
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):
    if method == 'acf':
        f0, timeInSec = track_pitch_acf(x,blockSize, hopSize, fs)
    elif method == 'hps':
        f0, timeInSec = track_pitch_hps(x,blockSize, hopSize, fs)
    elif method == 'max':
        f0, timeInSec = track_pitch_fftmax(x,blockSize, hopSize, fs)

    rmsDb = extract_rms(x)
    mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, mask)

    return f0Adj, timeInSec


# Question E6

# track acf function from assign 1
def comp_acf(inputVector, bIsNormalized=True):
    if bIsNormalized:
        norm = np.dot(inputVector, inputVector)
    else:
        norm = 1

    afCorr = np.correlate(inputVector, inputVector, "full") / norm
    afCorr = afCorr[np.arange(inputVector.size - 1, afCorr.size)]

    return (afCorr)


def get_f0_from_acf(r, fs):
    eta_min = 1
    afDeltaCorr = np.diff(r)
    eta_tmp = np.argmax(afDeltaCorr > 0)
    eta_min = np.max([eta_min, eta_tmp])

    f = np.argmax(r[np.arange(eta_min + 1, r.size)])
    f = fs / (f + eta_min + 1)

    return (f)


def track_pitch_acf(x, blockSize, hopSize, fs):
    # get blocks
    [xb, t] = block_audio(x, blockSize, hopSize, fs)

    # init result
    f0 = np.zeros(xb.shape[0]
                  )
    # compute acf
    for n in range(0, xb.shape[0]):
        r = comp_acf(xb[n, :])
        f0[n] = get_f0_from_acf(r, fs)

    return (f0, t)




def run_evaluation2(complete_path_to_data_folder):

    # init
    rmsAvg_1 = 0
    rmsAvg_2 = 0
    rmsAvg_3 = 0
    rmsAvg_4 = 0
    rmsAvg_5 = 0
    rmsAvg_6 = 0
    iNumOfFiles = 0

    # for loop over files
    for file in os.listdir(complete_path_to_data_folder):
        if file.endswith(".wav"):
            iNumOfFiles += 1
            # read audio
            [fs, afAudioData] = ToolReadAudio(complete_path_to_data_folder + file)

            # read ground truth (assume the file is there!)
            refdata = np.loadtxt(complete_path_to_data_folder + os.path.splitext(file)[0] + '.f0.Corrected.txt')
        else:
            continue

        # extract pitch
        [f0_1, t] = track_pitch(afAudioData, 1024, 512, fs, 'acf', -40)
        [f0_2, t] = track_pitch(afAudioData, 1024, 512, fs, 'hps', -40)
        [f0_3, t] = track_pitch(afAudioData, 1024, 512, fs, 'max', -40)
        [f0_4, t] = track_pitch(afAudioData, 1024, 512, fs, 'acf', -20)
        [f0_5, t] = track_pitch(afAudioData, 1024, 512, fs, 'hps', -20)
        [f0_6, t] = track_pitch(afAudioData, 1024, 512, fs, 'max', -20)
        # compute rms and accumulate
        rmsAvg_1 += eval_pitchtrack_v2(f0_1, refdata[:, 2])
        rmsAvg_2 += eval_pitchtrack_v2(f0_2, refdata[:, 2])
        rmsAvg_3 += eval_pitchtrack_v2(f0_3, refdata[:, 2])
        rmsAvg_4 += eval_pitchtrack_v2(f0_4, refdata[:, 2])
        rmsAvg_5 += eval_pitchtrack_v2(f0_5, refdata[:, 2])
        rmsAvg_6 += eval_pitchtrack_v2(f0_6, refdata[:, 2])

        rmsAvg = np.concatenate([rmsAvg_1, rmsAvg_2, rmsAvg_3, rmsAvg_4, rmsAvg_5, rmsAvg_6])

    if iNumOfFiles == 0:
        return -1

    return rmsAvg / iNumOfFiles


def errtest(f0,timeInSec):
    err_nonzero = []
    err = np.zeros(len(f0))
    for i in range(len(f0)):
        if 0 <= timeInSec[i] < 1:
            err[i] = abs(f0[i] - 441)
        elif timeInSec[i] >= 1:
            err[i] = abs(f0[i] - 882)
        np.append(err[i], err_nonzero)
    return err


def executeassign3():

    # Question E.1
    #
    # fs = 44100
    # sin441 = make_sin(fs, 441, 1)
    # sin882 = make_sin(fs, 882, 1)
    blockSize = 1024
    hopSize = 512
    #
    # sin = np.concatenate([sin441, sin882])
    # #xb, t = block_audio(sin, blockSize, hopSize, fs)
    # E1. generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec),

    fs = 44100
    timeA = np.linspace(start=0, stop=1, num=fs, endpoint=False)
    timeB = np.linspace(start=1, stop=2, num=fs, endpoint=False)

    # generate test signals at 441 Hz from 0 to 1 sec and 882 Hz from 1 to 2 sec
    testsignalA = np.sin(2 * np.pi * 441 * timeA)
    testsignalB = np.sin(2 * np.pi * 882 * timeB)

    # append arrays to create a 2 sec test signal
    sin = np.append(testsignalA, testsignalB)

    f0_fftmax, t_fftmax = track_pitch_fftmax(sin, blockSize, hopSize, fs)
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(f0_fftmax, t_fftmax)
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')

    # error441 = np.abs(f0[:len(f0)//2]-441*np.ones(len(f0//2)))
    # error882 = np.abs(f0[len(f0)//2:]-882*np.ones(len(f0//2)))
    # error = np.concatenate([error441, error882])




    error1=errtest(f0_fftmax,t_fftmax)

    plt.subplot(1, 2, 2)
    plt.plot(error1, t_fftmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')

    f0_hps, t_hps = track_pitch_hps(sin, blockSize, hopSize, fs)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.plot(f0_hps, t_hps)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')

    error2 = errtest(f0_hps, t_hps)
    plt.subplot(1, 2, 2)
    plt.plot(error2, t_hps)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')




    # Question E.2

    blockSize_2 = 2048
    hopSize_2 = 512
    f0_2, timeInSec_2 = track_pitch_fftmax(sin, blockSize_2, hopSize_2, fs)
    plt.subplot(1, 2, 1)
    plt.plot(f0_2, timeInSec_2)
    plt.xlabel('Time (s)')
    plt.ylabel('F0_2 (Hz)')

    # error441_2 = np.abs(f0[:len(f0_2) // 2] - 441 * np.ones(len(f0_2 // 2)))
    # error882_2 = np.abs(f0[len(f0_2) // 2:] - 882 * np.ones(len(f0_2 // 2)))
    # error_2 = np.concatenate([error441_2, error882_2])
    error3 = errtest(f0_2, timeInSec_2)
    plt.subplot(1, 2, 2)
    plt.plot(error3, timeInSec_2)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (Hz)')

    # Question E3,4,5,6
    #insert file path
    run_evaluation1('C:/Users/Laney/Documents/6201_comp_mus_analysis/Assignments/Assignment 3/data/trainData/')
    run_evaluation2('C:/Users/Laney/Documents/6201_comp_mus_analysis/Assignments/Assignment 3/data/trainData/')
    return ()


if __name__ == "__main__":
    executeassign3()


