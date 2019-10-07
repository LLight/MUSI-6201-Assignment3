##############################
#program name: assign3.py
#started 10/6/19
#MUSI 6201 Assignment 3: fundamental frequency detection/pitch tracking
#Evaluation of functions in ass3solution.py
##############################

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