'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

from __future__ import division
import numpy as np
from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from scipy.signal import blackmanharris
from scipy.fftpack import fft



from scipy.signal import butter, lfilter, filtfilt

Nan = float("nan") # Not-a-number capitalized like None, True, False
Inf = float("inf") # infinite value capitalized ...

eps = np.finfo("float32").eps


def mad(a, normalize=True, axis=0):
    
    from scipy.stats import norm
    c = norm.ppf(3/4.) if normalize else 1
    return np.median(np.abs(a - np.median(a)) / c, axis=axis)


def rssq(x):
    return np.sqrt(np.sum(np.abs(x)**2))


def peak2rms(x):
    num = max(abs(x))
    den = rms (x)
    return num/den

def rms(x):
    return np.sqrt(np.mean(x**2))

def range_bytes (win): 
    return range(win) 

def energy(x):
    energy = np.sum(x**2) / len(x) # axis = 1 is column sum
    return energy

def zcr_2(frame):
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(countZ) / np.float64(count-1.0))

def zcr(x):
    count = (np.diff(np.sign(x)) != 0).sum()
    rate = count/len(x)
    return rate

""" Frequency-domain features """

def peakfreq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) - 1 # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def parabolic(f, x):
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def spectralCentroidAndSpread(x, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    
    X = abs(fft(x)) # get fft magnitude
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)





def spectralRollOff(x, c, fs):
    """Computes spectral roll-off"""
    X = abs(fft(x)) # get fft magnitude
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position 
    # where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def chromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = np.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])    
    Cp = 27.50    
    nChroma = np.round(12.0 * np.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = np.zeros((nChroma.shape[0], ))

    uChroma = np.unique(nChroma)
    for u in uChroma:
        idx = np.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    
    return nChroma, nFreqsPerChroma


def chromaFeatures(x, fs, nChroma, nFreqsPerChroma):
    

    X = abs(fft(x)) # get fft magnitude
    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 
                   'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2    
    if nChroma.max()<nChroma.shape[0]:        
        C = np.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:        
        I = np.nonzero(nChroma>nChroma.shape[0])[0][0]        
        C = np.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec            
        C /= nFreqsPerChroma
    finalC = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12)
    #for i in range(12):
    #    finalC[i] = np.sum(C[i:C.shape[0]:12])
    finalC = np.matrix(np.sum(C2, axis=0)).T
    finalC /= spec.sum()


    return chromaNames, finalC

def recursive_sta_lta(a, nsta, nlta):

    """

    Recursive STA/LTA written in Python.

    .. note::

        There exists a faster version of this trigger wrapped in C

        called :func:`~obspy.signal.trigger.recursive_sta_lta` in this module!

    :type a: NumPy :class:`~numpy.ndarray`

    :param a: Seismic Trace

    :type nsta: int

    :param nsta: Length of short time average window in samples

    :type nlta: int

    :param nlta: Length of long time average window in samples

    :rtype: NumPy :class:`~numpy.ndarray`

    :return: Characteristic function of recursive STA/LTA

    .. seealso:: [Withers1998]_ (p. 98) and [Trnkoczy2012]_

    """

    try:

        a = a.tolist()

    except Exception:

        pass

    ndat = len(a)

    # compute the short time average (STA) and long time average (LTA)

    # given by Evans and Allen

    csta = 1. / nsta

    clta = 1. / nlta

    sta = 0.

    lta = 1e-99  # avoid zero division

    charfct = [0.0] * len(a)

    icsta = 1 - csta

    iclta = 1 - clta

    for i in range(1, ndat):

        sq = a[i] ** 2

        sta = csta * sq + icsta * sta

        lta = clta * sq + iclta * lta

        charfct[i] = sta / lta

        if i < nlta:

            charfct[i] = 0.

    return np.array(charfct)