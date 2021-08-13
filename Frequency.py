
# coding: utf-8

# In[90]:


import numpy
import wave
import os
import matplotlib.pyplot as plt
import matplotlib.mlab
from scipy.signal import fftconvolve, hann
from music21 import *

# In[143]:


def getFrequencies(recordedSignal, recordSampleRateIn):
    n = len(recordedSignal)
    nUniquePts = int(numpy.ceil((n+1)/2.0))
#     nUniquePts = n//2
#     nUniquePts = n

    recordedSignal = recordedSignal * hann(n, sym=0)
    fft = numpy.fft.fft(recordedSignal)
    fft = fft[0:nUniquePts]
    fft = numpy.abs(fft)
    fft = fft / float(n)
    fft = fft**2

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if n % 2 > 0: # we've got odd number of points fft
        fft[1:len(fft)] = fft[1:len(fft)] * 2
    else:
        fft[1:len(fft) -1] = fft[1:len(fft) - 1] * 2 # we've got even number of points fft

    print('\nfft:', fft.size)
    # print(fft)
    # print(numpy.argmax(fft), numpy.max(fft))
    freqMaxIdx = numpy.argmax(fft)
    print(fft.size, freqMaxIdx, numpy.max(fft))
    print('\nrms:', numpy.sqrt(numpy.mean(numpy.int32(recordedSignal)**2)), numpy.sqrt(numpy.sum(fft)))

    freqs = numpy.arange(0, nUniquePts, 1.0) * (recordSampleRateIn*1.0/n);
    print('\nfreqs:', freqMaxIdx, '/', freqs.size, freqs[freqMaxIdx])
#     print(freqs)

    plt.figure(figsize=(20,4))
    # plt.plot(freqs/1000, 10*numpy.log10(fft))
    plt.plot(freqs/1000, fft)
    plt.title('Frequecy')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power')
    plt.show()

    return freqs[freqMaxIdx]


# In[171]:


def find(condition):
    res, = numpy.nonzero(numpy.ravel(condition))
    return res

def autocorrelationFunction(recordedSignal, recordSampleRateIn):
    correlation = fftconvolve(recordedSignal, recordedSignal[::-1], mode='full')
    lengthCorrelation = len(correlation) // 2
    correlation = correlation[lengthCorrelation:]

    difference = numpy.diff(correlation) #  Calculates the difference between slots
    positiveDifferences = find(difference > 0)
    if len(positiveDifferences) == 0: # pylint: disable=len-as-condition
        finalResult = 10 # Rest
    else:
        beginning = positiveDifferences[0]
        peak = numpy.argmax(correlation[beginning:]) + beginning
        finalResult = recordSampleRateIn / peak

    n = len(recordedSignal)
#     print(n, lengthCorrelation)
    freqs = numpy.arange(0, lengthCorrelation+1, 1.0) * (recordSampleRateIn*1.0/n)
    plt.figure(figsize=(20,4))
    # plt.plot(freqs/1000, 10*numpy.log10(fft))
    plt.plot(freqs/1000, correlation)
    plt.title('Frequecy')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power')
    plt.show()

    return finalResult


# In[181]:


readFile = 'wav/canon_2.wav'
f = wave.open(readFile,'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
print(params)


# In[182]:


#################################
# WaveData
#################################
strData = f.readframes(nframes) #读取音频，字符串格式
f.close()
print('strData', len(strData))
waveData = numpy.frombuffer(strData, dtype=numpy.int16) #将字符串转化为int
print('waveData:', type(waveData), waveData.shape, numpy.max(waveData), numpy.min(waveData), numpy.mean(waveData))
print('waveData:', waveData[0:20])

waveData.shape = -1,2
waveData = waveData.T
print('waveData:', type(waveData), waveData.shape, numpy.max(waveData), numpy.min(waveData), numpy.mean(waveData))
print('waveData left:', waveData[0][0:10])
print('waveData right:', waveData[1][0:10])


# In[183]:


recordedSignal = numpy.array(waveData[0][framerate*0:framerate*1])

plt.figure(figsize=(20,4))
plt.plot(numpy.arange(recordedSignal.size), recordedSignal)
plt.title('Original wave')
plt.show()

freq = getFrequencies(recordedSignal, framerate)
print(freq)

correlation = autocorrelationFunction(recordedSignal, framerate)
print(correlation)

# print(audioSearch.normalizeInputFrequency(freq), audioSearch.normalizeInputFrequency(correlation))

# cMaj = scale.MajorScale('C4')
pitchesList = audioSearch.detectPitchFrequencies(numpy.array([freq, correlation]))
print(pitchesList)

(detectedPitches, listplot) = audioSearch.pitchFrequenciesToObjects(pitchesList)
print('\ndetectedPitches:', len(detectedPitches))
print(detectedPitches[0:10])
print('\nlistplot:', len(listplot))
print(listplot[0:10])

