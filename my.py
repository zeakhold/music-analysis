import numpy
import wave
import os
import matplotlib.pyplot as plt
import matplotlib.mlab
from scipy.signal import fftconvolve, hann
from music21 import audioSearch, scale

def getFrequencies(recordedSignal, recordSampleRateIn):
    '''转时域为频域

    输入为时域数据，输出为频域数据

    Args:
        recordedSignal: 时域数据
        recordSampleRateIn: 采样率

    Returns:
        最大频率（主音）
    '''
    n = len(recordedSignal)
    nUniquePts = int(numpy.ceil((n+1)/2.0))
#     nUniquePts = n//2
#     nUniquePts = n

    recordedSignal = recordedSignal * hann(n, sym=0)
    # 原序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    fft = numpy.fft.fft(recordedSignal)
    fft = fft[0:nUniquePts] # FFT具有对称性，一般只需要用N的一半
    fft = numpy.abs(fft) # 取复数的绝对值，即模，即振幅
    fft = fft / float(n)
    fft = fft**2

    if n % 2 > 0:
        fft[1:len(fft)] = fft[1:len(fft)] * 2
    else:
        fft[1:len(fft) -1] = fft[1:len(fft) - 1] * 2

    freqMaxIdx = numpy.argmax(fft) # 最大音量元素索引（主音）

#     print('fft：', fft)
#     print('fft sort：', numpy.argsort(fft))
#     print('振幅（频率）信息:', fft.size, freqMaxIdx, numpy.max(fft))

    # 频率
    freqs = numpy.arange(0, nUniquePts, 1.0) * (recordSampleRateIn*1.0/n)

#     plt.figure(figsize=(20,4))
#     plt.plot(freqs, fft)
#     plt.title('Frequecy')
#     plt.xlabel('Frequency')
#     plt.ylabel('Power')
#     plt.show()

    return freqs[freqMaxIdx]




w = wave.open('wav/canon_0.wav','rb')
params = w.getparams()

# 声道数量，字节长度，采样频率，音频总帧数
nchannels, sampwidth, framerate, nframes = params[:4]

print('音频信息：', params)

strData = w.readframes(nframes) #读取音频，字符串格式
w.close()
print('音频采样总数：', len(strData))
waveData = numpy.frombuffer(strData, dtype=numpy.int16) #将字符串转化为int
# print('waveData:', waveData[0:30])

waveData.shape = -1,2
waveData = waveData.T

# 取部分音频
recordedSignal = numpy.array(waveData[0][int(framerate*1):int(framerate*2)])

# plt.figure(figsize=(20,4))
# plt.plot(numpy.arange(recordedSignal.size), recordedSignal)
# plt.title('Original wave')
# plt.show()

freq = getFrequencies(recordedSignal, framerate)
print('检测到的频率:', freq)

correlation = audioSearch.autocorrelationFunction(recordedSignal, framerate)
print('校正频率:', correlation)


thresholds, pitches = audioSearch.prepareThresholds(scale.ChromaticScale('C4'))

# 部分音频
print('检测到的音高:', audioSearch.normalizeInputFrequency(freq, thresholds, pitches)[1])

# 整段音频
# for i in range(40):
#     rs = numpy.array(waveData[0][int(framerate*(i/2)):int(framerate*(i/2  + 0.5))])
# #     fr = getFrequencies(rs, framerate)
#     fr = audioSearch.autocorrelationFunction(rs, framerate)
#     print('检测到的音高:', audioSearch.normalizeInputFrequency(fr, thresholds, pitches)[1])
