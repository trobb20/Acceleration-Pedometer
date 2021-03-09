#acceleration pedometer project
#Teddy Robbins 2021

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#Read in data to pandas df
df = pd.read_csv('running.csv')
timeCol = df.columns[0]
absAccelCol = df.columns[-1]

#Get the data we really want into a np array
time = df[timeCol].to_numpy()
accel = df[absAccelCol].to_numpy()

dur = time[-1]
Fs = time.shape[0]/dur#sample rate

#Let's cut the data down to knock off the ends
newtime=time[int(Fs):-int(Fs)]
newaccel=accel[int(Fs):-int(Fs)]

#Do a fourier transform to find the frequency of stepping
transform = np.fft.fft(newaccel)[1:] #do fourier transform
length = transform.shape[0] #length of sample
transform = np.abs(transform/length) #normalize by total length
transform = transform[0:(length//2+1)] #clip off nyquist
transform[2:-2] = 2*transform[2:-2] #fold over array to capture energy on both sides
length = transform.shape[0] #reset length
f = np.linspace(0,Fs//2,length) #set up frequency bins

#Plotting
plt.figure()
plt.subplot(211)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s2)')
plt.plot(newtime,newaccel)

plt.subplot(212)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.plot(f,transform)
plt.axis([0, 10, 0, np.max(transform)+0.05*np.max(transform)])
plt.show()