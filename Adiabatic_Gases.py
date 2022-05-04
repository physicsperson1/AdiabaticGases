import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

#Import data from a csv file into 2 arrays
def getArrays(filename):
    file = open(filename)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    ts = np.array([])
    ys = np.array([])
    for row in csvreader:
        ts = np.append(ts,float(row[0]))
        ys = np.append(ys,float(row[1]))
    file.close()
    return ts,ys

#Trim two arrays between two timestamps
def trimArrays(ts,ys,t1,t2):
    initIndex = 0
    finalIndex = 0
    for time in ts:
        if time < t1:
            initIndex += 1
        else:
            break
    for time in ts:
        if time < t2:
            finalIndex += 1
        else:
            break
    newTs = ts[initIndex:finalIndex]
    newYs = ys[initIndex:finalIndex]
    return newTs,newYs

#Shift the time array so that its first value is zero
def zeroTimeArray(ts):
    return ts-ts[0]
    
#Find a tau value from an array of datapoints
#Approximate the main frequency of a signal through an FFT
def getMainFreqFFT(ts,ys):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    # Number of sample points
    N = len(ys)
    # sample spacing
    T = ts[1]-ts[0]
    yf = fft(ys)
    tf = fftfreq(N, T)[:N//40]
    plt.plot(tf[0:250], 2.0/N * np.abs(yf[0:250]))  #Replaced N//40 with 250 in yf
    ax.set_xlabel('Frequency / s^-1')
    ax.set_ylabel('Amplitude / arb. units')
    plt.grid()
    plt.show()
    loop = yf.size//2
    frequencies = []
    for i in range(loop):
        if(2.0/N * np.abs(yf[i]) >= np.max(2.0/N * np.abs(yf[0:250]))/2.0):
            frequencies.append(tf[i])
    avFreq = frequencies[len(frequencies)//2]
    return avFreq

#Get the mean average seperation of the peaks as well as the standard deviation
def getAvVals(ts,ys):
    ys -= np.mean(ys)
    inPositives = False    #Keeps track of whether last yValue was positive or not
    if(ys[0] >= 0.0):
        inPositives = True
    axisCrossTs = np.array([])
    i = 0
    for yValue in ys:
        if(inPositives and yValue < 0.0):
            inPositives = False
            axisCrossTs = np.append(axisCrossTs,ts[i])
        if(yValue >= 0.0):
            inPositives = True
        i += 1
    seperations = np.array([])
    indices = []
    for i in range(axisCrossTs.size - 1):
        seperations = np.append(seperations, axisCrossTs[i+1] - axisCrossTs[i])
    for i in range(seperations.size):
        if(np.abs(seperations[i] - np.mean(seperations)) / np.mean(seperations) >= 0.25):
            indices.append(i)
    seperations = np.delete(seperations,indices)
    avPeriod = np.round(np.mean(seperations), 4)
    stdPeriod = np.round(np.std(seperations), 4)
    avFreq = np.round(1.0 / np.mean(seperations), 4)
    stdFreq = np.round(avFreq * stdPeriod / avPeriod, 4)
    #print("Average period is {0} +/- {1} seconds".format(avPeriod,stdPeriod))
    print("Average frequency is {0} +/- {1} Hz".format(avFreq,stdFreq))
    return avPeriod,stdPeriod,avFreq,stdFreq

#Used in Volume model
def getdVdt(omega,Vmax,Vmin,V):
    return omega * np.power((Vmax - V)*(V - Vmin), 0.5)

#Volume model
def getVolumeModelYs(Vi,omega,Vmax,Vmin,leakRate,maxEMF,decayTime,datapoints):
    V = Vi
    dt = 0.0001
    leakStep = dt*leakRate
    t = 0.0
    isMovingUp = True
    times = np.array([])
    volumes = np.array([])
    dVdts = np.array([])
    for i in range(datapoints):
        times = np.append(times, t)
        volumes = np.append(volumes, V)
        dV = dt * getdVdt(omega, Vmax, Vmin, V)
        if(isMovingUp):
            V += dV - leakStep
            dVdts = np.append(dVdts,(dV-leakStep) / dt)
        else:
            V += -dV - leakStep
            dVdts = np.append(dVdts,-(dV-leakStep) / dt)
        if(V <= Vmin):
            V = Vmin + 10e-11
            isMovingUp = True
        if(V >= Vmax):
            V = Vmax - 10e-11
            isMovingUp = False
        t += dt
    volumes -= Vi
    volumes *= maxEMF * np.exp(-times/decayTime)/ (Vmax-Vi)
    dVdts *= maxEMF * np.exp(-times/decayTime)/ np.max(dVdts)
    return dVdts  #was volumes

#Plot an x-y graph with axes labelled
def plotGraphs(data):
    #Plot this pulse
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    for i in range(data.shape[0] // 3):
        ax.errorbar(data[3*i],           
                     data[3*i+1],               
                     marker='o',             
                     markersize = 1,
                     markerfacecolor = data[3*i+2],
                     color=data[3*i+2],          
                     linestyle='none',       
                     capsize=6,
                     )
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Induced emf / V')
    ax.tick_params(direction='in',length=7)
    plt.rcParams.update({'font.size':22})
    plt.show()

#Get array of maxima times and emfs
def getMaxima(ts,ys,approxDecayTime):
    maxTs = np.array([ts[0]])
    maxYs = np.array([ys[0]])   #First datapoint will always be a maximum
    inPositives = True
    foundMaximum = True
    maxima = 1
    maxVal = -1.0
    for i in range(ts.size):
        if(inPositives and ys[i] < 0.0):
            inPositives = False  #Gone from positive to negative y vals
            foundMaximum = False
            maxVal = -1.0
        if(not(inPositives) and ys[i] >= 0.0):
            inPositives = True   #Gone from negative to positive y vals
        if(not(foundMaximum) and inPositives):
            if(ys[i] == maxVal):
                percError = (np.exp((ts[i]-maxTs[maxima-1])/approxDecayTime)*maxYs[maxima-1] - ys[i]) / np.exp((ts[i]-maxTs[maxima-1])/approxDecayTime)*maxYs[maxima-1]
                if(percError < 0.04):
                    foundMaximum = True
                    maxVal = ys[i]
                    topIndex = i
                    for j in range(20):
                        if(ys[i-5+j] > maxVal):
                            topIndex = i-5+j
                            maxVal = ys[topIndex]
                    maxTs = np.append(maxTs,ts[topIndex])
                    maxYs = np.append(maxYs,ys[topIndex])
                    maxima += 1
            elif(ys[i] > maxVal):
                maxVal = ys[i]
    return maxTs,maxYs

def getMaxima2(ts,ys):
    maxTs = np.array([])
    maxYs = np.array([])    
    wavelengths = 0
    inPositives = True
    for i in range(ts.size):
        if(ys[i] < 0.0 and inPositives):
            wavelengths += 1
            inPositives = False
        if(ys[i] > 0.0 and not(inPositives)):
            inPositives = True
    indexMultiple = ts.size // wavelengths
    for i in range(wavelengths):
        maxTs = np.append(maxTs, ts[i*indexMultiple])
        maxYs = np.append(maxYs, ys[i*indexMultiple])
    return maxTs,maxYs

def analyse(filename,t1,t2,configMode,decayTime,leakage):
    t,y = getArrays(filename)
    t,y = trimArrays(t,y,t1,t2)
    t = zeroTimeArray(t)
    if configMode:
        plotGraphs(np.array([t,y,'red']))
    else:
        avPeriod,stdPeriod,avFreq,stdFreq = getAvVals(t,y)
        modelYsA = y[0]*np.cos(avFreq*2.0*3.14159*t)*np.exp(-t / decayTime)
        modelYsB = getVolumeModelYs(0.000070,avFreq*2.0*3.14159,0.000075,0.000065,leakage,y[0],decayTime,t.size)
        dataSetsA = np.array([t,y,'red',t,modelYsA,'black'])
        plotGraphs(dataSetsA)
        dataSetsB = np.array([t,y,'red',t,modelYsB,'black'])
        plotGraphs(dataSetsB)

#============ COMPLETED ANALYSIS ==============

#Feel free to have a play around with the parameters below!

#analyse('AirData001.csv',1.6563,2.532,False,0.59,0.0000005)
#analyse('AirData002.csv',0.6986,1.7,False,0.59,0.0000005)
#analyse('AirData003.csv',1.263,2.237,False,0.63,0.0000005)
#analyse('AirData004.csv',0.6222,1.595,False,0.62,0.0000005)
#analyse('CO2Data001.csv',0.0,3.0,True,0.5,0.0000005)      #Omitted from experiment
#analyse('CO2Data002.csv',0.6864,2.4906,False,0.81,0.0000005)
#analyse('CO2Data003.csv',1.6525,3.4,False,1.03,0.0000005)
#analyse('CO2Data004.csv',1.6303,3.4,False,1.00,0.0000005)
#analyse('HeData001.csv',1.918,2.35,False,0.21,0.0000005)
#analyse('HeData002.csv',0.8077,1.2,False,0.21,0.0000005)
#analyse('HeData003.csv',1.1056,1.6,False,0.21,0.0000005)
#analyse('N2Data001.csv',2.0009,3.1,False,0.64,0.0000005)
#analyse('N2Data002.csv',0.7637,1.7,False,0.59,0.0000005)
#analyse('N2Data003.csv',0.7013,1.7,False,0.67,0.0000005)