import os, glob
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit

from lmfit.models import GaussianModel,Model
from lmfit.models import PseudoVoigtModel,Model,VoigtModel



def extract(filename):
    '''
    This function extracts the the q and intensity values from the filename and outputs a matrix of (n,2) Q vs I
    '''

    # open files for extraction 
    f = open(filename, 'r')

    q = []
    I = []
    for i in f.readlines():
        dat =i.split(' ')
        if len(dat) > 4:
            continue

        q.append(float(dat[0]))
        I.append(float(dat[2]))

    q = np.array(q)
    I = np.array(I)

    data = np.concatenate((q.reshape(-1,1),I.reshape(-1,1)), axis = 1)

    return data

def gaus_fit1(data,int_guess): 

    x = data[:,0]
    y  =data[:,1]

    gauss1 = GaussianModel(prefix='g1_')
    pars = gauss1.guess(y,x=x)
    pars.update(gauss1.make_params())
    pars['g1_center'].set(value=int_guess[0], min=660, max=700)
    pars['g1_sigma'].set(value=int_guess[1])
    pars['g1_amplitude'].set(value=int_guess[2],min = 0)
    

    def line(x,slope,intercept):
        return x*slope+intercept
    l1 = Model(line)
    pars.update(l1.make_params(slope =int_guess[3], intercept = int_guess[4]))
    
    mod = gauss1 +l1
    out = mod.fit(y, pars, x=x)
    
    b1 = out.best_values 
    
    print('Residual is: ', np.linalg.norm(out.residual))

    return out

def voigt_fit(data,int1,int2):     

    x = data[:,0]
    y  =data[:,1]

    PV1 = PseudoVoigtModel


    V1 = VoigtModel(prefix='PV_')
    pars = V1.guess(y,x=x)
    pars.update(V1.make_params())

    pars['PV_center'].set(value=int1)
    pars['PV_amplitude'].set(value=int2)
    pars['PV_sigma'].set(value=.002,min = 0)
    pars['PV_gamma'].set(value=.002)
    
    # This is just a linear function
    def line(x,slope,intercept):
        return x*slope+intercept

    l1 = Model(line)
    pars.update(l1.make_params(slope =0, intercept = 100))

    mod = V1 +l1

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    y_out = out.best_fit#-np.amin(out.best_fit)
#     y = y-np.amin(y)
#     plt.figure(figsize = (9,7))
    ax = plt.gca()
    plt.plot(x,y,'-o',markersize =11)
#     plt.plot(x,y_out/np.amax(y_out),linewidth = 3)
    plt.plot(x,y_out,linewidth = 4)
    plt.ylabel('Intensity (a.u.)',fontsize = 40)
    plt.xlabel(r"${2\theta}$",fontsize = 40)
    plt.rc('axes', linewidth=2.5)
    ax.axes.yaxis.set_ticks([])
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='minor', length=5)
    ax.tick_params(which='major', length=10)
    plt.rcParams.update({'font.size': 40})
    plt.show()

    # outputting the fitted variable, since out.bestvalue is an ordereddict which is a specific structure class
    a = out.best_values
    return a
