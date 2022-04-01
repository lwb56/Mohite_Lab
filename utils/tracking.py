import os, glob
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import curve_fit

from lmfit.models import GaussianModel,Model
from lmfit.models import PseudoVoigtModel,Model,VoigtModel, ExponentialModel
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)



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

    # an matrix of [Q,I] where Q is the scattering vector and I is the intensity
    return data
    

def find_params(filename):
    name = filename.split('\\')[-1]
    surname = filename.split('/')[-2]
    return name[len(surname):].split('.dat')[0].split('_')



# line function
def line(x,slope,intercept):
    return x*slope+intercept


# this is the fitting function for the gaussian fit using lmfit package.
def gaus_fit1(data,int_guess, background = True):
    '''
    This function will fit and plot the raw data curve vs fitted curve.

    INPUT:
        data - an matrix of n x 2 where the first column is the Q value and the second is the intensity
        int_guess - a initial guess list [center value, sigma, and amplitude]
        line -  a booling for the background to be turned on, 
                if True then background is a line
                if false then background is a expotential

    OUTPUT:

        out -  the lmfit class object that contains all the fitted values (best_fit, best_params, fit_report)
    ''' 
    x = data[:,0]
    y  =data[:,1]

    gauss1 = GaussianModel(prefix='g1_')
    pars = gauss1.guess(y,x=x)
    pars.update(gauss1.make_params())
    pars['g1_center'].set(value=int_guess[0])
    pars['g1_amplitude'].set(value=int_guess[1],min = 0)
    pars['g1_sigma'].set(value=int_guess[2])
    
    
    if background == True:
        bg = Model(line)
        pars.update(bg.make_params(slope =int_guess[3], intercept = int_guess[4]))
    else:
        bg = ExponentialModel(prefix= 'exp_')
        pars = bg.guess(y, x=x)
    
    mod = gauss1 +bg
    out = mod.fit(y, pars, x=x)
    
    b1 = out.best_values 
    print('Residual is: ', np.linalg.norm(out.residual))

    return out


def pseudo_voigt_fit1(data,int_guess, background = True):     

    x = data[:,0]
    y  =data[:,1]

    PV1 = PseudoVoigtModel


    PV1 = PseudoVoigtModel(prefix='PV_')
    pars = PV1.guess(y,x=x)
    pars.update(PV1.make_params())

    # center of the peak
    pars['PV_center'].set(value=int_guess[0])
    # intensity of the peak
    pars['PV_amplitude'].set(value=int_guess[1])
    # the FWHM of the voigt function
    pars['PV_fwhm'].set(value=int_guess[2],min = 0)
    # the ratio of the two profiles
    pars['PV_fraction'].set(value=int_guess[3], min = .5)

    if background == True:
        bg = Model(line)
        pars.update(bg.make_params(slope =int_guess[4], intercept = int_guess[5]))
    else:
        bg = ExponentialModel(prefix= 'exp_')
        pars = bg.guess(y, x=x)

    mod = PV1 +bg

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    y_out = out.best_fit#-np.amin(out.best_fit)

    print('Residual is: ', np.linalg.norm(out.residual))

    # returning the lmfit class object with all the fits
    return out

    # outputting the fitted variable, since out.bestvalue is an ordereddict which is a specific structure class
    # best_params = out.best_values
    # return best_params


def voigt_fit1(data,int_guess, background = True):
    '''
    This is a voigt with a linear background

    '''

    x = data[:,0]
    y  =data[:,1]

    PV1 = VoigtModel
    PV1 = VoigtModel(prefix='PV_')
    pars = PV1.guess(y,x=x)
    pars.update(PV1.make_params())

    # center of the peak
    pars['PV_center'].set(value=int_guess[0])
    # intensity of the peak
    pars['PV_amplitude'].set(value=int_guess[1])
    # the FWHM of the voigt function
    pars['PV_sigma'].set(value=int_guess[2],min = 0.001)
    # the ratio of the two profiles
    pars['PV_gamma'].set(value=int_guess[3], min = .001)
    

    if background == True:
        bg = Model(line)
        pars.update(bg.make_params(slope =int_guess[4], intercept = int_guess[5]))
    else:
        bg = ExponentialModel(prefix= 'exp_')
        pars = bg.guess(y, x=x)

    mod = PV1 +bg

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    y_out = out.best_fit#-np.amin(out.best_fit)

    print('Residual is: ', np.linalg.norm(out.residual))


    # returning the lmfit class object with all the fits
    return out


def voigt_fit2(data,int_guess, background = True):
    '''
    This is a voigt with a linear background

    '''

    x = data[:,0]
    y  =data[:,1]

    PV1 = VoigtModel
    PV1 = VoigtModel(prefix='PV_')
    pars = PV1.guess(y,x=x)
    pars.update(PV1.make_params())

    print(pars)

    #['PV_amplitude', 'PV_center', 'PV_sigma', 'PV_gamma', 'PV_fwhm', 'PV_height']

    # center of the peak
    pars['PV_center'].set(value=int_guess[0])
    # intensity of the peak
    pars['PV_amplitude'].set(value=int_guess[1])
    # the FWHM of the voigt function
    pars['PV_sigma'].set(value=int_guess[2],min = 0.001)
    # the ratio of the two profiles
    pars['PV_gamma'].set(value=int_guess[3], min = .001)

    PV2 = VoigtModel
    PV2 = VoigtModel(prefix='PV2_')
    # pars = PV2.guess(y,x=x)
    pars.update(PV2.make_params())

    # center of the peak
    pars['PV2_center'].set(value=int_guess[4])
    # intensity of the peak
    pars['PV2_amplitude'].set(value=int_guess[5])
    # the FWHM of the voigt function
    pars['PV2_sigma'].set(value=int_guess[6],min = 0.001)
    # the ratio of the two profiles
    pars['PV2_gamma'].set(value=int_guess[7], min = .001)
    

    if background == 'linear':
        bg = Model(line)
        # pars = bg.guess(y, x=x)
        pars.update(bg.make_params(slope =int_guess[8], intercept = int_guess[9]))
    elif background == 'expotential':
        bg = ExponentialModel(prefix= 'exp_')
        # pars = bg.guess(y, x=x)
        pars.update(bg.make_params())

        pars['exp_amplitude'].set(value = int_guess[8], min = 0)
        pars['exp_decay'].set(value = int_guess[9])
    
    else:
        bg = 0


    mod = PV1 + bg

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    y_out = out.best_fit#-np.amin(out.best_fit)

    print('Residual is: ', np.linalg.norm(out.residual))


    # returning the lmfit class object with all the fits
    return out


def background_model(data,int_guess, linear= True):
    '''

    '''

    x = data[:,0]
    y  =data[:,1]


    if linear == True:
        bg = Model(line)
        pars = bg.guess(y, x=x)
        pars.update(bg.make_params(slope =int_guess[0], intercept = int_guess[1]))
    else:
        bg = ExponentialModel(prefix= 'exp_')
        pars = bg.guess(y, x=x)
        pars.update(bg.make_params())
        pars['exp_amplitude'].set(value = int_guess[0], min = 0)
        pars['exp_decay'].set(value = int_guess[1])

    mod = bg

    init = mod.eval(pars, x=x)
    out = mod.fit(y, pars, x=x)
    y_out = out.best_fit#-np.amin(out.best_fit)

    return out




def plotting_model(data, model):
    x = data[:,0]
    y  =data[:,1]
      
    plt.figure(figsize = (9,7))
    ax = plt.gca()
    plt.plot(x,y,'-o',markersize =11)
    plt.plot(x,model.best_fit ,linewidth = 4)
    plt.ylabel('Intensity (a.u.)',fontsize = 40)
    plt.xlabel(r"${Q(A^{-1})}$",fontsize = 40)
    plt.rc('axes', linewidth=2.5)
    # ax.axes.yaxis.set_ticks([])
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # ax.xaxis.set_minor_locator(MultipleLocator(.1))
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='minor', length=5)
    ax.tick_params(which='major', length=10)
    plt.rcParams.update({'font.size': 40})
    plt.show()