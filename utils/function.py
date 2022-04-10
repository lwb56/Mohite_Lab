import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt


def fwhm_pv(fG, fL):
    return (fG**5 + 2.69269*(fG**4)*fL + 2.42843*(fG**3)*(fL**2) + 4.47163*(fG**2)*(fL**3) + 0.07842*(fG)*(fL**4) + fL**5)**0.2


def cond2(p, ydata):

    c1 = abs(fwhm_pv(p[2], p[3]))
    c2 = abs(fwhm_pv(p[6], p[7]))

    c = abs(c1-c2)/(c1-.3)

    return c



def pseudovoigt(p,x):
    # % PSEUDOVOIGTTCH (Built-in Model) Pseudo-voigt type TCH    
    # %   Y=(OBJ/LINEFIT).PSEUDOVOIGTTCH(X,[AMP,X0,SIGMA,GAMMA])
    # %       AMP: Amplitude, X0: Location, SIGMA: Scale of Gaussian,
    # %       GAMMA: Scale of Lorentzian        
    # %
    # %   Reference:
    # %   (1) P. Thompson, et al., J. Appl. Cryst. 20, 79 (1987)

    amp = p[0]
    x0 = p[1]
    sigma = p[2] 
    gamma = p[3]
    # # FWHM of original G and L
    fG = 2*sigma*np.sqrt(2*np.log(2))
    fL = 2*gamma
    # FWHM and eta
    f = (fG**5 + 2.69269*(fG**4)*fL + 2.42843*(fG**3)*(fL**2) + 4.47163*(fG**2)*(fL**3) + 0.07842*(fG)*(fL**4) + fL**5)**0.2
    
    eta = 1.36603*(fL/f) - 0.47719*((fL/f)**2) + 0.11116*((fL/f)**3)
    # new sigma and gamma for new Gaussian and Lorentzian
    sigma = (f/2)/np.sqrt(2*np.log(2))
    gamma = f/2;    

    G = np.exp(np.divide(-1*np.power((x-x0),2),(2*(sigma**2))))*(1/m.sqrt(2*m.pi))*(1/sigma)
    L = np.divide(((1/m.pi)*gamma), ((x-x0)**2 +gamma**2))
    
    y = amp*(eta*L+(1-eta)*G)

    return y


def plotting_fit(data, func, p, x_eval, bg_fun, bp):

    x = data[:,0]
    y = data[:,1]

    
    plt.figure(figsize = (9,7))
    ax = plt.gca()
    plt.plot(x,y,'-o',markersize =11)
    plt.plot(x_eval,func(p, x_eval)+bg_fun(bp, x_eval) ,linewidth = 4)
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


