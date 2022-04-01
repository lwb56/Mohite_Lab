import scipy as sp
import numpy as np
import math as m

def pseudovoigt(p,x):
    # % PSEUDOVOIGTTCH (Built-in Model) Pseudo-voigt type TCH    
    # %   Y=(OBJ/LINEFIT).PSEUDOVOIGTTCH(X,[AMP,X0,SIGMA,GAMMA])
    # %       AMP: Amplitude, X0: Location, SIGMA: Scale of Gaussian,
    # %       GAMMA: Scale of Lorentzian        
    # %
    # %   Reference:
    # %   (1) P. Thompson, et al., J. Appl. Cryst. 20, 79 (1987)
    amp    = p[0]
    x0      = p[1]
    sigma   = p[2] 
    gamma   = p[3]
    # FWHM of original G and L
    fG = 2*sigma*np.sqrt(2*np.log(2));
    fL = 2*gamma;
    # FWHM and eta
    
    f = (fG**5 + 2.69269*(fG**4)*fL + 2.42843*(fG**3)*(fL**2) + 4.47163*(fG**2)*(fL**3) + 0.07842*(fG)*(fL**4) + fL**5)**0.2
    
    eta = 1.36603*(fL/f) - 0.47719*((fL/f)**2) + 0.11116*((fL/f)**3)
    # new sigma and gamma for new Gaussian and Lorentzian
    sigma = (f/2)/np.sqrt(2*np.log(2))
    gamma = f/2;    

    G = np.multiply((1/m.sqrt(2*m.pi))/sigma, np.exp(np.divide(-1*np.power((x-x0),2),(2*(sigma**2)))))
    L = np.divide(((1/m.pi)*gamma), ((x-x0)**2 +gamma**2))
    
    y = amp*(eta**L+(1-eta)**G)


    return y

