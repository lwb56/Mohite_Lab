import scipy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, minimize, fmin_slsqp
from utils.tracking import background_model, plotting_model


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


import math as m


def fit_xrdV1(data, range, bg_range, parms,extra2, adapt, plot, background_linear = False):

    '''
    This function will be used to fit a varity of GIWAXS/XRD data using a
    pseudo-Voigt functions. It will take in a range of data points and try to
    use least squares (no constraints) with contraints to fit the function. The outputs will be the
    fitted parameters, the fitted data, the FWHM, the peak location and then
    the peak area.

    INPUT:

    OUTPUT:
    
    '''


    # this is to do the background subtraction

    # finding the start and end of an array for the 040 plane background
    x1_bg = np.where(abs(data[:,0]-bg_range[0])<= .01)
    ind1_bg = x1_bg[0][0]
    x2_bg = np.where(abs(data[:,0]-bg_range[1])<= .01)
    ind2_bg =x2_bg[0][0]

    # finding the start and end of an array for the 404 plane
    x1_111 = np.where(abs(data[:,0]-range[0])<= .01)
    ind1 = x1_111[0][0]
    x2_111 = np.where(abs(data[:,0]-range[1])<= .01)
    ind2 =x2_111[0][0]
    
    # get the difference in indices
    indices = np.concatenate((np.arange(ind1_bg, ind1),np.arange(ind2, ind2_bg)),axis = 0 )
    bg_data = data[indices,:]

    # fitting the background to an expotential curve (line if needed)
    bg_model = background_model(bg_data, int_guess= [1000, 10], linear = background_linear)
    # plotting_model(bg_data, bg_model)

    bg_model = sp.interpolate.interp1d(bg_data[:,0], bg_data[:,1])

    ydata = data[ind1:ind2,1] - bg_model(data[ind1:ind2,0])
    xdata = data[ind1:ind2,0]

    raw_data = np.concatenate((xdata.reshape(-1,1),ydata.reshape(-1,1)), axis = 1)

    
    intperfun = sp.interpolate.interp1d(xdata, ydata)

    # iterpolate the data so that the datafit has more point for a more meanful residual value
    xdata = np.linspace(xdata[0],xdata[-1], 200)
    ydata = intperfun(xdata)

    
    ma = np.max(ydata)

    # hand derivative for calculating average slope
    aver_slope = np.mean(np.divide(np.diff(ydata), xdata[1:]))

    if range[2] == 1:
        
        def func(p0, x, y):
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            # minimization function (func - ydata) == 0
            return (pseudovoigt(p0, x)+p0[4]+p0[5]*x) - y

        fun1 = lambda p0: func(p0, xdata, ydata).reshape(-1,)

        print(fun1(parms).shape)

        print(ydata.shape)

        # upper and lower bound
        ub = [m.inf,parms[1]+.2,1,1,ma,m.inf]
        lb = [0,parms[1]-.2,.0002,0.0002, 0,-m.inf]

        # for a one fit problem, least squares is good enough
        res = least_squares(fun1, parms, bounds = (lb, ub), method='trf', max_nfev= 50000, verbose= 1)


        bg_fun = lambda p, xdata: p[0]+p[1]*xdata
        if plot:
            plotting_fit(raw_data, pseudovoigt, res.x, xdata,bg_fun , bp = res.x[4:])
        print('Cost function is :', res.cost)
        return res.x, xdata, ydata
    
    elif range[2] == 2:
        # this gets much more complicated since now we need to have a nonlinear constrained on both the peaks

        # ydata = ydata/np.max(ydata)

        def func(p,x):
            return pseudovoigt(p[0:4], x)+pseudovoigt(p[4:8],x)

        # fun2 = lambda p: np.sum((ydata-func(p,xdata)**2))#np.linalg.norm(func(p,xdata)-ydata, ord=2)

        fun2 = lambda p: (ydata-func(p,xdata)).reshape(-1,)
      
        # # upper and lower bound
        # ub = [m.inf,parms[1]+.3,1,1,
        #     m.inf,parms[1+4]+.3,1,1,
        #     ma,m.inf]

        # lb = [0,parms[1]-.3,.0002,0.00002,
        #     0,parms[1+4]-.3,.0002,0.00002,
        #     0,-m.inf]



        # res = minimize(fun2, parms, method = 'trust-constr', options = {'disp': True, 'maxiter': 50000}, bounds = bound_lims)

        if len(extra2) == 0:
            # upper and lower bound
            ub = [500,parms[1]+.2,1,1,
                500,parms[1+4]+.2,1,1]

            lb = [0.00000001,parms[1]-.2,.0002,0.00002,
                0.00000001,parms[1+4]-.2,.0002,0.00002]
            res = least_squares(fun2, parms, bounds = (lb, ub), method='trf', max_nfev= 50000, verbose= 0)
        else:
            # upper and lower bound
            ub = [extra2[0]*1.5,extra2[1]+.1,1,1,
                extra2[4]*1.5,extra2[1+4]+.1,1,1]

            lb = [extra2[0]*.5,extra2[1]-.2,.0002,0.00002,
                extra2[4]*.5,extra2[1+4]-.2,.0002,0.00002]
            res = least_squares(fun2, extra2, bounds = (lb, ub), method='trf', max_nfev= 50000, verbose= 0)

        

        pv2 = lambda p0, x: pseudovoigt(p0[0:4],x) + pseudovoigt(p0[4:8],x)
        bg_fun = lambda p, xdata: p[0]+p[1]*xdata

        fit_parms = res.x

        if plot:
            plotting_fit(raw_data, pv2, res.x, xdata, bg_fun , bp = [0,0])
        # plotting_fit(raw_data, pseudovoigt, res.x[0:4], xdata, bg_fun , bp = [0,0])
        # plotting_fit(raw_data, pseudovoigt, res.x[4:8], xdata, bg_fun , bp = [0,0])

        # if the fit every changes from first pk to the second pk
        if fit_parms[1] < fit_parms[1+4]:
            return fit_parms, xdata, ydata
        else:
            temp = fit_parms[0:4]
            fit_parms[0:4] = fit_parms[4:8]
            fit_parms[4:8] = temp
            return fit_parms, xdata, ydata

            


        # print('Cost function is :', res.cost)
        # return res.x

        
    return 0


def data_explore(data, range, bg_range, background_linear = False):

    '''
    This function will be used to fit a varity of GIWAXS/XRD data using a
    pseudo-Voigt functions. It will take in a range of data points and try to
    use least squares (no constraints) with contraints to fit the function. The outputs will be the
    fitted parameters, the fitted data, the FWHM, the peak location and then
    the peak area.

    INPUT:

    OUTPUT:
    
    '''


    # this is to do the background subtraction

    # finding the start and end of an array for the 040 plane background
    x1_bg = np.where(abs(data[:,0]-bg_range[0])<= .01)
    ind1_bg = x1_bg[0][0]
    x2_bg = np.where(abs(data[:,0]-bg_range[1])<= .01)
    ind2_bg =x2_bg[0][0]

    # finding the start and end of an array for the 404 plane
    x1_111 = np.where(abs(data[:,0]-range[0])<= .01)
    ind1 = x1_111[0][0]
    x2_111 = np.where(abs(data[:,0]-range[1])<= .01)
    ind2 =x2_111[0][0]
    
    # get the difference in indices
    indices = np.concatenate((np.arange(ind1_bg, ind1),np.arange(ind2, ind2_bg)),axis = 0 )
    bg_data = data[indices,:]

    # fitting the background to an expotential curve (line if needed)
    bg_model = background_model(bg_data, int_guess= [1000, 10], linear = background_linear)
    # plotting_model(bg_data, bg_model)

    bg_model = sp.interpolate.interp1d(bg_data[:,0], bg_data[:,1])

    ydata = data[ind1:ind2,1] - bg_model(data[ind1:ind2,0])
    xdata = data[ind1:ind2,0]

    raw_data = np.concatenate((xdata.reshape(-1,1),ydata.reshape(-1,1)), axis = 1)

    
    intperfun = sp.interpolate.interp1d(xdata, ydata)

    # iterpolate the data so that the datafit has more point for a more meanful residual value
    xdata = np.linspace(xdata[0],xdata[-1], 200)
    ydata = intperfun(xdata)

    return xdata, ydata