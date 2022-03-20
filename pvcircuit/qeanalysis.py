# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.QEanalysis    # function for QE analysis
"""

import math   #simple math
import copy
from time import time
from functools import lru_cache
import pandas as pd  #dataframes
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
from scipy.interpolate import interp1d
#from scipy.integrate import trapezoid
import scipy.constants as con   #physical constants
import ipywidgets as widgets
from IPython.display import display
from pvcircuit.junction import *

# constants
k_q = con.k/con.e
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]
nan=np.nan
nm2ev=con.h * con.c / con.e * 1e9   #1239.852 from Igor
Jconst=(1/100/100/nm2ev) #A/cm2

# standard data
path = '../data/'
dfrefspec = pd.read_csv(path+'ASTMG173.csv', index_col=0, header=2)
wvl=dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
AM0 = dfrefspec['space'].to_numpy(dtype=np.float64, copy=True)  # 1348.0 W/m2
AM15G = dfrefspec['global'].to_numpy(dtype=np.float64, copy=True) # 1000.5 W/m2
AM15D = dfrefspec['direct'].to_numpy(dtype=np.float64, copy=True) # 900.2 W/m2
refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  #all three reference spectra

def JintMD(EQE, xEQE, Pspec, xspec=wvl):
    '''
    calculate total power of spectra and Jsc of each junction from QE
    first column (0) is total power=int(Pspec)
	second column (1) is first Jsc = int(Pspec*QE[0]*lambda)

    integrate multidimentional QE(lambda)(junction) times MD reference spectra Pspec(lambda)(ispec)
    external quantum efficiency QE[unitless] x-units = nm, 
    reference spectra Pspec[W/m2/nm] x-units = nm
    optionally Pspec as string 'space', 'global', or 'direct'
    xEQE in nm, can optionally use (start, step) for equally spaced data
    default x values for Pspec from wvl
    '''
    
    if type(Pspec) is str: # optional string space, global, direct
        if Pspec in dfrefspec.columns:
            Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
        else:
            Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True) #just use refspec instead of error
    else:
        Pspec = np.array(Pspec, dtype=np.float64) #ensure numpy

    if Pspec.ndim == 1: #1D Pspec[lambda]
        nSlams,  = Pspec.shape
        nspecs = 1
    elif Pspec.ndim == 2:  #2D Pspec[lambda, ispec]
        nSlams, nspecs = Pspec.shape
    else:
        return 'dims in Pspec:' + str(Pspec.ndim)
        
    EQE = np.array(EQE)  #ensure numpy
    if EQE.ndim == 1: #1D EQE[lambda]
        nQlams,  = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  #2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return 'dims in EQE:' + str(EQE.ndim)
    
    if len(xEQE) == 2: # evenly spaced x-values (start, stop)
        start, stop = xEQE
        #stop =  start + step * (nQlams - 1)
        #xEQE, step = np.linspace(start, stop, nQlams, dtype=np.float64, retstep=True)
        xEQE= np.linspace(start, stop, nQlams, dtype=np.float64)
    else:   # arbitrarily spaced x-values
        xEQE = np.array(xEQE, dtype=np.float64)
        start = min(xEQE)
        stop = max(xEQE)
        #step = xEQE[1]-xEQE[0]  #first step
        
    if xspec.ndim != 1: # need 1D with same length as Pspec(lam)
        return 'dims in xspec:' + str(xspec.ndim) + '!=1'
    elif len(xspec) != nSlams:
        return 'nSlams:' + str(len(xspec.ndim)) + '!='+ str(nSlams)

    if xEQE.ndim != 1: # need 1D with same length as EQE(lam)
        return 'dims in xEQE:' + str(xEQE.ndim) + '!=1'
    elif len(xEQE) != nQlams:
        return 'nQlams:' + str(len(xEQE)) + '!='+ str(nQlams)
        
    #find start and stop index  of nSlams
    for i, lam in enumerate(xspec):
        if lam <= min(start,stop):
            n0 = i
        elif lam <= max(start,stop):            
            n1 = i
        else:
            break
    xrange = xspec[n0:n1+1] # range of xspec values within xEQE range
    EQEinterp = interp1d(xEQE, EQE, axis=0, fill_value=0) # interpolate alone axis=0
    Jintegral = np.zeros((nSlams, nspecs, (njuncs+1)), dtype=np.float64) #3D array
    if njuncs == 1:
        EQEfine = np.expand_dims((EQEinterp(xrange) * xrange), axis=1) * Jconst  # lambda*EQE(lambda)[lambda,1]
    else:
        EQEfine = EQEinterp(xrange) * xrange[:,np.newaxis] * Jconst # lambda*EQE(lambda)[lambda,junc]

    #print(xrange.shape, EQEfine.shape, EQE.shape, Pspec.shape, Jintegral.shape)
    if nspecs == 1:
        Jintegral[:,0,0] = Pspec.copy() #for Ptot
    else:
        Jintegral[:,:,0] = Pspec.copy() #for Ptot
        
    for ijunc in range(1,(njuncs+1)):
        Jintegral[n0:n1+1,:,ijunc] = Jintegral[n0:n1+1,:,0] * EQEfine[:, np.newaxis, ijunc-1]
        pass

    Jint = np.trapz(Jintegral, x=xspec, axis=0)     
    #print(nSlams, nspecs, njuncs, nQlams, start, stop, xspec[n0], xspec[n1], n0, n1)    
    return Jint

def JdbFromEg(TC,Eg,dbsides=1.,method=None):
    '''
    return the detailed balance dark current
    assuming a square EQE
    Eg[=]eV
    TK[=]K
    returns Jdb[=]A/cm2
    
    optional parameters
    method: 'gamma'
    dbsides:    single-sided->1.  bifacial->2.
    see special functions, Andrews p.71
    see Geisz EL&LC paper, King EUPVSEC
    '''
    EgkT = Eg / Vth(TC)
    TKlocal = TK(TC)

    if str(method).lower=='gamma':
        #use special function incomplete gamma
        #gamma(3)=2.0 not same incomplete gamma as in Igor
        Jdb = DB_PREFIX * TKlocal**3. * gammaincc(3., EgkT) * 2.0 * dbsides    
    else:
        #Jdb as in Geisz et al.
        Jdb = DB_PREFIX * TKlocal**3. * (EgkT*EgkT + 2.*EgkT + 2.) * np.exp(-EgkT) * dbsides    #units from DB_PREFIX

    return Jdb

def EgFromJdb(TC, Jdb, Eg=1.0, eps=1e-6, itermax=100, dbsides=1.):
    '''
    see GetData AT_Egcalc
    return the bandgap from the Jdb
    assuming a square EQE
    iterates using gammaInc(3,x)=2*exp(-x)*(1+x+x^2/2)
    see special functions, Andrews p.73  
      
    optional parameters
    Eg=1.0 eV    #initial guess
    eps=0.001    #tolerance
    itermax=100 #maximum iterations
    dbsides=1.  #bifacial->2.
    '''

    Vthlocal = Vth(TC)
    TKlocal = TK(TC)
    x0= Eg / Vthlocal
    off=np.log(2. * dbsides * DB_PREFIX * TKlocal**3. / Jdb)
    count=0

    while count < itermax:
        x1 = off + np.log(1.+x0+x0*x0/2.)
        try:
            tol=abs((x1-x0)/x0)
        except:
            tol=abs(x1-x0)
        if  tol < eps:
            return x1*Vthlocal
        x0=x1
        count+=1
    return None
