# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.qe    # functions for QE analysis
"""

import math   #simple math
import copy
from time import time
from functools import lru_cache
import pandas as pd  #dataframes
import numpy as np   #arrays
from cycler import cycler
import matplotlib.pyplot as plt   #plotting
import matplotlib as mpl   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
from scipy.interpolate import interp1d
#from scipy.integrate import trapezoid
import scipy.constants as con   #physical constants
import ipywidgets as widgets
from IPython.display import display
from pvcircuit.junction import *
from num2words import num2words
import os
from pvcircuit.multi2T import *

# colors
junctioncolors = [  ['black'], #0J
                    ['red'], #1J
                    ['blue', 'red'], #2J
                    ['blue', 'green', 'red'],  #3J
                    ['blue', 'green', 'orange', 'red'], #4J
                    ['purple', 'blue', 'green', 'orange', 'red'],  #5J
                    ['purple', 'blue', 'green', 'black', 'orange', 'red']] #6J

# constants
k_q = con.k/con.e
hc_k = con.h * con.c / con.k  * 1e9 #for wavelength (nm)
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]
nan=np.nan
nm2eV=con.h * con.c / con.e * 1e9   #1239.852 from Igor
JCONST=(1000/100/100/nm2eV) #mA/cm2
DBWVL_PREFIX = 2. * np.pi * con.c * con.e / 100 / 100 #A/cm2


# standard data
pvcpath = os.path.dirname(os.path.dirname(__file__))  
datapath = os.path.join(pvcpath, 'data/') # Data files here
#datapath = os.path.abspath(os.path.relpath('../data/', start=__file__))
#datapath = pvcpath.replace('/pvcircuit','/data/')
ASTMfile = os.path.join(datapath,'ASTMG173.csv')
try:
    dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)
    wvl=dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
    refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  #all three reference spectra
    refnames = ['space','global','direct']
    AM0 = refspec[:,0]  #dfrefspec['space'].to_numpy(dtype=np.float64, copy=True)  # 1348.0 W/m2
    AM15G = refspec[:,1]  #dfrefspec['global'].to_numpy(dtype=np.float64, copy=True) # 1000.5 W/m2
    AM15D = refspec[:,2]  #dfrefspec['direct'].to_numpy(dtype=np.float64, copy=True) # 900.2 W/m2
except:
    print(pvcpath)
    print(datapath)
    print(ASTMfile)

def JdbMD(EQE, xEQE, TC, Eguess = 1.0, kTfilter=3, bplot=False):
    '''
    calculate detailed-balance reverse saturation current 
    from EQE vs xEQE
    xEQE in nm, can optionally use (start, step) for equally spaced data
    debug on bplot
    '''
    Vthlocal = Vth(TC) #kT
    EQE = np.array(EQE)  #ensure numpy
    if EQE.ndim == 1: #1D EQE[lambda]
        nQlams,  = EQE.shape
        njuncs = 1        
    elif EQE.ndim == 2:  #2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return 'dims in EQE:' + str(EQE.ndim)

    Eguess = np.array([Eguess] * njuncs)
    
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

    if xEQE.ndim != 1: # need 1D with same length as EQE(lam)
        return 'dims in xEQE:' + str(xEQE.ndim) + '!=1'
    elif len(xEQE) != nQlams:
        return 'nQlams:' + str(len(xEQE)) + '!='+ str(nQlams)

    Egvect = np.vectorize(EgFromJdb)
    EkT = nm2eV / Vthlocal / xEQE
    blackbody = np.expand_dims(DBWVL_PREFIX / (xEQE*1e-9)**4 / np.expm1(EkT) , axis=1)
    
    for count in range(10): 
        nmfilter = nm2eV/(Eguess - Vthlocal * kTfilter) #MD [652., 930.]
        if njuncs == 1:
            EQEfilter = np.expand_dims(EQE.copy(), axis=1)
        else:
            EQEfilter = EQE.copy()
            
        for i, lam in enumerate(xEQE):
            EQEfilter[i,:] *= (lam < nmfilter) #zero EQE about nmfilter      
           
        DBintegral = blackbody * EQEfilter      
        Jdb = np.trapz(DBintegral, x=(xEQE*1e-9), axis=0)  
        Egnew = Egvect(TC, Jdb)
        if bplot: print(Egnew, max((Egnew-Eguess)/Egnew))
        if np.amax((Egnew-Eguess)/Egnew) < 1e-6:
            break
        else: 
            Eguess=Egnew
     
    if bplot:
        efig, eax = plt.subplots()
        eax.plot(xEQE, DBintegral[:,0], c='blue', lw=2, marker='.')
        if njuncs > 1:
            reax = eax.twinx() #right axis
            reax.plot(xEQE, DBintegral[:,1], c='red', lw=2, marker='.') 

    return Jdb, Egnew

def PintMD(Pspec, xspec=wvl):
    #optical power of spectrum over full range
    return JintMD(None, None, Pspec, xspec)
    
def JintMD(EQE, xEQE, Pspec, xspec=wvl):
    '''
    integrate over spectrum or spectra
    if EQE is None -> calculate Power in [W/m2]
        total power=int(Pspec) over xEQE range
    else             -> caluculate J = spectra * lambda * EQE(lambda)
        Jsc = int(Pspec*QE[0]*lambda) in [mA/cm2]
        EQE optionally scalar for constant over xEQE range
        integrate over full spectrum is xEQE is None
    
    integrate multidimentional EQE(lambda)(junction) times MD reference spectra Pspec(lambda)(ispec)
    external quantum efficiency EQE[unitless] x-units = nm, 
    reference spectra Pspec[W/m2/nm] x-units = nm
    optionally Pspec as string 'space', 'global', 'direct' or '' for all three
    xEQE in nm, can optionally 
        use (start, step) for equally spaced data
        use None for same as xspec
    default x values for Pspec from wvl
    '''
    
    #check spectra input
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
        
    #check EQE input
    EQE = np.array(EQE)  #ensure numpy
    if EQE.ndim == 0: #scalar or None
        if np.any(EQE): 
            nQlams=1    #scalar -> return current
            njuncs=1
        else:
            nQlams=1    #None or False -> return power
            njuncs=0
    elif EQE.ndim == 1: #1D EQE[lambda]
        nQlams,  = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  #2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return 'dims in EQE:' + str(EQE.ndim)
    
    #check x range input
    xEQE = np.array(xEQE, dtype=np.float64)
    if xEQE.ndim == 0: #scalar or None
        xEQE = xspec    #use spec range if no EQE range
    if xEQE.ndim == 1: #1D
        nxEQE,  = xEQE.shape 
        start = min(xEQE)
        stop = max(xEQE) 
        if nxEQE == 2: # evenly spaced x-values (start, stop)
            xEQE= np.linspace(start, stop, max(nQlams,2), dtype=np.float64)
    else:
        return 'dims in xEQE:' + str(xEQE.ndim)
           
    if nQlams == 1:
        EQE = np.full_like(xEQE,EQE)
        
    if xspec.ndim != 1: # need 1D with same length as Pspec(lam)
        return 'dims in xspec:' + str(xspec.ndim) + '!=1'
    elif len(xspec) != nSlams:
        return 'nSlams:' + str(len(xspec.ndim)) + '!='+ str(nSlams)

    if xEQE.ndim != 1: # need 1D with same length as EQE(lam)
        return 'dims in xEQE:' + str(xEQE.ndim) + '!=1'
    elif nQlams == 1:
        pass
    elif len(xEQE) != nQlams:
        return 'nQlams:' + str(len(xEQE)) + '!='+ str(nQlams)
        
    #find start and stop index  of nSlams
    n0=0
    n1=nSlams-1
    for i, lam in enumerate(xspec):
        if lam <= min(start,stop):
            n0 = i
        elif lam <= max(start,stop):            
            n1 = i
        else:
            break
    xrange = xspec[n0:n1+1] # range of xspec values within xEQE range
    nrange = abs(n1+1-n0)
    if njuncs == 0: #calculate power over xrange
        if nspecs == 1:
            Jintegral = Pspec.copy()[n0:n1+1] #for Ptot
        else:
            Jintegral = Pspec.copy()[n0:n1+1,:] #for Ptot
        
    else: #calculate J over xrange
        #print(xrange.shape, xEQE.shape, EQE.shape, Pspec.shape)
        EQEinterp = interp1d(xEQE, EQE, axis=0, fill_value=0) # interpolate along axis=0
        Jintegral = np.zeros((nrange, nspecs, njuncs), dtype=np.float64) #3D array
        if njuncs == 1:
            EQEfine = np.expand_dims((EQEinterp(xrange) * xrange), axis=1) * JCONST  # lambda*EQE(lambda)[lambda,1]
        else:
            EQEfine = EQEinterp(xrange) * xrange[:,np.newaxis] * JCONST # lambda*EQE(lambda)[lambda,junc]
        for ijunc in range(0,njuncs):
            if nspecs == 1:
                Jintegral[:,0,ijunc] = Pspec.copy()[n0:n1+1] #for Ptot
            else:
                Jintegral[:,:,ijunc] = Pspec.copy()[n0:n1+1,:] #for Ptot
            Jintegral[:,:,ijunc] *= EQEfine[:, np.newaxis, ijunc]

    Jint = np.trapz(Jintegral, x=xrange, axis=0)     
    #print(xrange.shape, EQEfine.shape, EQE.shape, Pspec.shape, Jintegral.shape)
    #print(nSlams, nspecs, njuncs, nQlams, start, stop, xspec[n0], xspec[n1], n0, n1)    
    return Jint

@lru_cache(maxsize = 100)
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

@lru_cache(maxsize = 100)
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

class EQE(object):
    '''
    EQE object
    '''
    def __init__(self, rawEQE, xEQE, name='EQE', sjuncs=None):
    
        #check EQE input
        rawEQE = np.array(rawEQE)  #ensure numpy
        if rawEQE.ndim == 0: #scalar or None
            if np.any(rawEQE): 
                nQlams=1    #scalar -> return current
                njuncs=1
            else:
                nQlams=1    #None or False -> return power
                njuncs=0
        elif rawEQE.ndim == 1: #1D rawEQE[lambda]
            nQlams,  = rawEQE.shape
            njuncs = 1
            rawEQE = np.expand_dims(rawEQE, axis=1)   # -> 2D
        elif rawEQE.ndim == 2:  #2D rawEQE[lambda, junction]
            nQlams, njuncs = rawEQE.shape
        else:
            return 'dims in rawEQE:' + str(rawEQE.ndim)
    
        #check x range input
        xEQE = np.array(xEQE, dtype=np.float64)
        if xEQE.ndim == 0: #scalar or None
            xEQE = xspec    #use spec range if no rawEQE range
        if xEQE.ndim == 1: #1D
            nxEQE,  = xEQE.shape 
            self.start = min(xEQE)
            self.stop = max(xEQE) 
            if nxEQE == 2: # evenly spaced x-values (start, stop)
                xEQE= np.linspace(self.start, self.stop, max(nQlams,2), dtype=np.float64)
        else:
            return 'dims in xEQE:' + str(xEQE.ndim)

        if nQlams == 1:
            rawEQE = np.full_like(xEQE,rawEQE)    # fill rawEQE across range with scalar input

        if xEQE.ndim != 1: # need 1D with same length as rawEQE(lam)
            return 'dims in xEQE:' + str(xEQE.ndim) + '!=1'
        elif nQlams == 1:
            pass
        elif len(xEQE) != nQlams:
            return 'nQlams:' + str(len(xEQE)) + '!='+ str(nQlams)

        if sjuncs == None:
            sjuncs = [num2words(junc+1, to = 'ordinal') for junc in range(njuncs)]

        #class attributes
        
        self.ui = None      
        self.debugout = widgets.Output() # debug output
        self.name = name        # name of EQE object
        self.rawEQE = rawEQE    # 2D(lambda)(junction) raw input rawEQE (not LC corrected)
        self.xEQE = xEQE        # wavelengths [nm] for rawEQE data
        self.njuncs = njuncs    # number of junctions
        self.sjuncs = sjuncs    # names of junctions
        self.nQlams = nQlams    # number of wavelengths in rawEQE data
 
        self.corrEQE = np.empty_like(self.rawEQE)  # luminescent coupling corrected EQE same size as rawEQE      
        self.etas = np.zeros((njuncs,3), dtype=np.float64) #LC factor for next three junctions
        self.LCcorr() #calculate LC with zero etas
        
    def LCcorr(self, junc=None, dist=None, val=None):
        # change one eta[junc,dist] value
        # calculate LC corrected EQE
        # using procedure from 
        # Steiner et al., IEEE PV, v3, p879 (2013)
        etas = self.etas
        #with self.debugout: print(junc,dist,val)
        if junc == None or dist == None or val == None:
            pass
        else:
            etas[junc,dist] = val   #assign value
            #with self.debugout: print('success')
        raw = self.rawEQE
        for ijunc in range(self.njuncs):
            if ijunc == 0: #1st ijunction
                self.corrEQE[:,ijunc] = raw[:,ijunc]
            elif ijunc == 1: #2nd ijunction
                denom=1.+etas[ijunc,0]
                self.corrEQE[:,ijunc] = raw[:,ijunc] * denom \
                    - raw[:,ijunc-1] * etas[ijunc,0] 
            elif ijunc == 2: #3rd ijunction
                denom=1.+etas[ijunc,0]*(1.+etas[ijunc,1])
                self.corrEQE[:,ijunc] = raw[:,ijunc] * denom \
                    - raw[:,ijunc-1] * etas[ijunc,0] \
                    - raw[:,ijunc-2] * etas[ijunc,0] * etas[ijunc,1]
            else: #higher ijunctions
                denom=1.+etas[ijunc,0]*(1.+etas[ijunc,1]*(1.+etas[ijunc,2]))
                self.corrEQE[:,ijunc] = raw[:,ijunc] * denom \
                    - raw[:,ijunc-1] * etas[ijunc,0] \
                    - raw[:,ijunc-2] * etas[ijunc,0] * etas[ijunc,1] \
                    - raw[:,ijunc-3] * etas[ijunc,0] * etas[ijunc,1] * etas[ijunc,2]
                
    def Jdb(self, TC, Eguess = 1.0, kTfilter=3, dbug=False):
        # calculate Jscs and Egs from self.corrEQE
        Vthlocal = Vth(TC) #kT   
        Eguess = np.array([Eguess] * self.njuncs)
        Egvect = np.vectorize(EgFromJdb)
        EkT = nm2eV / Vthlocal / self.xEQE
        blackbody = np.expand_dims(DBWVL_PREFIX / (self.xEQE*1e-9)**4 / np.expm1(EkT) , axis=1)
    
        for count in range(10): 
            nmfilter = nm2eV/(Eguess - Vthlocal * kTfilter) #MD [652., 930.]
            EQEfilter = self.corrEQE.copy()
            
            for i, lam in enumerate(self.xEQE):
                EQEfilter[i,:] *= (lam < nmfilter) #zero EQE about nmfilter      
           
            DBintegral = blackbody * EQEfilter      
            Jdb = np.trapz(DBintegral, x=(self.xEQE*1e-9), axis=0)  
            Egnew = Egvect(TC, Jdb)
            if dbug: print(Egnew, max((Egnew-Eguess)/Egnew))
            if np.amax((Egnew-Eguess)/Egnew) < 1e-6:
                break
            else: 
                Eguess=Egnew
            self.Egs = Egnew

        return Jdb, Egnew

    def Jint(self,  Pspec='global', xspec=wvl):
        # integrate junction currents = integrate (spectra * lambda * EQE(lambda))

        #check spectra input
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
            Pspec = np.expand_dims(Pspec, axis=1)   # -> 2D
        elif Pspec.ndim == 2:  #2D Pspec[lambda, ispec]
            nSlams, nspecs = Pspec.shape
        else:
            return 'dims in Pspec:' + str(Pspec.ndim)

        if xspec.ndim != 1: # need 1D with same length as Pspec(lam)
            return 'dims in xspec:' + str(xspec.ndim) + '!=1'
        elif len(xspec) != nSlams:
            return 'nSlams:' + str(len(xspec.ndim)) + '!='+ str(nSlams)

        #find start and stop index  of nSlams
        n0=0
        n1=nSlams-1
        for i, lam in enumerate(xspec):
            if lam <= min(self.start,self.stop):
                n0 = i
            elif lam <= max(self.start,self.stop):            
                n1 = i
            else:
                break
        xrange = xspec[n0:n1+1] # range of xspec values within xEQE range
        nrange = abs(n1+1-n0)
        
        #remember these for now
        self.Pspec = Pspec      # 2D(lambda)(spectrum) spectral irradiance [W/m2/nm]
        self.xspec = xspec      # wavelengths [nm] for spectra
        self.nSlams = nSlams    # number of wavelengths in spectra
        self.nspecs = nspecs    # number of spectra
        self.n0 = n0            # index of first spectral data that overlaps with rawEQE data
        self.n1 = n1            # index of last spectral data that overlaps with rawEQE data

        EQEinterp = interp1d(self.xEQE, self.corrEQE, axis=0, fill_value=0) # interpolate along axis=0
        Jintegral = np.zeros((nrange, nspecs, self.njuncs), dtype=np.float64) #3D array
        EQEfine = EQEinterp(xrange) * xrange[:,np.newaxis] * JCONST # lambda*EQE(lambda)[lambda,junc]
        for ijunc in range(self.njuncs):
            Jintegral[:,:,ijunc] = Pspec[n0:n1+1,:] * EQEfine[:, np.newaxis, ijunc]
        return np.trapz(Jintegral, x=xrange, axis=0)     
    
    def plot(self, Pspec='global', ispec=0, specname=None, xspec=wvl, size='x-large'):
        # plot EQE on top of a spectrum
        rnd2 =100
      
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=Multi2T.junctioncolors[self.njuncs])
        for i in range(self.njuncs):
            rlns = ax.plot(self.xEQE, self.rawEQE[:,i], lw=1, ls='--', marker='', label='_'+self.sjuncs[i])
            ax.plot(self.xEQE, self.corrEQE[:,i], lw=3, c=rlns[0].get_color(), marker='', label=self.sjuncs[i])
        ax.legend()
        ax.set_ylim(-0.1,1.1)      
        ax.set_xlim(math.floor(self.start/rnd2)*rnd2, math.ceil(self.stop/rnd2)*rnd2)
        ax.set_ylabel('EQE', size=size)  # Add a y-label to the axes.
        ax.set_xlabel('Wavelength (nm)', size=size)  # Add an x-label to the axes.
        ax.set_title(self.name + ' EQE', size=size)
        ax.axhline(0, lw=0.5, ls='--', c='black', label='_hzero')
        ax.axhline(1, lw=0.5, ls='--', c='black', label='_hone')

        rax = ax.twinx() #right axis
        #check spectra input
        if type(Pspec) is str: # optional string space, global, direct
            if Pspec in dfrefspec.columns:
                specname=Pspec
                Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
            else:
                specname=dfrefspec.columns[ispec]
                Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True) #just use refspec instead of error

        if np.any(Pspec):
            Pspec = np.array(Pspec, dtype=np.float64)            
            if not specname: specname='spectrum'+str(ispec)
            if Pspec.ndim == 2: Pspec = Pspec[:,ispec] #slice 2D numpy to 1D
            rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color='grey', label='fill')
            rax.plot(xspec, Pspec, c='grey', lw=0.5, marker='', label=specname)
            rax.set_ylabel('Irradiance (W/m2/nm)', size=size)  # Add a y-label to the axes.
            rax.set_ylim(0,2)
            #rax.legend(loc=7)
        return ax, rax
        
    def controls(self, Pspec='global', ispec=0, specname=None, xspec=wvl):
        '''
        use interactive_output for GUI in IPython
        '''
        tand_layout = widgets.Layout(width= '300px', height='40px')
        vout_layout = widgets.Layout(width= '180px', height='40px')
        junc_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    justify_content='space-around')
        multi_layout = widgets.Layout(display='flex', 
                    flex_flow='row',
                    justify_content='space-around')

        replot_types = [widgets.widgets.widget_float.BoundedFloatText, 
                        widgets.widgets.widget_int.BoundedIntText,
                        widgets.widgets.widget_int.IntSlider,
                        widgets.widgets.widget_float.FloatSlider,
                        widgets.widgets.widget_float.FloatLogSlider]

        def on_EQEchange(change):
            # function for changing values
            old = change['old'] #old value
            new = change['new'] #new value
            owner = change['owner'] #control
            value = owner.value
            desc = owner.description            
            #with self.debugout: print('Mcontrol: ' + desc + '->', value)
            #self.set(**{desc:value})

        def on_EQEreplot(change):
            # change info
            fast=True
            if type(change) is widgets.widgets.widget_button.Button:
                owner = change
            else: # other controls
                owner = change['owner'] #control 
                value = owner.value               
            desc = owner.description  
            if desc == 'Recalc': fast = False
              
            #recalculate            
            ts = time()  
            if desc[:3] == 'eta':
                junc, dist = parse('eta{:1d}{:1d}',desc)
                self.LCcorr(junc, dist, value) #replace one value and recalculate LC
                specname = None
            elif desc == 'spec':
                if value in dfrefspec.columns:
                    specname = value
                    Pspec = dfrefspec[specname].to_numpy(dtype=np.float64, copy=True)              
            else:
                VoutBox.clear_output()
                with VoutBox: print(desc)
                return 0

            with Rout: # right output device -> light
                #replot
                lines = ax.get_lines()
                for line in lines:
                    linelabel=line.get_label()
                    #with self.debugout: print(linelabel)
                    for i in range(self.njuncs):
                        if linelabel == self.sjuncs[i]:
                             line.set_data(self.xEQE, self.corrEQE[:,i]) #replot
                            
                rlines = rax.get_lines()
                for line in rlines:
                    linelabel=line.get_label()
                    #with self.debugout: print(linelabel)
                    if linelabel in refnames:
                        if specname == None: #desc == 'spec'
                             specname = linelabel
                             Pspec = specname
                        else:
                            line.set_data(xspec, Pspec) #replot spectrum 
                            for obj in rax.get_children():
                                if type(obj) is mpl.collections.PolyCollection: #contours
                                    if obj.get_label() == 'fill':
                                        obj.remove() #remove old fill
                            rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color='grey', label='fill')
                            line.set(label = specname) #relabel spectrum

            Jscs = self.Jint(Pspec, xspec)
            Jdbs, Egs = self.Jdb(25)
            OP = PintMD(Pspec, xspec)

            VoutBox.clear_output()
            with VoutBox: 
                stext = (specname+' {0:6.2f} W/m2').format(OP) 
                print('Eg = ',Egs, ' eV')
                print(stext)
                print('Jsc = ',Jscs[0], ' mA/cm2')

            te = time()
            dt=(te-ts)
            with VoutBox:   print('Calc Time: {0:>6.2f} s'.format(dt))

        # summary line
        VoutBox = widgets.Output()
        VoutBox.layout.height = '70px'
        #with VoutBox: print('Summary')
            
        # Right output -> EQE plot
        Rout = widgets.Output()
        with Rout: # output device
            if plt.isinteractive: 
                plt.ioff()
                restart = True
            else:
                restart = False
            ax, rax = self.plot(Pspec, ispec, specname, xspec)
            fig = ax.get_figure()
            fig.show()
            rlines = rax.get_lines()
            for line in rlines:
                linelabel=line.get_label()
                if linelabel in refnames:
                    specname = linelabel
            if restart: plt.ion()
        
        # tandem3T controls
        in_tit = widgets.Label(value='EQE: ', description='title')
        in_name = widgets.Text(value=self.name, description='name', layout=tand_layout,
            continuous_update=False)                        
        in_name.observe(on_EQEchange,names='value') #update values
        
        in_spec = widgets.Dropdown(value=specname, description='spec', layout=tand_layout,
            options=refnames)                        
        in_spec.observe(on_EQEreplot,names='value') #update values

        Hui = widgets.HBox([in_tit, in_name, in_spec])
        #in_Rs2T.observe(on_2Tchange,names='value') #update values

        in_eta = []
        elist0 = []
        elist1 = []
        elist2 = []
        # list of eta controls
        for i in range(self.njuncs) : 
            if i > 0:          
                in_eta.append(widgets.FloatSlider(value=self.etas[i,0], min=-0.2, max=1.5,step=0.001,
                    description='eta'+str(i)+"0",layout=junc_layout,readout_format='.4f'))
                j = len(in_eta)-1
                elist0.append(in_eta[j])
                in_eta[j].observe(on_EQEreplot,names='value')  #replot
            #if i > 1:          
                in_eta.append(widgets.FloatSlider(value=self.etas[i,1], min=-0.2, max=1.5,step=0.001,
                    description='eta'+str(i)+"1",layout=junc_layout,readout_format='.4f'))
                j = len(in_eta)-1
                elist1.append(in_eta[j])
                in_eta[j].observe(on_EQEreplot,names='value')  #replot
                if i > 1:
                    in_eta[j].observe(on_EQEreplot,names='value')  #replot
                else:
                    in_eta[j].disabled = True 
            #if i > 2:          
                in_eta.append(widgets.FloatSlider(value=self.etas[i,2], min=-0.2, max=1.5,step=0.001,
                    description='eta'+str(i)+"2",layout=junc_layout,readout_format='.4f'))
                j = len(in_eta)-1
                elist2.append(in_eta[j])
                if i > 2:
                    in_eta[j].observe(on_EQEreplot,names='value')  #replot
                else:
                    in_eta[j].disabled = True 
        etaui0 = widgets.HBox(elist0)
        etaui1 = widgets.HBox(elist1)
        etaui2 = widgets.HBox(elist2)
            
        #in_Rs2T.observe(on_2Treplot,names='value')  #replot
        #in_2Tbut.on_click(on_2Treplot)  #replot  
 
        #EQE_ui = widgets.HBox(clist)
        #eta_ui = widgets.HBox(jui) 
        
        ui = widgets.VBox([Rout, VoutBox, Hui, etaui0, etaui1, etaui2])
        self.ui = ui
        #in_2Tbut.click() #fill in MPP values

        # return entire user interface, dark and light graph axes for tweaking
        return ui, ax, rax
        