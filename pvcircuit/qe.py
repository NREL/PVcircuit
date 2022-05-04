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
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
from scipy.interpolate import interp1d
#from scipy.integrate import trapezoid
import scipy.constants as con   #physical constants
import ipywidgets as widgets
from IPython.display import display
from pvcircuit.junction import *
from num2words import num2words
import os, sys

# constants
k_q = con.k/con.e
hc_k = con.h * con.c / con.k  * 1e9 #for wavelength (nm)
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]
nan=np.nan
nm2eV=con.h * con.c / con.e * 1e9   #1239.852 from Igor
JCONST=(1000/100/100/nm2eV) #mA/cm2
DBWVL_PREFIX = 2. * np.pi * con.c * con.e / 100 / 100 #A/cm2


# standard data
pvcpath = os.path.dirname(__file__)  # Data files here
datapath = pvcpath.replace('/pvcircuit','/data/')
dfrefspec = pd.read_csv(datapath+'ASTMG173.csv', index_col=0, header=2)
wvl=dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  #all three reference spectra
refnames = ['space','global','direct']
AM0 = refspec[:,0]  #dfrefspec['space'].to_numpy(dtype=np.float64, copy=True)  # 1348.0 W/m2
AM15G = refspec[:,1]  #dfrefspec['global'].to_numpy(dtype=np.float64, copy=True) # 1000.5 W/m2
AM15D = refspec[:,2]  #dfrefspec['direct'].to_numpy(dtype=np.float64, copy=True) # 900.2 W/m2

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
        self.name = name        # name of EQE object
        self.rawEQE = rawEQE    # 2D(lambda)(junction) raw input rawEQE (not LC corrected)
        self.xEQE = xEQE        # wavelengths [nm] for rawEQE data
        self.njuncs = njuncs    # number of junctions
        self.sjuncs = sjuncs    # names of junctions
        self.nQlams = nQlams    # number of wavelengths in rawEQE data
 
        self.corrEQE = np.empty_like(self.rawEQE)  # luminescent coupling corrected EQE same size as rawEQE      
        self.etas = np.zeros((njuncs,3), dtype=np.float64) #LC factor for next three junctions
        self.LCcorr() #calculate LC with zero etas
        
    def LCcorr(self):
        # calculate LC corrected EQE
        # using procedure from 
        # Steiner et al., IEEE PV, v3, p879 (2013)
        etas = self.etas
        raw = self.rawEQE
        for junc in range(self.njuncs):
            if junc == 0: #1st junction
                self.corrEQE[:,junc] = raw[:,junc]
            elif junc == 1: #2nd junction
                denom=1.+etas[junc,0]
                self.corrEQE[:,junc] = raw[:,junc] * denom \
                    - raw[:,junc-1] * etas[junc,0] 
            elif junc == 2: #3rd junction
                denom=1.+etas[junc,0]*(1.+etas[junc,1])
                self.corrEQE[:,junc] = raw[:,junc] * denom \
                    - raw[:,junc-1] * etas[junc,0] \
                    - raw[:,junc-2] * etas[junc,0] * etas[junc,1]
            else: #higher junctions
                denom=1.+etas[junc,0]*(1.+etas[junc,1]*(1.+etas[junc,2]))
                self.corrEQE[:,junc] = raw[:,junc] * denom \
                    - raw[:,junc-1] * etas[junc,0] \
                    - raw[:,junc-2] * etas[junc,0] * etas[junc,1] \
                    - raw[:,junc-3] * etas[junc,0] * etas[junc,1] * etas[junc,2]
                
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
    
    def plot(self, Pspec='global', ispec=0, specname=None, xspec=wvl):
        # plot EQE on top of a spectrum
        rnd2 =100
        
        fig, ax = plt.subplots()
        for i in range(self.njuncs):
            ax.plot(self.xEQE, self.rawEQE[:,i], lw=1, marker='', label='_'+self.sjuncs[i])
            ax.plot(self.xEQE, self.corrEQE[:,i], lw=3, marker='', label=self.sjuncs[i])
        ax.legend()
        ax.set_ylim(0,1)      
        ax.set_xlim(math.floor(self.start/rnd2)*rnd2, math.ceil(self.stop/rnd2)*rnd2)
        ax.set_ylabel('EQE')  # Add a y-label to the axes.
        ax.set_xlabel('Wavelength (nm)')  # Add an x-label to the axes.

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
            rax.fill_between(xspec, Pspec, step="pre", alpha=0.2, color='grey')
            rax.plot(xspec, Pspec, c='grey', lw=0.5, marker='', label=specname)
            rax.set_ylabel('Irradiance (W/m2/nm)')  # Add a y-label to the axes.
            rax.set_ylim(bottom=0)
            #rax.legend(loc=7)
            Jscs = self.Jint(Pspec=Pspec, xspec=xspec)
            Jdbs, Egs = self.Jdb(25)
            OP = PintMD(Pspec=Pspec, xspec=xspec)
            stext=specname+'\n'+str(Jscs[0])+'mA/cm2\n'+str(OP)+'W/m2'
            #rax.text(1000,0.6,stext,bbox=dict())
            print(specname, OP, ' W/m2')
            print('Eg = ',Egs, ' eV')
            print('Jsc = ',Jscs, ' mA/cm2')
        
        return ax, rax