# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.QEanalysis    # function for QE analysis
"""

import math   #simple math
import copy
from time import time
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants
import ipywidgets as widgets
from IPython.display import display
from pvcircuit.junction import *

# constants
k_q = con.k/con.e
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]
nan=np.nan

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
