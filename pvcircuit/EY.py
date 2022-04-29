# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.EY     use Ripalda's Tandem proxy spectra for energy yield
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
import pvcircuit as pvc
from pvcircuit.junction import *
from pvcircuit.qe import *
from pvcircuit.multi2T import *
from pvcircuit.tandem3T import *
import os, sys
import glob
from tandems import sandia_T, physicaliam
from pprint import pprint

#  from 'Tandems' project
vectoriam = np.vectorize(physicaliam)
RIPpath = datapath.replace('/PVcircuit/data/','/Tandems/')  #assuming 'Tandems' is in parallel GitHub folders
FARMpath = RIPpath+"FARMS-NIT-clustered-spectra-USA/"

#list of data locations
clst_axis = glob.glob(FARMpath + "*axis.clusters.npz")
clst_tilt = glob.glob( FARMpath + "*tilt.clusters.npz")
tclst_axis = glob.glob(FARMpath + "*axis.timePerCluster.npz")
tclst_tilt = glob.glob( FARMpath + "*tilt.timePerCluster.npz")
clst_axis.sort()
tclst_axis.sort()
clst_tilt.sort()
tclst_tilt.sort()

def VMloss(type3T, bot, top, ncells):
    # calculates approximate loss factor for VM strings of 3T tandems
    if type3T == 'r':
        endloss = (max(bot,top) - 1)
    elif type3T == 's':
        endloss = (bot + top - 1)
    lossfactor = 1 - endloss / ncells
    return lossfactor
    
class TMY(object):
    '''
    typical meterological year at a specific location    
    '''
    
    def __init__(self, i, tilt=False):
    
        if not tilt:
            #Boulder i=497
            clst = clst_axis
            tclst = tclst_axis
        else:
            #Boulder i=491
            clst = clst_tilt
            tclst = tclst_tilt
        
        self.tilt=tilt
        self.index = i
        self.name =  clst[i].split("/")[-1][:-13]
        self.longitude = float(self.name.split('_')[0])
        self.latitude = float(self.name.split('_')[1])
        self.altitude = float(self.name.split('_')[2])
        self.zone = float(self.name.split('_')[3])

        d1 = np.load(clst[i])
        td1 = np.load(tclst[i])

        arr_0 = d1['arr_0']
        tarr_0 = td1['arr_0']

        spec = arr_0[0,154:172,:]
        tspec = tarr_0[0,154:172]
    
        self.Temp = spec[:,1].copy() * 1e6 #C
        self.Wind = spec[:,2].copy() * 1e6  #m/s
        self.DayTime = arr_0[0,0,-1].copy() * 24# scalar
        self.NTime = tspec # Fraction of 1
    
        aoi = spec[:, 5] * 1e6
        self.Angle = np.array(aoi.tolist())
        aim = vectoriam(aoi)
#        aim = physicaliam(aoi)
        self.AngleMOD = np.array(aim.tolist())
    
        spec[:,:5] = 0
        spec[:,-1] = 0
        self.Irradiance = np.array(spec.tolist()).transpose()  #transpose for integration with QE

        #calculate from spectral proxy data only
        #self.SpecPower = np.trapz(self.Irradiance, x=wvl, axis=0) # optical power of each spectrum 
        self.SpecPower = PintMD(self.Irradiance)   
        self.RefPower = PintMD(pvc.qe.refspec)   
        self.TempCell = sandia_T(self.SpecPower, self.Wind, self.Temp)
        self.inPower = self.SpecPower * self.NTime  # spectra power*(fractional time)
        self.YearlyEnergy = self.inPower.sum() * self.DayTime * 365.25 / 1000  #kWh/m2/yr
  
    def cellcurrents(self,EQE,xEQE):
        # subcell currents and Egs and under self TMY for a given EQE
           
        self.JscSTCs = JintMD(EQE, xEQE, pvc.qe.refspec)/1000.
        self.Jscs = JintMD(EQE, xEQE, self.Irradiance) /1000. 
        Jdbs, self.Egs = JdbMD(EQE, xEQE, 25)  #Eg from EQE same at all temperatures 
        
    def cellpower(self,model,oper,iref=1):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        #Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # iref = 0 -> space
        # iref = 1 -> global
        # iref = 2 -> direct
        #Outputs 
        #- EY energy yield of cell [kWh/m2/yr]
        #- EYeff energy yield efficiency = EY/YearlyEnergy
        #- STCeff efficiency of cell under reference spectrum (space,global,direct)
        self.outPower = self.inPower.copy() #initialize
        if isinstance(model,pvc.multi2T.Multi2T):
            T=2
            type3T = '2T'
        elif isinstance(model,pvc.tandem3T.Tandem3T):
            T=3
            if model.top.pn == model.bot.pn:
                type3T = 'r'
            else:
                type3T = 's'
        else:
            return 'unknown model' + type(model)

        for i in range(len(self.outPower)):
            if T == 2:
                model.j[0].set(Eg=self.Egs[0], Jext=self.Jscs[i,0], TC=self.TempCell[i])
                model.j[1].set(Eg=self.Egs[1], Jext=self.Jscs[i,1], TC=self.TempCell[i])                
                mpp_dict=model.MPP() #oper ignored for 2T
                Pmax = mpp_dict['Pmp']
            elif T == 3:
                model.top.set(Eg=self.Egs[0], Jext=self.Jscs[i,0], TC=self.TempCell[i])
                model.bot.set(Eg=self.Egs[1], Jext=self.Jscs[i,1], TC=self.TempCell[i])
                if oper == 'MPP':
                    iv3T = model.MPP()
                elif oper == 'CM':
                    ln, iv3T = model.CM()
                elif oper[:2] == 'VM':               
                    bot, top = parse('VM{:1d}{:1d}',oper)
                    #print(bot, top)
                    ln, iv3T = model.VM(bot,top)
                else:
                    print(oper+' not valid')
                    iv3T.Ptot[0] = 0
                Pmax = iv3T.Ptot[0]
            self.outPower[i] = Pmax * self.NTime[i] * 10000.

        EY = sum(self.outPower) * self.DayTime * 365.25/1000   #kWh/m2/yr
        EYeff = EY / self.YearlyEnergy

        #calc reference spectra efficiency
        if iref == 0:
            Tref = 28. #space
        else:
            Tref = 25. #global and direct
            
        if T == 2:  #Multi2T
            model.j[0].set(Eg=self.Egs[0], Jext=self.JscSTCs[iref,0], TC=Tref)
            model.j[1].set(Eg=self.Egs[1], Jext=self.JscSTCs[iref,1], TC=Tref)                
            mpp_dict=model.MPP() #oper ignored for 2T
            Pmax = mpp_dict['Pmp'] * 10.
            ratio = 0.
        elif T == 3:    #Tandem3T
            model.top.set(Eg=self.Egs[0], Jext=self.JscSTCs[iref,0], TC=Tref)
            model.bot.set(Eg=self.Egs[1], Jext=self.JscSTCs[iref,1], TC=Tref)
            if oper == 'MPP':
                iv3T = model.MPP()
                ratio = -0.5
            elif oper == 'CM':
                ln, iv3T = model.CM()
                ratio = 0.
            elif oper[:2] == 'VM':               
                bot, top = parse('VM{:1d}{:1d}',oper)
                #print(bot, top)
                ln, iv3T = model.VM(bot,top)
                ratio = top/bot
            else:
                print(oper+' not valid')
                iv3T.Ptot[0] = 0
                ratio = np.nan
            Pmax = iv3T.Ptot[0] * 10.
            
        STCeff = Pmax * 1000. / self.RefPower[iref]
        
        return EY, EYeff, STCeff, ratio, type3T
