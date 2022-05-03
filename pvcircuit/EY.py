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
from pvcircuit.iv3T import *
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

@lru_cache(maxsize = 100)
def VMloss(type3T, bot, top, ncells):
    # calculates approximate loss factor for VM strings of 3T tandems
    if bot == 0:
        endloss = 0
    elif top == 0:
        endloss = 0
    elif type3T == 'r':
        endloss = (max(bot,top) - 1)
    elif type3T == 's':
        endloss = (bot + top - 1)
    else:   # not Tandem3T
        endloss = 0
    lossfactor = 1 - endloss / ncells
    return lossfactor

@lru_cache(maxsize = 100)
def VMlist(mmax):
    # generate a list of VM configurations + 'MPP'=4T and 'CM'=2T
    # mmax < 10 for formating reasons
    
    sVM = ['MPP','CM','VM11']
    for m in range(mmax+1):
        for n in range(1,m):
            lcd = 2
            if m/lcd == round(m/lcd) and n/lcd == round(n/lcd):
                #print(n,m, 'skip2')
                continue
            lcd = 3
            if m/lcd == round(m/lcd) and n/lcd == round(n/lcd):
                #print(n,m, 'skip3')
                continue
            lcd = 5
            if m/lcd == round(m/lcd) and n/lcd == round(n/lcd):
                #print(n,m, 'skip5')
                continue
            #print(n,m, 'ok')
            sVM.append('VM'+str(m)+str(n))
    return sVM
    
def cellmodeldesc(model,oper):
    # return description of model and operation
    # cell 'model' can be 'Multi2T' or 'Tandem3T'
    #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
    #Outputs (ratio, type3T, loss)

    #loss = 1.
    bot = 0
    top = 0
    type3T = ''

    if isinstance(model,pvc.multi2T.Multi2T):
        type3T = '2T'
    elif isinstance(model,pvc.tandem3T.Tandem3T):
        if model.top.pn == model.bot.pn:
            type3T = 'r'
        else:
            type3T = 's'
            
        if oper == 'MPP':
            ratio = -0.5
        elif oper == 'CM':
            ratio = 0
        elif oper[:2] == 'VM':               
            bot, top = parse('VM{:1d}{:1d}',oper)
            #loss = VMloss(type3T, bot, top, ncells)
            ratio = bot/top
        else:
            print(oper+' not valid')
            ratio = np.nan
    else:
        print('unknown model' + str(type(model)))
        ratio = np.nan
        type3T = ''

    return bot, top, ratio, type3T

    
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
        self.longitude = float(self.name.split('_')[1])
        self.latitude = float(self.name.split('_')[0])
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
        self.SpecPower = np.trapz(self.Irradiance, x=wvl, axis=0) # optical power of each spectrum 
        self.RefPower = np.trapz(pvc.qe.refspec, x=wvl, axis=0) # optical power of each reference spectrum 
        #self.SpecPower = PintMD(self.Irradiance)   
        #self.RefPower = PintMD(pvc.qe.refspec)   
        self.TempCell = sandia_T(self.SpecPower, self.Wind, self.Temp)
        self.inPower = self.SpecPower * self.NTime  # spectra power*(fractional time)
        self.YearlyEnergy = self.inPower.sum() * self.DayTime * 365.25 / 1000  #kWh/m2/yr
  
    def cellbandgaps(self,EQE,TC=25):
        # subcell Egs for a given EQE class
        self.Jdbs, self.Egs = EQE.Jdb(TC)  #Eg from EQE same at all temperatures 

    def cellcurrents(self,EQE,STC=False):
        # subcell currents and Egs and under self TMY for a given EQE class
           
        #self.JscSTCs = JintMD(EQE, xEQE, pvc.qe.refspec)/1000.
        #self.Jscs = JintMD(EQE, xEQE, self.Irradiance) /1000. 
        if STC:
            self.JscSTCs = EQE.Jint(pvc.qe.refspec)/1000.
        else:
            self.Jscs = EQE.Jint(self.Irradiance) /1000. 
        
    def cellSTCeff(self,model,oper,iref=1):
        # max power of a cell under a reference spectrum
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        #Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # iref = 0 -> space
        # iref = 1 -> global
        # iref = 2 -> direct
        #Outputs 
        #- STCeff efficiency of cell under reference spectrum (space,global,direct)
        
        bot, top, ratio, type3T= cellmodeldesc(model,oper) #ncells does not matter here
            
        #calc reference spectra efficiency
        if iref == 0:
            Tref = 28. #space
        else:
            Tref = 25. #global and direct
            
        if type3T == '2T': #Multi2T
            for ijunc in range(model.njuncs):
                model.j[ijunc].set(Eg=self.Egs[ijunc], Jext=self.JscSTCs[iref,ijunc], TC=Tref)              
            mpp_dict=model.MPP() #oper ignored for 2T
            Pmax = mpp_dict['Pmp'] * 10.
            ratio = 0.
        elif type3T in ['s','r']:    #Tandem3T
            model.top.set(Eg=self.Egs[0], Jext=self.JscSTCs[iref,0], TC=Tref)
            model.bot.set(Eg=self.Egs[1], Jext=self.JscSTCs[iref,1], TC=Tref)
            if oper == 'MPP':
                iv3T = model.MPP()
            elif oper == 'CM':
                ln, iv3T = model.CM()
            elif oper[:2] == 'VM':               
                ln, iv3T = model.VM(bot,top)
            else:
                print(oper+' not valid')
                iv3T = IV3T('bogus')
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0] * 10.
        else:
            Pmax = 0.
                
        STCeff = Pmax * 1000. / self.RefPower[iref]

        return STCeff
        
    def cellEYeff(self,model,oper):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        #Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        #Outputs 
        #- EYeff energy yield efficiency = EY/YearlyEnergy
        #- EY energy yield of cell [kWh/m2/yr]
                
        bot, top, ratio, type3T = cellmodeldesc(model,oper) #ncells does not matter here

        # calc EY, etc
        self.outPower = np.empty_like(self.inPower) #initialize
        for i in range(len(self.outPower)):
            if type3T == '2T': #Multi2T
                for ijunc in range(model.njuncs):
                    model.j[ijunc].set(Eg=self.Egs[ijunc], Jext=self.Jscs[i,ijunc], TC=self.TempCell[i])
                mpp_dict=model.MPP() #oper ignored for 2T
                Pmax = mpp_dict['Pmp']
            elif type3T in ['s','r']:    #Tandem3T
                model.top.set(Eg=self.Egs[0], Jext=self.Jscs[i,0], TC=self.TempCell[i])
                model.bot.set(Eg=self.Egs[1], Jext=self.Jscs[i,1], TC=self.TempCell[i])
                if oper == 'MPP':
                    iv3T = model.MPP()
                elif oper == 'CM':
                    ln, iv3T = model.CM()
                elif oper[:2] == 'VM':               
                     ln, iv3T = model.VM(bot,top)
                else:
                    iv3T = IV3T('bogus')
                    iv3T.Ptot[0] = 0
                Pmax = iv3T.Ptot[0]
            else:
                Pmax = 0.

            self.outPower[i] = Pmax * self.NTime[i] * 10000.

        EY = sum(self.outPower) * self.DayTime * 365.25/1000   #kWh/m2/yr
        EYeff = EY / self.YearlyEnergy
        
        return EY, EYeff

