# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.Multi2T()    # properties of a 2T multijunction with arbitrary junctions
"""

import math   #simple math
from time import time
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants
from pvcircuit.junction import *
               
class Multi2T(object): 
    '''
    Multi2T class for optoelectronic model of two terminal multijunction
    '''
    update_now = True 
    
    def __init__(self, name='Multi2T', TC=TC_REF, Rser=0., area=1., Jext=0.014, \
                  Eg_list=[1.8,1.4], n=[1,2], J0ratio = [10., 10.]):
        # user inputs
          
        self.update_now = False
        
        self.name = name
        self.TC = TC
        self.Rser = Rser 
        self.njunc = len(Eg_list)
        CM = .03 / self.njunc
        self.Vmid = np.full(self.njunc, np.nan, dtype=np.float64)   #subcell voltages
        self.j = list()   #empty list of junctions
        for i, Eg in enumerate(Eg_list):
            jname='j['+str(i)+']'
            self.j.append(Junction(name=jname, Eg=Eg, TC=TC, \
                n=n, J0ratio = J0ratio, Jext=Jext, area=area))
                       
        self.update_now = True


    def copy(self):
        '''
        create a separate complete copy of a junction
        '''
        return copy.deepcopy(self)

    def copy3T(dev3T):
        '''
        create a Multi2T object that contains the values from a Tandem3T object
        input dev3T is Tandem3T object
        output is a Multi2T object
        '''
        top = dev3T.top
        bot = dev3T.bot
        
        dev2T = Multi2T(name=dev3T.name, TC=dev3T.TC, Eg_list=[top.Eg, bot.Eg])
        dev2T.j[0] = dev3T.top.copy()
        dev2T.j[1] = dev3T.bot.copy()
        dev2T.update(Rser = (top.Rser * top.totalarea + bot.Rser * bot.totalarea) / dev3T.totalarea)
        dev2T.j[0].Rser = 0.
        dev2T.j[1].Rser = 0.
        
        return dev2T  
            
    def __str__(self):
        '''
        format concise description of Multi2T object
        '''
         
        strout=self.name + ": <tandem.Multi2T class>"
    
        strout += '\nT = {0:.1f} C, Rser= {1:g} Ω cm2'\
            .format(self.TC,self.Rser)
            
        for i in range(self.njunc):
            strout += '\n\n'+str(self.j[i])
            
        return strout
    
    def __repr__(self):
        return str(self)
    
    #def __setattr__(self, key, value):
        #super(Multi2T, self).__setattr__(key, value)

    @property
    def lightarea(self):
        # largest junction light area
        areas = self.proplist('lightarea')
        return max(areas)

    @property
    def totalarea(self):
        # largest junction light area
        areas = self.proplist('totalarea')
        return max(areas)

    def update(self, **kwargs):
         for key, value in kwargs.items():
            if key == 'name':
                self.__dict__[key] = str(value)
            elif key == 'njunc':
                self.__dict__[key] = int(value)
            elif key == 'Vmid':
                self.__dict__[key] = np.array(value)
            elif key in ['j', 'update_now']: 
                self.__dict__[key] = value
            elif key in ['Rser', 'TC']:
                self.__dict__[key] = np.float64(value)
            
            if self.update_now:
                jlist = Junction.ATTR.copy()
                jlist.remove('Rser')
                if key in jlist :                    
                    for i, junc in enumerate(self.j):
                        if np.isscalar(value):
                            junc.__dict__[key] = value
                        else:
                            junc.__dict__[key] = value[i]
 
    def V2T(self,I):
        '''
        calcuate V(J) of 2T multijunction
        '''   
        
        I=np.float64(I)
        for i in range(self.njunc):
            if i > 0:    # previous LC
                self.j[i].JLC = self.j[i].beta * self.j[i-1].Jem(self.Vmid[i-1])
                if self.j[i-1].totalarea < self.j[i].totalarea: # distribute LC over total area
                    self.j[i].JLC *= self.j[i-1].totalarea / self.j[i].totalarea
                
            else:
                self.j[i].JLC = 0.    # no LC in top junction
                
            self.Vmid[i]  = self.j[i].Vdiode(I/self.j[i].totalarea) 
            
        return np.sum(self.Vmid) + self.Rser * I / self.totalarea
    
    def I2T(self, V):
        '''
        calculate J(V) of 2T multijunction
        using Dan's algorithm
        '''

        V = np.float64(V)

        #find max photocurrent at Voc
        Voc = self.Voc
        Jphotos = self.proplist('Jphoto')
        areas = self.proplist('lightarea')
        Imax = max([j*a for j,a in zip(Jphotos,areas)])            
        
        stepratio = 10
        count = 0
        if V <= Voc:   #Voc toward Jsc
            Idelta = - Imax/stepratio
        else:   #Voc toward forward
            Idelta = Imax
            
        Idelta_start = Idelta
        Iold = 0.   #start from Voc
        Itrace=[Iold]
        while abs(Idelta/Idelta_start) > 1e-7 :
            Inew = Iold + Idelta
            Vnew = self.V2T(Inew)
            if not math.isfinite(Vnew):
                Idelta /= stepratio                
            elif (Vnew < V) and (V <= Voc):
                Idelta /= stepratio
            elif (Vnew > V) and (V > Voc):
                Idelta /= stepratio
            else:
                Iold = Inew
                Itrace.append(Iold)
            
            count +=1
            if count > MAXITER: 
                Iold = np.nan
                break
        
        return Iold
    
    def proplist(self, key):
        #list of junction properties
        out = np.array([])
        for junc in self.j :
            try:
                value = getattr(junc, key)
            except:
                value = np.nan
                
            out = np.append(out, value)   
            
        return out
                    
    @property 
    def Voc(self):
        return self.V2T(0.)

    @property
    def Isc(self):
        return abs(self.I2T(0.))
       
    
    def MPP(self, pnts=11, bplot=False):
        # calculate maximum power point and associated IV, Vmp, Imp, FF     
        #res=0.001   #voltage resolution
 
        ts = time()
        Ilo = -self.Isc
        Ihi = 0.    #1mA forward
        #ndarray functions
        V2Tvect = np.vectorize(self.V2T)
      
        Jext_list = self.proplist('Jext')  #list external photocurrents at Voc
        if math.isclose(max(Jext_list), 0., abs_tol=1e-6) :
             self.Pmp = np.nan
             self.Vmp = np.nan
             self.Imp = np.nan
             self.FF = np.nan

        else:
            if bplot: # debug plot
                fig, ax = plt.subplots()
                ax.axhline(0, color='gray')
                ax.axvline(0, color='gray')
                ax.set_title(self.name + ' MPP')
                ax.set_xlabel('Voltage (V)')
                ax.set_ylabel('Current Density (A/cm2)')
                axr = ax.twinx()
                axr.set_ylabel('Power (W)',c='cyan')
        
            for i in range(5):
                Itemp = np.linspace(Ilo, Ihi, pnts)
                Vtemp = np.array([self.V2T(I) for I in Itemp])
                Vtemp = V2Tvect(Itemp)
                Ptemp = np.array([(-v*j) for v, j in zip(Vtemp, Itemp)])
                nmax = np.argmax(Ptemp)
                if bplot:
                    ax.plot(Vtemp, Itemp, marker='.', ls='')
                    axr.plot(Vtemp, Ptemp, marker='.', ls='', c='cyan')              
                    print(nmax, Ilo, Ihi, Ptemp[nmax])
                Ilo = Itemp[max(0,(nmax-1))]
                Ihi = Itemp[min((nmax+1),(pnts-1))]
 
                
            self.Pmp = Ptemp[nmax]
            self.Vmp = Vtemp[nmax]
            self.Imp = abs(Itemp[nmax])
            self.FF = abs((self.Vmp * self.Imp) / (self.Voc * self.Isc))
            
            self.Vpoints = np.array([0., self.Vmp, self.Voc])
            self.Ipoints = np.array([-self.Isc, -self.Imp, 0.])
            if bplot: 
                ax.plot(self.Vpoints,self.Ipoints,\
                marker='x',ls='', ms=12, c='black')  #special points
                axr.plot(self.Vmp, self.Pmp, marker='o', fillstyle='none', ms=12, c='black')
        
        mpp_dict = {"Voc":self.Voc, "Isc":self.Isc, "Vmp":self.Vmp, \
                    "Imp":self.Imp, "Pmp":self.Pmp,  "FF":self.FF}

        te = time()
        ds=(te-ts)
        print(f' {ds:2.4f} s')
        
        return mpp_dict
               
                                            
    def plot(self,title='', Vmin= -0.5, pnts=21, pplot=False):
        #plot a light IV of Multi2T
        
        ts = time()
        Jext_list = self.proplist('Jext') #remember list external photocurrents 
        areas = self.proplist('lightarea')  #list of junction areas
        Imax = max([j*a for j,a in zip(Jext_list,areas)])            
        Eg_list = self.proplist('Eg') #list of Eg 
        Egmax = sum(Eg_list)
        scale = 1000.

        #ndarray functions
        V2Tvect = np.vectorize(self.V2T)
        I2Tvect = np.vectorize(self.I2T)
        
        if self.name :
            title += self.name 
            
        # calc light IV
        if not math.isclose(Imax, 0., abs_tol=1e-6) :
           
            self.MPP()   #determine max power point
            #horizonal portion
            VxV = np.linspace(Vmin, self.Voc, pnts)
            IxV = I2Tvect(VxV)
            #vertical portion
            IxI = np.linspace(-Imax, Imax*2, pnts)
            VxI = V2Tvect(IxI)
            #combine
            Vboth = np.concatenate((VxV,VxI),axis=None)
            Iboth = np.concatenate((IxV,IxI),axis=None)
            #sort
            p = np.argsort(Vboth)
            Vlight = Vboth[p]
            Ilight = Iboth[p]
            Plight = np.array([(-v*j) for v, j in zip(Vlight,Ilight)])
            Vlight = np.array(Vlight)
            Ilight = np.array(Ilight) 
                     
        # calc dark IV
        self.update(Jext = 0., JLC = 0.)   # turn lights off
        lolog = -8
        hilog = 2
        pdec = 3
        dpnts=((hilog-lolog)*pdec+1)
        Idark = np.logspace(lolog, hilog, num=dpnts)
        Vdark = np.full(dpnts, np.nan, dtype=np.float64) # Vtotal
        Vdarkmid = np.full((dpnts,self.njunc), np.nan, dtype=np.float64) # Vmid[pnt, junc]
        for ii, I in enumerate(Idark):
            Vdark[ii] = self.V2T(I)  # also sets self.Vmid[i]
            for junc in range(self.njunc):
                Vdarkmid[ii,junc] = self.Vmid[junc]       
        self.update(Jext = Jext_list, JLC = 0.)  # turn lights back on
                
        #dark plot
        dfig, dax = plt.subplots()
        dax.plot(Vdark, Idark, lw=2, c='black')  #IV curve
        for junc in range(Vdarkmid.shape[1]):  #plot Vdiode of each junction
            dax.plot(Vdarkmid[:, junc], Idark, lw=2)
                 
        dax.set_yscale("log") #logscale   
        dax.set_xlim(0, Egmax*1.1)
        dax.grid(color='gray')
        dax.set_title(title + ' Dark')  # Add a title to the axes.
        dax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        dax.set_ylabel('Current (A)')  # Add a y-label to the axes.
     
        # light plot        
        lfig, lax = plt.subplots()
        lax.plot(Vdark, Idark*scale, lw=2, c='green')  # dark IV curve
        lax.plot(Vlight, Ilight*scale, lw=2, c='red')  #IV curve         
        lax.plot(self.Vpoints,self.Ipoints*scale,\
                marker='x',ls='', ms=12, c='black')  #special points
        if pplot:  # power curve
            laxr = lax.twinx()
            laxr.plot(Vlight, Plight*scale,ls='--',c='cyan',zorder=0)
            laxr.set_ylabel('Power (mW)',c='cyan')
        lax.set_xlim( (Vmin-0.1), min(Egmax,self.Voc*1.1))
        lax.set_ylim(-Imax*1.5*scale,Imax*1.5*scale)
        lax.set_title(title + ' Dark')  # Add a title to the axes.
        lax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        lax.set_ylabel('Current (mA)')  # Add a y-label to the axes.
        lax.axvline(0, ls='--', c='gray')
        lax.axhline(0, ls='--', c='gray')
        
        # annotate
        snote = 'T = {0:.1f} C, Rser = {1:g} Ω cm2, A = {2:g} cm2'.format(self.TC, self.Rser, self.lightarea) 
        snote += '\nEg = '+str(Eg_list) + ' eV'
        snote += '\nJext = '+str(Jext_list*1000) + ' mA/cm2'
        snote += '\nVoc = {0:.2f} V, Isc = {1:.1f} mA/cm2\nFF = {2:.1f}%, Pmp = {3:.1f} mW'\
            .format(self.Voc, self.Isc*1000, self.FF*100, self.Pmp*1000)
            
        lax.text(Vmin+0.1,Imax/2,snote,zorder=5,bbox=dict(facecolor='white'))
            
        te = time()
        ds=(te-ts)
        print(f' {ds:2.4f} s')

        return dfig, lfig, dax, lax
        
 