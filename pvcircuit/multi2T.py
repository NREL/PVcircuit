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
    
    def __init__(self, name='Multi2T', TC=TC_REF, Rser=0., Eg_list=[1.8,1.4], Jext=0.014, \
                 n=[1,2], J0ratio = [10., 10.]):
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
                n=n, J0ratio = J0ratio, Jext=Jext))
                       
        self.update_now = True
        
    def __str__(self):
        '''
        format concise description of Multi2T object
        '''
         
        strout=self.name + ": <tandem.Multi2T class>"
    
        strout += '\nT = {0:.1f} C, Rser= {1:g} Ω'\
            .format(self.TC,self.Rser)
            
        for i in range(self.njunc):
            strout += '\n\n'+str(self.j[i])
            
        return strout
    
    def __repr__(self):
        return str(self)
    
    #def __setattr__(self, key, value):
        #super(Multi2T, self).__setattr__(key, value)

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
 
    def V2T(self,J):
        '''
        calcuate V(J) of 2T multijunction
        '''   
        
        J=np.float64(J)
        for i in range(self.njunc):
            if i > 0:
                self.j[i].JLC = self.j[i].beta * self.j[i-1].Jem(self.Vmid[i-1])    # previous LC
            else:
                self.j[i].JLC = 0.    # no LC in top junction
                
            self.Vmid[i]  = self.j[i].Vdiode(J) 
            
        return np.sum(self.Vmid) + J * self.Rser
    
    def J2T(self, V):
        '''
        calculate J(V) of 2T multijunction
        using Dan's algorithm
        '''

        V = np.float64(V)

        #find max photocurrent at Voc
        Voc = self.Voc
        Jphoto_list = self.proplist('Jphoto')
        Jmax = max(Jphoto_list)            
        
        stepratio = 10
        count = 0
        if V <= Voc:   #Voc toward Jsc
            Jdelta = - Jmax/stepratio
        else:   #Voc toward forward
            Jdelta = Jmax
            
        Jdelta_start = Jdelta
        Jold = 0.   #start from Voc
        Jtrace=[Jold]
        while abs(Jdelta/Jdelta_start) > 1e-6 :
            Jnew = Jold + Jdelta
            Vnew = self.V2T(Jnew)
            if not math.isfinite(Vnew):
                Jdelta /= stepratio                
            elif (Vnew < V) and (V <= Voc):
                Jdelta /= stepratio
            elif (Vnew > V) and (V > Voc):
                Jdelta /= stepratio
            else:
                Jold = Jnew
                Jtrace.append(Jold)
            
            count +=1
            if count > MAXITER: 
                Jold = np.nan
                break
        
        return Jold
    
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
    def Jsc(self):
        return abs(self.J2T(0.))
       
    
    def MPP(self, pnts=11, bplot=False):
        # calculate maximum power point and associated IV, Vmp, Jmp, FF     
        #res=0.001   #voltage resolution
 
        ts = time()
        Jlo = -self.Jsc
        Jhi = 0.    #1mA forward
        #ndarray functions
        V2Tvect = np.vectorize(self.V2T)
      
        Jext_list = self.proplist('Jext')  #list external photocurrents at Voc
        if math.isclose(max(Jext_list), 0., abs_tol=1e-6) :
             self.Pmp = np.nan
             self.Vmp = np.nan
             self.Jmp = np.nan
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
                Jtemp = np.linspace(Jlo, Jhi, pnts)
                Vtemp = np.array([self.V2T(J) for J in Jtemp])
                Vtemp = V2Tvect(Jtemp)
                Ptemp = np.array([(-v*j) for v, j in zip(Vtemp, Jtemp)])
                nmax = np.argmax(Ptemp)
                if bplot:
                    ax.plot(Vtemp, Jtemp, marker='.', ls='')
                    axr.plot(Vtemp, Ptemp, marker='.', ls='', c='cyan')              
                    print(nmax, Jlo, Jhi, Ptemp[nmax])
                Jlo = Jtemp[max(0,(nmax-1))]
                Jhi = Jtemp[min((nmax+1),(pnts-1))]
 
                
            self.Pmp = Ptemp[nmax]
            self.Vmp = Vtemp[nmax]
            self.Jmp = abs(Jtemp[nmax])
            self.FF = abs((self.Vmp * self.Jmp) / (self.Voc * self.Jsc))
            
            self.Vpoints = [0., self.Vmp, self.Voc]
            self.Jpoints = [-self.Jsc, -self.Jmp, 0.]
            if bplot: 
                ax.plot(self.Vpoints,self.Jpoints,\
                marker='x',ls='', ms=12, c='black')  #special points
                axr.plot(self.Vmp, self.Pmp, marker='o', fillstyle='none', ms=12, c='black')
        
        mpp_dict = {"Voc":self.Voc, "Jsc":self.Jsc, "Vmp":self.Vmp, \
                    "Jmp":self.Jmp, "Pmp":self.Pmp,  "FF":self.FF}

        te = time()
        ds=(te-ts)
        print(f' {ds:2.4f} s')
        
        return mpp_dict
               
                                            
    def plot(self,title='', Vmin= -0.5, pnts=21, pplot=False):
        #plot a light IV of Multi2T
        
        ts = time()
        Jext_list = self.proplist('Jext') #remember list external photocurrents 
        Jmax = max(Jext_list)
        Eg_list = self.proplist('Eg') #list of Eg 
        Egmax = sum(Eg_list)

        #ndarray functions
        V2Tvect = np.vectorize(self.V2T)
        J2Tvect = np.vectorize(self.J2T)
        
        if self.name :
            title += self.name 
            
        # calc light JV
        if not math.isclose(Jmax, 0., abs_tol=1e-6) :
           
            self.MPP()   #determine max power point
            #horizonal portion
            VxV = np.linspace(Vmin, self.Voc, pnts)
            #JxV = np.array([self.J2T(V) for V in VxV])
            JxV = J2Tvect(VxV)
            #vertical portion
            JxJ = np.linspace(-Jmax, Jmax*2, pnts)
            #VxJ = np.array([self.V2T(J) for J in JxJ])
            VxJ = V2Tvect(JxJ)
            #combine
            Vboth = np.concatenate((VxV,VxJ),axis=None)
            Jboth = np.concatenate((JxV,JxJ),axis=None)
            #sort
            p = np.argsort(Vboth)
            Vlight = Vboth[p]
            Jlight = Jboth[p]
            Plight = np.array([(-v*j) for v, j in zip(Vlight,Jlight)])
            
        # calc dark JV
        self.update(Jext = 0., JLC = 0.)   # turn lights off
        lolog = -8
        hilog = 2
        pdec = 3
        dpnts=((hilog-lolog)*pdec+1)
        Jdark = np.logspace(lolog, hilog, num=dpnts)
        Vdark = np.full(dpnts, np.nan, dtype=np.float64) # Vtotal
        Vdarkmid = np.full((dpnts,self.njunc), np.nan, dtype=np.float64) # Vmid[pnt, junc]
        for i, J in enumerate(Jdark):
            Vdark[i] = self.V2T(J)  # also sets self.Vmid[i]
            for junc in range(self.njunc):
                Vdarkmid[i,junc] = self.Vmid[junc]       
        #self.Vdark = np.array([self.V2T(J) for J in self.Jdark])
        #self.Vdark = V2Tvect(self.Jdark)
        self.update(Jext = Jext_list, JLC = 0.)  # turn lights back on
                
        #dark plot
        dfig, dax = plt.subplots()
        dax.plot(Vdark, Jdark, lw=2, c='black')  #JV curve
        for junc in range(Vdarkmid.shape[1]):  #plot Vdiode of each junction
            dax.plot(Vdarkmid[:, junc], Jdark, lw=2)
                 
        dax.set_yscale("log") #logscale   
        dax.set_xlim(0, 4.3) #Egmax*1.1)
        dax.grid(color='gray')
        dax.set_title(title + ' Dark')  # Add a title to the axes.
        dax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        dax.set_ylabel('Current Density (A/cm2)')  # Add a y-label to the axes.
     
        # light plot        
        lfig, lax = plt.subplots()
        lax.plot(Vdark, Jdark, lw=2, c='green')  # dark JV curve
        lax.plot(Vlight, Jlight, lw=2, c='red')  #JV curve         
        lax.plot(self.Vpoints,self.Jpoints,\
                marker='x',ls='', ms=12, c='black')  #special points
        if pplot:  # power curve
            laxr = lax.twinx()
            laxr.plot(Vlight, Plight,ls='--',c='cyan')
            laxr.set_ylabel('Power (W)',c='cyan')
        lax.set_xlim( (Vmin-0.1), min(Egmax,self.Voc*1.1))
        lax.set_ylim(-Jmax*1.5,Jmax*1.5)
        lax.set_title(title + ' Dark')  # Add a title to the axes.
        lax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        lax.set_ylabel('Current Density (A/cm2)')  # Add a y-label to the axes.
        lax.axvline(0, ls='--', c='gray')
        lax.axhline(0, ls='--', c='gray')
        
        # annotate
        snote = 'T = {0:.1f} C, Rser = {1:g} Ω'.format(self.TC, self.Rser) 
        snote += '\nEg = '+str(Eg_list) + ' eV'
        snote += '\nJext = '+str(Jext_list*1000) + ' mA/cm2'
        snote += '\nVoc = {0:.2f} V, Jsc = {1:.1f} mA/cm2\nFF = {2:.1f}%, Pmp = {3:.1f} mW'\
            .format(self.Voc, self.Jsc*1000, self.FF*100, self.Pmp*1000)
            
        lax.text(Vmin+0.1,Jmax/2,snote,bbox=dict(facecolor='white'))
            
        te = time()
        ds=(te-ts)
        print(f' {ds:2.4f} s')

        return dfig, lfig, dax, lax
        
 