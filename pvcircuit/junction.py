# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.Junction()   # properties and methods for each junction
"""

import math   #simple math
from time import time
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants

# constants
k_q = con.k/con.e
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]

# Junction defaults
Eg_DEFAULT=1.1    #[eV]
TC_REF=25.0   #[C]
AREA_DEFAULT = 1.   #[cm2] note: if A=1, then I->J
BETA_DEFAUlT = 15.  # unitless

# numerical calculation parameters
VLIM_REVERSE=10.
VLIM_FORWARD=3.
VTOL= 0.0001
EPSREL=1e-15
MAXITER=1000

def TK(TC): return TC + con.zero_Celsius
    #convert degrees celcius to kelvin
         
class Junction(object):
    """
    Class for PV junctions.

    :param Rs: series resistance [ohms]
    """
    
    ATTR = ['Eg', 'TC', 'Gsh', 'Rser', 'area', 'Jext', 'JLC', 'beta', 'gamma']          

    def __init__(self, name='junc', Eg=Eg_DEFAULT, TC=TC_REF, \
                 Gsh=0., Rser=0., area=AREA_DEFAULT, \
                 n=[1.,2.], J0ratio=[10.,10.], J0ref=None, \
                 RBB=None, Jext=0.04, JLC=0., \
                 pn=-1, beta=BETA_DEFAUlT, gamma=0. ):
        
        # user inputs
        self.name = name    # remember my name
        self.Eg = np.float64(Eg)  #: [eV] junction band gap
        self.TC = np.float64(TC)  #: [C] junction temperature
        self.Jext = np.float64(Jext)   #: [A/cm2] photocurrent density
        self.Gsh = np.float64(Gsh)  #: [mho] shunt conductance=1/Rsh
        self.Rser = np.float64(Rser)  #: [ohm] series resistance
        self.area = np.float64(area)   # [cm2] junction area
        #used for tandems only
        self.pn = int(pn)     # p-on-n=1 or n-on-p=-1
        self.beta = np.float64(beta)    # LC parameter
        self.gamma = np.float64(gamma)    # PL parameter from Lan
        self.JLC = np.float64(JLC)   # LC current from other cell JLC=beta(this)*Jem(other)
        
        # multiple diodes
        # n=1 bulk, n=m SNS, and n=2/3 Auger mechanisms
        self.n = np.array(n)   #diode ideality list e.g. [n0, n1]
        if J0ref == None:
            self.J0ratio = np.array(J0ratio)    #diode J0/Jdb^(1/n) ratio list for T dependence            
        else:
            self._J0init(J0ref)  # calculate self.J0ratio from J0ref at current self.TC
         
        if RBB == 'JFG':
            self.RBB_dict =  {'method':'JFG', 'mrb':10., 'J0rb':0.5, 'Vrb':0.}
        elif RBB == 'bishop':
            self.RBB_dict = {'method':'bishop','mrb':3.28, 'avalanche':1., 'Vrb':-5.5}
        else:
            self.RBB_dict =  {'method': None}  #no RBB

    def __str__(self):
        #attr_list = self.__dict__.keys()
        #attr_dict = self.__dict__.items()
        #print(attr_list)
        
        strout = self.name+": <tandem.Junction class>"
                    
        strout += '\n Eg = {0:.2f} eV, TC = {1:.1f} C, Jext = {2:.1f} mA/cm2' \
            .format(self.Eg, self.TC, self.Jext*1000.)

        strout += '\n Gsh = {0:g} S, Rser = {1:g} Ω, area = {2:g} cm2' \
            .format(self.Gsh, self.Rser, self.area)
            
        strout += '\n pn = {0:d}, beta = {1:g}, gamma = {2:g}, JLC = {3:.1f}' \
            .format(self.pn, self.beta, self.gamma, self.JLC)

        i=0
        strout += '\n   {0:^5s} {1:^10s} {2:^10s}' \
            .format('n','J0ratio', 'J0')
        strout += '\n   {0:^5s} {1:^10.0f} {2:^10.3e}' \
            .format('db', 1., self.Jdb)
        for ideality_factor,ratio, saturation_current in zip(self.n, self.J0ratio, self.J0):
            strout += '\n   {0:^5.2f} {1:^10.2f} {2:^10.3e}' \
                .format(self.n[i], self.J0ratio[i], self.J0[i])
            i+=1
        
        if self.RBB_dict['method'] :
            strout+=' \nRBB_dict: '+str(self.RBB_dict)
 
        return strout

    def __repr__(self):
        return str(self)

    '''def __setattr__(self, key, value):
        super(Junction, self).__setattr__(key, value) 
        self.update(key = value)
   '''
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'name':
                value = str(value)
            elif key == 'pn':
                value = int(value)
            elif key == 'RBB_dict':
                value = value
            elif key in ['n','J0ratio']:
                value = np.array(value)
            else:
                value = np.float64(value)
            
            self.__dict__[key] = value
                                
    @property
    def Jphoto(self): return self.Jext + self.JLC 
        #total photocurrent
               
    @property
    def TK(self): return TK(self.TC)

    @property
    def Vth(self): return k_q * self.TK
        #Thermal voltage in volts = kT/q

    @property
    def Jdb(self):
        """
        detailed balance saturation current
        """
 
        EgkT = self.Eg / self.Vth
        
        #Jdb from Geisz et al.
        return DB_PREFIX * self.TK**3. * (EgkT*EgkT + 2.*EgkT + 2.) * np.exp(-EgkT)    #units from DB_PREFIX
    
    @property
    def J0(self):
        '''
        dynamically calculated J0(T)
        return np.ndarray [J0(n0), J0(n1), etc]
        '''
        if (type(self.n) is np.ndarray) and (type(self.J0ratio) is np.ndarray):
            if self.n.size == self.J0ratio.size:
                #return (self.Jdb)**(1./self.n) * self.J0ratio  # ratio in A/cm2
                return (self.Jdb*1000.)**(1./self.n) * self.J0ratio / 1000. # mA same as Igor
            else:
                return np.nan   # different sizes
        else:
           return np.nan    # not numpy.ndarray
    
    def _J0init(self,J0ref):
        '''
        initialize self.J0ratio from J0ref
        return np.ndarray [J0(n0), J0(n1), etc]
        '''
        J0ref = np.array(J0ref)
        if (type(self.n) is np.ndarray) and (type(J0ref) is np.ndarray):
            if self.n.size == J0ref.size:
                self.J0ratio = J0ref / self.Jdb**(1./self.n)
                return 0   # success
            else:
                return 1   # different sizes
        else:
           return 2    # not numpy.ndarray
    
    def Jem(self,Vmid):
        '''
        light emitted from junction by reciprocity
        quantified as current density
        '''
        if Vmid > 0.:
            Jem = self.Jdb  * (np.exp(Vmid / self.Vth) - 1.)  # EL Rau
            Jem += self.gamma * self.Jphoto   # PL Lan and Green
            return Jem
        else:
            return 0.
 
    def Jmultidiodes(self,Vdiode):
        '''
        calculate recombination current density from 
        multiple diodes self.n, self.J0 numpy.ndarray
        two-diodes:
        n  = [1, 2]  #two diodes
        J0 = [10,10]  #poor cell
        detailed balance:
        n  = [1]
        J0 = [1]
        three-diodes
        n = [1, 1.8, (2/3)]
        '''     
        Jrec = np.float64(0.)
        for ideality_factor, saturation_current in zip(self.n, self.J0):
            if ideality_factor>0. and math.isfinite(saturation_current):
                try:
                    Jrec += saturation_current \
                        * (np.exp(Vdiode / self.Vth / ideality_factor) - 1.)
                except:
                    continue
     
        return Jrec        

    def JshuntRBB(self, Vdiode):
        '''
        return shunt + reverse-bias breakdown current
    
            RBB_dict={'method':None}   #None
    
            RBB_dict={'method':'JFG', mrb'':10., 'J0rb':1., 'Vrb':0.}
    
            RBB_dict={'method':'bishop','mrb'':3.28, 'avalanche':1, 'Vrb':-5.5}
    
            RBB_dict={'method':'pvmismatch','ARBD':arbd,'BRBD':brbd,'VRBD':vrb,'NRBD':nrbd:
    
        Vdiode without Rs
        Vth = kT
        Gshunt
        '''
         
        RBB_dict = self.RBB_dict
        method=RBB_dict['method']
        JRBB=np.float64(0.)
        
        if method=='JFG' :
            Vrb=RBB_dict['Vrb']
            J0rb=RBB_dict['J0rb']
            mrb=RBB_dict['mrb']
            if Vdiode <= Vrb and mrb != 0. : 
                #JRBB = -J0rb * (self.Jdb)**(1./mrb) * (np.exp(-Vdiode / self.Vth / mrb) - 1.0)
                JRBB = -J0rb * (self.Jdb*1000)**(1./mrb) / 1000. \
                   * (np.exp(-Vdiode / self.Vth / mrb) - 1.0)
            
        elif method=='bishop':
            Vrb=RBB_dict['Vrb']
            a=RBB_dict['avalanche']
            mrb=RBB_dict['mrb']
            if Vdiode <= 0. and Vrb !=0. :  
                JRBB =  Vdiode * self.Gsh  * a * (1. - Vdiode / Vrb)**(-mrb)
                 
        elif method=='pvmismatch':
            JRBB=np.float64(0.) 
            
    
        return Vdiode * self.Gsh + JRBB
    
    def Jparallel(self,Vdiode,Jtot):
        '''
        circuit equation to be zeroed to solve for Vi
        for voltage across parallel diodes with shunt and reverse breakdown
        '''

        JLED = self.Jmultidiodes(Vdiode)
        JRBB = self.JshuntRBB(Vdiode)
        #JRBB = JshuntRBB(Vdiode, self.Vth, self.Gsh, self.RBB_dict)
        return Jtot - JLED  - JRBB

    def Vdiode(self,Jdiode):
        '''
        Jtot = Jphoto + J
        for junction self of class Junction
        return Vdiode(Jtot)
        no Rseries here
        '''

        Jtot=self.Jphoto+Jdiode
        
        try: 
            Vdiode = brentq(self.Jparallel, -VLIM_REVERSE, VLIM_FORWARD, args=(Jtot),
                           xtol=VTOL, rtol=EPSREL, maxiter=MAXITER,
                           full_output=False, disp=True)
        except:
            return np.nan
            #print("Exception:",err)
                    
        return Vdiode

    def _dV(self, Vmid, Vtot):
        '''
        see singlejunction
        circuit equation to be zeroed (returns voltage difference) to solve for Vmid
        single junction circuit with series resistance and parallel diodes
        '''
        
        J = self.Jparallel(Vmid, self.Jphoto)
        dV = Vtot - Vmid  + J * self.Rser
        return dV

    def Vmid(self,Vtot):
        '''
        see Vparallel
        find intermediate voltage in a single junction diode with series resistance
        Given Vtot=Vparallel + Rser * Jparallel
        '''
 
        try:        
            Vmid = brentq(self._dV, -VLIM_REVERSE, VLIM_FORWARD, args=(Vtot),
                           xtol=VTOL, rtol=EPSREL, maxiter=MAXITER,
                           full_output=False, disp=True)
           
        except:
            return np.nan
            #print("Exception:",err)
      
        return Vmid
 
    def Jdiode(self,Vtot):
        '''
        return J for Vtot
        '''
        return  self.Jparallel(self.Vmid(Vtot),self.Jphoto)       

    #single junction cell 

    def Jcell(self,Vcell):
        #Vcell, Jcell are iterable

        Jcell=[]    #empty list
        
        for Vtot in Vcell:
             Jcell.append(-self.Jdiode(Vtot)) 
          
        return Jcell

    def Vcell(self,Jcell):
        #Vcell, Jcell are iterable

        Vcell=[]    #empty list
        
        for Jdiode in Jcell:
            try:
                Vtot=self.Vdiode(Jdiode) + Jdiode * self.Rser
            except:
                Vtot=None
                
            Vcell.append(Vtot) 
          
        return Vcell

    @property
    def Voc(self):
        #Jdiode=0
        return self.Vdiode(0.) 
        
    @property
    def Jsc(self):
        #Vtot=0
        return self.Jdiode(0.)
    
    @property
    def MPP(self):
        # calculate maximum power point and associated IV, Vmp, Jmp, FF
        
        ts = time()
        res=0.001   #voltage resolution
        if self.Jphoto > 0.: #light JV
            self.Vsingle = list(np.arange(-0.2, (self.Voc*1.02), res))
            self.Jsingle = self.Jcell(self.Vsingle)
            self.Psingle = [(-v*j) for v, j in zip(self.Vsingle,self.Jsingle)]
            nmax = np.argmax(self.Psingle)
            self.Vmp = self.Vsingle[nmax]
            self.Jmp = self.Jsingle[nmax]
            self.Pmp = self.Psingle[nmax]
            self.FF = abs((self.Vmp * self.Jmp) / (self.Voc * self.Jsc))
            
            self.Vpoints = [0., self.Vmp, self.Voc]
            self.Jpoints = [-self.Jsc, self.Jmp, 0.]
            
        else:   #dark JV
            self.Jsingle = list(np.logspace(-13., 7., num=((13+7)*3+1)))
            self.Vsingle = self.Vcell(self.Jsingle)
            self.Vmp = None
            self.Jmp = None
            self.Pmp = None
            self.FF = None
            self.Vpoints = None
            self.Jpoints = None
 
        mpp_dict = {"Voc":self.Voc, "Jsc":self.Jsc, \
                    "Vmp":self.Vmp,"Jmp":self.Jmp, \
                    "Pmp":self.Pmp, "FF":self.FF}
                        
        te = time()
        ms=(te-ts)*1000.
        #print(f'MPP: {res:g}V , {ms:2.4f} ms')
        
        return mpp_dict
                                            
    def plot(self,title=None):
        #plot a single junction
        
        if title == None:
            title = self.name
                       
        self.MPP   #generate IV curve and analysis
        
        fig, ax = plt.subplots()
        ax.plot(self.Vsingle,self.Jsingle)  #JV curve
        ax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
        ax.set_ylabel('Current Density (A/cm2)')  # Add a y-label to the axes.
        ax.grid()
        if self.Jphoto > 0.: #light JV
            ax.plot(self.Vpoints,self.Jpoints,\
                    marker='o',ls='',ms=12,c='#000000')  #special points
            #ax.scatter(self.Vpoints,self.Jpoints,s=100,c='#000000',marker='o')  #special points
            axr = ax.twinx()
            axr.plot(self.Vsingle, self.Psingle,ls='--',c='red')
            axr.set_ylabel('Power (W)')

            snote = 'Eg = {0:.2f} eV, Jpc = {1:.1f} mA/cm2, T = {2:.1f} C'\
                .format(self.Eg, self.Jphoto*1000, self.TC)
            snote += '\nGsh = {0:.1e} S, Rser = {1:g} Ω, A = {2:g} cm2 '\
                .format(self.Gsh, self.Rser, self.area)
            snote += '\nVoc = {0:.2f} V, Jsc = {1:.1f} mA/cm2, FF = {2:.1f}%'\
                .format(self.Voc, self.Jsc*1000, self.FF*100)
            ax.text(-0.2,0,snote,bbox=dict(facecolor='white'))
            ax.set_title(title + " Light")  # Add a title to the axes.
        else:
            ax.set_yscale("log")
            ax.set_title(title + " Dark")  # Add a title to the axes.
    
        return fig
