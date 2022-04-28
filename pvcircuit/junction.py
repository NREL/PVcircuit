# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.Junction()   # properties and methods for each junction
"""

import math   #simple math
import copy
from time import time
from functools import lru_cache
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from parse import *
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants
import ipywidgets as widgets
from IPython.display import display

# constants
k_q = con.k/con.e
DB_PREFIX = 2. * np.pi * con.e * (con.k/con.h)**3 / (con.c)**2 /1.e4    #about 1.0133e-8 for Jdb[A/cm2]
nan=np.nan

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

@lru_cache(maxsize = 100)
def TK(TC): return TC + con.zero_Celsius
    #convert degrees celcius to kelvin
    
@lru_cache(maxsize = 100)
def Vth(TC): return k_q * TK(TC)
    #Thermal voltage in volts = kT/q

@lru_cache(maxsize = 100)
def Jdb(TC, Eg):
    #detailed balance saturation current

    EgkT = Eg / Vth(TC)
    
    #Jdb from Geisz et al.
    return DB_PREFIX * TK(TC)**3. * (EgkT*EgkT + 2.*EgkT + 2.) * np.exp(-EgkT)    #units from DB_PREFIX
        
class Junction(object):
    """
    Class for PV junctions.

    :param Rs: series resistance [ohms]
    """
    
    ATTR = ['Eg','TC','Gsh','Rser','lightarea','totalarea','Jext','JLC','beta','gamma','pn'
            ,'Jphoto','TK','Jdb']          
    ARY_ATTR = ['n','J0ratio','J0']
    J0scale = 1000. # mA same as Igor, changes J0ratio because of units

    def __init__(self, name='junc', Eg=Eg_DEFAULT, TC=TC_REF, \
                 Gsh=0., Rser=0., area=AREA_DEFAULT, \
                 n=[1.,2.], J0ratio=[10.,10.], J0ref=None, \
                 RBB=None, Jext=0.04, JLC=0., \
                 pn=-1, beta=BETA_DEFAUlT, gamma=0. ):
        
        self.ui = None  
        self.debugout = widgets.Output() # debug output
        self.RBB_dict = {}

        # user inputs
        self.name = name    # remember my name
        self.Eg = np.float64(Eg)  #: [eV] junction band gap
        self.TC = np.float64(TC)  #: [C] junction temperature
        self.Jext = np.float64(Jext)   #: [A/cm2] photocurrent density
        self.Gsh = np.float64(Gsh)  #: [mho] shunt conductance=1/Rsh
        self.Rser = np.float64(Rser)  #: [ohm] series resistance
        self.lightarea = np.float64(area)   # [cm2] illuminated junction area
        self.totalarea = np.float64(area)   # [cm2] total junction area including shaded areas
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
         
        self.set(RBB=RBB) 

    def copy(self):
        '''
        create a copy of a Junction
        need deepcopy() to separate lists, dicts, etc but crashes
        '''
        
        return copy.copy(self)

    def __str__(self):
        #attr_list = self.__dict__.keys()
        #attr_dict = self.__dict__.items()
        #print(attr_list)
        
        strout = self.name+": <pvcircuit.junction.Junction class>"
                    
        strout += '\nEg = {0:.2f} eV, TC = {1:.1f} C' \
            .format(self.Eg, self.TC)
            
        strout += '\nJext = {0:.1f} , JLC = {1:.1f} mA/cm2' \
            .format( self.Jext*1000., self.JLC*1000.)

        strout += '\nGsh = {0:g} S/cm2, Rser = {1:g} Ωcm2' \
            .format(self.Gsh, self.Rser)
            
        strout += '\nlightA = {0:g} cm2, totalA = {1:g} cm2' \
            .format(self.lightarea, self.totalarea)
            
        strout += '\npn = {0:d}, beta = {1:g}, gamma = {2:g}' \
            .format(self.pn, self.beta, self.gamma, self.JLC)

        strout += '\n {0:^5s} {1:^10s} {2:^10s}' \
            .format('n','J0ratio', 'J0(A/cm2)')
        strout += '\n {0:^5s} {1:^10.0f} {2:^10.3e}' \
            .format('db', 1., self.Jdb)

        i=0
        for ideality_factor,ratio, saturation_current in zip(self.n, self.J0ratio, self.J0):
            strout += '\n {0:^5.2f} {1:^10.2f} {2:^10.3e}' \
                .format(self.n[i], self.J0ratio[i], self.J0[i])
            i+=1
        
        if self.RBB_dict['method'] :
            strout+=' \nRBB_dict: '+str(self.RBB_dict)
 
        return strout

    def __repr__(self):
        return str(self)

    '''
    def __setattr__(self, key, value):
        # causes problems
        super(Junction, self).__setattr__(key, value) 
        self.set(key = value)
    '''
    
    def update(self):
        # update Junction self.ui controls

        if self.ui:  # junction user interface has been created
            if self.RBB_dict:
                if self.RBB_dict['method']:
                    RBB_keys =  list(self.RBB_dict.keys())
                else:
                    RBB_keys = []

            cntrls = self.ui.children
            for cntrl in cntrls:
                desc = cntrl._trait_values.get('description','nodesc')  # control description
                cval = cntrl._trait_values.get('value','noval')  # control value
                if desc == 'nodesc' or cval == 'noval':
                    break
                elif desc.endswith(']') and desc.find('[') > 0 :
                    key, ind = parse('{}[{:d}]',desc)
                else:
                    key = desc
                    ind = None
 
                if key in self.ATTR:   # Junction scalar controls to update
                    attrval = getattr(self, key)  # current value of attribute
                    if cval != attrval:
                        with self.debugout: print('Jupdate: ' + desc, attrval)
                        cntrl.value = attrval
                elif key in self.ARY_ATTR:   # Junction array controls to update
                    attrval = getattr(self, key)  # current value of attribute
                    if type(ind) is int:
                        if type(attrval) is np.ndarray:
                            if cval != attrval[ind]:
                                with self.debugout: print('Jupdate: ' + desc, attrval[ind])
                                cntrl.value = attrval[ind]
                elif key in RBB_keys:
                    attrval = self.RBB_dict[key]
                    if cval != attrval:
                        with self.debugout: print('Jupdate: ' + desc, attrval)
                        cntrl.value = attrval  
                                              
    def set(self, **kwargs):
        # controlled update of Junction attributes

        #with self.debugout: print('Jset: ', list(kwargs.keys()))
                        
        for testkey, value in kwargs.items():
            if testkey.endswith(']') and testkey.find('[') > 0 :
                key, ind = parse('{}[{:d}]',testkey)   #set one element of array e.g. 'n[0]'
            else:
                key = testkey
                ind = None

            if self.RBB_dict:
                if self.RBB_dict['method']:
                    RBB_keys =  list(self.RBB_dict.keys())
                else:
                    RBB_keys = []
                
            if key == 'RBB' or key == 'method':
                # this change requires redrawing self.ui
                if value == 'JFG': # RBB shortcut
                    self.__dict__['RBB_dict'] =  {'method':'JFG', 'mrb':10., 'J0rb':0.5, 'Vrb':0.}
                elif value == 'bishop':
                    self.__dict__['RBB_dict'] = {'method':'bishop','mrb':3.28, 'avalanche':1., 'Vrb':-5.5}
                else:
                    self.__dict__['RBB_dict'] =  {'method': None}  #no RBB                     
                if self.ui:  # junction user interface has been created
                    #ui = self.controls()    # redraw junction controls 
                    pass                                   
            elif key in RBB_keys: #RBB parameters
                self.RBB_dict[key] = np.float64(value) 
            elif key == 'area': # area shortcut
                self.__dict__['lightarea'] = np.float64(value) 
                self.__dict__['totalarea'] = np.float64(value) 
            elif key == 'name': # strings
                self.__dict__[key] = str(value)
            elif key == 'pn': # integers
                self.__dict__[key] = int(value)
            elif key == 'RBB_dict':
                self.__dict__[key] = value
            elif key in ['n','J0ratio']: # diode parameters (array)
                if type(ind) is int and np.isscalar(value) :
                    attrval = getattr(self, key)  # current value of attribute
                    localarray = attrval.copy()
                    if type(localarray) is np.ndarray:
                        if ind < localarray.size:
                            localarray[ind] = np.float64(value) #add new value
                            self.__dict__[key] = localarray
                            with self.debugout: print('scalar',key, ind, localarray)
                else:
                    self.__dict__[key] = np.array(value)
                    with self.debugout: print('array', key, value)
            elif key in self.ATTR: # scalar float
                self.__dict__[key] = np.float64(value)
            else:
                with self.debugout: print('no Junckey',key)
                
    @property
    def Jphoto(self): return self.Jext * self.lightarea / self.totalarea + self.JLC 
        # total photocurrent
        # external illumination is distributed over total area
               
    @property
    def TK(self): 
        #temperature in (K)
        return TK(self.TC)

    @property
    def Vth(self): 
        #Thermal voltage in volts = kT/q
        return Vth(self.TC)

    @property
    def Jdb(self): 
        #detailed balance saturation current
        return Jdb(self.TC, self.Eg)
    
    @property
    def J0(self):
        #dynamically calculated J0(T)
        #return np.ndarray [J0(n0), J0(n1), etc]
    
        if (type(self.n) is np.ndarray) and (type(self.J0ratio) is np.ndarray):
            if self.n.size == self.J0ratio.size:
                return (self.Jdb * self.J0scale)**(1./self.n) * self.J0ratio / self.J0scale 
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
                self.J0ratio = self.J0scale * J0ref / (self.Jdb * self.J0scale)**(1./self.n)
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

    def notdiode(self):
        '''
        is this junction really a diode
        or just a resistor
        sum(J0) = 0 -> not diode
        pn = 0 -> not diode
        '''
        if self.pn == 0:
            return True
            
        jsum = np.float64(0.)
        for saturation_current in self.J0:
            jsum +=saturation_current
            
        return (jsum == np.float64(0.))
        
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

        if self.notdiode():  # sum(J0)=0 -> no diode
            return Jtot

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

        if self.notdiode():  # sum(J0)=0 -> no diode
            return 0.

        Jtot = self.Jphoto + Jdiode
        
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

        if self.notdiode():  # sum(J0)=0 -> no diode
            return 0.
 
        try:        
            Vmid = brentq(self._dV, -VLIM_REVERSE, VLIM_FORWARD, args=(Vtot),
                           xtol=VTOL, rtol=EPSREL, maxiter=MAXITER,
                           full_output=False, disp=True)
           
        except:
            return np.nan
            #print("Exception:",err)
      
        return Vmid 
 
    def controls(self):
        '''
        use interactive_output for GUI in IPython
        '''
        
        cell_layout = widgets.Layout(display='inline_flex',
                            flex_flow='row',
                            justify_content='flex-end',
                            width='300px')  
        # controls 
        in_name = widgets.Text(value=self.name,description='name',layout=cell_layout, 
                    continuous_update=False)                        
        in_Eg = widgets.FloatSlider(value=self.Eg, min=0.1,max=3.0,step=0.01,
            description='Eg',layout=cell_layout,readout_format='.2f')
        in_TC = widgets.FloatSlider(value=self.TC, min=-40, max=200.,step=2,
            description='TC',layout=cell_layout,readout_format='.1f')
        in_Jext = widgets.FloatSlider(value=self.Jext, min=0., max=.080,step=0.001,
            description='Jext',layout=cell_layout,readout_format='.4f')
        in_JLC = widgets.FloatSlider(value=self.JLC, min=0., max=.080,step=0.001,
            description='JLC',layout=cell_layout,readout_format='.4f',disabled=True)
        in_Gsh = widgets.FloatLogSlider(value=self.Gsh, base=10, min=-10, max=3 ,step=0.01,
            description='Gsh',layout=cell_layout,readout_format='.2e')
        in_Rser= widgets.FloatLogSlider(value=self.Rser, base=10, min=-7, max=3, step=0.01,
            description='Rser',layout=cell_layout,readout_format='.2e')           
        in_lightarea = widgets.FloatLogSlider(value=self.lightarea, base=10, min=-6, max=3.,step=0.1,
            description='lightarea',layout=cell_layout)
        in_totalarea = widgets.FloatSlider(value=self.totalarea, min=self.lightarea, max=1e3, step=0.1,
            description='totalarea',layout=cell_layout)
        in_beta = widgets.FloatSlider(value=self.beta, min=0., max=50.,step=0.1,
            description='beta',layout=cell_layout)
        in_gamma = widgets.FloatSlider(value=self.gamma, min=0., max=3.0, step=0.1,
            description='gamma',layout=cell_layout,readout_format='.2e')
        in_pn = widgets.IntSlider(value=self.pn, min=-1, max=1, step=1,
            description='pn',layout=cell_layout)
            
        #linkages
        arealink = widgets.jslink((in_lightarea,'value'), (in_totalarea,'min')) #also jsdlink works
            
        attr = ['name']+self.ATTR.copy()
        cntrls = [in_name, in_Eg,in_TC,in_Gsh,in_Rser,in_lightarea,in_totalarea,
                in_Jext,in_JLC,in_beta,in_gamma,in_pn]
        sing_dict = dict(zip(attr,cntrls))
        #singout = widgets.interactive_output(self.set, sing_dict)  #all at once

        def on_juncchange(change):
            # function for changing values
            old = change['old'] #old value
            new = change['new'] #new value
            owner = change['owner'] #control
            value = owner.value
            desc = owner.description            
              
            if new == old:
                with self.debugout: print('Jcontrol: ' + desc + '=', value)
            else:
                with self.debugout: print('Jcontrol: ' + desc + '->', value)
                self.set(**{desc:value})
                
            #iout.clear_output()
            #with iout: print(self)

                

        # diode array
        in_tit = widgets.Label(value='Junction', description='Junction')
        in_diodelab = widgets.Label(value='diodes:', description='diodes:')
        diode_layout = widgets.Layout(flex_flow='column',align_items='center')    
        
        cntrls.append(in_diodelab)
        in_n = []  # empty list of n controls
        in_ratio = [] # empty list of Jratio controls
        hui = []
        diode_dict = {} 
        for i in range(len(self.n)):
            in_n.append(widgets.FloatLogSlider(value=self.n[i], base=10, min=-1, max=1, step=0.001,
                description='n['+str(i)+']',layout=cell_layout))
            in_ratio.append(widgets.FloatLogSlider(value=self.J0ratio[i], base=10, min=-6, max=6, step=0.1,
                description='J0ratio['+str(i)+']',layout=cell_layout))
            cntrls.append(in_n[i])
            cntrls.append(in_ratio[i])
            diode_dict['n['+str(i)+']'] = in_n[i]
            diode_dict['J0ratio['+str(i)+']'] = in_ratio[i]  
            #hui.append(widgets.HBox([in_n[i],in_ratio[i]])) 
            #cntrls.append(hui[i])
          
        #diodeout = widgets.interactive_output(self.set, diode_dict)  #all at once
  
        if self.RBB_dict:
            RBB_keys =  list(self.RBB_dict.keys())
            in_rbblab = widgets.Label(value='RBB:', description='RBB:')
            cntrls.append(in_rbblab) 
            in_rbb = []  # empty list of n controls
            for i, key in enumerate(RBB_keys):
                with self.debugout: print(i,key)
                if key == 'method':       
                    in_rbb.append(widgets.Dropdown(options=['','JFG','bishop'],value=self.RBB_dict[key],
                        description=key, layout=cell_layout, continuous_update=False))
                else:
                    in_rbb.append(widgets.FloatLogSlider(value=self.RBB_dict[key], base = 10, min=-10, max=5, step=0.1,
                        description=key,layout=cell_layout))
                cntrls.append(in_rbb[i])
                    
                         
        for cntrl in cntrls:
            cntrl.observe(on_juncchange,names='value')

        #output
        iout = widgets.Output()
        iout.layout.height = '5px'
        #with iout: print(self)
        cntrls.append(iout)
        
        # user interface        
        box_layout = widgets.Layout(display='flex',
                            flex_flow='column',
                            align_items='center',
                            border='1px solid black',
                            width='320px',
                            height = '350px')
                            
        ui = widgets.VBox([in_tit] + cntrls,layout=box_layout)
        self.ui = ui    # make it an attribute
        
        return ui
