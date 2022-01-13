# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.IV3T()       # many forms of operational conditions of 3T tandems
"""

import math   #simple math
from time import time
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants
from pvcircuit.junction import *

# conversion matrices
SQ2 = math.sqrt(2.)
SQ3 = math.sqrt(3.)
SQ6 = math.sqrt(6.)
# note all matrices tranposed relative to Igor entry
M_isoC = np.transpose(np.array([[SQ3,1,SQ2],[0,2,-SQ2],[-SQ3,1,SQ2]]))	#two rotations
M_isoC /= SQ6
M_isoB = np.transpose(np.array([[1,0,0],[0,0,0],[0,-1,0]]))	#two rotations
dev2hex = M_isoB @ M_isoC	#octant viewpoint
hex2dev = dev2hex
#hex2dev = np.linalg.inv(dev2hex)   #inverse of M_isometric....singular
CR_Vload2dev = np.transpose(np.array([[1,-1,0],[-1,0,1],[0,0,0]]))
CR_Iload2dev = np.transpose(np.array([[-1,1,0],[-1,0,1],[0,0,0]]))
CR_Vdev2load = np.transpose(np.array([[0,0,0],[-1,0,0],[0,1,0]]))
CR_Idev2load = np.transpose(np.array([[0,0,0],[1,0,0],[0,1,0]]))
CT_Vload2dev = np.transpose(np.array([[0,1,-1],[1,-1,0],[0,0,0]]))
CT_Iload2dev = np.transpose(np.array([[1,0,-1],[0,1,-1],[0,0,0]]))
CT_Vdev2load = np.transpose(np.array([[0,1,0],[0,0,0],[-1,0,0]]))
CT_Idev2load = np.transpose(np.array([[1,0,0],[0,1,0],[0,0,0]]))
CZ_Vload2dev = np.transpose(np.array([[0,1,-1],[-1,0,1],[0,0,0]]))
CZ_Iload2dev = np.transpose(np.array([[1,-1,0],[0,-1,1],[0,0,0]]))
CZ_Vdev2load = np.transpose(np.array([[0,-1,0],[1,0,0],[0,0,0]]))
CZ_Idev2load = np.transpose(np.array([[1,0,0],[0,0,0],[0,1,0]]))

'''
CR:A = ZR,B = TR
	Jro = -(JB+JA)
	Jto = JB
	Jzo = JA
	Vzt = VA-VB
	Vtr = VB
	Vrz = -VA
CF:A = RF,B = ZF
CT:A = RT,B = ZT
	Jro = JA
	Jto = -(JB+JA)
	Jzo = JB
	Vzt = VB
	Vtr = -VA
	Vrz = VA-VB
CZ:A = RZ, B = TZ
	Jro = JA
	Jto = JB
	Jzo = -(JB+JA)
	Vzt = -VB
	Vtr = VB-VA
	Vrz = VA
'''

class IV3T(object):
    '''
    operational state of 3T tandem
    (Iro,Izo,Ito), (Vzt,Vrz,Vtr), etc
    '''
    
    arraykeys = ['Iro', 'Izo', 'Ito', 'Vzt', 'Vrz', 'Vtr', \
            'IA', 'IB', 'VA', 'VB', 'Ptot', \
            'Ixhex', 'Iyhex', 'Vxhex', 'Vyhex']
    Idevlist = ['Iro','Izo','Ito']
    Vdevlist = ['Vzt','Vrz','Vtr']
        
     
    def __init__(self, name = 'iv3T', meastype='CZ', shape=1, fillname=''):
        
        self.name = name
        self.meastype = meastype
        
        # force shape to be tuple
        if np.ndim(shape) == 0:   #not iterable
            self.shape = (shape, )
        else:   #iterable
            self.shape = tuple(shape)
        
        for key in self.arraykeys:
           setattr(self, key, np.full(shape, np.nan, dtype=np.float64))
 
        size = getattr(self, key).size  #size of the last arraykey
        self.names = [fillname for i in range(size)]  # list of names of point in flat arrays
        
    def __str__(self):
        #   string description of IV3T object
        strout=self.name + ": <tandem.IV3T class>"
        attr_list = dict(self.__dict__.items()).copy()  #all attributes
        nshape = len(str(self.shape))  # number letters in largest index string
        nnames = len(max(self.names, key=len)) #number of letters in larges names[i] string
        nshape = max(nshape,nnames)
        title = '#'.center(nshape)
        sizes = self.sizes(self.arraykeys)
        nmin,nmax = sizes
        width = 9
        prntmax = 50
        fmtV = '{0:^'+str(width)+'.3f}'
        fmtI = '{0:^'+str(width)+'.2f}'
        
        if 'x' in attr_list:
            attr_list['x']=(np.min(self.x), np.max(self.x))
        if 'y' in attr_list:
            attr_list['y']=(np.min(self.y), np.max(self.y))
                        
        del attr_list['names']
        for key in self.arraykeys:
            if key in attr_list: del attr_list[key] 
            
            if key[:1] == 'V':
                title += key.center(width)
            else:   #append 'm'
                title += ('m'+key).center(width)
                           
        strout += '\n' + str(attr_list) 
        strout += '\nsizes' + str(sizes)
        strout += '\n\n' + title
        
        for i, index in enumerate(np.ndindex(self.shape)):
            if hasattr(self,'names'):
                if i < len(self.names) and self.names[i]:
                    strout += '\n' + self.names[i].center(nshape)
                else:
                    strout += '\n' + str(index).center(nshape)
            else:
                strout += '\n' + str(index).center(nshape)
                
            for key in self.arraykeys:
                array = getattr(self, key)
                if i < array.size:
                    if key[:1] == 'V':
                        #sval = fmtV.format(array.flat[i])
                        sval = fmtV.format(array[index])
                    else:   #print mA or mW
                        sval = fmtI.format(1000 * array[index])
                else:
                    sval = ''.center(width)
                strout += sval
                
            if i > prntmax: break
            
        return strout
    
    def __repr__(self):
        return str(self)

    def update(self, **kwargs):
        """
        Update variables
        """
        for key, value in kwargs.items():
            if key in set(self.arraykeys):   #this should be numpy.ndarray
                if np.isscalar(value):  # value = scalar
                    #shape = getattr(self, key).size
                    setattr(self, key, np.full(self.shape, value, dtype=np.float64))
                else:   #value = full array
                    setattr(self, key, np.array(value))
            else:
                setattr(self, key, value)
    
    def line(self, xkey, x0, x1, xn, ykey, yconstraint):
        '''
        create a 1D ndarray on xke with evenly spaced values
        ykey is constrained to xkey with eval expression using 'x'
        '''
        
        shape = (xn,)
        x = np.linspace(x0, x1, xn, dtype=np.float64)
        
        for key in self.arraykeys:
            if key == xkey:   
                setattr(self, key, x)
            else:
                setattr(self, key, np.full(shape, np.nan, dtype=np.float64)) 
        
        y = eval(yconstraint)
        if type(y) is not np.array:
            y = np.full(shape, y, dtype=np.float64)

        setattr(self, ykey, y)
        self.kirchhoff([xkey, ykey])    # calculate everything
        
        #remember
        self.shape = shape
        self.xkey = xkey
        self.ykey = ykey
        self.x = x
        self.y = y
        
    def box(self, xkey, x0, x1, xn, ykey, y0, y1, yn):
        '''
        create a 2D ndarray for xkey and ykey with shape (xn, yn)
        with evenly spaced values
        '''
        shape = (xn, yn)
        x = np.linspace(x0, x1, xn, dtype=np.float64)
        y = np.linspace(y0, y1, yn, dtype=np.float64)
        xx, yy = np.meshgrid(x , y)
        
        for key in self.arraykeys:
            if key == xkey:   
                setattr(self, key, xx)
            elif key == ykey:   
                setattr(self, key, yy)
            else:
                setattr(self, key, np.full(shape, np.nan, dtype=np.float64))
                
        self.kirchhoff([xkey, ykey])    # calculate everything
        
        #remember
        self.shape = shape
        self.xkey = xkey
        self.ykey = ykey
        self.x = x
        self.y = y

    def MPP(self,name=''):
        '''
        find max power point of existing IV3T class datapoints
        '''

        temp3T = IV3T(name='MPP'+name, shape=1, meastype = self.meastype)
        nmax = np.argmax(self.Ptot)  

        for key in self.arraykeys:
            sarray = getattr(self, key)
            tarray = getattr(temp3T, key)
            tarray[0] = sarray.flat[nmax]
        
        return temp3T
      
    def sizes(self,klist):
        #array length
        # for all use .sizes(self.arraykeys)
            
        sizes = []   #initialize list of array lengths 
        allowlist = [key for key in klist if key in self.arraykeys]
        #ignore items that are not array attributes
        for key in allowlist:
           num = getattr(self, key).size 
           sizes.append(num)

        nmin = np.amin(sizes)    
        nmax = np.amax(sizes) 
        return (nmin,nmax)

    def resize(self, shape, fillname = ''):
        '''
        resize arrays and clear values
        '''
        
        # force shape to be tuple
        if np.ndim(shape) == 0:   #not iterable
            self.shape = (shape, )
        else:   #iterable
            self.shape = tuple(shape)
        
        for key in self.arraykeys:
            array = getattr(self, key)
            setattr(self, key, np.resize(array,shape))
            
        oldsize = len(self.names)
        newsize =  getattr(self, key).size  # last array
        
        if newsize > oldsize :
            for i in range(oldsize,newsize):
                self.names.append(fillname)  #append
        elif newsize < oldsize :
            for i in range(oldsize,newsize,-1):
               del self.names[-1]   #delete
               
        return oldsize, newsize

    def append(self, iv3T):
        '''
        append another Junction object to self
        '''
        nmin, nmax = self.sizes(self.arraykeys)
        if nmin != nmax:
            print(nmin, nmax, 'not same size')
            return 1
        addmin, addmax = iv3T.sizes(self.arraykeys)
        newsize = (nmax + addmax)
                    
        self.resize(newsize, fillname = iv3T.name)  #resize and flatten
        
        for i in range(addmax):
            if iv3T.names[i] :  #not null string
                self.names[i+nmax] = iv3T.names[i]    #add names            

        for key in self.arraykeys:
            selfarray = getattr(self, key)
            addarray = getattr(iv3T, key)
            
            for i in range(addmax):
                selfarray[i+nmax] = addarray[i]  #add key values
                
        if self.meastype != iv3T.meastype:  # correct all load values
            self.convert('V','dev2load')
            self.convert('I','dev2load')
            #print('meastypes different', self.meastype , iv3T.meastype)
            
        return 0
        
    def init(self,inlist,outlist):
        #initialize output arrays to nan if input arrays are consistent

        nmin, nmax = self.sizes(inlist) 
        if  nmin == nmax:    #everything in inlist has same length
            # record shape of inlist
            self.shape = getattr(self,inlist[0]).shape
            
            #initialize output
            for key in outlist:
                setattr(self, key, np.full(self.shape, np.nan, dtype=np.float64))
                
            return 0
       
        else:
            return 4    #inlist arrays not same length

    def kirchhoff(self,two):
        '''
        apply or check kirchoff's law on        
        [Iro,Izo,Ito] or [Vzt,Vrz,Vtr]
        input 2 or 3 of the device input keys:
            2 -> calculate the third device value from other two knowns
            3 -> check the validity of 3 device parameters
        '''
        
        stwo=set(two)
        if stwo.issubset(set(self.Vdevlist)):
            klist = self.Vdevlist.copy()   #['Vzt','Vrz','Vtr']
        elif stwo.issubset(set(self.Idevlist)):
            klist = self.Idevlist.copy()   #['Iro','Izo','Ito']            
        else:
            #print(str(two) + ' not device parameters')
            return 1    #not Vdev or Idev

        for key in two: klist.remove(key)
        if klist:
            scalc = klist[0]   #remaining item to calculate
            
        ltwo = len(two)
        nmin, nmax = self.sizes(two) 
        if  nmin == nmax:    #everything in two has same length
            array0 = getattr(self, two[0])
            array1 = getattr(self, two[1])
            if ltwo == 2:  #calc third
                calcarray = - array0 - array1
                setattr(self,scalc,calcarray)
            elif ltwo == 3:  #check
                array2 = getattr(self, two[2])
                akirch = array0 + array1 + array2
                for i, kzero in enumerate(akirch.flat):
                    if not math.isclose(kzero, 0., abs_tol=1e-3):
                        array0.flat[i] = np.nan
                        array1.flat[i] = np.nan
                        array2.flat[i] = np.nan
                        
        else:
            print(str(two) + ' arrays not same length')
            return 3    #klist arrays not same length
            
        return 0    #success
        
    def Pcalc(self, oper='dev2load', meastype=None):
        '''
        calculate Ptot using oper = 'dev2load' or 'load2dev'
        '''
        
        if meastype != None:  #optionally change the attribute here
            self.meastype = meastype
        
        if oper == 'dev2load': 
            devlist = self.Idevlist.copy() + self.Vdevlist.copy()
            #['Iro', 'Izo', 'Ito', 'Vzt', 'Vrz', 'Vtr'])
            nmin, nmax = self.sizes(devlist)  
        elif oper == 'load2dev':
            nmin, nmax = self.sizes(['IA', 'IB', 'VA', 'VB'])
        else:
            return 1   #invalid oper
        
        if nmin == nmax:
            #calc either direction using self.meastype
            self.convert('V',oper, meastype=meastype)
            self.convert('I',oper, meastype=meastype)
            self.convert('V','dev2hex', meastype=meastype)
            self.convert('I','dev2hex', meastype=meastype)
        else:
            return 2   #need all input values of same length

        self.Ptot = - self.IA * self.VA -  self.IB * self.VB
        np.nan_to_num(self.Ptot, copy=False, nan = -100.)
                             
        return 0

    def convert(self, VorI, oper, meastype=None):
        '''
        calculate hexagonal coordinates
        VorI: 'V' or 'I'
        oper: 'hex', 'load2dev', 'dev2load'
        can optionally set the meastype here
        meastype: 'CR','CT','CZ','CF' 
        add an extra character to swap the loads: 'CRo','CTo','CZo', 'CFo' 
        '''
        
        oper = oper.lower()
        VorI = VorI.upper()
        
        if VorI == "V":
            devlist = self.Vdevlist.copy()   #['Vzt','Vrz','Vtr']
            loadlist = ['VA','VB']
            hexlist = ['Vxhex','Vyhex']          
        elif VorI == "I":
            devlist = self.Idevlist.copy()   #['Iro','Izo','Ito'] 
            loadlist = ['IA','IB']
            hexlist = ['Ixhex','Iyhex']
        else:
            print('VorI err', VorI)
            return 1    #invalid VorI
        
        if meastype != None:  #change the attribute here
            self.meastype = meastype
                       
        smatrix = self.meastype[0:2].upper().replace('F','T') \
            + '_' + VorI + oper

        if len(self.meastype) > 2:  #swap loads for alternate measurements, e.g CTo
            loadlist = loadlist[::-1]
            print(self.meastype, '->', loadlist, ' swapped')

        if oper == 'dev2hex':
            inlist = devlist
            outlist = hexlist
            smatrix = oper
        elif oper == 'hex2dev':
            inlist = hexlist
            outlist = devlist
            smatrix = oper
        elif oper == 'load2dev':
            inlist = loadlist
            outlist = devlist           
        elif oper== 'dev2load':
            inlist = devlist
            outlist = loadlist
        else:
            print('oper err', oper)
            return 2    #invalid oper

        try:
            matrix = eval(smatrix)
        except:
            print('matrix err', smatrix)
            return 3   #could not evaluate smatrix
        
        nmin, nmax = self.sizes(inlist) 
        if  nmin == nmax:    #everything in klist has same length
            inarray0 = getattr(self, inlist[0])
            inarray1 = getattr(self, inlist[1])
            if len(inlist)==3:
                inarray2 = getattr(self, inlist[2])
            else:
                inarray2 = np.zeros(self.shape)

            #initialize local empty output to same size
            outarray0 = np.full(self.shape, np.nan, dtype=np.float64)
            outarray1 = np.full(self.shape, np.nan, dtype=np.float64)
            outarray2 = np.full(self.shape, np.nan, dtype=np.float64)
        else:
            return 4    #klist arrays not same length
            
        i=0
        for in0,in1,in2 in zip(inarray0.flat, inarray1.flat, inarray2.flat):
            vector_in = np.array([in0,in1,in2])
            vector_out = matrix @ vector_in
            outarray0.flat[i] = vector_out[0]
            outarray1.flat[i] = vector_out[1]
            outarray2.flat[i] = vector_out[2]
            i += 1
        
        setattr(self,outlist[0],outarray0)
        setattr(self,outlist[1],outarray1)
        if len(outlist)==3:   # otherwise they are just zeros
            setattr(self,outlist[2],outarray2)
        
        return 0
   
    def plot(self):
        '''
        plot IV3T line or box
        '''
        
        dim=len(self.shape)
        if dim == 2:   # box
            x = self.x
            y = self.y
            z = self.Ptot*1000
            xlab = self.xkey
            ylab = self.ykey
            extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
            levels = [0,5,10,15,20,25,30]
            
            fig, ax = plt.subplots()
            imag = ax.imshow(z, vmin=0, vmax=25, origin='lower', 
                             extent = extent, cmap='terrain')         
            
            cont = ax.contour(x, y, z, colors = 'black',
                           levels = levels)
            ax.clabel(cont, inline=True, fontsize=10)
            
            ax.axis('scaled')
            ax.set_xlabel(xlab)  # Add an x-label to the axes.
            ax.set_ylabel(ylab)  # Add a y-label to the axes.
            ax.axhline(0, color='gray')
            ax.axvline(0, color='gray')
             
        else:  # line or points
            fig, ax = plt.subplots()
            ax.plot(self.VA, self.IA, marker='.',c='green')  #JV curve
            ax.plot(self.VB, self.IB, marker='.',c='red')  #JV curve
            ax.set_title('title')  # Add a title to the axes.
            ax.set_xlabel('Voltage (V)')  # Add an x-label to the axes.
            ax.set_ylabel('Current Density (A/cm2)')  # Add a y-label to the axes.
            
        return fig
