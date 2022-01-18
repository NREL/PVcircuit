# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package. 
    pvcircuit.Tandem3T()   # properties of a 3T tandem including 2 junctions
"""

import math   #simple math
from time import time
import numpy as np   #arrays
import matplotlib.pyplot as plt   #plotting
from scipy.interpolate import interp1d
from scipy.optimize import brentq    #root finder
#from scipy.special import lambertw, gammaincc, gamma   #special functions
import scipy.constants as con   #physical constants
from pvcircuit.junction import *
from pvcircuit.iv3T import *
                
class Tandem3T(object): 
    '''
    Tandem3T class for optoelectronic model
    '''

    update_now = True 

    def __init__(self, name='Tandem3T', TC=TC_REF, Rz=1, Eg_list=[1.8,1.4], pn=[-1,1], Jext=0.014):
        # user inputs
        # default s-type n-on-p
        
        update_now = False
        
        self.name = name
        self.TC = TC
        self.Rz = Rz
        self.top = Junction(name='top', Eg=Eg_list[0], TC=TC, \
                            Jext = Jext, pn=pn[0], beta=0.)
        self.bot = Junction(name='bot', Eg=Eg_list[1], TC=TC, \
                            Jext = Jext, pn=pn[1])

        update_now = True               

    def __str__(self):
        '''
        format concise description of Tandem3T object
        '''
        
        #attr_list = self.__dict__.keys()
        #attr_dict = self.__dict__.items()
        #print(attr_list)
        
        strout=self.name + ": <tandem.Tandem3T class>"
        strout += '\nT = {0:.1f} C, Rz= {1:g} Ω, Rt= {2:g} Ω, Rr = {3:g} Ω'\
            .format(self.TC,self.Rz,self.top.Rser,self.bot.Rser)
        strout += '\n\n'+str(self.top)
        strout += '\n\n'+str(self.bot)
        return strout
    
    def __repr__(self):
        return str(self)
 
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'name':
                self.__dict__[key] = str(value)
            elif key == 'njunc':
                self.__dict__[key] = int(value)
            elif key == 'Vmid':
                self.__dict__[key] = np.array(value)
            elif key in ['top', 'bot', 'update_now']: 
                self.__dict__[key] = value
            elif key in ['Rz', 'TC']:
                self.__dict__[key] = np.float64(value)

            if self.update_now:  # put appropriate values into junction attributes
                jlist = Junction.ATTR.copy()
                jlist.remove('Rser')
                if key in jlist :                    
                    for i, junc in enumerate(self.j):
                        if np.isscalar(value):
                            junc.__dict__[key] = value
                        else:
                            junc.__dict__[key] = value[i]
    
    @property
    def area(self):
        # largest junction area
        return max(self.top.area,self.bot.area)
    
    def V3T(self,iv3T):
        '''
        calcuate iv3T.(Vzt,Vrz,Vtr) from iv3T.(Iro,Izo,Ito)
        input class tandem.IV3T object 'iv3T'
        '''
        
        top = self.top   # top Junction
        bot = self.bot   # bot Junction 
  
        inlist = iv3T.Idevlist.copy()
        outlist = iv3T.Vdevlist.copy()
        err = iv3T.init(inlist,outlist)    # initialize output
        if err: return err

        i = 0
        for index in np.ndindex(iv3T.shape):
        #for Ito, Iro, Izo in zip(iv3T.Ito.flat, iv3T.Iro.flat, iv3T.Izo.flat):
            # loop through points
            
            Ito = iv3T.Ito.flat[i]
            Iro = iv3T.Iro.flat[i]
            Izo = iv3T.Izo.flat[i]
                # loop through points
            kzero = Ito + Iro + Izo
            if not math.isclose(kzero, 0., abs_tol=1e-5):
                #print(i, 'Kirchhoff violation', kzero)
                iv3T.Vzt[i] = np.nan
                iv3T.Vrz[i] = np.nan
                iv3T.Vtr[i] = np.nan
                i += 1
                continue
           
            #input current densities
            Jt = Ito / top.area
            Jr = Iro / bot.area
            Jz = Izo / self.area
            
            # top Junction
            top.JLC = 0.
            Vtmid = top.Vdiode(Jt * top.pn) * top.pn
            Vt = Vtmid + Jt * top.Rser
            
            # bot Junction
            bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)    # top to bot LC
            Vrmid = bot.Vdiode(Jr * bot.pn) * bot.pn
            Vr = Vrmid + Jr * bot.Rser
            
            if top.beta > 0.:   # repeat if backwards LC
                # top Junction
                top.JLC = top.beta * bot.Jem(Vrmid * bot.pn)    # bot to top LC
                Vtmid = top.Vdiode(Jt * top.pn) * top.pn
                Vt = Vtmid + Jt * top.Rser
                
                # bot Junction
                bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)    # top to bot LC
                Vrmid = bot.Vdiode(Jr * bot.pn) * bot.pn
                Vr = Vrmid + Jr * bot.Rser
                
            
            # extra Z contact
            Vz = Jz * self.Rz
            
            #items in array = difference of local variable
            iv3T.Vzt.flat[i] = Vz - Vt  
            iv3T.Vrz.flat[i] = Vr - Vz
            iv3T.Vtr.flat[i] = Vt - Vr
            i += 1
         
        iv3T.Pcalc()    #dev2load defaults
        
        return 0       
    
    def J3Tabs(self,iv3T):
        '''
        calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
        from ABSOLUTE (Vz,Vr,Vt) mapped <- iv3T.(Vzt,Vrz,Vtr) 
        input class tandem.IV3T object 'iv3T'
        '''
        
        top = self.top   # top Junction
        bot = self.bot   # bot Junction 
  
        inlist = iv3T.Vdevlist.copy()   #['Vzt','Vrz','Vtr']
        outlist = iv3T.Idevlist.copy()
        err = iv3T.init(inlist,outlist)   # initialize output
        if err: return err

        i = 0
        for index in np.ndindex(iv3T.shape):
        #for Vz, Vr, Vt in zip(iv3T.Vzt.flat, iv3T.Vrz.flat, iv3T.Vtr.flat):
            # loop through points
            # ABSOLUTE (Vz,Vr,Vt) mapped <- iv3T.(Vzt,Vrz,Vtr)

            Vz = iv3T.Vzt.flat[i]
            Vr = iv3T.Vrz.flat[i]
            Vt = iv3T.Vtr.flat[i]
            # top Junction
            top.JLC = 0.
            if top.notdiode():  # top resistor only
                Vtmid = 0.
                Jt = Vt / top.Rser
            else: # top diode
                Vtmid = top.Vmid(Vt * top.pn) * top.pn  
                Jt = -top.Jparallel(Vtmid * top.pn,top.Jphoto) * top.pn
         
            # bot Junction
            bot.JLC = bot.beta * top.Jem(Vtmid)   # top to bot LC
            if bot.notdiode():  # bot resistor only
                Vrmid = 0.
                Jr = Vr / bot.Rser
            else: # bot diode
                Vrmid = bot.Vmid(Vr * bot.pn) * bot.pn  
                Jr = -bot.Jparallel(Vrmid * bot.pn ,bot.Jphoto) * bot.pn
            
            if top.beta > 0.:   # repeat if backwards LC
                # top Junction
                top.JLC = top.beta * bot.Jem(Vrmid)    # bot to top LC
                if top.notdiode():  # top resistor only
                    Vtmid = 0.
                    Jt = Vt / top.Rser
                else: # top diode
                    Vtmid = top.Vmid(Vt * top.pn) * top.pn  
                    Jt = -top.Jparallel(Vtmid * top.pn,top.Jphoto) * top.pn
         
                # bot Junction
                bot.JLC = bot.beta * top.Jem(Vtmid)   # top to bot LC
                if bot.notdiode():  # bot resistor only
                    Vrmid = 0.
                    Jr = Vr / bot.Rser
                else: # bot diode
                    Vrmid = bot.Vmid(Vr * bot.pn) * bot.pn  
                    Jr = -bot.Jparallel(Vrmid * bot.pn ,bot.Jphoto) * bot.pn
 
            # extra Z contact
            if self.Rz == 0.:
                Jz = (-Jt* top.area - Jr* bot.area ) / self.area   # determine from kirchhoff    
            else:
                Jz = Vz / self.Rz   # determine from Rz
            
            # output (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
            iv3T.Iro.flat[i] = Jr
            iv3T.Izo.flat[i] = Jz
            iv3T.Ito.flat[i] = Jt
            i += 1
                
        return 0
    
    def _dI(self,Vz,Vzt,Vrz,temp3T):
        '''
        return dI = Iro + Izo + Ito
        function solved for dI(Vz)=0 in I3rel
        input Vzt, Vrz, temp3T <IV3T class> container for calculation
        '''
        top = self.top   # top Junction
        bot = self.bot   # bot Junction 

        Vt = Vz - Vzt
        Vr = Vrz + Vz

        temp3T.update(Vzt=Vz, Vrz=Vr, Vtr=Vt)
        
        self.J3Tabs(temp3T)   #calcuate (Jro,Jzo,Jto) from (Vz,Vr,Vt)

        # (Jro,Jzo,Jto)  -> (Iro,Izo,Ito)
        Jro = temp3T.Iro[0]
        Jzo = temp3T.Izo[0]
        Jto = temp3T.Ito[0]
        
        Iro = Jro * bot.area
        Izo = Jzo * self.area
        Ito = Jto * top.area
        
        return Iro + Izo + Ito
    
    def I3Trel(self,iv3T):
        '''
        calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
        from RELATIVE iv3T.(Vzt,Vrz,Vtr) ignoring Vtr
        input class tandem.IV3T object 'iv3T'
        '''
        
        top = self.top   # top Junction
        bot = self.bot   # bot Junction 


        inlist = iv3T.Vdevlist.copy()   #['Vzt','Vrz','Vtr']
        outlist = iv3T.Idevlist.copy()
        err = iv3T.init(inlist,outlist)   # initialize output
        if err: return err

        temp3T = IV3T(name='temp3T', shape=1, meastype = iv3T.meastype)
        
        # remember resistances
        Rz = self.Rz
        Rt = top.Rser
        Rr = bot.Rser

        i = 0
        for index in np.ndindex(iv3T.shape):
        #for Vzt, Vrz in zip(iv3T.Vzt.flat, iv3T.Vrz.flat):
            # loop through points  
            
            Vzt = iv3T.Vzt.flat[i]
            Vrz = iv3T.Vrz.flat[i]
            # initial guess Rz=0 -> Vz=0
            self.Rz = 0.
            top.Rser = Rt + Rz
            bot.Rser = Rr + Rz
            Vz = 0.
            Vt = Vz - Vzt
            Vr = Vrz + Vz
            temp3T.update(Vzt=Vz, Vrz=Vr, Vtr=Vt)
            self.J3Tabs(temp3T)
            
            Jro = temp3T.Iro[0]
            Jzo = temp3T.Izo[0]
            Jto = temp3T.Ito[0]
            Vzmax = Jzo * Rz
            
            # reset resistance
            self.Rz = Rz
            top.Rser = Rt
            bot.Rser = Rr
            
            # iterate with varying Vz
            # bracket
            Vmin = 0.5
            if math.isfinite(Vzmax) and (abs(Vzmax) > VTOL):
                Vlo = min(0, 1.2 * Vzmax)
                Vhi = max(0, 1.2 * Vzmax)
            else:
                Vlo = -Vmin
                Vhi = Vmin
            dIlo = self._dI(Vlo,Vzt,Vrz,temp3T)
            dIhi = self._dI(Vhi,Vzt,Vrz,temp3T)
            
            count = 0
            while dIlo * dIhi > 0.:
                #print('I3Trel not bracket', i, count, Vlo, Vhi, dIlo, dIhi)
                Vlo -= 0.1
                Vhi += 0.1
                dIlo = self._dI(Vlo,Vzt,Vrz,temp3T)
                dIhi = self._dI(Vhi,Vzt,Vrz,temp3T)
                if count > 2: break
                count += 1
          
            if dIlo * dIhi > 0.:
                #print('I3Trel not bracket', i, count, Vlo, Vhi, dIlo, dIhi)
                i += 1
                continue
                
            if Rz > 0.:      
                try:   #find Vz that satisfies dJ(Vz) = Jro+Jzo+Jto = 0
                    Vz = brentq(self._dI, Vlo, Vhi, args=(Vzt,Vrz,temp3T),
                                   xtol=VTOL, rtol=EPSREL, maxiter=MAXITER,
                                   full_output=False, disp=True)
                    #dI0 = self._dI(Vz,Vzt,Vrz,temp3T)
                    
                    Jro = temp3T.Iro[0]
                    Jzo = temp3T.Izo[0]
                    Jto = temp3T.Ito[0]
                  
                except: 
                    Jro = np.nan
                    Jzo = np.nan
                    Jto = np.nan
                    
            #output
            iv3T.Iro.flat[i] = Jro * bot.area
            iv3T.Izo.flat[i] = Jzo * self.area
            iv3T.Ito.flat[i] = Jto * top.area
            i += 1
            
        iv3T.kirchhoff(['Vzt', 'Vrz'])   # Vtr not used so make sure consistent
        iv3T.kirchhoff(iv3T.Idevlist.copy())   # check for bad results
        iv3T.Pcalc()    #dev2load defaults
                   
        return 0    
   
    def Voc3(self, meastype='CZ'):
        '''
        triple Voc of 3T tandem
        (Vzt, Vrz, Vtr) of (Iro = 0, Izo = 0, Ito = 0)
        '''
        
        temp3T = IV3T(name='Voc3', shape=1, meastype=meastype)
        temp3T.update(Iro = 0, Izo = 0, Ito = 0)
        self.V3T(temp3T)
        
        #return (temp3T.Vzt[0], temp3T.Vrz[0], temp3T.Vtr[0])
        return temp3T
      
    def Isc3(self, meastype='CZ'):
        '''
        triple Isc of 3T tandem
        (Iro, Izo, Ito ) of (Vzt = 0, Vrz = 0, Vtr = 0) 
        '''
        
        temp3T = IV3T(name='Isc3', shape=1, meastype=meastype)
        temp3T.update(Vzt = 0, Vrz = 0, Vtr = 0)
        self.I3Trel(temp3T)
        
        #return (temp3T.Iro[0], temp3T.Izo[0], temp3T.Ito[0])
        return temp3T
 
    def MPP(self, pnts=31, VorI= 'I', less = 2., bplot=False):
        '''
        iteratively find MPP from lines
        as experimentally done
        varying I is faster than varying V
        but initial guess is not as good
        'less' must be > 1.0
        if FF is really bad, may need larger 'less'
        bplot for debugging information
        '''
        
        ts = time()
        meastype = 'CZ'
        Voc3 = self.Voc3(meastype)
        Isc3 = self.Isc3(meastype)
        Isct = Isc3.Ito[0]
        Iscr = Isc3.Iro[0]
        pnt = self.top.pn
        pnr = self.bot.pn
        tol = 1e-5
        # initial guesses
        Vmpr = 0.
        Impr = Iscr
        Impt = Isct
        # create IV3T classes for top and rear
        lnt = IV3T(name = 'lnMPPt', meastype = meastype)        
        lnr = IV3T(name = 'lnMPPr', meastype = meastype)

        if bplot:
            fig, ax = plt.subplots()
            ax.axhline(0, color='gray')
            ax.axvline(0, color='gray')
            ax.set_title(self.name + ' MPP calc.')
            ax.plot(Voc3.Vrz, 0, marker='o')
            ax.plot(Voc3.Vzt, 0, marker='o')
            ax.plot(0, Isc3.Iro*pnr, marker='o')
            ax.plot(0, Isc3.Ito*pnt, marker='o')
       
        Pmpo = 0.
        for i in range(5):
            #iterate
            if VorI == 'V':
                lnt.line('Vzt', 0, Voc3.Vzt[0], pnts, 'Vrz', str(Vmpr)) 
                self.I3Trel(lnt)
            else:
                lnt.line('Ito', Isct, Isct/less, pnts, 'Iro', str(Impr))
                self.V3T(lnt)
            aPt = getattr(lnt, 'Ptot')
            aVzt = getattr(lnt, 'Vzt')
            aIto = getattr(lnt, 'Ito')
            nt = np.argmax(aPt)
            Vmpt = aVzt[nt]
            Impt = aIto[nt]
            Pmpt = aPt[nt]
            if bplot: 
                ax.plot(aVzt, aIto*pnt, marker='.')
                ax.plot(Vmpt, Impt*pnt,marker='o')
                print(i, 'T', Vmpt, Impt, Pmpt)

            if VorI == 'V':
                lnr.line('Vrz', 0, Voc3.Vrz[0], pnts, 'Vzt', str(Vmpt)) 
                self.I3Trel(lnr)
            else:
                lnr.line('Iro', Iscr, Iscr/less, pnts, 'Ito', str(Impt))
                self.V3T(lnr)
            aPr = getattr(lnr, 'Ptot')
            aVrz = getattr(lnr, 'Vrz')
            aIro = getattr(lnr, 'Iro')
            nr = np.argmax(aPr)
            Vmpr = aVrz[nr]
            Impr = aIro[nr]
            Pmpr = aPr[nr]
            if bplot: 
                ax.plot(aVrz, aIro*pnr, marker='.')
                ax.plot(Vmpr, Impr*pnr, marker='o')
                print(i, 'R', Vmpr, Impr, Pmpr)
            
            if (Pmpr - Pmpo)/Pmpr < tol : break
            Pmpo = Pmpr
            VorI = 'I'  #switch to 'I' after first iteration
 
        # create one line solution
        pt = IV3T(name = 'MPP', meastype = meastype, shape=1)
        pt.Vzt[0] = Vmpt
        pt.Vrz[0] = Vmpr
        pt.kirchhoff(['Vzt', 'Vrz'])
        self.I3Trel(pt)

        te = time()
        dt=(te-ts)
        if bplot: print('MPP: {0:d}pnts , {1:2.4f} s'.format(pnts,dt))
        
        return pt
               
    def VIpoint(self, zerokey, varykey, crosskey, meastype='CZ', pnts=11, bplot=False):
        '''
        solve for mixed (V=0, I=0) zero power points
        '''
        
        ts = time()
        if varykey[0] == 'V':
            Voc3 = self.Voc3(meastype)
            x0 = getattr(Voc3, varykey)[0]
            dx = 1
            abs_tol = 1e-5 #.01 mA
        else:
            Isc3 = self.Isc3(meastype)
            x0 = getattr(Isc3, varykey)[0]
            #dx = abs(getattr(Isc3, 'Izo')[0])
            dx = .1
            abs_tol = 1e-3  #1 mV

        ln = IV3T(name = 'ln'+zerokey+'_0', meastype = meastype)
        
        growth = (pnts - 1.)

        if bplot:
            fig, ax = plt.subplots()
            ax.axhline(0, color='gray')
            ax.set_title('VIpoint: '+zerokey+crosskey+'  '+zerokey+'=0')
            ax.set_xlabel(varykey)
            ax.set_ylabel(crosskey)
            ax.plot(x0,0,marker='o',fillstyle='none', ms=8, color='black')

        for i in range(4):
            if bplot: print('x0 =',x0)
            ln.line(varykey, x0-dx, x0+dx, pnts, zerokey, '0') 
            if varykey[0] == 'V':
                self.I3Trel(ln)
            else:
                self.V3T(ln)  
            xx = getattr(ln, varykey)
            yy = getattr(ln, crosskey)
            
            try:
                fn_interp = interp1d(yy,xx) # scipy interpolate function
                xguess = fn_interp(0.)
            except:
                xguess = np.interp(0., yy, xx)  # y=zero crossing
            if not np.isnan(xguess):
                x0 = xguess
            dx /= growth
            if bplot: ax.plot(xx,yy,marker='.')
              
        # create one line solution
        pt = IV3T(name = zerokey+crosskey, meastype = meastype, shape=1)
        xp = getattr(pt, varykey)
        xp[0] = x0
        zp = getattr(pt, zerokey)
        zp[0] = 0.
        pt.kirchhoff([zerokey, varykey])            
        if varykey[0] == 'V':          
            self.I3Trel(pt)
        else:
            self.V3T(pt)
            
        yp = getattr(pt, crosskey)
        if not math.isclose(yp[0], 0., abs_tol=1e-3):  # test if it worked
            pt.nanpnt(0)

        te = time()
        dt=(te-ts)
        if bplot: print('VIpoint: ' + pt.name + ' {0:d}pnts , {1:2.4f} s'.format(pnts,dt))

        return pt

    def specialpoints(self, meastype = 'CZ', bplot=False):
        '''
        compile all the special zero power points
        and fast MPP estimate
        '''

        sp = self.Voc3(meastype=meastype) #Voc3 = 0
        sp.names[0] = 'Voc3'
        sp.name = 'SpecialPoints'
        sp.append(self.Isc3(meastype=meastype)) #Isc3 = 1

        # (Izo = 0, Vtr =0)        
        sp.append(self.VIpoint('Izo','Ito','Vtr', meastype=meastype, bplot=bplot)) # Izo = 0, x = Ito, y = Vtr      
        sp.append(self.VIpoint('Vtr','Vzt','Izo', meastype=meastype, bplot=bplot)) # Vtr = 0, x = Vzt, y = Izo
        sp.append(self.VIpoint('Vtr','Vrz','Izo', meastype=meastype, bplot=bplot)) # Vtr = 0, x = Vzt, y = Izo
       
        # (Ito = 0, Vrz = 0)        
        sp.append(self.VIpoint('Ito','Iro','Vrz', meastype=meastype, bplot=bplot)) # Ito = 0, x = Iro, y = Vrz
        sp.append(self.VIpoint('Vrz','Vtr','Ito', meastype=meastype, bplot=bplot)) # Vrz = 0, x = Vzt, y = Ito
        sp.append(self.VIpoint('Vrz','Vzt','Ito', meastype=meastype, bplot=bplot)) # Vrz = 0, x = Vzt, y = Ito

        # (Iro = 0, Vzt = 0)       
        sp.append(self.VIpoint('Iro','Ito','Vzt', meastype=meastype)) # Iro = 0, x = Ito, y = Vzt      
        sp.append(self.VIpoint('Vzt','Vrz','Iro', meastype=meastype, bplot=bplot)) # Vzt = 0, x = Vrz , y = Iro
        sp.append(self.VIpoint('Vzt','Vtr','Iro', meastype=meastype, bplot=bplot)) # Vzt = 0, x = Vrz , y = Iro
        
        sp.append(self.MPP(bplot=bplot))
        
        return sp
        
    def plot(self, pnts=31, meastype='CZ', oper = 'load2dev', cmap='terrain'):
        '''
        plots characterizing the Tandem3T devices 'self'
        '''
        #oper = oper.lower()
        
        #bounding points
        factor = 1.2
        cmap = plt.cm.get_cmap(cmap)  # start with existing cmap
        cmap.set_under(color='white')  # white for Ptot < 0 and nan
        sp = self.specialpoints(meastype)
        Vmax = max(abs(sp.Vzt[0]), abs(sp.Vrz[0]), abs(sp.Vtr[0])) 
        Imax = max(abs(sp.Iro[1]), abs(sp.Izo[1]), abs(sp.Ito[1]))
        levels = [0,5,10,15,20,25,30]
        Pmax = 25
                   
        iv = list()  #empty list to contain IV3T structures
        axs = list()  #empty list to contain axis of each figure
        #fig, ax = plt.subplots(1,1)  #,gridspec_kw={'width_ratios': [5,5,1]})
        #axs[1].set_title(meastype + '-mode ' + self.name, loc='center')               
        #plt.tight_layout()

        for i, VorI in enumerate(['V','I']):
            
            ts = time()
            # create box IV3T instance
            name = VorI + 'plot'
            common = meastype[1].lower()
            if VorI == "V":
                devlist = IV3T.Vdevlist.copy()   #['Vzt','Vrz','Vtr']
                factor = 1.1
                xmax = Vmax * factor
                ymax = Vmax * factor           
                Vfig, ax = plt.subplots(1,1)  #,gridspec_kw={'width_ratios': [5,5,1]})
                               
            elif VorI == "I":
                devlist = IV3T.Idevlist.copy()   #['Iro','Izo','Ito'] 
                factor = 3.0
                xmax = Imax * factor
                ymax = Imax * factor
                Ifig, ax = plt.subplots(1,1)  #,gridspec_kw={'width_ratios': [5,5,1]})
            if oper == 'load2dev':
                xkey = VorI + 'A'
                ykey = VorI + 'B'
                ax.set_title(self.name + ' P-'+VorI+'-'+VorI + ' ' + meastype + '-mode ' , loc='center')
            elif oper == 'dev2load':
                xkey = devlist[0]
                ykey = devlist[1]
                ax.set_title(self.name + ' P-'+VorI+'-'+VorI , loc='center')
            elif oper == 'dev2hex':
                xkey = devlist[0]
                ykey = devlist[1]
                xscat = VorI + 'xhex'
                yscat = VorI + 'yhex'
                ax.set_title(self.name + ' P-'+VorI+'-'+VorI+'-'+VorI + ' Hexagonal', loc='center')
            elif oper == 'hex2dev':
                xkey = VorI + 'xhex'
                ykey = VorI + 'yhex'
            
            #xmax = abs(getattr(limpnt, xkey)[0]) * factor
            #ymax = abs(getattr(limpnt, ykey)[0]) * factor
            axs.append(ax)
            iv.append(IV3T(name = name, meastype = meastype))  #add another IV3T class to iv list
            iv[i].box(xkey,-xmax, xmax, pnts, ykey, -ymax, ymax, pnts)
            if oper:
                iv[i].convert(VorI, oper)
             
            #return iv3T
        
            if VorI == 'V':    
                self.I3Trel(iv[i])   # calculate I(V)
            else:
                self.V3T(iv[i])   # calculate V(I)
              
            sp.append(iv[i].MPP(VorI))  #append MPP of current iv[i] to special points
            
            #print(iv3T)
            #return iv3T
            # plot 2D iv3T
            if VorI == 'I':
                #use mA for current
                scale = 1000.
                xlab = iv[i].loadlabel(xkey, meastype=meastype) + ' (mA)'
                ylab = iv[i].loadlabel(ykey, meastype=meastype) + ' (mA)'
                step = 20
            else:
                scale = 1.
                xlab = iv[i].loadlabel(xkey, meastype=meastype) + ' (V)'
                ylab = iv[i].loadlabel(ykey, meastype=meastype) + ' (V)'
                if Vmax < 1.:
                    step = 0.5
                else:
                    step = 1.0
                
            x = iv[i].x * scale # 1D
            y = iv[i].y * scale # 1D
            z = iv[i].Ptot * 1000 # 2D
            extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
            #subplot
            #ax = axs[i]
            ax.set_aspect(1)
          
            if oper.find('hex') < 0:
                #cartesian grids
                ax.set_xlabel(xlab)  # Add an x-label to the axes.
                ax.set_ylabel(ylab)  # Add a y-label to the axes.
                ax.axhline(0, ls= '--', color='gray')
                ax.axvline(0, ls= '--', color='gray')
                #image
                imag = ax.imshow(z, vmin=0, vmax=Pmax, origin='lower', 
                                 extent = extent, cmap=cmap)           
                #contour            
                cont = ax.contour(x, y, z, colors = 'black',
                               levels = levels)
                ax.clabel(cont, inline=True, fontsize=10)
            else:
                ax.set_axis_off()   # turn off confusing x-axis and y-axis
                #add hexgrids 
                iv[i].hexgrid(ax, VorI, step)               
                #scatter
                xkey = VorI + 'xhex'
                ykey = VorI + 'yhex'
                xx = getattr(iv[i],xkey) * scale
                yy = getattr(iv[i],ykey) * scale
                imag = ax.scatter(xx, yy, s=100, c=z, marker='h', cmap=cmap, \
                    vmin=0, vmax=Pmax)
                #contour
                cont = ax.contour(xx, yy, z, colors = 'black',
                               levels = levels)
                ax.clabel(cont, inline=True, fontsize=10)
                
                       
            #add points           
            xp = getattr(sp, xkey) * scale
            yp = getattr(sp, ykey) * scale
            #ax.plot(xp, yp, marker='o', fillstyle='none', mew=2, ls='', ms=6, c='red')
            ax.scatter(xp, yp, marker='^', s=100, c='red', edgecolors='black')
            #colorbar
            cb = plt.colorbar(imag, ax=ax)
            cb.set_label('Power (mW)')
           
            te = time()
            dt=(te-ts)
            print('axs[{0:g}]: {1:d}pnts , {2:2.4f} s'.format(i,pnts,dt))
 
        return Vfig, Ifig, axs, iv, sp
 