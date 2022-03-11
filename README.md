# pvcircuit 
*pvcircuit* contains objects that are building blocks for PV modeling.

Based on publications:

[J. F. Geisz, et al., IEEE Journal of Photovoltaics 5, p. 1827 (2015).](http://dx.doi.org/10.1109/JPHOTOV.2015.2478072)

[J. F. Geisz, et al., Cell Reports Physical Science 2, p. 100677 (2021).](https://doi.org/10.1016/j.xcrp.2021.100677)

### Installation
- install GitHub Desktop
- on this https://github.com/NREL/PVcircuit page click the green "Code" button and "Open with GitHub Desktop"
- cd to the PVcircuit directory in terminal and type "pip install -e ."

### Packages needed 
- pandas
- numpy
- matplotlib
- scipy
- ipywidgets
- ipympl
- parse

# Junction( ) Class
A single *Junction* that can be combined together to form tandem and multijunction solar cells.

![Junction Schematic](https://github.com/NREL/PVcircuit/blob/master/images/junction%20equations.png) 

The *Junction* is modeled by an arbitrary number of parallel diodes such as n=1, n=2, n=2/3, or any other ideality factor.
The temperature dependance of each diode saturation current J0n is determined relative to the Jdb(Eg,T). 

## Junction.attributes
### Junction.name
Name of this junction *string*

### Junction.Eg
Junction bandgap (eV)

### Junction.TC
Junction temperature (°C)

### Junction.Gsh
Junction shunt conductance (S/cm2)

### Junction.Rser
Junction series resistance (Ohm cm2)

### Junction.lightarea
Illuminated area (cm2)

### Junction.totalarea
Total area including any dark area (cm2)
always greater than or equal to .lightarea

### Junction.Jext
External photocurrent (A/cm2)

### Junction.JLC
Luminescent coupling photocurrent (A/cm2)

### Junction.beta
Electro luminescent coupling factor (unitless)
Amount of light collected as current in an adjectent junction relative to Jdb

### Junction.gamma
Photo luminescent coupling factor (unitless)

### Junction.pn
Direction of pn diode
    -1  n-on-p
    0   simple resistor
    1   p-on-n

### Junction.n[n0, n1, etc]
Ideality factor *numpy.array*

### Junction.J0ratio[J0ratio0, J0ratio1, etc]
Ratio of saturation current densities relative to Jdb  *numpy.array*

    J0ratio = J0 / (Jdb)^(1/n)

### Junction.ui
User interface created by .controls()
Consists of plot, widget controls, and other outputs

### Junction.debugout
Hidden output of for debugging

## Junction.properties 
### Junction.TK
Temperature (K) from TC

### Junction.Vth
Thermal voltage = kT/q from TK

### Junction.Jdb
Detailed balance current (A/cm2).
If J0ratio for n=1 only this is Shockley-Quiesser limit

### Junction.Jphoto
Total photocurrent = Jext+JLC

### Junction.J0[J00, J01, etc]
Reverse saturation currents (A/cm2) corresponding to each n[] and J0ratio[] *numpy.array*

## Junction.methods()

### Junction.copy()
Create a copy of a Junction

### Junction.set(\**kwargs)
Controlled change of Junction attributes

### Junction.controls()
Create a user interface Junction.ui which consists of widget controls that are mainly used to build up Multi2T.ui and Tandem3T.ui

### Junction.update()
Update Junction attributes into Junction.ui controls if they exist

### Junction._J0init(J0ref)
initialize self.J0ratio from J0ref

returns *np.ndarray* [J0(n0), J0(n1), etc]

### Junction.Jem(Vmid)
Light emitted from junction by reciprocity, quantified as current density (A/cm2)

### Junction.notdiode()
*boolean* whether the junction is a junction (False) or resistor (True)

### Junction.Jmultidiodes(Vdiode)
Calculate total recombination current density from multiple parallel diodes: n[], J0[] given voltage across diode *without* series resistance.

### Junction.JshuntRBB(Vdiode)
returns shunt + reverse-bias breakdown current

    RBB_dict={'method':None}
    RBB_dict={'method':'JFG', mrb'':10., 'J0rb':1., 'Vrb':0.}
    RBB_dict={'method':'bishop','mrb'':3.28, 'avalanche':1, 'Vrb':-5.5}
    RBB_dict={'method':'pvmismatch','ARBD':arbd,'BRBD':brbd,'VRBD':vrb,'NRBD':nrbd:  

### Junction.Jparallel(Vdiode,Jtot)
Circuit equation to be zeroed to solve for Vi for voltage across parallel diodes with shunt and reverse breakdown.

### Junction.Vdiode(Jdiode)
Calculate voltage across diode without series resistance as a function current density through the diode.

### Junction._dV(Vmid, Vtot)
Circuit equation to be zeroed (returns voltage difference) to solve for Vmid. Single junction circuit with series resistance and parallel diodes. 
*internal use only*

### Junction.Vmid(Vtot)
Find intermediate voltage in a single junction diode with series resistance given Vtot including series resistance

    Vtot = Vparallel + Rser * Jparallel

## Multi2T( ) Class
Two terminal multijunction device composed of any number of series connected *Junctions*. The sum of all Rser is an attribute of the *Multi2T* object and the Rser attributes of each sub *Junction* are ignored

![2T Multijunction Schematic](https://github.com/NREL/PVcircuit/blob/master/images/MultijunctionSchematic.png)

## Multi2T.attributes
### Multi2T.name
Name of this junction *string*

### Multi2T.Rs2T
Total series resistance of Multi2T. (Junction.Rser are ignored)

### Multi2T.njunc
Number of junctions contained in Multi2T

### Multi2T.j
List of Junction objects series-connected within Multi2T object

### Multi2T.Vmid[ ]
List of intermediate voltages between series-connected junctions

### Multi2T.ui
User interface created by Multi2T.controls() which consists of plot, widget controls, and other outputs

### Multi2T.debugout
Hidden output of for debugging

## Multi2T.properties
### Multi2T.TC
Maximum of the contained Junction.TCs
### Multi2T.lightarea
Maximum of the contained Junction.lightareas
### Multi2T.totalarea
Maximum of the contained Junction.totalareas

## Multi2T.methods()
### Multi2T.copy()
Create a copy of this Multi2T object

### Multi2T.copy3T()
Create a Multi2T object from a Tandem3T object

### Multi2T.single(junc, copy=True)
Create a Multi2T object from a Junction object

### Multi2T.set(\**kwargs)
Controlled change of Multi2T and its Junction's attributes 

### Multi2T.proplist(key)
Create a list of the scalar attributes or properties of the Junctions within a Multi2T object

### Multi2T.controls()
Create a user interface Multi2T.ui which consists of plot, widget controls, and other outputs

### Multi2T.update()
Update Multi2T and its Junction's attributes into Multi2T.ui controls if they exist

### Multi2T.V2T(I)
Calculates the total Multi2T voltage as a function of the series-connected current

*Also sets Multi2T.Vmid[]*

Inputs scalar and outputs scalar.

Can be vectorized by

     V2Tvect = np.vectorize(self.V2T)
     
### Multi2T.I2T(V)
Calculates the series-connected current as a function of total current Multi2T voltage.

*Also sets Multi2T.Vmid[]*

Inputs scalar and outputs scalar.

Can be vectorized by

     I2Tvect = np.vectorize(self.I2T)

### Multi2T.Imaxrev()
maximum reverse-bias current (A) without Gsh or breakdown

### Multi2T.Voc()
Open-circuit voltate (V) of Muli2T object

### Multi2T.Isc()
Short-circuit current (A) of Multi2T object

### Multi2T.MPP(pnts=11, bplot=False, timer=False)
Fast method to calculate the maximum power point of Multi2T object.
Outputs dictionary:

    {"Voc":Voc, "Isc":Isc, "Vmp":Vmp, "Imp":Imp, "Pmp":Pmp,  "FF":FF}

### Multi2T.calcDark(hilog = 3, pdec = 5, timer=False)
Calculate a dark IV curve outputs: *(Idark, Vdark, Vdarkmid)*

### Multi2T.calcLight(pnts=21, Vmin=-0.5, timer=False, fast=False)
Calculate a light IV curve outputs: *(Vlight, Ilight, Plight, MPP)*

### Multi2T.plot(title='', pplot=False, dark=None, pnts=21, Vmin= -0.5, lolog = -8, hilog = 7, pdec = 5)
Create a light or dark plot modeled from Multi2T parameters. Outputs: *(mpl.Figure, mpl.Axes)*

## Tandem3T( ) Class
Three terminal (3T) tandem composed of two *Junctions*.
Four terminal (4T) tandems can be modeled as 3T tandems with no resistive coupling (Rz=0) but still require luminescent coupling. The 4T shunt (or breakdown) between the subcells is not treated but could become important for large voltage differences.

![3T and 4T Tandem Schematic](https://github.com/NREL/PVcircuit/blob/master/images/TandemsSchematic.png)

## Tandem3T.attributes
### Tandem3T.name
Name of this junction *string*

### Tandem3T.Rz
Common series resistance of Tandem3T z-terminal

### Tandem3T.top
Top Junction object of Tandem3T object

### Tandem3T.bot
Bottom Junction object of Tandem3T object

### Tandem3T.ui
User interface created by Tandem3T.controls() which consists of plot, widget controls, and other outputs

### Tandem3T.debugout
Hidden output of for debugging

## Tandem3T.properties
### Tandem3T.TC
Maximum of the contained Junction.TCs
### Tandem3T.lightarea
Maximum of the contained Junction.lightareas
### Tandem3T.totalarea
Maximum of the contained Junction.totalareas

## Tandem3T.methods()
### Tandem3T.copy()
Create a copy of this Tandem3T object

### Tandem3T.set(\**kwargs)
Controlled change of Tandem3T and its Junction's attributes 

### Tandem3T.controls()
Create a user interface Tandem3T.ui which consists of plot, widget controls, and other outputs

### Tandem3T.update()
Update Tandem3T and its Junction's attributes into Tandem3T.ui controls if they exist

### Tandem3T.V3T(iv3T)
Calcuate iv3T.(Vzt,Vrz,Vtr) from iv3T.(Iro,Izo,Ito) 

*input/output IV3T object*

### Tandem3T.J3Tabs(iv3T)
Calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito) from ABSOLUTE (Vz,Vr,Vt) mapped <- iv3T.(Vzt,Vrz,Vtr) 

*input/output IV3T object*

### Tandem3T.I3Trel(iv3T)
calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito) from RELATIVE iv3T.(Vzt,Vrz,Vtr) ignoring Vtr

*input/output IV3T object*

### Tandem3T._dI(Vz,Vzt,Vrz,temp3T)
Return dI = Iro + Izo + Ito function solved for dI(Vz)=0 in I3rel

*input Vzt, Vrz, IV3T object*

### Tandem3T.Voc3(meastype='CZ')
Triple Voc of 3T tandem returns IV3T object.

*returns IV3T object with one point*


        (Vzt, Vrz, Vtr) of (Iro = 0, Izo = 0, Ito = 0)
        
### Tandem3T.Isc3(meastype='CZ')
Triple Isc of 3T tandem returns IV3T object.

*returns IV3T object with one point*


        (Iro, Izo, Ito ) of (Vzt = 0, Vrz = 0, Vtr = 0) 
        
### Tandem3T.MPP(pnts=31, VorI= 'I', less = 2., bplot=False)
Iteratively find MPP from lines as experimentally done. Varying I is faster than varying V but initial guess is not as good.

*returns IV3T object with one point*

'less' must be > 1.0. If FF is really bad, may need larger 'less'

Use 'bplot' for debugging information

### Tandem3T.VI0(VIname, meastype='CZ')
Solve for mixed (V=0, I=0) zero power points using separate diodes for quick solutions

*returns IV3T object with one point*

### Tandem3T.VIpoint(zerokey, varykey, crosskey, meastype='CZ', pnts=11, bplot=False)
Solve for mixed (V=0, I=0) zero power points *does not work well*

### Tandem3T.specialpoints(meastype = 'CZ', bplot=False, fast=False)
Compile all the special zero power points into a labeled IV3T object

### Tandem3T.plot(pnts=31, meastype='CZ', oper = 'load2dev', cmap='terrain')
Calculate and plot Tandem3T device

## IV3T( ) Class
Structure to contain all the information about the operational state of a 3T tandem.

![3T Measurement Equipment](https://github.com/NREL/PVcircuit/blob/master/images/3Tmeasurementequipment.png)

Device parameters calculated for a 'Tandem3T' object.

    (Iro, Izo, Ito) <-> (Vzt, Vrz, Vtr)

Device paramters converted to load parameters for given measurement configuration (CZ, CR, CT).

    (Iro, Izo, Ito) <-> (IA, IB) 

    (Vzt, Vrz, Vtr) <-> (VA, VB)

Hexagonal representation of 3 device parameters in 2 dimensions.

    (Iro, Izo, Ito) <-> (Ixhex, Iyhex) 

    (Vzt, Vrz, Vtr) <-> (Vxhex, Vyhex)

## IV3T.attributes
### IV3T.name
*string* Name of this IV3T 

### IV3T.meastype
*string* Measurement type of IV3T object  'CZ', 'CR', 'CT', 'CF', 'CRo', 'CTo', 'CZo', or 'CFo' 

### IV3T.shape
Shape of numpy.array that contains array points of IV3T object. *tuple* either 1D ie. (npts,) or 2D (xpnts, ypnts)

### IV3T.xkey
*string* Name of the arraykey that is systematically varied.

### IV3T.ykey
*string* Name of the orthoganal arraykey that is systematically varied by IV3T.box() or constrained by IV3T.line() 

### IV3T.x
1D *ndarray* of *xkey* values of box or line

### IV3T.y 
1D *ndarray* of *ykey* values of box or line

### IV3T.names[ ]
List of labels of each point within IV3T using a *string* (optional but used by special points)

### IV3T.Iro[ ], IV3T.Izo[ ], IV3T.Ito[ ], IV3T.IA[ ], IV3T.IB[ ]
1D or 2D *ndarray* of current (A) values for each Tandem3T operating point.

[Iro,Izo,Ito] are device currents.

[IA, IB] are load currents relative to *meastype*

### IV3T.Vzt[ ], IV3T.Vrz[ ], IV3T.Vtr[ ], IV3T.VA[ ], IV3T.VB[ ]
1D or 2D *ndarray* of device voltage (V) values for each Tandem3T operating point

[Vzt,Vrz,Vtr] are device voltages.

[VA, VB] are load voltage relative to *meastype*

### IV3T.Ixhex[ ], IV3T.Iyhex[ ], IV3T.Vxhex[ ], IV3T.Vyhex[ ]
1D or 2D *ndarray* of device voltage (V) or current (A) values mapped onto a 2D isometric hexagonal representation.

### IV3T.Ptot[ ]
1D or 2D *ndarray* of total Tandem3T powers

## IV3T.methods()
### IV3T.copy()
Create a separate complete copy of a IV3T

### IV3T.set()

### IV3T.line(xkey, x0, x1, xn, ykey, yconstraint)
Create a 1D ndarray on xkey with evenly spaced values in IV3T..x
        
ykey is constrained to xkey with eval expression using 'x'

### IV3T.box(xkey, x0, x1, xn, ykey, y0, y1, yn)
Create a 2D ndarray for xkey and ykey with shape (xn, yn) with evenly spaced values. IV3T.x are values in one dimention. IV3T.y are values in orthogonal dimention

### IV3T.hexgrid(ax, VorI, step, xn=2, maxlines=10)
Add hexagonal grid lines to mpl.Axes. Range determined from self box IV3T object

### IV3T.nanpnt(index)
Make indexed point in each keyarray an *nan* where *index* is a tuple i = (i, ) or (i,j)

### IV3T.MPP(name='')
Find the MPP from within points of IV3T and return a new IV3T object with one point.

### IV3T.sizes(klist)
Return a range of sizes of the array attributes in klist. Returns *tuple* (nmin,nmax)

### IV3T.resize(self, shape, fillname = '')
Resize IV3T arrays and clears the values

### IV3T.append(iv3T)
Appends one IV3T object onto another

### IV3T.init(inlist,outlist)
Initialize output arrays to nan if input arrays are consistent

### IV3T.kirchhoff(two)
Apply or check Kirchoff's law on [Iro,Izo,Ito] or [Vzt,Vrz,Vtr]

Input a list of 2 or 3 of the device input keys:

            2 -> calculate the third device value from other two knowns
            3 -> check the validity of 3 device parameters
            
### IV3T.Pcalc(oper='dev2load', meastype=None)
Calculate Ptot after converting using oper = 'dev2load' or 'load2dev'

### IV3T.loadlabel(load, meastype=None)
Return descriptive axis label for load variables. Add an extra character to swap the loads: 'CRo','CTo','CZo', 'CFo'

### IV3T.convert(VorI, oper, meastype=None)
Calculate some array values from other array values. Can optionally set the meastype here.

VorI: 'V' or 'I'

oper: 'load2dev', 'dev2load', 'dev2hex', 'hex2dev' (not developed yet)

meastype: 'CR','CT','CZ','CF' or swap the loads: 'CRo','CTo','CZo', 'CFo' 

### IV3T.loadcsv(name, path, fileA, fileB, VorI, meastype, Iscale=1000.)
import csv file as data table into iv3T object

two 2D arrays with x and y index on top and left

    load variables:
    VA(IA,IB) & VB(IA,IB) .......... VorI='I'
    or
    IA(VA,VB) & IB(VA,VB) .......... VorI='V'
    Iscale converts current mA -> A or mA/cm2-> A
        
### IV3T.plot(xkey = None, ykey = None, zkey = None, inplot = None, cmap='terrain', ccont = 'black', bar = True)
Plot 2D IV3T object zkey(xkey,ykey) as image if evenly spaced or randomly spaced with contours

### IV3T.addpoints(ax, xkey, ykey, \**kwargs)
Plot IV3T points or lines onto existing axes
