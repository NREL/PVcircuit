# pvcircuit 
*pvcircuit* contains objects that are building blocks for PV modeling.

Based on publications:

[J. F. Geisz, et al., IEEE Journal of Photovoltaics 5, p. 1827 (2015).](http://dx.doi.org/10.1109/JPHOTOV.2015.2478072)

[J. F. Geisz, et al., Cell Reports Physical Science 2, p. 100677 (2021).](https://doi.org/10.1016/j.xcrp.2021.100677)

### Junction( ) Class
A single *Junction* that can be combined together to form tandem and multijunction solar cells.

![Junction Schematic](https://github.nrel.gov/jgeisz/PVcircuit/blob/master/images/junction%20equations.png) 

The *Junction* is modeled by an arbitrary number of parallel diodes such as n=1, n=2, n=2/3, or any other ideality factor.
The temperature dependance of each diode saturation current J0n is determined relative to the Jdb(Eg,T). 

### Multi2T( ) Class
Two terminal multijunction device composed of any number of series connected *Junctions*. The sum of all Rser is an attribute of the *Multi2T* object and the Rser attributes of each sub *Junction* are ignored

![2T Multijunction Schematic](https://github.nrel.gov/jgeisz/PVcircuit/blob/master/images/MultijunctionSchematic.png)

### Tandem3T( ) Class
Three terminal (3T) tandem composed of two *Junctions*.
Four terminal (4T) tandems can be modeled as 3T tandems with no resistive coupling (Rz=0) but still require luminescent coupling. The 4T shunt (or breakdown) between the subcells is not treated but could become important for large voltage differences.

![3T and 4T Tandem Schematic](https://github.nrel.gov/jgeisz/PVcircuit/blob/master/images/TandemsSchematic.png)

### IV3T( ) Class
Structure to contain all the information about the operational state of a 3T tandem.

![3T Measurement Equipment](https://github.nrel.gov/jgeisz/PVcircuit/blob/master/images/3Tmeasurementequipment.png)

Device parameters calculated for a 'Tandem3T' object.

    (Iro, Izo, Ito) <-> (Vzt, Vrz, Vtr)

Device paramters converted to load parameters for given measurement configuration (CZ, CR, CT).

    (Iro, Izo, Ito) <-> (IA, IB) 

    (Vzt, Vrz, Vtr) <-> (VA, VB)

Hexagonal representation of 3 device parameters in 2 dimensions.

    (Iro, Izo, Ito) <-> (Ixhex, Iyhex) 

    (Vzt, Vrz, Vtr) <-> (Vxhex, Vyhex)

