# # 4J IMM solar cell (MM927) from:
# ###  J. F. Geisz, et al., IEEE Journal of Photovoltaics __5__, 1827 (2015). 
import pandas as pd
import numpy as np
import pvcircuit as pvc
# import ipywidgets as widgets

# %%
totalarea=1.15
MM927 = pvc.Multi2T(name='MM927',Eg_list = [1.83,1.404,1.049,0.743], Jext=.012, Rser=0.1,area=1)
MM927.j[0].update(Jext=0.01196,n=[1,1.6],J0ratio=[31,4.5],totalarea=totalarea)
MM927.j[1].update(Jext=0.01149,n=[1,1.8],J0ratio=[17,42],beta=14.3,totalarea=totalarea)
MM927.j[2].update(Jext=0.01135,n=[1,1.4],J0ratio=[51,14],beta=8.6,totalarea=totalarea)
MM927.j[3].update(Jext=0.01228,n=[1,1.5],J0ratio=[173,79],beta=10.5,totalarea=totalarea)
MM927.j[3].RBB_dict={'method':'JFG', 'mrb':43., 'J0rb':0.3, 'Vrb':0.}

# %%
plotit = False
if plotit:
    import matplotlib.pyplot as plt
    dfig, lfig, dax, lax, Vlight, Ilight = MM927.plot()
    plt.show()
# %%

info = MM927.MPP()

print(info)

# {'Voc': 3.425330977574876,
 # 'Isc': 0.011350991117526023,
 # 'Vmp': 3.0129358068534247,
 # 'Imp': 0.011073118854968986,
 # 'Pmp': 0.033362596291679855,
 # 'FF': 0.8580715725119747}

assert np.isclose(info['Voc'] , 3.425330977574876, rtol=1e-5)

assert np.isclose(info['Isc'] , 0.011350991117526023, rtol=1e-5)

assert np.isclose(info['Vmp'], 3.0129358068534247, rtol=1e-5)

assert np.isclose(info['Imp'], 0.011073118854968986, rtol=1e-5)

# %%
