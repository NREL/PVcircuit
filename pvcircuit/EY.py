# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.EY     use Ripalda's Tandem proxy spectra for energy yield
"""

import copy
import glob
import multiprocessing as mp
import os
import warnings
from functools import lru_cache

import numpy as np  # arrays
import pandas as pd
from parse import parse
from scipy import constants
from tqdm import tqdm, trange

import pvcircuit as pvc

warnings.warn(
    "The 'EY.py' module is deprecated and will be change in future version.",
    DeprecationWarning,
    stacklevel=2
)

#  from 'Tandems' project
# vectoriam = np.vectorize(physicaliam)
GITpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RIPpath = os.path.join(GITpath, "Tandems")  # assuming 'Tandems' is in parallel GitHub folders
FARMpath = os.path.join(RIPpath, "FARMS-NIT-clustered-spectra-USA", "")

# list of data locations
clst_axis = glob.glob(FARMpath + "*axis.clusters.npz")
clst_tilt = glob.glob(FARMpath + "*tilt.clusters.npz")
tclst_axis = glob.glob(FARMpath + "*axis.timePerCluster.npz")
tclst_tilt = glob.glob(FARMpath + "*tilt.timePerCluster.npz")
clst_axis.sort()
tclst_axis.sort()
clst_tilt.sort()
tclst_tilt.sort()


@lru_cache(maxsize=100)
def VMloss(type3T, bot, top, ncells):
    # calculates approximate loss factor for VM strings of 3T tandems
    if bot == 0:
        endloss = 0
    elif top == 0:
        endloss = 0
    elif type3T == "r":
        endloss = max(bot, top) - 1
    elif type3T == "s":
        endloss = bot + top - 1
    else:  # not Tandem3T
        endloss = 0
    lossfactor = 1 - endloss / ncells
    return lossfactor


# @lru_cache(maxsize=100)
def VMlist(mmax):
    # generate a list of VM configurations + 'MPP'=4T and 'CM'=2T
    # mmax < 10 for formating reasons

    sVM = ["MPP", "CM", "VM11"]
    for m in range(mmax + 1):
        for n in range(1, m):
            lcd = 2
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip2')
                continue
            lcd = 3
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip3')
                continue
            lcd = 5
            if m / lcd == round(m / lcd) and n / lcd == round(n / lcd):
                # print(n,m, 'skip5')
                continue
            # print(n,m, 'ok')
            sVM.append("VM" + str(m) + str(n))
    return sVM

def sandia_T(poa_global, wind_speed, temp_air):
    """ Sandia solar cell temperature model
    Adapted from pvlib library to avoid using pandas dataframes
    parameters used are those of 'open_rack_cell_polymerback'
    """

    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell

# def _calc_yield_async(i, bot, top, type3T, Jscs, Egs, TempCell, devlist, oper):
#     model = devlist[i]
#     if type3T == "2T":  # Multi2T
#         for ijunc in range(model.njuncs):
#             # model.j[ijunc].set(Eg=Egs[ijunc], Jext=Jscs[i, ijunc], TC=TempCell[i])
#             model.j[ijunc].set(Eg=Egs[i,ijunc], Jext=Jscs[i, ijunc], TC=25)
#         mpp_dict = model.MPP()  # oper ignored for 2T
#         Pmax = mpp_dict["Pmp"]
#     elif type3T in ["s", "r"]:  # Tandem3T
#         model.top.set(Eg=Egs[i,0], Jext=Jscs[i, 0], TC=TempCell[i])
#         # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
#         model.bot.set(Eg=Egs[i,1], Jext=Jscs[i, 1], TC=TempCell[i])
#         # model.bot.set(Eg=Egs[1], Jext=Jscs[i, 1], TC=25)
#         if oper == "MPP":
#             tempRz = model.Rz
#             model.set(Rz=0)
#             iv3T = model.MPP()
#             model.set(Rz=tempRz)
#         elif oper == "CM":
#             ln, iv3T = model.CM()
#         elif oper[:2] == "VM":
#             ln, iv3T = model.VM(bot, top)
#         else:
#             iv3T = pvc.iv3T.IV3T("bogus")
#             iv3T.Ptot[0] = 0
#         Pmax = iv3T.Ptot[0]
#     else:
#         Pmax = 0.0
#     # outPowerMP[i] = Pmax *1e4
#     return Pmax * 1e4

def _calc_yield_async(bot, top, type3T, Jscs, Egs, TempCell, devlist, oper):
    Pmax_out = np.zeros(len(Jscs))
    for i in range(len(Jscs)):
        model = devlist[i]
        if type3T == "2T":  # Multi2T
            for ijunc in range(model.njuncs):
                # model.j[ijunc].set(Eg=Egs[ijunc], Jext=Jscs[i, ijunc], TC=TempCell[i])
                model.j[ijunc].set(Eg=Egs[i,ijunc], Jext=Jscs[i, ijunc], TC=25)
            mpp_dict = model.MPP()  # oper ignored for 2T
            Pmax = mpp_dict["Pmp"]
        elif type3T in ["s", "r"]:  # Tandem3T
            model.top.set(Eg=Egs[i,0], Jext=Jscs[i, 0], TC=TempCell.iloc[i])
            # model.top.set(Eg=Egs[0], Jext=Jscs[i, 0], TC=TempCell)
            model.bot.set(Eg=Egs[i,1], Jext=Jscs[i, 1], TC=TempCell.iloc[i])
            # model.bot.set(Eg=Egs[1], Jext=Jscs[i, 1], TC=25)
            if oper == "MPP":
                tempRz = model.Rz
                model.set(Rz=0)
                iv3T = model.MPP()
                model.set(Rz=tempRz)
            elif oper == "CM":
                ln, iv3T = model.CM()
            elif oper[:2] == "VM":
                ln, iv3T = model.VM(bot, top)
            else:
                iv3T = pvc.iv3T.IV3T("bogus")
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0]
        else:
            Pmax = 0.0
        # outPowerMP[i] = Pmax *1e4
        Pmax_out[i] = Pmax * 1e4

    return Pmax_out

def cellmodeldesc(model, oper):
    # return description of model and operation
    # cell 'model' can be 'Multi2T' or 'Tandem3T'
    #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
    # Outputs (bot, top, ratio, type3T)

    # loss = 1.
    bot = 0
    top = 0
    type3T = ""

    if isinstance(model, pvc.multi2T.Multi2T):
        type3T = "2T"
    elif isinstance(model, pvc.tandem3T.Tandem3T):
        if model.top.pn == model.bot.pn:
            type3T = "r"
        else:
            type3T = "s"

        if oper == "MPP":
            ratio = -1.0
        elif oper == "CM":
            ratio = 0
        elif oper[:2] == "VM":
            bot, top = parse("VM{:1d}{:1d}", oper)
            # loss = VMloss(type3T, bot, top, ncells)
            ratio = bot / top
        else:
            print(oper + " not valid")
            ratio = np.nan
    else:
        print("unknown model" + str(type(model)))
        ratio = np.nan
        type3T = ""

    return bot, top, ratio, type3T


class TMY(object):
    """
    typical meterological year at a specific location
    """

    def __init__(self, i, tilt=False):

        if not tilt:
            # Boulder i=497
            clst = clst_axis
            tclst = tclst_axis
        else:
            # Boulder i=491
            clst = clst_tilt
            tclst = tclst_tilt

        self.tilt = tilt
        self.index = i
        _, tail = os.path.split(clst[i])
        self.name = tail.replace(".clusters.npz", "")
        # self.name =  clst[i].split("/")[-1][:-13]  #posix only
        self.longitude = float(self.name.split("_")[1])
        self.latitude = float(self.name.split("_")[0])
        self.altitude = float(self.name.split("_")[2])
        self.zone = float(self.name.split("_")[3])

        d1 = np.load(clst[i])
        td1 = np.load(tclst[i])

        arr_0 = d1["arr_0"]
        tarr_0 = td1["arr_0"]

        spec = arr_0[0, 154:172, :]
        tspec = tarr_0[0, 154:172]

        self.Temp = spec[:, 1].copy() * 1e6  # C
        self.Wind = spec[:, 2].copy() * 1e6  # m/s
        self.DayTime = arr_0[0, 0, -1].copy() * 24  # scalar
        self.NTime = tspec  # Fraction of 1

        aoi = spec[:, 5] * 1e6
        self.Angle = np.array(aoi.tolist())
        # aim = vectoriam(aoi)
        #        aim = physicaliam(aoi)
        # self.AngleMOD = np.array(aim.tolist())

        spec[:, :5] = 0
        spec[:, -1] = 0
        self.Irradiance = np.array(spec.tolist()).transpose()  # transpose for integration with QE

        # standard data
        pvcpath = os.path.dirname(os.path.dirname(__file__))
        datapath = os.path.join(pvcpath, "data", "")  # Data files here
        # datapath = os.path.abspath(os.path.relpath('../data/', start=__file__))
        # datapath = pvcpath.replace('/pvcircuit','/data/')
        ASTMfile = os.path.join(datapath, "ASTMG173.csv")
        dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)

        self.ref_wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)

        # calculate from spectral proxy data only
        self.SpecPower = np.trapz(self.Irradiance, x=self.ref_wvl, axis=0)  # optical power of each spectrum
        self.RefPower = np.trapz(pvc.qe.refspec, x=self.ref_wvl, axis=0)  # optical power of each reference spectrum
        # self.SpecPower = PintMD(self.Irradiance)
        # self.RefPower = PintMD(pvc.qe.refspec)
        self.TempCell = sandia_T(self.SpecPower, self.Wind, self.Temp)
        self.inPower = self.SpecPower * self.NTime  # spectra power*(fractional time)
        self.YearlyEnergy = self.inPower.sum() * self.DayTime * 365.25 / 1000  # kWh/m2/yr

        self.outPower = np.empty_like(self.inPower)  # initialize outPower

        self.Jdbs = None
        self.Egs = None
        self.JscSTCs = None
        self.Jscs = None

    def cellbandgaps(self, EQE, TC=25):
        # subcell Egs for a given EQE class
        self.Jdbs, self.Egs = EQE.Jdb(TC)  # Eg from EQE same at all temperatures

    def cellcurrents(self, EQE, STC=False):
        # subcell currents and Egs and under self TMY for a given EQE class

        # self.JscSTCs = JintMD(EQE, xEQE, pvc.qe.refspec)/1000.
        # self.Jscs = JintMD(EQE, xEQE, self.Irradiance) /1000.
        if STC:
            self.JscSTCs = EQE.Jint(pvc.qe.refspec) / 1000.0
        else:
            self.Jscs = EQE.Jint(self.Irradiance) / 1000.0

    def cellSTCeff(self, model, oper, iref=1):
        # max power of a cell under a reference spectrum
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # iref = 0 -> space
        # iref = 1 -> global
        # iref = 2 -> direct
        # Outputs
        # - STCeff efficiency of cell under reference spectrum (space,global,direct)

        bot, top, _, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc reference spectra efficiency
        if iref == 0:
            Tref = 28.0  # space
        else:
            Tref = 25.0  # global and direct

        if type3T == "2T":  # Multi2T
            for ijunc in range(model.njuncs):
                model.j[ijunc].set(Eg=self.Egs[ijunc], Jext=self.JscSTCs[iref, ijunc], TC=Tref)
            mpp_dict = model.MPP()  # oper ignored for 2T
            Pmax = mpp_dict["Pmp"] * 10.0
        elif type3T in ["s", "r"]:  # Tandem3T
            model.top.set(Eg=self.Egs[0], Jext=self.JscSTCs[iref, 0], TC=Tref)
            model.bot.set(Eg=self.Egs[1], Jext=self.JscSTCs[iref, 1], TC=Tref)
            if oper == "MPP":
                iv3T = model.MPP()
            elif oper == "CM":
                _, iv3T = model.CM()
            elif oper[:2] == "VM":
                _, iv3T = model.VM(bot, top)
            else:
                print(oper + " not valid")
                iv3T = pvc.iv3T.IV3T("bogus")
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0] * 10.0
        else:
            Pmax = 0.0

        STCeff = Pmax * 1000.0 / self.RefPower[iref]

        return STCeff

    def cellEYeff(self, model, oper):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # Outputs
        # - EYeff energy yield efficiency = EY/YearlyEnergy
        # - EY energy yield of cell [kWh/m2/yr]

        bot, top, _, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc EY, etc
        for i in trange(len(self.inPower), leave=True, desc='single core'):
            if type3T == "2T":  # Multi2T
                for ijunc in range(model.njuncs):
                    model.j[ijunc].set(Eg=self.Egs[ijunc], Jext=self.Jscs[i, ijunc], TC=self.TempCell.iloc[i])
                mpp_dict = model.MPP()  # oper ignored for 2T
                Pmax = mpp_dict["Pmp"]
            elif type3T in ["s", "r"]:  # Tandem3T
                model.top.set(Eg=self.Egs[0], Jext=self.Jscs[i, 0], TC=self.TempCell[i])
                model.bot.set(Eg=self.Egs[1], Jext=self.Jscs[i, 1], TC=self.TempCell[i])
                if oper == "MPP":
                    iv3T = model.MPP()
                elif oper == "CM":
                    _, iv3T = model.CM()
                elif oper[:2] == "VM":
                    _, iv3T = model.VM(bot, top)
                else:
                    iv3T = pvc.iv3T.IV3T("bogus")
                    iv3T.Ptot[0] = 0
                Pmax = iv3T.Ptot[0]
            else:
                Pmax = 0.0

            self.outPower[i] = Pmax * self.NTime[i] * 10000.0

        EY = sum(self.outPower) * self.DayTime * 365.25 / 1000  # kWh/m2/yr
        EYeff = EY / self.YearlyEnergy

        return EY, EYeff




class Meteo(object):
    """
    Meteorological environmental data and spectra
    """

    def __init__(self, wavelength, spectra, ambient_temperature, wind, daytime):

        self.temp = ambient_temperature  # [degC]
        self.wind = wind  # [m/s]
        self.daytime = daytime  # daytime vector

        self.wavelength = wavelength
        self.spectra = spectra  # transpose for integration with QE


        # standard data
        pvcpath = os.path.dirname(os.path.dirname(__file__))
        datapath = os.path.join(pvcpath, "data", "")  # Data files here
        # datapath = os.path.abspath(os.path.relpath('../data/', start=__file__))
        # datapath = pvcpath.replace('/pvcircuit','/data/')
        ASTMfile = os.path.join(datapath, "ASTMG173.csv")
        dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)

        self.ref_wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)

         # calculate from spectral proxy data only
        self.SpecPower = pd.Series(np.trapz(spectra, x=wavelength), index=spectra.index) # optical power of each spectrum
        self.RefPower = np.trapz(pvc.qe.refspec, x=self.ref_wvl, axis=0)  # optical power of each reference spectrum
        self.TempCell = sandia_T(self.SpecPower, self.wind, self.temp)
        self.inPower = self.SpecPower  # * self.NTime  # spectra power*(fractional time)
        self.outPower = None
        # construct a results dataframe for better handling
        self.models=[]
        self.operation_modes=[]
        self.tandem_types=[]
        self.EnergyIn = np.trapz(self.SpecPower, self.daytime.values.astype(np.int64)) / 1e9 / 60  # kWh/m2/yr

        self.average_photon_energy = None # is calcluated when running calc_ape

    def cellbandgaps(self, EQE, TC=25):
        # subcell Egs for a given EQE class
        self.Jdbs, self.Egs = EQE.Jdb(TC)  # Eg from EQE same at all temperatures

    def cellcurrents(self, EQE, STC=False):
        # subcell currents and Egs and under self TMY for a given EQE class

        if STC:
            self.JscSTCs = EQE.Jint(pvc.qe.refspec) / 1000.0
        else:
            self.Jscs = EQE.Jint(self.spectra.T, xspec=self.wavelength) / 1000.0

    def cellSTCeff(self, model, oper, iref=1):
        # max power of a cell under a reference spectrum
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # iref = 0 -> space
        # iref = 1 -> global
        # iref = 2 -> direct
        # Outputs
        # - STCeff efficiency of cell under reference spectrum (space,global,direct)

        bot, top, ratio, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc reference spectra efficiency
        if iref == 0:
            Tref = 28.0  # space
        else:
            Tref = 25.0  # global and direct

        if type3T == "2T":  # Multi2T
            for ijunc in range(model.njuncs):
                model.j[ijunc].set(TC=Tref)
            mpp_dict = model.MPP()  # oper ignored for 2T
            Pmax = mpp_dict["Pmp"] * 10.0
            ratio = 0.0
        elif type3T in ["s", "r"]:  # Tandem3T
            model.top.set(TC=Tref)
            model.bot.set(TC=Tref)
            if oper == "MPP":
                tempRz = model.Rz
                model.set(Rz=0)
                iv3T = model.MPP()
                model.set(Rz=tempRz)

            elif oper == "CM":
                ln, iv3T = model.CM()
            elif oper[:2] == "VM":
                ln, iv3T = model.VM(bot, top)
            else:
                print(oper + " not valid")
                iv3T = pvc.iv3T.IV3T("bogus")
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0] * 10.0
        else:
            Pmax = 0.0

        STCeff = Pmax * 1000.0 / self.RefPower[iref]

        return STCeff

    def cellEYeff(self, model, oper):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # Outputs
        # - EYeff energy yield efficiency = EY/YearlyEnergy
        # - EY energy yield of cell [kWh/m2/yr]

        bot, top, ratio, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc EY, etc
        outPower = np.empty((self.inPower.shape[0],1))  # initialize

        for i in trange(len(outPower)):

            if type3T == "2T":  # Multi2T
                for ijunc in range(model.njuncs):
                    model.j[ijunc].set(Eg=self.Egs[i, ijunc], Jext=self.Jscs[i, ijunc], TC=self.TempCell.iloc[i])
                mpp_dict = model.MPP()  # oper ignored for 2T
                Pmax = mpp_dict["Pmp"]
            elif type3T in ["s", "r"]:  # Tandem3T
                model.top.set(Eg=self.Egs[i, 0], Jext=self.Jscs[i, 0], TC=self.TempCell.iloc[i])
                model.bot.set(Eg=self.Egs[i, 1], Jext=self.Jscs[i, 1], TC=self.TempCell.iloc[i])
                if oper == "MPP":
                    tempRz = model.Rz
                    model.set(Rz=0)
                    iv3T = model.MPP()
                    model.set(Rz=tempRz)
                elif oper == "CM":
                    ln, iv3T = model.CM()
                elif oper[:2] == "VM":
                    ln, iv3T = model.VM(bot, top)
                else:
                    iv3T = pvc.iv3T.IV3T("bogus")
                    iv3T.Ptot[0] = 0
                Pmax = iv3T.Ptot[0]
            else:
                Pmax = 0.0

            outPower[i] = Pmax * 10000

        if self.outPower is None:
            self.outPower = outPower
        elif self.outPower.shape[0] == outPower.shape[0]:
            self.outPower = np.concatenate([self.outPower, outPower], axis=1)

        self.models.append(model)
        self.operation_modes.append(oper)
        self.tandem_types.append(type3T)

        # EnergyOut = np.trapz(self.outPower, self.daytime.values.astype(int)) / 1e9 / 60  # kWh/m2/yr
        EnergyOut = np.trapz(outPower, self.daytime.values.astype(np.int64), axis=0) / 1e9 / 60  # kWh/m2/yr
        EYeff = EnergyOut / self.EnergyIn

        return EnergyOut, EYeff

    # def cellEYeffMP(self, model, oper):
    #     # max power of a cell under self TMY
    #     # self.Jscs and self.Egs must be calculate first using cellcurrents
    #     # Inputs
    #     # cell 'model' can be 'Multi2T' or 'Tandem3T'
    #     #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
    #     # Outputs
    #     # - EYeff energy yield efficiency = EY/YearlyEnergy
    #     # - EY energy yield of cell [kWh/m2/yr]

    #     bot, top, ratio, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

    #     # calc EY, etc
    #     outPowerMP = np.empty_like(self.inPower)  # initialize

    #     cpu_count = mp.cpu_count()
    #     print(f"running multiprocess with {cpu_count} pools")
    #     with tqdm(total=len(self.inPower), leave=True, desc = f"Multi processing {oper}") as pbar:

    #         dev_list = [copy.deepcopy(model) for _ in range(len(self.Jscs))]
    #         with mp.Pool(cpu_count) as pool:
    #             def callback(*args):
    #                 # callback
    #                 pbar.update()
    #                 return

    #             results = list(
    #                 pool.apply_async(_calc_yield_async, args=(i, bot, top, type3T, self.Jscs, self.Egs, self.TempCell, dev_list, oper), callback=callback)
    #                 for i, _ in enumerate(self.inPower)
    #             )
    #             results = [r.get() for r in results]

    #     self.outPowerMP = results

    #     EnergyOut = np.trapz(self.outPowerMP, self.daytime.values.astype(np.int64)) / 1e9 / 60  # kWh/m2/yr
    #     EYeff = EnergyOut / self.EnergyIn
    #     return EnergyOut, EYeff

    def cellEYeffMP(self, model, oper):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # Outputs
        # - EYeff energy yield efficiency = EY/YearlyEnergy
        # - EY energy yield of cell [kWh/m2/yr]

        bot, top, ratio, type3T = cellmodeldesc(model, oper)  # ncells does not matter here

        # calc EY, etc
        outPowerMP = np.empty_like(self.inPower)  # initialize

        # Split data into chunks for workers
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.Jscs))
        chunk_size =  min(len(chunk_ids) // cpu_count, max_chunk_size)

        chunks = [chunk_ids[i:i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]

        print(f"running multiprocess with {cpu_count} pools")
        with tqdm(total=len(self.inPower), leave=True, desc = f"Multi processing {oper}") as pbar:

            dev_list = np.array([copy.deepcopy(model) for _ in range(len(self.Jscs))])
            with mp.Pool(cpu_count) as pool:
                def callback(*args):
                    # callback
                    pbar.update(len(args[0]))
                    return

                # Assign tasks to workers
                jobs = [pool.apply_async(_calc_yield_async, args=(bot, top, type3T, self.Jscs[chunk], self.Egs[chunk], self.TempCell.iloc[chunk], dev_list[chunk], oper), callback=callback) for chunk in chunks]
                # Get results from workers
                results = [item for job in jobs for item in job.get()]

        self.outPowerMP = results

        EnergyOut = np.trapz(self.outPowerMP, self.daytime.values.astype(np.int64)) / 1e9 / 60  # kWh/m2/yr
        EYeff = EnergyOut / self.EnergyIn
        return EnergyOut, EYeff


    def calc_ape(self):
        """
        Calcualtes the average photon energy (APE) of the spectra
        """

        phi = self.spectra * (self.wavelength * 1e-9) / constants.h / constants.c
        self.average_photon_energy = np.trapz(x=self.wavelength, y=self.spectra.values) / constants.e / np.trapz(x=self.wavelength, y=phi.values)


    def filter_ape(self, min_ape:float = 0, max_ape:float = 10):
        """
        filter the average photon energy (APE)

        Args:
            min_ape (float, optional): min value of th APE. Defaults to 0.
            max_ape (float, optional): max value of the APE. Defaults to 10.
        """
        if self.average_photon_energy is None:
            self.calc_ape()

        self_copy = copy.deepcopy(self)
        ape_mask = (self_copy.average_photon_energy > min_ape) & (self_copy.average_photon_energy < max_ape)

        self_copy.daytime = self_copy.daytime[ape_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[ape_mask]
        self_copy.spectra = self_copy.spectra[ape_mask]
        self_copy.SpecPower = self_copy.SpecPower[ape_mask]
        self_copy.TempCell = self_copy.TempCell[ape_mask]

        assert len(self_copy.spectra) == len(self_copy.SpecPower) == len(self_copy.TempCell) == len(self_copy.average_photon_energy)
        return self_copy



    def filter_spectra(self, min_spectra:float = 0, max_spectra:float = 10):
        """
        spectral data

        Args:
            min_spectra (float, optional): min value of the spectra. Defaults to 0.
            max_spectra (float, optional): max value of the spectra. Defaults to 10.
        """


        self_copy = copy.deepcopy(self)
        spectra_mask = (self_copy.spectra >= min_spectra).all(axis=1) & (self_copy.spectra < max_spectra).all(axis=1)
        self_copy.daytime = self_copy.daytime[spectra_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[spectra_mask]
        self_copy.spectra = self_copy.spectra[spectra_mask]
        self_copy.SpecPower = self_copy.SpecPower[spectra_mask]
        self_copy.TempCell = self_copy.TempCell[spectra_mask]

        assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy




    def filter_custom(self, filter_array:bool):
        """
        Applys a custom filter ot the meteo data
        Args:
            filter_array (bool): Filter array to apply to the data
        """
        # assert len(filter_array) == len(self.spectra) == len(self.SpecPower) == len(self.TempCell)


        self_copy = copy.deepcopy(self)

        # self_copy.average_photon_energy = self_copy.average_photon_energy[filter_array]
        # self_copy.spectra = self_copy.spectra[filter_array]
        # self_copy.SpecPower = self_copy.SpecPower[filter_array]
        # self_copy.TempCell = self_copy.TempCell[filter_array]

        for attr_name in vars(self):
            if hasattr(getattr(self_copy, attr_name), '__len__'):
                attr = getattr(self_copy, attr_name)
                if len(attr) == len(filter_array):
                    setattr(self_copy,attr_name,attr[filter_array])


        # assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy

    def reindex(self, index:bool, method="nearest", tolerance=pd.Timedelta(seconds=30)):
        """
        Reindex according to indexer
        Args:
            filter_array (bool): Filter array to apply to the data
        """

        self_copy = copy.deepcopy(self)

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                setattr(self_copy,attr_name,attr.reindex(index=index, method=method, tolerance=tolerance))


        return self_copy
