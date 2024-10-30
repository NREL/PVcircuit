# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.qe    # functions for QE analysis
"""

import copy
import math  # simple math
import os
from functools import lru_cache
from pathlib import Path
from time import time

import ipywidgets as widgets
import matplotlib as mpl  # plotting
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd  # dataframes
from cycler import cycler
from IPython.display import display

# from scipy.integrate import trapezoid
from scipy import constants  # physical constants
from scipy.integrate import trapezoid

# from scipy.special import lambertw, gammaincc, gamma   #special functions
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import (
    brentq,  # root finder
    curve_fit,
    fsolve,
)
from scipy.special import erfc

from pvcircuit.junction import *
from pvcircuit.multi2T import *

# colors
junctioncolors = [
    ["black"],  # 0J
    ["red"],  # 1J
    ["blue", "red"],  # 2J
    ["blue", "green", "red"],  # 3J
    ["blue", "green", "orange", "red"],  # 4J
    ["purple", "blue", "green", "orange", "red"],  # 5J
    ["purple", "blue", "green", "black", "orange", "red"],
]  # 6J

# constants
k_q = constants.k / constants.e
hc_k = constants.h * constants.c / constants.k * 1e9  # for wavelength (nm)
DB_PREFIX = 2.0 * np.pi * constants.e * (constants.k / constants.h) ** 3 / (constants.c) ** 2 / 1.0e4  # about 1.0133e-8 for Jdb[A/cm2]
nm2eV = constants.h * constants.c / constants.e * 1e9
JCONST = 1000 / 100 / 100 / nm2eV  # mA/cm2
DBWVL_PREFIX = 2.0 * np.pi * constants.c * constants.e / 100 / 100  # A/cm2


# standard data
pvcpath = Path(__file__).parent
datapath = pvcpath.joinpath("data")  # Data files here
ASTMfile = datapath.joinpath("ASTMG173.csv")

try:
    dfrefspec = pd.read_csv(ASTMfile, index_col=0, header=2)
    wvl = dfrefspec.index.to_numpy(dtype=np.float64, copy=True)
    refspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # all three reference spectra
    refnames = ["space", "global", "direct"]
    AM0 = refspec[:, 0]  # dfrefspec['space'].to_numpy(dtype=np.float64, copy=True)  # 1348.0 W/m2
    AM15G = refspec[:, 1]  # dfrefspec['global'].to_numpy(dtype=np.float64, copy=True) # 1000.5 W/m2
    AM15D = refspec[:, 2]  # dfrefspec['direct'].to_numpy(dtype=np.float64, copy=True) # 900.2 W/m2
except:
    print(pvcpath)
    print(datapath)
    print(ASTMfile)


def ordinal(n):
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = suffixes.get(n % 10, "th")
    return f"{n}{suffix}"


def wavelength_to_photonenergy(wavelength):
    return constants.h * constants.c / (wavelength * 1e-9) / constants.e


def photonenergy_to_wavelength(photonenergy):
    return constants.h * constants.c / (photonenergy * 1e-9) / constants.e


def _normalize(eqe: pd.DataFrame) -> pd.DataFrame:
    eqe_min = np.nanmin(eqe)
    eqe_max = np.nanmax(eqe)
    return (eqe - eqe_min) / (eqe_max - eqe_min)


def _eq_solve_Eg(Eg, *data):
    x, y = data
    return trapezoid(x * y, x) / trapezoid(y, x) - Eg


def _gaussian(x, a, x0, sigma):
    return 1 * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def JdbMD(EQE, xEQE, TC, Eguess=1.0, kTfilter=3, bplot=False):
    """
    calculate detailed-balance reverse saturation current
    from EQE vs xEQE
    xEQE in nm, can optionally use (start, step) for equally spaced data
    debug on bplot
    """
    Vthlocal = Vth(TC)  # kT
    EQE = np.array(EQE)  # ensure numpy
    if EQE.ndim == 1:  # 1D EQE[lambda]
        (nQlams,) = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  # 2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return "dims in EQE:" + str(EQE.ndim)

    Eguess = np.array([Eguess] * njuncs)

    if len(xEQE) == 2:  # evenly spaced x-values (start, stop)
        start, stop = xEQE
        # stop =  start + step * (nQlams - 1)
        # xEQE, step = np.linspace(start, stop, nQlams, dtype=np.float64, retstep=True)
        xEQE = np.linspace(start, stop, nQlams, dtype=np.float64)
    else:  # arbitrarily spaced x-values
        xEQE = np.array(xEQE, dtype=np.float64)
        start = min(xEQE)
        stop = max(xEQE)
        # step = xEQE[1]-xEQE[0]  #first step

    if xEQE.ndim != 1:  # need 1D with same length as EQE(lam)
        return "dims in xEQE:" + str(xEQE.ndim) + "!=1"
    elif len(xEQE) != nQlams:
        return "nQlams:" + str(len(xEQE)) + "!=" + str(nQlams)

    Egvect = np.vectorize(EgFromJdb)
    EkT = nm2eV / Vthlocal / xEQE
    blackbody = np.expand_dims(DBWVL_PREFIX / (xEQE * 1e-9) ** 4 / np.expm1(EkT), axis=1)

    for count in range(10):
        nmfilter = nm2eV / (Eguess - Vthlocal * kTfilter)  # MD [652., 930.]
        if njuncs == 1:
            EQEfilter = np.expand_dims(EQE.copy(), axis=1)
        else:
            EQEfilter = EQE.copy()

        for i, lam in enumerate(xEQE):
            EQEfilter[i, :] *= lam < nmfilter  # zero EQE about nmfilter

        DBintegral = blackbody * EQEfilter
        Jdb = trapezoid(DBintegral, x=(xEQE * 1e-9), axis=0)
        Egnew = Egvect(TC, Jdb)
        if bplot:
            print(Egnew, max((Egnew - Eguess) / Egnew))
        if np.amax((Egnew - Eguess) / Egnew) < 1e-6:
            break
        else:
            Eguess = Egnew

    if bplot:
        efig, eax = plt.subplots()
        eax.plot(xEQE, DBintegral[:, 0], c="blue", lw=2, marker=".")
        if njuncs > 1:
            reax = eax.twinx()  # right axis
            reax.plot(xEQE, DBintegral[:, 1], c="red", lw=2, marker=".")

    return Jdb, Egnew


def PintMD(Pspec, xspec=wvl):
    # optical power of spectrum over full range
    return JintMD(None, None, Pspec, xspec)


def JintMD(EQE, xEQE, Pspec, xspec=wvl):
    """
    integrate over spectrum or spectra
    if EQE is None -> calculate Power in [W/m2]
        total power=int(Pspec) over xEQE range
    else             -> caluculate J = spectra * lambda * EQE(lambda)
        Jsc = int(Pspec*QE[0]*lambda) in [mA/cm2]
        EQE optionally scalar for constant over xEQE range
        integrate over full spectrum is xEQE is None

    integrate multidimentional EQE(lambda)(junction) times MD reference spectra Pspec(lambda)(ispec)
    external quantum efficiency EQE[unitless] x-units = nm,
    reference spectra Pspec[W/m2/nm] x-units = nm
    optionally Pspec as string 'space', 'global', 'direct' or '' for all three
    xEQE in nm, can optionally
        use (start, step) for equally spaced data
        use None for same as xspec
    default x values for Pspec from wvl
    """

    # check spectra input
    if type(Pspec) is str:  # optional string space, global, direct
        if Pspec in dfrefspec.columns:
            Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
        else:
            Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # just use refspec instead of error
    else:
        Pspec = np.array(Pspec, dtype=np.float64)  # ensure numpy

    if Pspec.ndim == 1:  # 1D Pspec[lambda]
        (nSlams,) = Pspec.shape
        nspecs = 1
    elif Pspec.ndim == 2:  # 2D Pspec[lambda, ispec]
        nSlams, nspecs = Pspec.shape
    else:
        return "dims in Pspec:" + str(Pspec.ndim)

    # check EQE input
    EQE = np.array(EQE)  # ensure numpy
    if EQE.ndim == 0:  # scalar or None
        if np.any(EQE):
            nQlams = 1  # scalar -> return current
            njuncs = 1
        else:
            nQlams = 1  # None or False -> return power
            njuncs = 0
    elif EQE.ndim == 1:  # 1D EQE[lambda]
        (nQlams,) = EQE.shape
        njuncs = 1
    elif EQE.ndim == 2:  # 2D EQE[lambda, junction]
        nQlams, njuncs = EQE.shape
    else:
        return "dims in EQE:" + str(EQE.ndim)

    # check x range input
    xEQE = np.array(xEQE, dtype=np.float64)
    if xEQE.ndim == 0:  # scalar or None
        xEQE = xspec  # use spec range if no EQE range
    if xEQE.ndim == 1:  # 1D
        (nxEQE,) = xEQE.shape
        start = min(xEQE)
        stop = max(xEQE)
        if nxEQE == 2:  # evenly spaced x-values (start, stop)
            xEQE = np.linspace(start, stop, max(nQlams, 2), dtype=np.float64)
    else:
        return "dims in xEQE:" + str(xEQE.ndim)

    if nQlams == 1:
        EQE = np.full_like(xEQE, EQE)

    if xspec.ndim != 1:  # need 1D with same length as Pspec(lam)
        return "dims in xspec:" + str(xspec.ndim) + "!=1"
    elif len(xspec) != nSlams:
        return "nSlams:" + str(len(xspec.ndim)) + "!=" + str(nSlams)

    if xEQE.ndim != 1:  # need 1D with same length as EQE(lam)
        return "dims in xEQE:" + str(xEQE.ndim) + "!=1"
    elif nQlams == 1:
        pass
    elif len(xEQE) != nQlams:
        return "nQlams:" + str(len(xEQE)) + "!=" + str(nQlams)

    # find start and stop index  of nSlams
    n0 = 0
    n1 = nSlams - 1
    for i, lam in enumerate(xspec):
        if lam <= min(start, stop):
            n0 = i
        elif lam <= max(start, stop):
            n1 = i
        else:
            break
    xrange = xspec[n0 : n1 + 1]  # range of xspec values within xEQE range
    nrange = abs(n1 + 1 - n0)
    if njuncs == 0:  # calculate power over xrange
        if nspecs == 1:
            Jintegral = Pspec.copy()[n0 : n1 + 1]  # for Ptot
        else:
            Jintegral = Pspec.copy()[n0 : n1 + 1, :]  # for Ptot

    else:  # calculate J over xrange
        # print(xrange.shape, xEQE.shape, EQE.shape, Pspec.shape)
        EQEinterp = interp1d(xEQE, EQE, axis=0, fill_value=0)  # interpolate along axis=0
        Jintegral = np.zeros((nrange, nspecs, njuncs), dtype=np.float64)  # 3D array
        if njuncs == 1:
            EQEfine = np.expand_dims((EQEinterp(xrange) * xrange), axis=1) * JCONST  # lambda*EQE(lambda)[lambda,1]
        else:
            EQEfine = EQEinterp(xrange) * xrange[:, np.newaxis] * JCONST  # lambda*EQE(lambda)[lambda,junc]
        for ijunc in range(0, njuncs):
            if nspecs == 1:
                Jintegral[:, 0, ijunc] = Pspec.copy()[n0 : n1 + 1]  # for Ptot
            else:
                Jintegral[:, :, ijunc] = Pspec.copy()[n0 : n1 + 1, :]  # for Ptot
            Jintegral[:, :, ijunc] *= EQEfine[:, np.newaxis, ijunc]

    Jint = trapezoid(Jintegral, x=xrange, axis=0)
    # print(xrange.shape, EQEfine.shape, EQE.shape, Pspec.shape, Jintegral.shape)
    # print(nSlams, nspecs, njuncs, nQlams, start, stop, xspec[n0], xspec[n1], n0, n1)
    return Jint


@lru_cache(maxsize=100)
def JdbFromEg(TC, Eg, dbsides=1.0, method=None):
    """
    return the detailed balance dark current
    assuming a square EQE
    Eg[=]eV
    TK[=]K
    returns Jdb[=]A/cm2

    optional parameters
    method: 'gamma'
    dbsides:    single-sided->1.  bifacial->2.
    see special functions, Andrews p.71
    see Geisz EL&LC paper, King EUPVSEC
    """
    EgkT = Eg / Vth(TC)
    TKlocal = TK(TC)

    if str(method).lower == "gamma":
        # use special function incomplete gamma
        # gamma(3)=2.0 not same incomplete gamma as in Igor
        Jdb = DB_PREFIX * TKlocal**3.0 * gammaincc(3.0, EgkT) * 2.0 * dbsides
    else:
        # Jdb as in Geisz et al.
        Jdb = DB_PREFIX * TKlocal**3.0 * (EgkT * EgkT + 2.0 * EgkT + 2.0) * np.exp(-EgkT) * dbsides  # units from DB_PREFIX

    return Jdb


@lru_cache(maxsize=100)
def EgFromJdb(TC, Jdb, Eg=1.0, eps=1e-6, itermax=100, dbsides=1.0):
    """
    see GetData AT_Egcalc
    return the bandgap from the Jdb
    assuming a square EQE
    iterates using gammaInc(3,x)=2*exp(-x)*(1+x+x^2/2)
    see special functions, Andrews p.73

    optional parameters
    Eg=1.0 eV    #initial guess
    eps=0.001    #tolerance
    itermax=100 #maximum iterations
    dbsides=1.  #bifacial->2.
    """

    Vthlocal = Vth(TC)
    TKlocal = TK(TC)
    x0 = Eg / Vthlocal
    off = np.log(2.0 * dbsides * DB_PREFIX * TKlocal**3.0 / Jdb)
    count = 0

    while count < itermax:
        x1 = off + np.log(1.0 + x0 + x0 * x0 / 2.0)
        try:
            tol = abs((x1 - x0) / x0)
        except:
            tol = abs(x1 - x0)
        if tol < eps:
            return x1 * Vthlocal
        x0 = x1
        count += 1
    return None


def shift_eqe_tcbc(lam, tc_eqe_ref, tc_bandgap_25, tc_sigma_25, tc_bandgaps, tc_sigmas, bc_eqe_ref, bc_bandgap_25, bc_sigma_25, bc_bandgaps, bc_sigmas, spectra):

    vec_erfc = np.vectorize(erfc)
    tc_trans = None
    # in case values are percentages
    Ey = constants.h * constants.c / (lam * 1e-9) / constants.e  # [eV]

    tc_lam_eqe_saturation_idx = np.argmax(tc_eqe_ref * lam)
    tc_eqe_saturation = tc_eqe_ref[tc_lam_eqe_saturation_idx]
    # using 25 degC EQE for saturation
    # tc_eqe_saturation = tc_eqe_ref[lam > photonenergy_to_wavelength(tc_bandgap_25 + 2 * tc_sigma_25)][0]

    bc_lam_eqe_saturation_idx = np.argmax(bc_eqe_ref * lam)
    bc_eqe_saturation = bc_eqe_ref[bc_lam_eqe_saturation_idx]
    # using 25 degC EQE for saturation
    # bc_eqe_saturation = bc_eqe_ref[lam > photonenergy_to_wavelength(bc_bandgap_25 + 2 * bc_sigma_25)][0]

    tc_bandgaps_arr = np.tile(tc_bandgaps, [len(Ey), 1])
    tc_sigmas_arr = np.tile(tc_sigmas, [len(Ey), 1])
    tc_erfc_arr = (tc_bandgaps_arr - Ey.reshape(-1, 1)) / (tc_sigmas_arr * np.sqrt(2))
    tc_eqe_filter = np.tile(lam, [len(tc_bandgaps), 1]).T > photonenergy_to_wavelength(tc_bandgaps_arr + 2 * tc_sigmas_arr)
    tc_eqe_new_arr = np.tile(tc_eqe_ref, [len(tc_bandgaps), 1]).T
    tc_abs_arr = vec_erfc(tc_erfc_arr) * 0.5 * tc_eqe_saturation
    # tc_abs_arr = vec_erfc(tc_erfc_arr) * 0.5 * np.tile(tc_eqe_new_arr[tc_eqe_filter.argmax(axis=0),:].reshape(1,-1), (tc_eqe_new_arr.shape[0],1))
    tc_eqe_new_arr = tc_eqe_new_arr * ~tc_eqe_filter + tc_abs_arr * tc_eqe_filter

    # fig,ax = plt.subplots()
    # ax.plot(lam,tc_abs_arr[:,0], "k--")
    # ax.plot(lam,tc_eqe_new_arr[:,0] * ~tc_eqe_filter[:,0], "r--")
    # ax.plot(lam,tc_abs_arr[:,0] * tc_eqe_filter[:,0], "b--")
    # ax.plot(lam,tc_eqe_ref, "m-")

    tc_trans = None
    if tc_trans is None:
        tc_trans = 1 - _normalize(tc_eqe_new_arr)

    eqe_max_idx = np.argmax(tc_eqe_new_arr, axis=0)
    filter_idx = (tc_eqe_new_arr < 0.01) & (tc_eqe_new_arr > eqe_max_idx)
    tc_trans[filter_idx] = 1
    # tc_trans[~tc_eqe_filter] = 0

    bc_bandgaps_arr = np.tile(bc_bandgaps, [len(Ey), 1])
    bc_sigmas_arr = np.tile(bc_sigmas, [len(Ey), 1])
    bc_erfc_arr = (bc_bandgaps_arr - Ey.reshape(-1, 1)) / (bc_sigmas_arr * np.sqrt(2))
    bc_eqe_filter = np.tile(lam, [len(bc_bandgaps), 1]).T > photonenergy_to_wavelength(bc_bandgaps_arr + 2 * bc_sigmas_arr)
    bc_eqe_new_arr = np.tile(bc_eqe_ref, [len(bc_bandgaps), 1]).T
    bc_abs_arr = vec_erfc(bc_erfc_arr) * 0.5 * bc_eqe_saturation
    # bc_abs_arr = vec_erfc(bc_erfc_arr) * 0.5 * np.tile(bc_eqe_new_arr[bc_eqe_filter.argmax(axis=0),:].reshape(1,-1), (bc_eqe_new_arr.shape[0],1))
    bc_eqe_new_arr = bc_eqe_new_arr * ~bc_eqe_filter + bc_abs_arr * bc_eqe_filter

    # Apply TC transmittance to BC

    # fig,ax = plt.subplots()
    # ax.plot(lam,tc_eqe_new_arr, "r--")
    # ax.plot(lam,bc_eqe_new_arr, "k--")
    # bc_eqe_new_arr = bc_eqe_new_arr * tc_trans
    # ax.plot(lam,bc_eqe_new_arr, "g--")
    # ax.plot(lam,bc_eqe_new_arr[:,0] * ~bc_eqe_filter[:,0], "r--")
    # ax.plot(lam,bc_abs_arr[:,0] * bc_eqe_filter[:,0], "b--")
    # ax.plot(lam,bc_eqe_ref, "m-")

    tc_jscs = np.trapz(y=tc_eqe_new_arr * spectra / wavelength_to_photonenergy(lam).reshape(-1, 1), x=lam.reshape(-1, 1), axis=0) / 10
    bc_jscs = np.trapz(y=bc_eqe_new_arr * spectra / wavelength_to_photonenergy(lam).reshape(-1, 1), x=lam.reshape(-1, 1), axis=0) / 10

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(lam, tc_eqe_new_arr)
    # ax[0].plot(lam, tc_eqe_ref, "r--")

    # ax[1].plot(lam, bc_eqe_new_arr)
    # ax[1].plot(lam, bc_eqe_ref, "r--")
    # ax.plot(tc_eqe_new_arr)
    # bc_eqe_filtered = tc_trans * bc_eqe_ref

    # get the bandgap to cut-off eqe of bottom cell where eqe of the top cell is weak
    # lam_cut_psc = photonenergy_to_wavelength(tc_bandgap_25 + 2 * tc_sigma_25)

    # bc_eqe_filtered[lam <= lam_cut_psc] = 0

    return tc_jscs, bc_jscs


def ensure_numpy_2drow(array):

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(1, -1) if array.ndim == 1 else array
    return array


def ensure_numpy_2dcol(array):

    # ensure numpy
    array = np.array(array)
    # ensure 2D
    array = array.reshape(-1, 1) if array.ndim == 1 else array
    return array


class EQE(object):
    """
    EQE object
    It creates a class containing nth junctions EQEs and Luminescent Coupling
    between junctions. The contribution can be studied interectively using
    ipywidgets under a notebook.
    The methods
    self.control create the self.ui to adjust the LC
    self.plot function to plot
    sle.Jdb: it gets the reverse saturation current.

    """

    def __init__(self, wavelength, eqe, name="EQE", sjuncs=None):
        """It creats the EQE class. ntegrate over spectrum or spectra
        rawEQE (numpy.array):  2D(lambda)(junction) raw input rawEQE (not LC corrected)
        xEQE(numpy.array)      xEQE        # wavelengths [nm] for rawEQE data
        name (str):            name of EQE object, sample
        sjuncs [list of str]:          labels for the junctions, if None it is self generated

        The number of junctions is created from the dimension of the EQE
        self.njuncs = njuncs    # number of junctions
        """
        # Number of wavelength and EQE values must match
        if wavelength.shape[0] != eqe.shape[0]:
            raise ValueError(f"shape of wavelength and eqe doesn't match: {wavelength.shape[0]} != {eqe.shape[0]}")

        self.wavelength = ensure_numpy_2dcol(wavelength)
        self.eqe = ensure_numpy_2dcol(eqe)

        self.name = name

        self.njuncs = self.eqe.shape[1]  # number of junction

        if sjuncs == None:
            self.sjuncs = [ordinal(junc + 1) for junc in range(self.njuncs)]
        else:
            self.sjuncs = sjuncs  # names of junctions

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # LC factor for next junctions
        self.LCcorr()  # calculate LC with zero etas
        self.spectra = None

    def add_spectra(self, wavelength=None, spectra=None):
        """
        Add spectral data

        Args:
            spectra (optional): array with wavelength and spectral data in W/m^2/nm. Defaults to AM1.5G.
        """

        if wavelength is None and spectra is None:
            wavelength = wvl
            spectra = AM15G

        if wavelength is None or spectra is None:
            raise ValueError("Provide wavelength and spectra")

        if wavelength.shape[0] != spectra.shape[0]:
            raise ValueError("Wavelength and Spectra need to be column vectors")

        # ensure numpy and 2D
        wavelength = np.array(wavelength)
        spectra = ensure_numpy_2dcol(spectra)

        # interpolate spectra to eqe
        f_spec_interp = interp1d(wavelength, spectra, kind="linear", axis=0)
        spectra_interp = f_spec_interp(self.wavelength.flatten())
        self.spectra = spectra_interp

    def add_eqe(self, wavelength_add, eqe_add, sjuncs=None):
        """
        Add eqe. Merge wavelength, inpolate eqe and fill extrapolation with 0. Assumes that the all EQE are part of a multijunction device.

        Args:
            eqe (optional): Wavelength of the new EQE
            eqe (optional): New EQE data
        """
        # ensure numpy
        wavelength_add = ensure_numpy_2dcol(wavelength_add)
        eqe_add = ensure_numpy_2dcol(eqe_add)

        # Combine old and new wavelength
        combined_wavelength = np.sort(np.unique(np.concatenate((self.wavelength, wavelength_add))))

        # Create interpolation functions with fill_value=0 for extrapolation
        f_interp_eqe = interp1d(self.wavelength.flatten(), self.eqe, kind="linear", fill_value=0, bounds_error=False, axis=0)
        f_interp_eqe_add = interp1d(wavelength_add.flatten(), eqe_add, kind="linear", fill_value=0, bounds_error=False, axis=0)

        # Interpolate and fill missing values
        eqe_interp = f_interp_eqe(combined_wavelength)
        eqe_add_interp = f_interp_eqe_add(combined_wavelength)

        # Assign new values
        self.wavelength = ensure_numpy_2dcol(combined_wavelength)
        self.eqe = np.hstack([eqe_interp, eqe_add_interp])

        self.njuncs = self.eqe.shape[1]  # number of junction

        if sjuncs is None:
            self.sjuncs.append(ordinal(self.njuncs + 1))
        else:
            self.sjuncs.append(sjuncs)

        self.corrEQE = np.empty_like(self.eqe)  # luminescent coupling corrected EQE same size as rawEQE
        self.etas = np.zeros((self.njuncs, 3), dtype=np.float64)  # LC factor for next junctions
        self.LCcorr()  # calculate LC with zero etas

    def calc_Eg_Rau(self, return_sigma=True, fit_gaussian=True):
        # using [1] U. Rau, B. Blank, T. C. M. Müller, and T. Kirchartz,
        # “Efficiency Potential of Photovoltaic Materials and Devices Unveiled by Detailed-Balance Analysis,”
        # Phys. Rev. Applied, vol. 7, no. 4, p. 044016, Apr. 2017, doi: 10.1103/PhysRevApplied.7.044016.
        # extended by gaussian fit

        bandgaps = []
        sigmas = []
        # Define the Gaussian function
        for i in range(self.eqe.shape[1]):
            y = self.eqe[:, i]
            x = wavelength_to_photonenergy(self.wavelength.flatten())

            # convert wavelength to photon energy
            y_grad = -1 * np.gradient(y)

            # filter tail to avoid eqe dips at end/beginning of measurement
            # y_grad = y_grad[(x < x[len(x) // 2])]
            # x = x[(x < x[len(x) // 2])]
            data_filter = x < (x.max() + x.min()) / 2
            x = x[data_filter]
            y = y[data_filter]
            y_grad = y_grad[data_filter]

            # we only need declining EQE to determine bandgaps
            data_filter = y_grad > 0
            x = x[data_filter]
            y = y[data_filter]
            y_grad = y_grad[data_filter]

            # y_grad = np.abs(np.diff(y.values, prepend=np.nan))
            # y_grad = y.diff().abs().values

            # # filter tail to avoid eqe dips at end/beginning of measurement
            # y_grad = y_grad[(x < (x.max() + x.min()) / 2)]
            # y = y[(x < (x.max() + x.min()) / 2)]
            # x = x[(x < (x.max() + x.min()) / 2)]

            # normalize data
            y_grad = _normalize(y_grad)
            # get the index of the maximum
            y_diff_max_idx = np.nanargmax(y_grad)
            # get the max coordinates
            x_diff_max = x[y_diff_max_idx]
            y_diff_max = y_grad[y_diff_max_idx]

            # define lower threshold
            p_ab = np.exp(-2) * y_diff_max
            thres = 0.5
            # p_ab = thres * y_diff_max
            # find the index of the low-energy side where P(a) is max(P(Eg)/2)
            a_cond = np.where((y_grad < p_ab) & (x < x_diff_max))[0]
            if len(a_cond > 0):
                a_idx = np.nanmin(a_cond)
            else:
                a_idx = len(x) - 1
            a = x[a_idx]
            p_a = y_grad[a_idx]
            # find the index of the high-energy side where P(b) is max(P(Eg)/2)
            b_idx = np.nanmax(np.where((y_grad < p_ab) & (x > x_diff_max))[0])
            b = x[b_idx]
            p_b = y_grad[b_idx]

            x_target = x[a_idx : b_idx - 1 : -1]
            y_target = y_grad[a_idx : b_idx - 1 : -1]

            if fit_gaussian:

                # initial guesses from weighted arithmetic mean and weighted sample sigma
                mean = sum(x_target * y_target) / sum(y_target)
                sigma = np.sqrt(sum(y_target * (x_target - mean) ** 2) / sum(y_target))

                fit_res = curve_fit(_gaussian, x_target, y_target, p0=[max(y_target), mean, sigma], method="trf")
                x_fit = np.linspace(x[b_idx], x[a_idx], 100)
                y_fit = _gaussian(x_fit, *fit_res[0])

                # fig, ax = plt.subplots(1)
                # ax.plot(x, y)
                # ax.plot(x, y_grad, "--")
                # ax.plot(x_fit, y_fit)
                # ax.plot(x_target, y_target, ".r")
                # ax.plot(x_diff_max, y_diff_max, "r*")
                # ax.plot(a, p_a, "g*")
                # ax.plot(b, p_b, "b*")
                # plt.plot(x_fit - fit_res[0][1], y_fit)
                # plt.plot(x_fit, y_fit)
                # ax.set_xlim(1.1, 1.8)
                # ax.set_ylabel(r"$\frac{{\mathrm{d}}EQE}{{\mathrm{d}}E_{\nu}}$")
                # ax.set_xlabel(r"Photon energy $E_{\nu}$ [eV]")

                x_fit = x_fit[y_fit >= thres * y_fit.max()]
                y_fit = y_fit[y_fit >= thres * y_fit.max()]
                sigma = fit_res[0][2]

            else:
                fit_res = [None]
                x_fit = x_target
                y_fit = y_target
                sigma = None

            res = fsolve(_eq_solve_Eg, 1.0, args=(x_fit, y_fit))
            bandgap = res[0]

            bandgaps.append(bandgap)
            sigmas.append(sigma)
        return bandgaps, sigmas

    def LCcorr(self, junc=None, dist=None, val=None):
        """
        Applies the correction of the Luminescent coupling to the QE junc using procedure
        from Steiner et al., IEEE PV, v3, p879 (2013)
        """
        # change one eta[junc,dist] value
        # calculate LC corrected EQE
        etas = self.etas
        # with self.debugout: print(junc,dist,val)
        if junc == None or dist == None or val == None:
            pass
        else:
            etas[junc, dist] = val  # assign value
            # with self.debugout: print('success')
        raw = self.eqe
        ## TODO: This should be a nested loop, with a break for 10th junctions?
        if self.njuncs == 1:
            self.corrEQE = raw
        else:

            for ijunc in range(self.njuncs):
                if ijunc == 0:  # 1st ijunction
                    self.corrEQE[:, ijunc] = raw[:, ijunc]
                elif ijunc == 1:  # 2nd ijunction
                    denom = 1.0 + etas[ijunc, 0]
                    self.corrEQE[:, ijunc] = raw[:, ijunc] * denom - raw[:, ijunc - 1] * etas[ijunc, 0]
                elif ijunc == 2:  # 3rd ijunction
                    denom = 1.0 + etas[ijunc, 0] * (1.0 + etas[ijunc, 1])
                    self.corrEQE[:, ijunc] = raw[:, ijunc] * denom - raw[:, ijunc - 1] * etas[ijunc, 0] - raw[:, ijunc - 2] * etas[ijunc, 0] * etas[ijunc, 1]
                else:  # higher ijunctions
                    denom = 1.0 + etas[ijunc, 0] * (1.0 + etas[ijunc, 1] * (1.0 + etas[ijunc, 2]))
                    self.corrEQE[:, ijunc] = (
                        raw[:, ijunc] * denom
                        - raw[:, ijunc - 1] * etas[ijunc, 0]
                        - raw[:, ijunc - 2] * etas[ijunc, 0] * etas[ijunc, 1]
                        - raw[:, ijunc - 3] * etas[ijunc, 0] * etas[ijunc, 1] * etas[ijunc, 2]
                    )

    def Jdb(self, TC, Eguess=1.0, kTfilter=3, dbug=False):
        """It calculate Jscs and Egs from self.corrEQE"""
        Vthlocal = Vth(TC)  # kT
        Eguess = np.array([Eguess] * self.njuncs)
        Egvect = np.vectorize(EgFromJdb)
        EkT = nm2eV / Vthlocal / self.wavelength
        blackbody = DBWVL_PREFIX / (self.wavelength * 1e-9) ** 4 / np.expm1(EkT)

        for count in range(10):
            nmfilter = nm2eV / (Eguess - Vthlocal * kTfilter)  # MD [652., 930.]
            EQEfilter = self.corrEQE.copy()

            for i, lam in enumerate(self.wavelength):
                EQEfilter[i, :] *= lam < nmfilter  # zero EQE about nmfilter

            DBintegral = blackbody * EQEfilter
            Jdb = trapezoid(DBintegral, x=(self.wavelength * 1e-9), axis=0)
            Egnew = Egvect(TC, Jdb)
            if dbug:
                print(Egnew, max((Egnew - Eguess) / Egnew))
            if np.amax((Egnew - Eguess) / Egnew) < 1e-6:
                break
            else:
                Eguess = Egnew
            self.Egs = Egnew

        return Jdb, Egnew

    def Jint(self):
        """It integrates over spectrum or spectra
          J = spectra * lambda * EQE(lambda)
        Jsc = int(Pspec*QE[0]*lambda) in [mA/cm2]
        EQE optionally scalar for constant over xEQE range
        integrate over full spectrum is xEQE is None
        """

        if self.spectra is None:
            raise ValueError("Load spectral information first.")

        integrand = np.einsum("ni,nj->nij", (self.eqe / wavelength_to_photonenergy(self.wavelength) * 1e-1), self.spectra)

        jsc = trapezoid(
            y=integrand,
            x=self.wavelength.flatten(),
            axis=0,
        )

        return jsc

    def plot(self, Pspec="global", ispec=0, specname=None, xspec=wvl, size="x-large"):
        # plot EQE on top of a spectrum
        rnd2 = 100

        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=Multi2T.junctioncolors[self.njuncs])
        for i in range(self.njuncs):
            rlns = ax.plot(self.wavelength, self.eqe[:, i], lw=1, ls="--", marker="", label="_" + self.sjuncs[i])
            ax.plot(self.wavelength, self.corrEQE[:, i], lw=3, c=rlns[0].get_color(), marker="", label=self.sjuncs[i])
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(math.floor(self.wavelength[0] / rnd2) * rnd2, math.ceil(self.wavelength[-1] / rnd2) * rnd2)
        ax.set_ylabel("EQE", size=size)  # Add a y-label to the axes.
        ax.set_xlabel("Wavelength (nm)", size=size)  # Add an x-label to the axes.
        ax.set_title(self.name + " EQE", size=size)
        ax.axhline(0, lw=0.5, ls="--", c="black", label="_hzero")
        ax.axhline(1, lw=0.5, ls="--", c="black", label="_hone")

        rax = ax.twinx()  # right axis
        # check spectra input
        if type(Pspec) is str:  # optional string space, global, direct
            if Pspec in dfrefspec.columns:
                specname = Pspec
                Pspec = dfrefspec[Pspec].to_numpy(dtype=np.float64, copy=True)
            else:
                specname = dfrefspec.columns[ispec]
                Pspec = dfrefspec.to_numpy(dtype=np.float64, copy=True)  # just use refspec instead of error

        if np.any(Pspec):
            Pspec = np.array(Pspec, dtype=np.float64)
            if not specname:
                specname = "spectrum" + str(ispec)
            if Pspec.ndim == 2:
                Pspec = Pspec[:, ispec]  # slice 2D numpy to 1D
            rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color="grey", label="fill")
            rax.plot(xspec, Pspec, c="grey", lw=0.5, marker="", label=specname)
            rax.set_ylabel("Irradiance (W/m2/nm)", size=size)  # Add a y-label to the axes.
            rax.set_ylim(0, 2)
            # rax.legend(loc=7)
        return ax, rax

    def plot_sr(self):
        # plot EQE on top of a spectrum

        sr = self.eqe * self.wavelength
        fig, ax = plt.subplots()
        ax.plot(self.wavelength, sr)
        return fig, ax

    # def controls(self, Pspec='global', ispec=0, specname=None, xspec=wvl):
    #     '''
    #     use interactive_output for GUI in IPython
    #     '''
    #     tand_layout = widgets.Layout(width= '300px', height='40px')
    #     vout_layout = widgets.Layout(width= '180px', height='40px')
    #     junc_layout = widgets.Layout(display='flex',
    #                 flex_flow='row',
    #                 justify_content='space-around')
    #     multi_layout = widgets.Layout(display='flex',
    #                 flex_flow='row',
    #                 justify_content='space-around')

    #     replot_types = [widgets.widgets.widget_float.BoundedFloatText,
    #                     widgets.widgets.widget_int.BoundedIntText,
    #                     widgets.widgets.widget_int.IntSlider,
    #                     widgets.widgets.widget_float.FloatSlider,
    #                     widgets.widgets.widget_float.FloatLogSlider]

    #     def on_EQEchange(change):
    #         # function for changing values
    #         old = change['old'] #old value
    #         new = change['new'] #new value
    #         owner = change['owner'] #control
    #         value = owner.value
    #         desc = owner.description
    #         #with self.debugout: print('Mcontrol: ' + desc + '->', value)
    #         #self.set(**{desc:value})

    #     def on_EQEreplot(change):
    #         # change info
    #         fast=True
    #         if type(change) is widgets.widgets.widget_button.Button:
    #             owner = change
    #         else: # other controls
    #             owner = change['owner'] #control
    #             value = owner.value
    #         desc = owner.description
    #         if desc == 'Recalc': fast = False

    #         #recalculate
    #         ts = time()
    #         if desc[:3] == 'eta':
    #             junc, dist = parse('eta{:1d}{:1d}',desc)
    #             self.LCcorr(junc, dist, value) #replace one value and recalculate LC
    #             specname = None
    #         elif desc == 'spec':
    #             if value in dfrefspec.columns:
    #                 specname = value
    #                 Pspec = dfrefspec[specname].to_numpy(dtype=np.float64, copy=True)
    #         else:
    #             VoutBox.clear_output()
    #             with VoutBox: print(desc)
    #             return 0

    #         with Rout: # right output device -> light
    #             #replot
    #             lines = ax.get_lines()
    #             for line in lines:
    #                 linelabel=line.get_label()
    #                 #with self.debugout: print(linelabel)
    #                 for i in range(self.njuncs):
    #                     if linelabel == self.sjuncs[i]:
    #                          line.set_data(self.wavelength, self.corrEQE[:,i]) #replot

    #             rlines = rax.get_lines()
    #             for line in rlines:
    #                 linelabel=line.get_label()
    #                 #with self.debugout: print(linelabel)
    #                 if linelabel in refnames:
    #                     if specname == None: #desc == 'spec'
    #                          specname = linelabel
    #                          Pspec = specname
    #                     else:
    #                         line.set_data(xspec, Pspec) #replot spectrum
    #                         for obj in rax.get_children():
    #                             if type(obj) is mpl.collections.PolyCollection: #contours
    #                                 if obj.get_label() == 'fill':
    #                                     obj.remove() #remove old fill
    #                         rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color='grey', label='fill')
    #                         line.set(label = specname) #relabel spectrum

    #         Jscs = self.Jint(Pspec, xspec)
    #         Jdbs, Egs = self.Jdb(25)
    #         OP = PintMD(Pspec, xspec)

    #         VoutBox.clear_output()
    #         with VoutBox:
    #             stext = (specname+' {0:6.2f} W/m2').format(OP)
    #             print('Eg = ',Egs, ' eV')
    #             print(stext)
    #             print('Jsc = ',Jscs[0], ' mA/cm2')

    #         te = time()
    #         dt=(te-ts)
    #         with VoutBox:   print('Calc Time: {0:>6.2f} s'.format(dt))

    #     # summary line
    #     VoutBox = widgets.Output()
    #     VoutBox.layout.height = '70px'
    #     #with VoutBox: print('Summary')

    #     # Right output -> EQE plot
    #     Rout = widgets.Output()
    #     with Rout: # output device
    #         if plt.isinteractive:
    #             plt.ioff()
    #             restart = True
    #         else:
    #             restart = False
    #         ax, rax = self.plot(Pspec, ispec, specname, xspec)
    #         fig = ax.get_figure()
    #         fig.show()
    #         rlines = rax.get_lines()
    #         for line in rlines:
    #             linelabel=line.get_label()
    #             if linelabel in refnames:
    #                 specname = linelabel
    #         if restart: plt.ion()

    #     # tandem3T controls
    #     in_tit = widgets.Label(value='EQE: ', description='title')
    #     in_name = widgets.Text(value=self.name, description='name', layout=tand_layout,
    #         continuous_update=False)
    #     in_name.observe(on_EQEchange,names='value') #update values

    #     in_spec = widgets.Dropdown(value=specname, description='spec', layout=tand_layout,
    #         options=refnames)
    #     in_spec.observe(on_EQEreplot,names='value') #update values

    #     Hui = widgets.HBox([in_tit, in_name, in_spec])
    #     #in_Rs2T.observe(on_2Tchange,names='value') #update values

    #     in_eta = []
    #     elist0 = []
    #     elist1 = []
    #     elist2 = []
    #     # list of eta controls
    #     for i in range(self.njuncs) :
    #         if i > 0:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,0], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"0",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist0.append(in_eta[j])
    #             in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #         #if i > 1:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,1], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"1",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist1.append(in_eta[j])
    #             in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             if i > 1:
    #                 in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             else:
    #                 in_eta[j].disabled = True
    #         #if i > 2:
    #             in_eta.append(widgets.FloatSlider(value=self.etas[i,2], min=-0.2, max=1.5,step=0.001,
    #                 description='eta'+str(i)+"2",layout=junc_layout,readout_format='.4f'))
    #             j = len(in_eta)-1
    #             elist2.append(in_eta[j])
    #             if i > 2:
    #                 in_eta[j].observe(on_EQEreplot,names='value')  #replot
    #             else:
    #                 in_eta[j].disabled = True
    #     etaui0 = widgets.HBox(elist0)
    #     etaui1 = widgets.HBox(elist1)
    #     etaui2 = widgets.HBox(elist2)

    #     #in_Rs2T.observe(on_2Treplot,names='value')  #replot
    #     #in_2Tbut.on_click(on_2Treplot)  #replot

    #     #EQE_ui = widgets.HBox(clist)
    #     #eta_ui = widgets.HBox(jui)

    #     ui = widgets.VBox([Rout, VoutBox, Hui, etaui0, etaui1, etaui2])
    #     self.ui = ui
    #     #in_2Tbut.click() #fill in MPP values

    #     # return entire user interface, dark and light graph axes for tweaking
    #     return ui, ax, rax
