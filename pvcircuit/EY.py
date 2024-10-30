# -*- coding: utf-8 -*-
"""
Package to simulate energy yield
"""

import copy
import glob
import multiprocessing as mp
import os
import warnings
from functools import lru_cache
from typing import Union

import numpy as np  # arrays
import pandas as pd
from parse import parse
from scipy import constants
from scipy.integrate import trapezoid
from scipy.special import erfc
from tqdm import tqdm, trange

import pvcircuit as pvc

warnings.warn("The 'EY.py' module is deprecated and will be change in future version.", DeprecationWarning, stacklevel=2)


def VMloss(model: Union['pvc.Tandem3T', 'pvc.Multi2T'], oper: str, ncells: int):

    if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
        return 1

    elif isinstance(model, pvc.Tandem3T):  # Tandem3T
        tandem_type = oper.split("-")

        if tandem_type[0] == "MPP" or tandem_type[0] == "CM":
            return 1
        elif tandem_type[0] == "VM":
            if len(tandem_type) != 3:
                raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
            vm_ratio = tuple(map(int, tandem_type[1]))

            if tandem_type[2] == "r":
                endloss = max(vm_ratio) - 1
            elif tandem_type[2] == "s":
                endloss = sum(vm_ratio) - 1
            else:
                raise ValueError("Unknown model")
    else:
        raise ValueError("Unknown model")

    lossfactor = max(0,1 - endloss / ncells)
    return lossfactor


# @lru_cache(maxsize=100)
def VMlist(mmax):
    # generate a list of VM configurations + 'MPP'=4T and 'CM'=2T
    # mmax < 10 for formating reasons
    if mmax > 9:
        raise ValueError("mmmax must be smaller than 10")

    sVM = ["MPP", "CM"]  # Initialize sVM with predefined elements
    primes = [2, 3, 5]
    for m in range(mmax + 1):
        for n in range(1, m):
            if any(m % p == 0 and n % p == 0 for p in primes):
                continue
            sVM.append(f"VM{m}{n}")
    return sVM


def sandia_T(poa_global, wind_speed, temp_air):
    """Sandia solar cell temperature model
    Adapted from pvlib library to avoid using pandas dataframes
    parameters used are those of 'open_rack_cell_polymerback'
    """

    a = -3.56
    b = -0.075
    deltaT = 3

    E0 = 1000.0  # Reference irradiance

    temp_module = poa_global * np.exp(a + b * wind_speed) + temp_air

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return temp_cell


def _calc_yield_async(Jscs, Egs, TempCell, devlist, oper):
    Pmax_out = np.zeros(len(Jscs))

    for i in range(len(Jscs)):
        model = devlist[i]
        if isinstance(model, pvc.Multi2T):  # Multi2T or current matched 2-junction tandem
            for ijunc in range(model.njuncs):
                model.j[ijunc].set(Eg=Egs[ijunc], Jext=Jscs[i, ijunc], TC=TempCell[i])

            mpp_dict = model.MPP()
            Pmax = mpp_dict["Pmp"]

        elif isinstance(model, pvc.Tandem3T):  # Tandem3T
            tandem_type = oper.split("-")

            model.top.set(Eg=Egs[i, 0], Jext=Jscs[i, 0], TC=TempCell.iloc[i])
            model.bot.set(Eg=Egs[i, 1], Jext=Jscs[i, 1], TC=TempCell.iloc[i])
            if tandem_type[0] == "MPP":
                tempRz = model.Rz
                model.set(Rz=0)
                iv3T = model.MPP()
                model.set(Rz=tempRz)
            elif tandem_type[0] == "CM":
                ln, iv3T = model.CM()
            elif tandem_type[0] == "VM":
                if len(tandem_type) != 3:
                    raise ValueError("3T voltage matched operation must be VM-[bc/tc ratio]-[r/s-type], e.g. VM-21-r")
                model.bot.pn = -1 if tandem_type[2] == "r" else 1
                ln, iv3T = model.VM(*map(int, tandem_type[1]))
            else:
                iv3T = pvc.iv3T.IV3T("bogus")
                iv3T.Ptot[0] = 0
            Pmax = iv3T.Ptot[0]
        else:
            Pmax = 0.0

        Pmax_out[i] = Pmax

    return Pmax_out  # Pmax in [W]


def si_eg_shift(temperature: float, bandgap_25: float) -> float:
    """
    Temperature dependence of bandgap of a silicon cell.
    See pvc.qe.EQE class for details about bandgap and sigma determination

    Args:
        temperature (float): target temperature
        bandgap_25: bandgap at 25degC

    Returns: bandgap at target temperature
    """
    p = [-6.47214956e-04, 1.01632828e00]
    return (p[0] * temperature + p[1]) * bandgap_25


def si_sigma_shift(temperature: float, sigma_25: float) -> float:
    """
    Temperature dependence of sigma of a silicon cell.
    See pvc.qe.EQE class for details about bandgap and sigma determination

    Args:
        temperature (float): target temperature
        sigma_25: sigma at 25degC

    Returns: sigma at target temperature
    """
    p = [0.00959188, 0.76558903]
    return (p[0] * temperature + p[1]) * sigma_25


def psc_eg_shift(temperature: float, bandgap_25: float) -> float:
    """
    Temperature dependence of bandgap of a metal halide perovskite.
    Assumes a piecewise linear function.
    See pvc.qe.EQE class for details about bandgap and sigma determination

    Args:
        temperature (float): target temperature
        bandgap_25: bandgap at 25degC

    Returns: bandgap at target temperature
    """
    t_split = 32
    p = [2.59551019e-04, 9.91138163e-01]
    res = np.zeros_like(temperature, dtype=np.float64)

    t_filter = temperature > t_split
    res[t_filter] = p[0] * temperature[t_filter] + p[1]
    res[~t_filter] = p[0] * t_split + p[1]
    res = res * bandgap_25
    res = pd.Series(res, index=temperature.index)
    return res


def psc_sigma_shift(temperature: float, sigma_25: float) -> float:
    """
    Temperature dependence of sigma of a metal halide perovskite.
    See pvc.qe.EQE class for details about bandgap and sigma determination

    Args:
        temperature (float): target temperature
        sigma_25 (float): sigma at 25degC

    Returns: sigma at target temperature
    """
    p = [0.00358866, 0.50074156]
    return (p[0] * temperature + p[1]) * sigma_25


def wavelength_to_photonenergy(wavelength: float):
    """
    Convert wavelength [nm] to photon energy [eV]

    Args:
        wavelength (float): wavelength in [nm]

    Returns:
        float: photon energy in [eV]
    """
    return constants.h * constants.c / (wavelength * 1e-9) / constants.e


def photonenergy_to_wavelength(photonenergy):
    return constants.h * constants.c / (photonenergy * 1e-9) / constants.e


def _normalize(eqe: pd.DataFrame) -> pd.DataFrame:
    eqe_min = np.nanmin(eqe)
    eqe_max = np.nanmax(eqe)
    return (eqe - eqe_min) / (eqe_max - eqe_min)


class Meteo(object):
    """
    Meteorological environmental data and spectra
    """

    def __init__(self, wavelength, spectra, ambient_temperature, wind, datetime):

        # replace nan is spctra with 0
        spectra = spectra.fillna(0)
        # drop any other nans
        ffilter = (np.all(np.isfinite(spectra), axis=1)) & (np.isfinite(ambient_temperature)) & (np.isfinite(wind))
        self.temp = ambient_temperature[ffilter]  # [degC]
        self.wind = wind[ffilter]  # [m/s]
        self.datetime = datetime[ffilter]  # daytime vector

        self.wavelength = wavelength
        self.spectra = spectra[ffilter]  # transpose for integration with QE

        # calculate from spectral proxy data only
        self.irradiance = pd.Series(trapezoid(y=self.spectra, x=self.wavelength), index=self.datetime)  # optical power of each spectrum
        self.cell_temp = sandia_T(self.irradiance, self.wind, self.temp)
        self.energy_in = trapezoid(y=self.irradiance, x=self.datetime.astype(np.int64)) / 1e9 / 3600 / 1000  # [kWh/m2/yr]

        self.average_photon_energy = None  # is calcluated when running calc_ape

    def calc_currents(self, eqe):

        eqe.add_spectra(self.wavelength, self.spectra.T)

        lam = eqe.wavelength.flatten()
        spectra = eqe.spectra

        tandem_bandgaps, tandem_sigmas = eqe.calc_Eg_Rau()
        tc_bandgap_25 = tandem_bandgaps[0]
        tc_sigma_25 = tandem_sigmas[0]
        bc_bandgap_25 = tandem_bandgaps[1]
        bc_sigma_25 = tandem_sigmas[1]

        tc_bandgaps = psc_eg_shift(self.cell_temp, tc_bandgap_25)
        tc_sigmas = psc_sigma_shift(self.cell_temp, tc_sigma_25)
        bc_bandgaps = si_eg_shift(self.cell_temp, bc_bandgap_25)
        bc_sigmas = si_sigma_shift(self.cell_temp, bc_sigma_25)

        tc_eqe = eqe.eqe[:, 0]
        bc_eqe = eqe.eqe[:, 1]

        vec_erfc = np.vectorize(erfc)
        tc_trans = None
        Ey = constants.h * constants.c / (lam * 1e-9) / constants.e  # [eV]

        tc_lam_eqe_saturation_idx = np.argmax(tc_eqe * lam)
        tc_eqe_saturation = tc_eqe[tc_lam_eqe_saturation_idx]
        # using 25 degC EQE for saturation
        # tc_eqe_saturation = tc_eqe[lam > photonenergy_to_wavelength(tc_bandgap_25 + 2 * tc_sigma_25)][0]

        bc_lam_eqe_saturation_idx = np.argmax(bc_eqe * lam)
        bc_eqe_saturation = bc_eqe[bc_lam_eqe_saturation_idx]
        # using 25 degC EQE for saturation
        # bc_eqe_saturation = bc_eqe[lam > photonenergy_to_wavelength(bc_bandgap_25 + 2 * bc_sigma_25)][0]

        tc_bandgaps_arr = np.tile(tc_bandgaps, [len(Ey), 1])
        tc_sigmas_arr = np.tile(tc_sigmas, [len(Ey), 1])
        tc_erfc_arr = (tc_bandgaps_arr - Ey.reshape(-1, 1)) / (tc_sigmas_arr * np.sqrt(2))
        tc_eqe_filter = np.tile(lam, [len(tc_bandgaps), 1]).T > photonenergy_to_wavelength(tc_bandgaps_arr + 2 * tc_sigmas_arr)
        tc_eqe_new_arr = np.tile(tc_eqe, [len(tc_bandgaps), 1]).T
        tc_abs_arr = vec_erfc(tc_erfc_arr) * 0.5 * tc_eqe_saturation
        tc_eqe_new_arr = tc_eqe_new_arr * ~tc_eqe_filter + tc_abs_arr * tc_eqe_filter

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
        bc_eqe_new_arr = np.tile(bc_eqe, [len(bc_bandgaps), 1]).T
        bc_abs_arr = vec_erfc(bc_erfc_arr) * 0.5 * bc_eqe_saturation
        bc_eqe_new_arr = bc_eqe_new_arr * ~bc_eqe_filter + bc_abs_arr * bc_eqe_filter

        tc_jscs = np.trapz(y=tc_eqe_new_arr * spectra / wavelength_to_photonenergy(lam).reshape(-1, 1), x=lam.reshape(-1, 1), axis=0) / 10
        bc_jscs = np.trapz(y=bc_eqe_new_arr * spectra / wavelength_to_photonenergy(lam).reshape(-1, 1), x=lam.reshape(-1, 1), axis=0) / 10

        # replace negative currents with 0
        tc_jscs[tc_jscs < 0] = 0
        bc_jscs[bc_jscs < 0] = 0

        self.Jscs = np.hstack([tc_jscs.reshape(-1, 1) / 1000, bc_jscs.reshape(-1, 1) / 1000])
        self.Egs = np.hstack([tc_bandgaps.values.reshape(-1, 1), bc_bandgaps.values.reshape(-1, 1)])

        return tc_jscs, bc_jscs

    def run_ey(self, model, oper, multiprocessing=True):
        # max power of a cell under self TMY
        # self.Jscs and self.Egs must be calculate first using cellcurrents
        # Inputs
        # cell 'model' can be 'Multi2T' or 'Tandem3T'
        #'oper' describes operation method unconstrained 'MPP', series-connected 'CM', parallel-configurations 'VM'
        # Outputs
        # - EYeff energy yield efficiency = EY/YearlyEnergy
        # - EY energy yield of cell [kWh/m2/yr]

        # Split data into chunks for workers
        max_chunk_size = 200
        cpu_count = mp.cpu_count()
        chunk_ids = np.arange(len(self.Jscs))
        chunk_size = min(len(chunk_ids) // cpu_count + 1, max_chunk_size)

        chunks = [chunk_ids[i : i + chunk_size] for i in range(0, len(chunk_ids), chunk_size)]
        dev_list = np.array([copy.deepcopy(model) for _ in range(len(self.Jscs))])

        with tqdm(total=len(self.datetime), leave=True, desc=f"Running {model.name} in mode {oper}") as pbar:

            def update_tqdm(*args):
                # callback
                pbar.update(len(args[0]))
                return

            if multiprocessing:
                print(f"Running with {cpu_count} parallel processes")

                with mp.Pool(cpu_count) as pool:

                    # Assign tasks to workers
                    jobs = [pool.apply_async(_calc_yield_async, args=(self.Jscs[chunk], self.Egs[chunk], self.cell_temp.iloc[chunk], dev_list[chunk], oper), callback=update_tqdm) for chunk in chunks]
                    # Get results from workers
                    results = [item for job in jobs for item in job.get()]

            else:
                print("Running sequentially without multiprocessing")

                results = []
                for chunk in chunks:
                    chunk_result = _calc_yield_async(self.Jscs[chunk], self.Egs[chunk], self.cell_temp.iloc[chunk], dev_list[chunk], oper)
                    results.extend(chunk_result)
                    pbar.update(len(chunk))

        self.outPowerMP = results

        EnergyOut = trapezoid(self.outPowerMP, self.datetime.values.astype(np.int64)) / 1e9  # [Ws/cm2/yr]
        # convert to [kWh/m2/yr]
        EnergyOut = EnergyOut / 3.6e3 / 1e3 * 1e4

        # calculate energy harvesting efficieny
        EYeff = EnergyOut / self.energy_in
        return EnergyOut, EYeff

    def calc_ape(self):
        """
        Calcualtes the average photon energy (APE) of the spectra
        """

        phi = self.spectra * (self.wavelength * 1e-9) / constants.h / constants.c
        self.average_photon_energy = trapezoid(x=self.wavelength, y=self.spectra.values) / constants.e / trapezoid(x=self.wavelength, y=phi.values)

    def filter_ape(self, min_ape: float = 0, max_ape: float = 10):
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

        self_copy.datetime = self_copy.datetime[ape_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[ape_mask]
        self_copy.spectra = self_copy.spectra[ape_mask]
        self_copy.irradiance = self_copy.irradiance[ape_mask]
        self_copy.cell_temp = self_copy.cell_temp[ape_mask]

        assert len(self_copy.spectra) == len(self_copy.irradiance) == len(self_copy.cell_temp) == len(self_copy.average_photon_energy)
        return self_copy

    def filter_spectra(self, min_spectra: float = 0, max_spectra: float = 10):
        """
        spectral data

        Args:
            min_spectra (float, optional): min value of the spectra. Defaults to 0.
            max_spectra (float, optional): max value of the spectra. Defaults to 10.
        """

        self_copy = copy.deepcopy(self)
        spectra_mask = (self_copy.spectra >= min_spectra).all(axis=1) & (self_copy.spectra < max_spectra).all(axis=1)
        self_copy.datetime = self_copy.datetime[spectra_mask]
        self_copy.average_photon_energy = self_copy.average_photon_energy[spectra_mask]
        self_copy.spectra = self_copy.spectra[spectra_mask]
        self_copy.irradiance = self_copy.irradiance[spectra_mask]
        self_copy.cell_temp = self_copy.cell_temp[spectra_mask]

        assert len(self.spectra) == len(self.irradiance) == len(self.cell_temp) == len(self.average_photon_energy)
        return self_copy

    def filter_custom(self, filter_array: bool):
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
            if hasattr(getattr(self_copy, attr_name), "__len__"):
                attr = getattr(self_copy, attr_name)
                if len(attr) == len(filter_array):
                    setattr(self_copy, attr_name, attr[filter_array])

        # assert len(self.spectra) == len(self.SpecPower) == len(self.TempCell) == len(self.average_photon_energy)
        return self_copy

    def reindex(self, index: bool, method="nearest", tolerance=pd.Timedelta(seconds=30)):
        """
        Reindex according to indexer
        Args:
            filter_array (bool): Filter array to apply to the data
        """

        self_copy = copy.deepcopy(self)

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, pd.DataFrame) or isinstance(attr, pd.Series):
                setattr(self_copy, attr_name, attr.reindex(index=index, method=method, tolerance=tolerance))

        return self_copy
