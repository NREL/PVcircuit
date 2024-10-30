from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pvcircuit as pvc
from pvcircuit import EQE


def get_measured_eqe():
    path = Path(pvc.datapath)
    eqe_file = "MM927Bn5CEQE.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    eqe = pvc.EQE(data, data.index, "TestEQE")
    return eqe


@pytest.fixture
def ideal_eqe():
    wvl = np.arange(280, 4000)
    return pvc.EQE(wvl, np.ones_like(wvl))


@pytest.fixture
def example_eqe():

    path = Path(pvc.datapath)
    eqe_file = "MM927Bn5CEQE.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    eqe = pvc.EQE(data, data.index, "TestEQE")
    return eqe


def test_eqe(ideal_eqe):
    ideal_eqe.add_spectra()
    np.testing.assert_almost_equal(68.98763776603343, ideal_eqe.Jint())


def test_Jint():
    waves = np.arange(300, 1200)
    eq = np.ones_like(waves)
    eqe = EQE(waves, eq)

    eqe.add_spectra(pvc.qe.wvl, pvc.qe.AM15G.T)
    test_res = np.array([[46.42154037]])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())

    eqe.add_spectra(pvc.qe.wvl, np.tile(test_res, [5, 1]).T)
    test_res = np.array([[46.42154037, 46.42154037, 46.42154037, 46.42154037, 46.42154037]])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())

    eqe.add_eqe(waves, eq * 0.5)
    test_res = np.vstack([test_res, test_res / 2])
    np.testing.assert_array_almost_equal(test_res, eqe.Jint())


if __name__ == "__main__":

    eqe = get_measured_eqe()

    eqe.plot()
    plt.show()
