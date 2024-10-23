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

    return EQE(np.array([1, 1]), np.array([1, 2]))

@pytest.fixture
def example_eqe():

    path = Path(pvc.datapath)
    eqe_file = "MM927Bn5CEQE.csv"
    data = pd.read_csv(path.joinpath(eqe_file), index_col=0)
    eqe = pvc.EQE(data, data.index, "TestEQE")
    return eqe


def test_eqe(ideal_eqe):
    assert 1 == 1


if __name__ == "__main__":

    eqe = get_measured_eqe()

    eqe.plot()
    plt.show()
