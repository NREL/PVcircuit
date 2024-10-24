import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import IV3T, Tandem3T


@pytest.fixture
def dev3T():
    return Tandem3T()


@pytest.fixture
def iv3t():
    return IV3T()


@pytest.fixture
def junction():
    return pvc.junction.Junction()


def test_iv3t_str(iv3t):


    test_file = "iv3t_str.txt"

    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_iv3t_setter(dev3T, iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)
    for k in iv3t.arraykeys:
        iv3t.set(**{k: 1})

    test_file = "iv3t_setter.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_dev2load(dev3T,iv3t):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)
    iv3t2 = iv3t.copy()
    iv3t.init(["IA","IB"],["IA","IB"])
    iv3t.convert("I", "dev2load")
    iv3t.__str__() == iv3t2.__str__()

    iv3t.line("Vzt", 0, 1.2, 11, "Vrz", "0")


def test_line(iv3t):

    iv3t.line("Vzt", 0, 1.2, 11, "Vrz", "0")

    test_file = "iv3t_line.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_box(iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)

    test_file = "iv3t_box.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_nanpnt(iv3t):

    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)

    array = getattr(iv3t,"IA")
    np.testing.assert_almost_equal(array[0,0],-25e-3)
    iv3t.nanpnt((0,0))
    np.testing.assert_almost_equal(array[0,0],np.nan)


def test_mpp(iv3t, dev3T):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    # mpp = iv3t.MPP()

    test_file = "iv3t_mpp.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_resize(iv3t, dev3T):
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    iv3t.resize([5,10])

    np.testing.assert_array_equal(iv3t.sizes(["VA"]), [50,50])

    test_file = "iv3t_resize.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_calc(iv3t,dev3T):

    # Create voltage line range
    iv3t.line("Vzt", 0, 1.5, 20, "Vrz", "0")

    # Copy for comparinson
    iv3t2 = iv3t.copy()

    # Convert dev2load for voltages
    iv3t.convert("V", "dev2load")
    # Calculate current based on abs values
    dev3T.J3Tabs(iv3t)
    # convert dev2load for currents
    iv3t.convert("I", "dev2load")
    # Calculate power
    iv3t.Pcalc()
    # Check kirchhoff
    iv3t.kirchhoff(iv3t.Idevlist)
    # recalculate currents
    iv3t.convert("I", "load2dev")

    # Calculate everything fo iv3t2 directly. Should be the same?
    dev3T.I3Trel(iv3t2)
    iv3t.append(iv3t2.MPP())

    iv3t.delete(5)

    test_file = "iv3t_calc.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))

if __name__ == "__main__":
    iv3t = IV3T()
    dev3T = Tandem3T()
    iv3t.line("Vzt", 0, 1.5, 20, "Vrz", "0")
    iv3t.convert("V", "dev2load")
    dev3T.J3Tabs(iv3t)
    iv3t.convert("I", "dev2load")
    iv3t.Pcalc()
    dev3T.I3Trel(iv3t)


    v = getattr(iv3t,"VB")
    i = getattr(iv3t,"IB")
    plt.plot(v,i)
    plt.show()