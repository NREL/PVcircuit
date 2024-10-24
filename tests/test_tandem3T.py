import itertools
import re
from pathlib import Path

import numpy as np
import pytest

import pvcircuit as pvc
from pvcircuit import IV3T, Multi2T, Tandem3T


@pytest.fixture
def dev2T():
    return Multi2T()


@pytest.fixture
def dev3T():
    return Tandem3T()


@pytest.fixture
def iv3t():
    return IV3T()


@pytest.fixture
def junction():
    return pvc.junction.Junction()


def test_tandem3T_str(dev3T):

    test_file = "Tandem3T_str.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(dev3T.__str__())

    # read fixed test case for s-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",test_str))


def test_tandem3T_maxsetters(dev3T):
    # test the area and TC setter

    top_area = dev3T.top.totalarea
    bot_area = dev3T.bot.totalarea

    tc = max(dev3T.top.TC, dev3T.bot.TC)

    np.testing.assert_almost_equal(dev3T.totalarea, max(top_area, bot_area))
    np.testing.assert_almost_equal(dev3T.TC, tc)

    set_area = max(top_area, bot_area) * 100
    dev3T.top.set(area=set_area)
    np.testing.assert_almost_equal(dev3T.totalarea, set_area)
    np.testing.assert_almost_equal(dev3T.lightarea, set_area)

    set_TC = tc * 200
    dev3T.top.set(TC=set_TC)
    np.testing.assert_almost_equal(dev3T.TC, set_TC)


def test_V3T(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("IA", -25e-3, 25e-3, 55, "IB", -30e-3, 30e-3, 55)
    iv3t.convert("I", "load2dev")
    dev3T.V3T(iv3t)

    test_file = "Tandem3T_V3T-s.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding='utf8') as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))

    dev3T.bot.set(Jext=20e-3, pn=-1)
    dev3T.V3T(iv3t)

    test_file = "Tandem3T_V3T-r.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding='utf8') as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_J3Tabs(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("VA", -1.5, 0.2, 30, "VB", -1.5, 0.2, 30)
    iv3t.convert("V", "load2dev")
    dev3T.J3Tabs(iv3t)

    test_file = "Tandem3T_J3TAabs-s.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))

    dev3T.bot.set(pn=-1)
    dev3T.J3Tabs(iv3t)

    test_file = "Tandem3T_J3TAabs-r.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_I3Trel(dev3T, iv3t):
    # test 3T voltage from current
    iv3t.box("VA", -1.5, 0.2, 30, "VB", -1.5, 0.2, 30)
    iv3t.convert("V", "load2dev")
    dev3T.I3Trel(iv3t)

    test_file = "Tandem3T_I3Trel-s.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for s-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))

    dev3T.bot.set(pn=-1)
    dev3T.I3Trel(iv3t)

    test_file = "Tandem3T_I3Trel-r.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(iv3t.__str__())

    # read fixed test case for r-type
    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_VM(dev3T):

    ratios = [(1, 1), (2, 1), (3, 2)]
    for ratio in ratios:
        iv3t_vm, iv3t_vmpp = dev3T.VM(*ratio)

        vm_fname = "Tandem3T_iv3t_vm_{}.txt".format("".join(map(str, ratio)))
        vmpp_fname = "Tandem3T_iv3t_vmpp_{}.txt".format("".join(map(str, ratio)))
        # write test case
        # with open(Path().cwd().joinpath("tests","test_files", vm_fname), "w", encoding="utf8") as fout:
        #     fout.write(iv3t_vm.__str__())
        # with open(Path().cwd().joinpath("tests","test_files", vmpp_fname), "w", encoding="utf8") as fout:
        #     fout.write(iv3t_vmpp.__str__())

        # read fixed test case for s-type
        with open(Path().cwd().joinpath("tests","test_files", vm_fname), "r", encoding="utf8") as fin:
            test_vm = fin.read()
        with open(Path().cwd().joinpath("tests","test_files", vmpp_fname), "r", encoding="utf8") as fin:
            test_vmpp = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+"," ",test_vm), re.sub(r"\s+"," ",iv3t_vm.__str__()))
        np.testing.assert_string_equal(re.sub(r"\s+"," ",test_vmpp), re.sub(r"\s+"," ",iv3t_vmpp.__str__()))


def test_CM(dev3T):
    dev2T = Multi2T.from_3T(dev3T)
    lnout, mpp = dev3T.CM()
    v2t = lnout.VA - lnout.VB
    i2t = []
    for v in v2t:
        i2t.append(dev2T.I2T(v))
    np.testing.assert_array_almost_equal(i2t, lnout.IA)

    mpp2t = dev2T.MPP()
    np.testing.assert_almost_equal(mpp2t["Pmp"], mpp.Ptot[0], decimal=6)
    np.testing.assert_almost_equal(mpp2t["Vmp"], mpp.VA - mpp.VB, decimal=3)
    np.testing.assert_almost_equal(mpp2t["Imp"], -mpp.IA, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], mpp.IB, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], -mpp.Iro, decimal=5)
    np.testing.assert_almost_equal(mpp2t["Imp"], mpp.Ito, decimal=5)


def test_MPP(dev3T):
    # Test the 4T MPP operating point.
    mpp = dev3T.MPP()

    # Use single junctions for comparison
    tc = Multi2T.from_single_junction(dev3T.top)
    bc = Multi2T.from_single_junction(dev3T.bot)

    tc_mpp = tc.MPP(pnts=30)
    bc_mpp = bc.MPP(pnts=30)

    mpp = dev3T.MPP(pnts=30)

    np.testing.assert_almost_equal(tc_mpp["Pmp"] + bc_mpp["Pmp"], mpp.Ptot[0], decimal=4)
    np.testing.assert_almost_equal(tc_mpp["Imp"], mpp.Ito[0], decimal=4)
    np.testing.assert_almost_equal(bc_mpp["Imp"], -mpp.Iro[0], decimal=4)


def test_VI0(dev3T):

    for point in [
        "VztIro",
        "VrzIto",
        "VtrIzo",
    ]:

        iv3t = dev3T.VI0(point)

        test_file = f"Tandem3T_VI0_{point}.txt"
        # write test case
        # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
        #     fout.write(iv3t.__str__())

        # read fixed test case for s-type
        with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
            test_str = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))


def test_VIpoints(dev3T):

    iv3t = IV3T()

    current_keys = [k for k in iv3t.arraykeys if k.startswith("I") and len(k) > 2]
    voltage_keys = [k for k in iv3t.arraykeys if k.startswith("V") and len(k) > 2]

    iv3t = dev3T.VIpoint("Iro", "Izo", "Vzt")
    # test a few randomly
    combs = list(itertools.product(range(len(current_keys)), range(len(current_keys)), range(len(voltage_keys))))
    combs = [combo for combo in combs if combo[0] != combo[1]]
    combids = [4, 42, 21, 43, 23]
    for combid in combids:
        combo = combs[combid]
        iv3t = dev3T.VIpoint(current_keys[combo[0]], current_keys[combo[1]], voltage_keys[combo[2]])

        test_file = f"Tandem3T_VIpoint_{iv3t.name}.txt"
        # write test case
        # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
        #     fout.write(iv3t.__str__())

        with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
            test_str = fin.read()

        np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",iv3t.__str__()))

    #     iv3t_vals = np.concatenate([getattr(iv3t,k) for k in iv3t.arraykeys])
    #     if all(~np.isnan(iv3t_vals)):
    #         testids.append(combid)

    # print(testids)
    # import random
    # random_ids = random.sample(testids, 5)


def test_specialpoints(dev3T):

    special_points = dev3T.specialpoints()

    test_file = "Tandem3T_specialpoints.txt"
    # write test case
    # with open(Path().cwd().joinpath("tests","test_files", test_file), "w", encoding="utf8") as fout:
    #     fout.write(special_points.__str__())

    with open(Path().cwd().joinpath("tests","test_files", test_file), "r", encoding="utf8") as fin:
        test_str = fin.read()

    np.testing.assert_string_equal(re.sub(r"\s+"," ",test_str), re.sub(r"\s+"," ",special_points.__str__()))


if __name__ == "__main__":
    dev3T = Tandem3T()
    iv3t = IV3T()

    current_keys = [k for k in iv3t.arraykeys if k.startswith("I") and len(k) > 2]
    voltage_keys = [k for k in iv3t.arraykeys if k.startswith("V") and len(k) > 2]

    # test a few randomly
    combs = list(itertools.product(range(len(current_keys)), range(len(current_keys)), range(len(voltage_keys))))
    combs = [combo for combo in combs if combo[0] != combo[1]]
    combids = range(len(combs))
    testids = []
    testnames = []
    for combid in combids:
        combo = combs[combid]
        iv3t = dev3T.VIpoint(current_keys[combo[0]], current_keys[combo[1]], voltage_keys[combo[2]])
        iv3t_vals = np.concatenate([getattr(iv3t, k) for k in iv3t.arraykeys])
        if all(~np.isnan(iv3t_vals)):
            if iv3t.name not in testnames:
                testids.append(combid)
                testnames.append(iv3t.name)

    print(testnames)
    print(testids)
    import random

    random_ids = random.sample(testids, 5)
    print(random_ids)
    print(testnames)