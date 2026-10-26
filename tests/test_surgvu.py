import numpy as np
import pandas as pd
import pytest

from swm.data import surgvu as sv


def test_hms_parses_clock_and_numbers():
    assert sv.hms("01:02:03.5") == pytest.approx(3723.5)
    assert sv.hms("226.189629") == pytest.approx(226.189629)
    assert sv.hms(12) == 12.0


def test_canonical_names():
    assert sv.canonical_instrument("nan(camera in)") == sv.CAMERA
    assert sv.canonical_instrument("clip applier ") == "clip applier"
    assert sv.canonical_instrument(np.nan) == sv.UNKNOWN
    assert sv.canonical_task("suturing") == "Suturing"
    with pytest.raises(KeyError):
        sv.canonical_instrument("laser")


def test_part_offsets_and_missing_part():
    tl = sv.Timeline("case_x", offsets={1: 0.0, 2: 100.0}, duration=250.0)
    assert tl.to_global(2, "00:00:10") == pytest.approx(110.0)
    assert tl.to_local(110.0) == (2, pytest.approx(10.0))
    missing = sv.Timeline("case_y", offsets={2: 0.0}, duration=50.0)   # part 1 never delivered
    assert np.isnan(missing.to_global(1, 5.0))
    df = pd.DataFrame(dict(t0=[np.nan, np.nan, 1.0], t1=[20.0, np.nan, 3.0]))
    out, qc = sv._clip_missing_parts(df, pd.Series([2, 1, 2]), missing)
    assert qc["missing_part"] == 1
    assert out["t0"].tolist() == [0.0, 1.0]     # spill-over interval starts at part 2's start


def test_arm_occupancy_latest_install_wins():
    tools = pd.DataFrame(dict(arm=["USM1", "USM1", "USM2"], occupant=["needle driver", "stapler", "camera"],
                              t0=[0.0, 3.0, 0.0], t1=[9.0, 5.0, 9.0]))
    occ = sv.arm_occupancy(tools, duration=10.0, hz=1.0)
    assert occ[2, 0] == sv.OCC_INDEX["needle driver"]
    assert occ[4, 0] == sv.OCC_INDEX["stapler"]
    assert (occ[:, 1] == sv.OCC_INDEX[sv.CAMERA]).sum() == 10
