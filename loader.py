"""Loader for the Second-batch SUMO experiment data (.mat files).

File naming: data_{A}_{UC}_{QIN}_{True|False}.mat, where A >= 100 encodes
assertiveness A/100 (e.g. 150 -> 1.5). UC is the controlled-vehicle slow
reference speed [m/s], QIN the background inflow [veh/h].

Variables inside each file, per repetition r in 0..4 (suffix _r) plus a
rep-averaged copy without the suffix:

  densities_{A}_{UC}_{QIN}[_r]   (TEND, 300)   [veh/km], t = (1..TEND)*10 s,
                                               cells of 100 m (x = 100..30000 m)
  flows_{A}_{UC}_{QIN}[_r]       (TEND, 300)   [veh/h]
  updown_{A}_{UC}_{QIN}[_r]      (4, TEND+1)   rows: rho-, q-, rho+, q+
                                               (up/downstream of the CAV),
                                               t = (0..TEND)*10 s
  overtake_{A}_{UC}_{QIN}[_r]    (1, TEND+1)   overtaking flow [veh/h]
  trajectory_flow{i}_{A}_{UC}_{QIN}_{t0}_{r}   (3, N) rows x[m], v[m/s], lane
  trajectory_vehicle_S_{n1}_{UC}_{QIN}_{t0}_{r}(3, N) controlled vehicle
                                               (n1 is NOT A; constant 3)

TEND = 100 for "True" files (1000 s), 70 for "False" files (700 s).
Timeline (ECC22): CAV enters at 100 s, slows to UC at 250 s, resumes 750 s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

T_SAMPLE = 10.0          # [s]
CELL_LEN = 100.0         # [m]
N_CELL = 300
T_SLOW, T_FAST = 250.0, 750.0   # [s] CAV slow phase

_FNAME_RE = re.compile(r"data_(\d+)_(\d+)_(\d+)_(True|False)\.mat$")


@dataclass
class Trajectory:
    t: np.ndarray            # [s]
    x: np.ndarray            # [m]
    v: np.ndarray            # [m/s]
    lane: np.ndarray | None  # 0 = right lane, 1 = left lane


@dataclass
class Rep:
    rho: np.ndarray | None = None       # (TEND, 300) [veh/km]
    q: np.ndarray | None = None         # (TEND, 300) [veh/h]
    updown: np.ndarray | None = None    # (4, TEND+1)
    overtake: np.ndarray | None = None  # (TEND+1,)
    ctrl: Trajectory | None = None
    trajs: list[Trajectory] = field(default_factory=list)


@dataclass
class Scenario:
    path: Path
    A: float                 # assertiveness (lcAssertive)
    uc: float                # [m/s] CAV slow reference speed
    qin: float               # [veh/h] inflow
    flag: bool               # True/False file variant
    tend: int                # number of field samples (100 or 70)
    reps: dict[int, Rep] = field(default_factory=dict)

    @property
    def t_field(self) -> np.ndarray:
        """Times of the density/flow field rows [s]."""
        return (np.arange(self.tend) + 1) * T_SAMPLE

    @property
    def t_updown(self) -> np.ndarray:
        return np.arange(self.tend + 1) * T_SAMPLE

    @property
    def x_cells(self) -> np.ndarray:
        """Cell centers... cell j covers ((j)*100, (j+1)*100]; use upper edge
        like the MATLAB code (XS = (1:NCELL)*L)."""
        return (np.arange(N_CELL) + 1) * CELL_LEN


def parse_filename(path: Path):
    m = _FNAME_RE.search(path.name)
    if not m:
        return None
    a_raw, uc, qin, flag = m.groups()
    A = int(a_raw) / 100.0 if int(a_raw) >= 100 else float(a_raw)
    return A, float(uc), float(qin), flag == "True"


def list_scenarios(second_dir: str | Path, flag: bool | None = True):
    """All scenario files, optionally filtered to True/False variant."""
    out = []
    for p in sorted(Path(second_dir).glob("data_*.mat")):
        parsed = parse_filename(p)
        if parsed is None:
            continue
        A, uc, qin, fl = parsed
        if flag is not None and fl is not flag:
            continue
        out.append((p, A, uc, qin, fl))
    return out


def load_scenario(path: str | Path, fields: bool = True, ctrl: bool = True,
                  trajs: bool = False) -> Scenario:
    """Load one .mat file. `fields` = densities/flows/updown/overtake,
    `ctrl` = controlled-vehicle trajectories, `trajs` = all background
    trajectories (heavy: thousands of variables)."""
    path = Path(path)
    A, uc, qin, fl = parse_filename(path)
    tend = 100 if fl else 70
    sc = Scenario(path=path, A=A, uc=uc, qin=qin, flag=fl, tend=tend)

    names = [n for n, _, _ in sio.whosmat(str(path))]
    want = []
    for n in names:
        if fields and re.fullmatch(r"(densities|flows|updown|overtake)_\d+_\d+_\d+_\d+", n):
            want.append(n)
        elif ctrl and n.startswith("trajectory_vehicle_S"):
            want.append(n)
        elif trajs and n.startswith("trajectory_flow"):
            want.append(n)
    data = sio.loadmat(str(path), variable_names=want)

    def rep_of(n: int) -> Rep:
        return sc.reps.setdefault(n, Rep())

    for n, arr in data.items():
        if n.startswith("__"):
            continue
        if n.startswith(("densities_", "flows_", "updown_", "overtake_")):
            r = int(n.rsplit("_", 1)[1])
            rep = rep_of(r)
            if n.startswith("densities_"):
                rep.rho = np.asarray(arr, float)
            elif n.startswith("flows_"):
                rep.q = np.asarray(arr, float)
            elif n.startswith("updown_"):
                rep.updown = np.asarray(arr, float)
            else:
                rep.overtake = np.asarray(arr, float).ravel()
        elif n.startswith("trajectory_vehicle_S"):
            nums = [int(s) for s in re.findall(r"\d+", n)]
            t0, r = nums[-2], nums[-1]
            arr = np.asarray(arr, float)
            tt = (t0 + np.arange(arr.shape[1])) * T_SAMPLE
            lane = arr[2] if arr.shape[0] > 2 else None
            rep_of(r).ctrl = Trajectory(t=tt, x=arr[0], v=arr[1], lane=lane)
        elif n.startswith("trajectory_flow"):
            nums = [int(s) for s in re.findall(r"\d+", n)]
            t0, r = nums[-2], nums[-1]
            arr = np.asarray(arr, float)
            tt = (t0 + np.arange(arr.shape[1])) * T_SAMPLE
            rep_of(r).trajs.append(
                Trajectory(t=tt, x=arr[0], v=arr[1], lane=arr[2]))
    return sc


def ctrl_position(sc: Scenario, r: int, t: np.ndarray) -> np.ndarray:
    """CAV position [m] interpolated at times t [s]; NaN outside its lifetime."""
    c = sc.reps[r].ctrl
    return np.interp(t, c.t, c.x, left=np.nan, right=np.nan)
