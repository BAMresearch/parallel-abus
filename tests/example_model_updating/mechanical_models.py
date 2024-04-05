"""Mechanical models for the static and dynamic behaviour of the concrete beam."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import openseespy.opensees as ops


@dataclass(frozen=True)
class GenericSection:
    """Generic section with properties area, centre of gravity, moments of inertia."""

    A: float
    y_c: float
    z_c: float
    I_yy: float
    I_zz: float
    I_yz: float

# Definition of units

# define basic units
meter = 1.0
sec = 1.0
newton = 1.0
rad = 1.0

# define dependent units
cm = meter / 100
mm = meter / 1000

mrad = rad / 1000

kN = newton * 10**3
MN = newton * 10**6

# 1 Pa = N/m²
Pa = newton / (meter**2)
kPa = Pa * 10**3
MPa = Pa * 10**6
GPa = Pa * 10**9

kg = newton * sec**2 * meter
ton = kg * 10**3

g = 9.81 * meter / (sec**2)

# thermal models

QUERSCHNITT = GenericSection(
    A=20 * 40 * cm**2,
    y_c=0.0 * cm,
    z_c=10 * cm,
    I_yy=(40.0 * (20**3) / 12) * cm**4,
    I_zz=((40.0**3) * 20 / 12) * cm**4,
    I_yz=0.0 * cm**4,
)

QUERSCHNITT_mu = (0.2 * 0.4 * meter**2) * (2500 * kg / meter**3)


def flexural_stiffness_at_temperature_model_1(
    EI_0: float, alpha: float, T: float
) -> float:
    """Calculate the temperature dependent flexural stiffness at a given temperature.

    Arguments:
        EI_0 {float} -- Flexural stiffness at T=0°C in [Mm²]
        alpha {float} -- Temperature dependency in [Mm²/°C]
        T {float} -- Given temperature in [°C]

    Returns:
        float or np.ndarray(n_temperatures x n_EI_0) -- Flexural stiffness at temperature T
    """
    if (
        isinstance(alpha, np.ndarray)
        and isinstance(EI_0, np.ndarray)
        and isinstance(T, np.ndarray)
    ):
        EI = EI_0 + alpha[np.newaxis, :] * T[:, np.newaxis]
    else:
        EI = EI_0 + alpha * T
    return EI


# x-coordinates

# bearing_l: 0.00m
# sensor_W1: 0.68m
# mirrored load_position 3: 1.03m
# midpoint: 1.36m
# load_position 3: 1.69m
# sensor_W3: 2.04m
# bearing_r: 2.72m

# geophones:
# geohpone: 0.31m
# mirrored shaker: 0.40m
# geohpone: 0.86m
# geohpone: 1.36m
# geohpone: 1.84m
# shaker: 2.32m
# geohpone: 2.41m

# evenly spaced grid:
# additional_node_1: 0.15m
# additional_node_2: 0.54m
# additional_node_3: 1.19m
# additional_node_4: 1.53m
# additional_node_5: 2.18m
# additional_node_6: 2.57m

x_nodes = (
    np.array(
        [
            0.00,
            0.15,
            0.31,
            0.40,
            0.54,
            0.68,
            0.86,
            1.03,
            1.19,
            1.36,
            1.53,
            1.69,
            1.84,
            2.04,
            2.18,
            2.32,
            2.41,
            2.57,
            2.72,
        ]
    )
    * meter
)
n_nodes = x_nodes.shape[0]

n_nodes_fine = 273
x_nodes_fine = np.linspace(0, 2.72, n_nodes_fine)


i_node_left_bearing = 0
i_node_right_bearing = 18
i_node_W1 = 5
i_node_W2 = 9
i_node_W3 = 13


i_node_left_bearing_fine = 0
i_node_right_bearing_fine = 272
i_node_W1_fine = 68
i_node_W2_fine = 136
i_node_W3_fine = 204

# node numbers consistent with numpy 0-based indexing:
node_numbers = np.arange(0, n_nodes, 1, int)
node_numbers_fine = np.arange(0, n_nodes_fine, 1, int)


def _ops_model_undamaged_timoshenko_fine(
    EI: float, poissons_ratio: float = 0.2
) -> None:
    # 1. wipe model
    ops.wipe()
    # ops.logFile("logfile.txt", "-append")

    # 2. model generation
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # 3. create nodes
    for i_n, x_n in zip(node_numbers_fine, x_nodes_fine):
        ops.node(int(i_n), x_n, 0.0)

    # 4. boundary conditions
    ops.fix(int(node_numbers_fine[i_node_left_bearing_fine]), 1, 1, 0)
    ops.fix(int(node_numbers_fine[i_node_right_bearing_fine]), 0, 1, 0)

    # 5. settings for element creation (integration, coordinate Transformation)
    coordTransf = "Linear"
    ops.geomTransf(coordTransf, 1)
    # nPoints = 5  # number of integration-points per beam element
    # # beamIntegration('Lobatto', tag, secTag, N)
    # ops.beamIntegration("Lobatto", 1, 1, nPoints)

    E = EI * MN * meter**2 / QUERSCHNITT.I_yy

    # In ModelCode2010: poissons ratio varies between 0.14 and 0.26
    G = E / (2 * (1 + poissons_ratio))

    def create_standard_element(i_n: int, E: float, G: float) -> None:
        # assumes shear area as total area!
        i_n = int(i_n)
        # element('ElasticTimoshenkoBeam', eleTag, *eleNodes, E_mod, G_mod, Area, Iz, Avy, transfTag, <'-mass', massDens>, <'-cMass'>)
        ops.element(
            "ElasticTimoshenkoBeam",
            i_n,
            i_n,
            i_n + 1,
            E,
            G,
            QUERSCHNITT.A,
            QUERSCHNITT.I_yy,
            QUERSCHNITT.A,
            1,
            "-mass",
            QUERSCHNITT_mu,
        )
        return None

    # 6. Element creation
    for i_n in node_numbers_fine[:-1]:
        create_standard_element(i_n, E, G)


def model_1_timoshenko_ops_model_mech_fine(
    EI: float, nu: float, F: float, x_F: float
) -> Tuple[float, ...]:
    """Return the vector of predictions of the 5 measured quantities given EI, F, x_F."""
    # 1) build opensees model
    _ops_model_undamaged_timoshenko_fine(EI, nu)

    position = int(np.argmin(np.abs(x_nodes_fine - x_F)))

    # create TimeSeries
    ops.timeSeries("Constant", 1)

    # create a plain load pattern
    ops.pattern("Plain", 1, 1)

    # Create the nodal load - command: load nodeID xForce yForce
    ops.load(position, 0.0, F, 0.0)

    # ------------------------------
    # Start of analysis generation
    # ------------------------------

    # create SOE
    ops.system("BandSPD")

    # create DOF number
    ops.numberer("RCM")

    # create constraint handler
    ops.constraints("Plain")

    # create integrator
    ops.integrator("LoadControl", 1.0)

    # create algorithm
    ops.algorithm("Linear")

    # create analysis object
    ops.analysis("Static")

    # perform the analysis
    ops.analyze(1)

    displacements = (
        ops.nodeDisp(i_node_W1_fine, 2),
        ops.nodeDisp(i_node_W2_fine, 2),
        ops.nodeDisp(i_node_W3_fine, 2),
        ops.nodeDisp(i_node_left_bearing_fine, 3),
        ops.nodeDisp(i_node_right_bearing_fine, 3),
    )

    ops.wipeAnalysis()
    ops.setTime(0.0)
    ops.remove("loadPattern", 1)

    return displacements


def global_model_m1_undamaged_stat_timoshenko_fine(
    x_F: float, EI_0: float, alpha: float, T: float, nu: float = 0.2, F=3 * kN
) -> np.ndarray:
    """Calculate the displacements and rotations for an undamaged openSEES Timoshenko beam.

    This model has a small discretization.

    Args:
        x_F (float): Position of the moving load in m.
        EI_0 (float): Flexural stiffness of the cross-section at 0°C in MNm².
        alpha (float): Temperature coefficient in MNm²/K.
        T (float): Temperature in °C.
        nu (float): Poissons' ratio. Defaults to 0.2.
        F (_type_, optional): Force in N. Defaults to 3 kN.

    Returns:
        np.ndarray: [disp_W1, disp_W2, disp_W3, rot_N1, rot_N2]
    """
    EI = flexural_stiffness_at_temperature_model_1(EI_0, alpha, T)
    return np.array(model_1_timoshenko_ops_model_mech_fine(EI, nu, F, x_F))
