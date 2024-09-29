# file that holds the dynamics class of the drone.
# this class is only to be used along with the option argument of
# the reset function of the environment, to inject arbitrary dynamics
# irrespective of the scale variant nature of the drone.

from dataclasses import dataclass


@dataclass
class CustomDynamics:
    """
    This class is only suppoesed to be used with the argument
    of the reset function of the environment, to inject arbitrary
    dynamics irrespective of the scale variant nature of the drone.
    """

    arm_length: float
    mass: float
    ixx: float
    iyy: float
    izz: float
    km_kf: float
