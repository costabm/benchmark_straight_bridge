import re
import os
import h5py
import numpy as np
import json
import OrcFxAPI

from typing import List
from math import atan2, atan, cos, sin
from scipy.interpolate import interp1d, LinearNDInterpolator

SIMULATION_DATA = "SimulationData"
FORCE_INDEX = {"AppliedForceX": 0,
               "AppliedForceY": 1,
               "AppliedForceZ": 2,
               "AppliedMomentX": 3,
               "AppliedMomentY": 4,
               "AppliedMomentZ": 5}


def ramping_function(ramping_duration: float, time: np.ndarray):
    """Return ramping function for a given time sample"""
    if ramping_duration > 0:
        r = np.clip((time - time[0]) / ramping_duration, 0, 1)
        return r ** 3 * (6 * r ** 2 - 15 * r + 10.)
    return np.ones_like(time)


class WindCoefficients:
    def __init__(self, beta, alpha, wind_coefficients):
        if beta is not None:
            if not isinstance(beta, list):
                raise Exception("beta must be a list of float values.")
        if alpha is not None:
            if not isinstance(alpha, list):
                raise Exception("alpha must be a list of float values.")

        self._wind_coefficients = [lambda *x: 0] * 6
        self._value = np.zeros(6, dtype=float)
        if (beta is not None) and (alpha is not None):
            if not len(alpha) == len(beta):
                raise Exception("alpha and beta must have identical length.")
            for i in [0, 1, 2, 3, 4, 5]:
                if wind_coefficients[i] is not None:
                    if not isinstance(wind_coefficients[i], list):
                        raise Exception("3D aerodynamic coefficient must be a list of float values.")
                    if not len(wind_coefficients[i]) == len(alpha):
                        raise Exception("3D aerodynamic coefficient must have identical length as alpha and beta.")
                    self._wind_coefficients[i] = WindCoefficient3D(beta, alpha, wind_coefficients[i])

        elif alpha is not None:
            for i in [0, 1, 5]:
                if wind_coefficients[i] is not None:
                    if not isinstance(wind_coefficients[i], list):
                        raise Exception("2D aerodynamic coefficient must be a list of float values.")
                    if not len(wind_coefficients[i]) == len(alpha):
                        raise Exception("2D aerodynamic coefficient must have identical length as alpha.")
                    self._wind_coefficients[i] = WindCoefficient2D(alpha, wind_coefficients[i])

        else:
            for i in [0, 1]:
                if wind_coefficients[i] is not None:
                    if not isinstance(wind_coefficients[i], (float, int)):
                        raise Exception("Constant drag coefficient must be a float value")
                    self._wind_coefficients[i] = WindCoefficient1D(wind_coefficients[i], i)

    def __call__(self, beta: float, alpha: float, force_index):
        for i in force_index:
            self._value[i] = self._wind_coefficients[i](beta, alpha)
        return self._value


class WindCoefficient1D:
    """This class is used to represent wind coefficients that are constant for all alpha angles. This is
    the standard drag force formulation as defined in OrcaFlex."""

    def __init__(self, wind_coefficient: float, index: int):
        self.cd = wind_coefficient
        if index == 0:  # x-direction, such that Fx is proportional to Cd * cos(alpha)
            self.alpha_function = cos
        elif index == 1:  # y-direction, such that Fy is proportional to Cd * sin(alpha)
            self.alpha_function = sin

    def __call__(self, beta: float, alpha: float):
        return self.cd * self.alpha_function(alpha) * cos(beta) ** 2


class WindCoefficient2D:
    """This class is used to represent wind coefficients that are defined for alpha angles only. This formulation
    is given in "Bridge buffeting by skew winds: A quasi-steady case study." eq(12)"""

    def __init__(self, alpha: List[float], wind_coefficient: List[float]):
        alpha, wind_coefficient = np.array(alpha), np.array(wind_coefficient)
        alpha %= 2 * np.pi
        mask = alpha == 0
        alpha = np.append(alpha, 2 * np.pi)
        wind_coefficient = np.concatenate([wind_coefficient, wind_coefficient[mask]])
        arg = np.argsort(alpha)
        self.alpha = alpha[arg]
        self.wind_coefficient = wind_coefficient[arg]

    def __call__(self, beta: float, alpha: float):
        return np.interp(alpha, self.alpha, self.wind_coefficient) * cos(beta) ** 2


class WindCoefficient3D(LinearNDInterpolator):
    """This class is used to represent wind coefficients that are defined for both alpha and beta angles."""

    def __init__(self, beta: List[float], alpha: List[float], wind_coefficient: List[float]):
        beta, alpha, wind_coefficient = np.array(beta), np.array(alpha), np.array(wind_coefficient)
        alpha %= 2 * np.pi
        mask = alpha == 0
        alpha = np.append(alpha, alpha[mask] + 2 * np.pi)
        beta = np.append(beta, beta[mask])
        wind_coefficient = np.concatenate([wind_coefficient, wind_coefficient[mask]])
        super().__init__((beta, alpha), wind_coefficient)


class WindData:
    force: np.ndarray
    wind_cofficients: WindCoefficients
    factor: np.ndarray
    wind_speed: interp1d

    def __init__(self, info):
        self.time_shift = 0.0
        self.info = info
        self.force_index = []

    @staticmethod
    def alpha(u: List[float]):
        return atan2(u[1], u[0]) % (2 * np.pi)

    @staticmethod
    def beta(u: List[float]):
        return atan(u[2] / (u[0] ** 2 + u[1] ** 2) ** 0.5)

    def update_wind_force(self):
        """Calculate the local aerodynamic forces based on the instantaneous relative wind speed and the
        aerodynamic load coefficients.
        """
        model_data = self.info.InstantaneousCalculationData
        simulation_time = self.info.SimulationTime - self.time_shift
        u_g = self.wind_speed(simulation_time) - model_data.Velocity  # relative wind speed vector in global coordinate system.
        tr_g2l = np.array(model_data.NodeOrientation)  # transformation matrix global -> local coordinate system.
        u_l = tr_g2l.dot(u_g)  # relative wind speed vector in local coordinate system.
        self.force = u_g.dot(u_g) * self.wind_coefficients(self.beta(u_l), self.alpha(u_l), self.force_index) * self.factor


class SimulationData:
    def __init__(self):
        self.static_state = True

    def update_force(self, new_time_step: bool) -> bool:
        """Return True if OrcaFlex is starting a new time step or if model OrcaFlex model is in static state."""
        if new_time_step:
            self.static_state = False
            return True
        return self.static_state


class WindForce:
    force_index: int
    workspace_key: str

    def Initialise(self, info):
        """Initialise the wind force model. Wind speed time series (extracted from the WindSim result h5 file) and
        aerodynamic load coefficients (read from tags on the shape objects) are stored on WindData objects, which are
        accessed from the Calculation method during static and dynamic simulation."""

        if info.Model.state == OrcFxAPI.ModelState.CalculatingStatics:
            info.UpdateDuringStatics = True
            model_object_name = info.ModelObject.Name
            object_data_name = re.split("\\[|\\]", info.DataName)
            variable_index = int(object_data_name[1]) - 1
            factor = 0.5 * info.Model["Environment"].AirDensity
            self.force_index = FORCE_INDEX[object_data_name[0]]
            self.workspace_key = f"{model_object_name} {variable_index}"  # All forces/moments in one applied load point will share a common workspace key.

            if SIMULATION_DATA not in info.Workspace:
                simulation_data = SimulationData()
                windsim_path = info.Model.environment.tags.windsim
                if not os.path.isabs(windsim_path):
                    windsim_path = os.path.join(info.ModelDirectory, windsim_path)
                simulation_data.windsim_file = h5py.File(windsim_path, "r")  # h5 file object containing WindSim time series.
                simulation_data.time_shift = json.loads(info.Model.environment.tags.get("time_shift", "true").lower())  # If True, the Taylor frozen hypothesis is used to calculate the time shift.
                simulation_data.windsim_time = info.Model.simulationStartTime + simulation_data.windsim_file["Time"][:]  # Array containing time sample from WindSim.
                if simulation_data.windsim_time[-1] < info.Model.simulationStopTime:
                    raise ValueError('WindSim time series cannot be shorter than the total OrcaFlex simulation time.')
                ramping_duration = info.Model.general.ActualRampFinishTime - info.Model.general.ActualRampStartTime
                simulation_data.ramping_factor = ramping_function(ramping_duration, simulation_data.windsim_time)  # Ramping factor that increase from 0 to 1 during ramping in Orcaflex.
                info.Workspace[SIMULATION_DATA] = simulation_data
            simulation_data = info.Workspace[SIMULATION_DATA]

            if self.workspace_key not in info.Workspace:
                wind_data = WindData(info)
                delta_length = simulation_data.windsim_file[model_object_name]["Element Length"][variable_index]
                mean_wind_speed = simulation_data.windsim_file[model_object_name]["Mean Wind Speed"][variable_index]
                gust_wind_speed = simulation_data.windsim_file[model_object_name]["Gust Wind Speed"][variable_index]
                line_type = simulation_data.windsim_file[model_object_name]["Line Type"][variable_index].decode()
                arclength = simulation_data.windsim_file[model_object_name]["Local Position"][variable_index, -1]
                wind_speed = mean_wind_speed + gust_wind_speed * simulation_data.ramping_factor[:, None]
                wind_data.wind_speed = interp1d(simulation_data.windsim_time, wind_speed,
                                                axis=0, copy=False, assume_sorted=True)
                if simulation_data.time_shift:
                    wind_data.time_shift_factor = mean_wind_speed / mean_wind_speed.dot(mean_wind_speed)
                else:
                    wind_data.time_shift_factor = np.zeros(3)

                tags = info.Model[line_type].tags
                orcaflex_linetype = info.ModelObject.lineTypeAt(arclength)
                B = float(tags.get("B", orcaflex_linetype.tags.get("B", orcaflex_linetype.NormalDragLiftDiameter)))
                if B == OrcFxAPI.OrcinaDefaultReal():
                    B = orcaflex_linetype.OD
                wind_data.factor = factor * delta_length * np.array([B, B, B, B * B, B * B, B * B])

                if line_type not in info.Workspace:
                    if (beta := tags.get("beta")) is not None:
                        beta = np.radians(json.loads(beta)).tolist()
                    if (alpha := tags.get("alpha")) is not None:
                        alpha = np.radians(json.loads(alpha)).tolist()
                    if (cx := tags.get("cx")) is not None:
                        cx = json.loads(cx)
                    if (cy := tags.get("cy")) is not None:
                        cy = json.loads(cy)
                    if (cz := tags.get("cz")) is not None:
                        cz = json.loads(cz)
                    if (crx := tags.get("crx")) is not None:
                        crx = json.loads(crx)
                    if (cry := tags.get("cry")) is not None:
                        cry = json.loads(cry)
                    if (crz := tags.get("crz")) is not None:
                        crz = json.loads(crz)
                    info.Workspace[line_type] = WindCoefficients(beta, alpha, (cx, cy, cz, crx, cry, crz))

                wind_data.wind_coefficients = info.Workspace[line_type]
                info.Workspace[self.workspace_key] = wind_data

            wind_data: WindData = info.Workspace[self.workspace_key]
            wind_data.force_index.append(self.force_index)

    def Calculate(self, info):
        """Update wind forces and return value if info.NewTimeStep is True or if OrcaFlex model is in static state."""
        simulation_data = info.Workspace[SIMULATION_DATA]
        if simulation_data.update_force(info.NewTimeStep):  # Access if new time step in OrcaFlex. Will not be accessed for each iteration.
            wind_data = info.Workspace[self.workspace_key]
            if self.force_index == wind_data.force_index[0]:  # Access only once per applied load.
                if simulation_data.static_state:
                    wind_data.update_wind_force()
                    wind_data.static_position = np.array(info.InstantaneousCalculationData.Position)
                else:
                    offset = np.array(info.InstantaneousCalculationData.Position) - wind_data.static_position
                    wind_data.time_shift = wind_data.time_shift_factor.dot(offset)
                    wind_data.update_wind_force()
            info.Value = wind_data.force[self.force_index]  # Update the local