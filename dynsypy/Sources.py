from abc import ABC, abstractmethod
from DynaSys import System

import numpy as np


class Source(System, ABC):

    def __init__(self, number_of_inputs, number_of_outputs,
                 dt0=1.5e-5, t0=0, x0=0):
        """

        Parameters
        ----------
        dt0: float
        t0: float
        x0: numpy.ndarray/float
        number_of_inputs: int
        number_of_outputs: int
        """

        super().__init__(dt0=dt0, t0=t0, x0=x0,
                         number_of_inputs=number_of_inputs,
                         number_of_outputs=number_of_outputs,
                         allowed_error=1e-6, dt_max=1e-2)

    def step(self, end_of_pool_step):

        while self.t < end_of_pool_step:
            self.t = self.t + self.dt

            if self.t > end_of_pool_step:
                self.t = end_of_pool_step

            self.update_input()
            self.update_state()
            self.update_output()

            self.data_archiving()

    def adaptive_step(self, end_of_pool_step):

        self.step(end_of_pool_step)

    def update_state(self):

        self.x = self.source_function(self.t)

    @abstractmethod
    def source_function(self, t):

        pass

    def update_output(self):

        # self.y = self.source_function(self.t)
        self.y = self.x


# ----------------------------------------------------------------------------


class UncontrolledSource(Source, ABC):

    def __init__(self, number_of_outputs,
                 dt0=1.5e-5, t0=0, x0=0):
        """

        Parameters
        ----------
        dt0: float
        t0: float
        x0: numpy.ndarray/float
        number_of_outputs: int
        """

        super().__init__(number_of_inputs=0,
                         number_of_outputs=number_of_outputs,
                         dt0=dt0, t0=t0, x0=x0)

    def step(self, end_of_pool_step):

        while self.t < end_of_pool_step:
            self.t = self.t + self.dt

            if self.t > end_of_pool_step:
                self.t = end_of_pool_step

            # self.update_input()
            self.update_state()
            self.update_output()
            # self.archive_x = np.append(self.archive_x, self.x, axis=1)
            self.archive_x = np.append(self.archive_x, self.x)
            self.archive_y = np.append(self.archive_y, self.y)
            # self.archive_u = np.append(self.archive_u, self.u)
            self.archive_t = np.append(self.archive_t, self.t)

    def output(self, t):

        return self.source_function(t)


# ----------------------------------------------------------------------------


class ControlledSource(Source, ABC):

    def __init__(self, number_of_inputs, number_of_outputs,
                 dt0=1.5e-5, t0=0, x0=0):
        """

        Parameters
        ----------
        dt0: float
        t0: float
        x0: numpy.ndarray/float
        number_of_inputs: int
        number_of_outputs: int
        """

        super().__init__(number_of_inputs=number_of_inputs,
                         number_of_outputs=number_of_outputs,
                         dt0=dt0, t0=t0, x0=x0)

# ----------------------------------------------------------------------------


class HarmonicFunctions(UncontrolledSource, ABC):

    def __init__(self, parameters, dt0, t0):

        super().__init__(1, dt0, t0)

        self.amplitude = parameters["amplitude"]
        self.frequency = parameters["frequency"]
        self.phase = parameters["phase"]


# ----------------------------------------------------------------------------


class Sine(HarmonicFunctions):

    def __init__(self, parameters, dt0=1.5e-5, t0=0):

        super().__init__(parameters, dt0, t0)

    def source_function(self, t):

        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)


# ----------------------------------------------------------------------------


class Cosine(HarmonicFunctions):

    def __init__(self, parameters, dt0=1.5e-5, t0=0):

        super().__init__(parameters, dt0, t0)

    def source_function(self, t):

        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)


# ----------------------------------------------------------------------------


class UnitStep(UncontrolledSource):

    def __init__(self, final_value=1, initial_value=0.0, step_time=0, dt0=1.5e-5, t0=0):

        super().__init__(1, dt0, t0)

        self.step_time = step_time
        self.initial_value = initial_value
        self.final_value = final_value

    def source_function(self, t):

        if t >= self.step_time:
            return self.final_value
        else:
            return self.initial_value


# ----------------------------------------------------------------------------


class ControlledSine(ControlledSource):     # changes the order of phases

    def __init__(self, parameters, dt0=1.5e-5, t0=0):

        super().__init__(number_of_inputs=2, number_of_outputs=1,
                         dt0=dt0, t0=t0, x0=np.array([[0]]))

        self.amplitude = parameters["amplitude"]
        self.frequency = parameters["frequency"]
        self.phase = parameters["phase"]
        self.beta = parameters["phase"]

    def update_input(self):
        super(ControlledSine, self).update_input()

        self.amplitude = self.u[0, 0]
        self.frequency = self.u[1, 0]

    def update_beta(self):
        self.beta = self.beta + 2 * np.pi * self.dt * self.frequency

        if self.beta > np.pi:
            self.beta = self.beta - 2 * np.pi

        if self.beta < -np.pi:
            self.beta = self.beta + 2 * np.pi

    def source_function(self, t):

        self.update_beta()

        return np.array([[self.amplitude * np.sin(self.beta + self.phase)]])


# ----------------------------------------------------------------------------


class ControlledNPhaseSine(ControlledSource):

    def __init__(self, parameters, dt0=1.5e-5, t0=0):

        self.amplitude = parameters["amplitude"]
        self.frequency = parameters["frequency"]

        self.number_of_phases = parameters["number_of_phases"]

        self.phase = 0
        self.beta = 0

        self.x0 = 0

        self.source_initialize(parameters["phase"], t0)

        super().__init__(number_of_inputs=2,
                         number_of_outputs=self.number_of_phases,
                         dt0=dt0, t0=t0,
                         x0=self.x0)

    @staticmethod
    def angle_range_adjustment(angle):

        if angle > np.pi:
            angle = angle - 2 * np.pi

        if angle < -np.pi:
            angle = angle + 2 * np.pi

        return angle

    def source_initialize(self, phase, t0):

        self.phase = np.zeros((self.number_of_phases, 1)) + phase

        for i in range(1, self.number_of_phases):
            self.phase[i, 0] = (self.number_of_phases - i) * 2 * np.pi / self.number_of_phases
            self.phase[i, 0] = self.angle_range_adjustment(self.phase[i, 0])

        self.x0 = self.source_function(t0)

    def update_input(self):

        super(ControlledNPhaseSine, self).update_input()

        self.amplitude = self.u[0, 0]
        self.frequency = self.u[1, 0]

    def update_beta(self):

        self.beta = self.beta + 2 * np.pi * self.dt * self.frequency

    def update_state(self):

        self.update_beta()

        super(ControlledNPhaseSine, self).update_state()

    def source_function(self, t):

        self.beta = self.angle_range_adjustment(self.beta)
        beta = self.beta + self.phase

        return self.amplitude * np.sin(beta)
