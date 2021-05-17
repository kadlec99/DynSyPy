from abc import ABC, abstractmethod
from DynaSys import System

import numpy as np


class Source(System, ABC):

    def __init__(self, dt0=1.5e-5):

        super().__init__(dt0,
                         x0=0, t0=0,
                         number_of_inputs=0, number_of_outputs=1,
                         allowed_error=1e-6, dt_max=1e-2)

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

    def adaptive_step(self, end_of_pool_step):

        self.step(end_of_pool_step)

    def update_state(self):

        self.x = self.source_function(self.t)

    @abstractmethod
    def source_function(self, t):

        pass

    def update_output(self):

        self.y = self.source_function(self.t)

    def output(self, t):

        return self.source_function(t)


# ----------------------------------------------------------------------------


class TrigonometricFunctions(Source, ABC):

    def __init__(self, amplitude, frequency, phase, dt0):

        super().__init__(dt0)

        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase


# ----------------------------------------------------------------------------


class Sine(TrigonometricFunctions):

    def __init__(self, amplitude, frequency, phase, dt0=1.5e-5):

        super().__init__(amplitude, frequency, phase, dt0)

    def source_function(self, t):

        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)


# ----------------------------------------------------------------------------


class Cosine(TrigonometricFunctions):

    def __init__(self, amplitude, frequency, phase, dt0=1.5e-5):

        super().__init__(amplitude, frequency, phase, dt0)

    def source_function(self, t):

        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)


# ----------------------------------------------------------------------------


class UnitStep(Source):

    def __init__(self, parameter=1, dt0=1.5e-5):

        super().__init__(dt0)

        self.parameter = parameter

    def source_function(self, t):

        return self.parameter
