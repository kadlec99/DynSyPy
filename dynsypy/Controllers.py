from abc import ABC
from DynaSys import NonStateSpaceSystem

import numpy as np


class Controllers(NonStateSpaceSystem, ABC):

    def __init__(self, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=2, number_of_outputs=2,
                 allowed_error=1e-6, dt_max=1e-2):
        """

        Parameters
        ----------
        dt0: float
        t0: float
        x0: numpy.ndarray/float
        number_of_inputs: int
        number_of_outputs: int
        allowed_error: float
        dt_max: float
        """

        super().__init__(dt0, t0, x0,
                         number_of_inputs, number_of_outputs,
                         allowed_error, dt_max)


# ----------------------------------------------------------------------------


class ASMScalarControl(Controllers):

    def __init__(self, parameters, dt0=1.5e-5, t0=0, x0=np.zeros([2, 1]),
                 number_of_inputs=2, number_of_outputs=2,
                 allowed_error=1e-6, dt_max=1e-2):

        super().__init__(dt0, t0, x0,
                         number_of_inputs, number_of_outputs,
                         allowed_error, dt_max)

        self.K_U = (parameters["U_s_n"] * np.sqrt(2)) / parameters["f_s_n"]
        self.K_f_r =\
            (parameters["U_s_n"] * np.sqrt(2) * parameters["R_s"]) / (parameters["f_s_n"] * parameters["R_r"])

        self.rad_to_deg = 1 / (2 * np.pi)
        self.mech_rad_to_el_deg = parameters["p_p"] / (2 * np.pi)

        self.U_s_max = parameters["U_s_n"] * np.sqrt(2)

    def update_state(self):

        self.x = self.system_function(self.u)

    def system_function(self, u):

        x = np.zeros([2, 1])

        f_r = self.rad_to_deg * u[0][0]
        f_me = self.mech_rad_to_el_deg * u[1][0]

        x[1][0] = f_me + f_r
        x[0][0] = self.K_f_r * abs(f_r) + self.K_U * abs(x[1][0])

        if x[0][0] > self.U_s_max:
            x[0][0] = self.U_s_max

        return x

    def update_output(self):

        self.y = self.x

    def output(self, t):

        index = np.searchsorted(self.archive_t, t)

        if index >= len(self.archive_t):
            self.last_used_archive_index = len(self.archive_t) - 2
            return self.system_function(self.input_linear_regression(t, len(self.archive_t) - 1))
        else:
            if self.archive_t[index] == t:
                self.last_used_archive_index = index
                return self.system_function(self.archive_u[:, [self.last_used_archive_index]])
            elif self.archive_t[index] > t:
                self.last_used_archive_index = index - 1
                return self.system_function(self.input_linear_regression(t, index))
            else:
                self.last_used_archive_index = index - 2
                return self.system_function(self.input_linear_regression(t, index - 1))


# ----------------------------------------------------------------------------


class PIController(Controllers):

    def __init__(self, parameters, dt0=1.5e-5, t0=0, x0=np.zeros([3, 1]),
                 number_of_inputs=2, number_of_outputs=1,
                 allowed_error=1e-6, dt_max=1e-2):

        super().__init__(dt0, t0, x0,
                         number_of_inputs, number_of_outputs,
                         allowed_error, dt_max)

        self.K = parameters["K"]
        self.T_i = parameters["T_i"]
        self.T_r = 0.5 * parameters["T_i"]

        self.saturation_value = parameters["saturation_value"]

        self.output_matrix = np.array([[0, 0, 1]])

    def update_state(self):

        self.x = self.system_function(self.u, self.x, self.dt)

    def saturation(self, y):

        if y > self.saturation_value:
            return self.saturation_value
        elif y < -self.saturation_value:
            return -self.saturation_value
        else:
            return y

    def system_function(self, u, x, dt):

        e = u[0][0] - u[1][0]

        proportional_part = self.K * e
        x[0][0] = x[0][0] + (self.K / self.T_i * e - x[1][0]) * dt

        y = proportional_part + x[0][0]

        x[2][0] = self.saturation(y)

        x[1][0] = (y - x[2][0]) / self.T_r  # anti-windup

        return x

    def update_output(self):

        self.y = self.output_matrix @ self.x

    def output(self, t):

        index = np.searchsorted(self.archive_t, t)

        if index >= len(self.archive_t):
            self.last_used_archive_index = len(self.archive_t) - 2
            return self.output_matrix @ self.system_function(
                self.input_linear_regression(t, len(self.archive_t) - 1),
                self.archive_x[:, [-1]],
                self.archive_t[-1] - t)
        else:
            if self.archive_t[index] == t:
                self.last_used_archive_index = index
                return self.output_matrix @ self.system_function(
                    self.archive_u[:, [self.last_used_archive_index]],
                    self.archive_x[:, [index - 1]],
                    self.archive_t[index] - self.archive_t[index - 1])
            elif self.archive_t[index] > t:
                self.last_used_archive_index = index - 1
                return self.output_matrix @ self.system_function(
                    self.input_linear_regression(t, index),
                    self.archive_x[:, [index - 1]],
                    self.archive_t[index - 1] - t)
            else:
                self.last_used_archive_index = index - 2
                return self.output_matrix @ self.system_function(
                    self.input_linear_regression(t, index - 1),
                    self.archive_x[:, [index - 1]],
                    self.archive_t[index - 1] - t)
