from abc import ABC     # , abstractmethod
from DynaSys import PartiallyNonlinearSystem, Matrix

import numpy as np


class Machine(PartiallyNonlinearSystem, ABC):

    def __init__(self, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=3, number_of_outputs=1,
                 allowed_error=1e-6, dt_max=1e-2):

        super().__init__(dt0, t0, x0,
                         number_of_inputs, number_of_outputs,
                         allowed_error, dt_max)

        self.C_p = 2 / 3 * np.array([[1, -1 / 2, -1 / 2],
                                     [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]])

        self.C_p_inv = np.array([[1, 0],
                                 [-1 / 2, np.sqrt(3) / 2],
                                 [-1 / 2, -np.sqrt(3) / 2]])

    def clarke_transformation(self, signal_3_phase, reverse=False):

        if not reverse:
            return 2 / 3 * self.C_p @ signal_3_phase
        else:
            return 2 / 3 * self.C_p_inv @ signal_3_phase

    @staticmethod
    def park_transformation(signal_alpha_beta, theta, reverse=False):

        if not reverse:
            r_dq = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])

            return r_dq @ signal_alpha_beta
        else:
            r_dq_inv = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])

            return r_dq_inv @ signal_alpha_beta


# ----------------------------------------------------------------------------


class AsynchronousMachine(Machine):

    def __init__(self, parameters, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=4, allowed_error=1e-6, dt_max=1e-2):

        number_of_outputs = np.size(x0)

        super().__init__(dt0, t0, x0,
                         number_of_inputs, number_of_outputs,
                         allowed_error, dt_max)

        self.R_s = parameters["R_s"]    # 1.617
        self.R_r = parameters["R_r"]    # 1.609
        self.L_s_sigma = parameters["L_s_sigma"]    # 8.5e-3
        self.L_r_sigma = parameters["L_r_sigma"]    # 8.5e-3
        self.L_h = parameters["L_h"]    # 134.4e-3
        self.p_p = parameters["p_p"]    # 2
        self.N_n = parameters["N_n"]    # 1420
        self.U_n = parameters["U_n"]    # 150  # 380    # 3x380
        self.f_s_n = parameters["f_s_n"]    # 50
        self.I_s_n = parameters["I_s_n"]    # 8.5
        self.J = parameters["J"]        # 0.03
        self.k_p = parameters["k_p"]    # 1.5

        self.L_s = self.L_s_sigma + self.L_h
        self.L_r = self.L_r_sigma + self.L_h

        lmb = self.L_s_sigma + self.L_s_sigma * (self.L_h / self.L_r)

        self.alpha = (self.R_s + self.R_r * ((self.L_h ** 2) / (self.L_r ** 2))) / lmb
        self.beta = (self.R_r * (self.L_h / (self.L_r ** 2))) / lmb
        self.gamma = ((self.L_h / self.L_r) / lmb) * self.p_p
        self.delta = 1 / lmb
        self.epsilon = ((self.k_p * self.p_p) / self.J) * (self.L_h / self.L_r)

        self.M_z = 0  # Load torque

        self.Theta = 0

        a_init = np.array(
            [[-self.alpha, 0, self.beta, self.gamma * self.x[4, 0], 0],
             [0, -self.alpha, -self.gamma * self.x[4, 0], self.beta, 0],
             [self.R_r * (self.L_h / self.L_r), 0, -self.R_r / self.L_r, -self.p_p * self.x[4, 0], 0],
             [0, self.R_r * (self.L_h / self.L_r), self.p_p * self.x[4, 0], -self.R_r / self.L_r, 0],
             [-self.epsilon * self.x[3, 0], self.epsilon * self.x[2, 0], 0, 0, 0]])

        non_linearity_functions =\
            [self.non_linearity_03, self.non_linearity_12, self.non_linearity_23,
             self.non_linearity_32, self.non_linearity_40, self.non_linearity_41]

        non_linearity_indexes = [[0, 3], [1, 2], [2, 3], [3, 2], [4, 0], [4, 1]]

        self.A = Matrix(a_init, non_linearity_functions, non_linearity_indexes)
        self.B = np.array([[self.delta, 0, 0],
                           [0, self.delta, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, -1 / self.J]])

        self.u_transformed = np.zeros([np.shape(self.B)[1], 1])

    def non_linearity_03(self, x):
        return self.gamma * x[4, 0]

    def non_linearity_12(self, x):
        return -self.gamma * x[4, 0]

    def non_linearity_23(self, x):
        return -self.p_p * x[4, 0]

    def non_linearity_32(self, x):
        return self.p_p * x[4, 0]

    def non_linearity_40(self, x):
        return -self.epsilon * x[3, 0]

    def non_linearity_41(self, x):
        return self.epsilon * x[2, 0]

    def transform(self):

        self.u_transformed = np.vstack(
            (self.clarke_transformation(self.u[0:3, :]), self.u[-1, :]))

    def update_input(self):

        super(AsynchronousMachine, self).update_input()

        self.transform()

        # self.u_transformed = self.clarke_transformation(self.u)

    # def equation_of_state(self, t, x):
    #
    #     x1 = x[0, 0]
    #     x2 = x[1, 0]
    #     x3 = x[2, 0]
    #     x4 = x[3, 0]
    #     x5 = x[4, 0]
    #
    #     # u_ab = self.clarke_transformation(self.u)
    #
    #     u_alpha = self.u_transformed[0, 0]
    #     u_beta = self.u_transformed[1, 0]
    #
    #     # disx = -alpha * isx + beta * psirx + gamma * wm * psiry + delta * usx;
    #     # disy = -alpha * isy - gamma * wm * psirx + beta * psiry + delta * usy;
    #     # dpsirx = R_r * (L_h / L_r) * isx - (R_r / L_r) * psirx - p_p * wm * psiry;
    #     # dpsiry = R_r * (L_h / L_r) * isy + p_p * wm * psirx - (R_r / L_r) * psiry;
    #     # dwm = -epsilon * psiry * isx + epsilon * psirx * isy - Mz / J;
    #
    #     dx1 = -self.alpha * x1 + self.beta * x3 + self.gamma * x5 * x4 + self.delta * u_alpha
    #     dx2 = -self.alpha * x2 - self.gamma * x5 * x3 + self.beta * x4 + self.delta * u_beta
    #     dx3 = self.R_r * (self.L_h / self.L_r) * x1 - (self.R_r / self.L_r) * x3 - self.p_p * x5 * x4
    #     dx4 = self.R_r * (self.L_h / self.L_r) * x2 + self.p_p * x5 * x3 - (self.R_r / self.L_r) * x4
    #     dx5 = -self.epsilon * x4 * x1 + self.epsilon * x3 * x2 - self.M_z / self.J
    #
    #     dx = np.array([[dx1],
    #                    [dx2],
    #                    [dx3],
    #                    [dx4],
    #                    [dx5]])
    #
    #     return dx

    def equation_of_state(self, t, x):

        self.A.eval(x)

        dx = self.A.matrix @ x + self.B @ self.u_transformed

        return dx

    def update_output(self):

        # self.y = np.array([[1, 0, 0, 0, 0]]) @ self.x

        self.y = np.eye(5) @ self.x
