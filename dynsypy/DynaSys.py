from abc import ABC, abstractmethod

from scipy import signal

import numpy as np


class System(ABC):

    def __init__(self, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=1, number_of_outputs=1,
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
        self.sources = [self.null_input]
        self.source_output_indexes = []

        self.number_of_inputs = number_of_inputs

        self.u = np.full((number_of_inputs, 1), self.sources[0](0))
        self.x = np.array(x0)
        self.y = np.full((number_of_outputs, 1), 0)

        self.t0 = t0
        self.t = self.t0
        self.dt0 = dt0
        self.dt = dt0
        self.dt_max = dt_max

        self.archive_u = np.array(self.u)
        self.archive_x = np.array(self.x)
        self.archive_y = np.array(self.y)
        self.archive_t = np.array([self.t])

        self.number_of_outputs = number_of_outputs
        self.output_index = 0

        self.allowed_error = allowed_error

        self.last_used_archive_index = 0

    def data_archiving(self):

        self.archive_x = np.append(self.archive_x, self.x, axis=1)
        self.archive_y = np.append(self.archive_y, self.y, axis=1)
        self.archive_u = np.append(self.archive_u, self.u, axis=1)
        self.archive_t = np.append(self.archive_t, [self.t])

    def connect(self, source, position, source_output_indexes=None):
        """

        Parameters
        ----------
        source : function
        position : int
        source_output_indexes : List[int]/Tuple[int]/None
        """

        try:
            self.sources[position] = source
        except IndexError:
            extension_size = ((position + 1) - len(self.sources))
            extension_list = [self.null_input] * extension_size

            self.sources = self.sources + extension_list

            self.sources[position] = source

        try:
            self.source_output_indexes[position] = source_output_indexes
        except IndexError:
            extension_size = ((position + 1) - len(self.source_output_indexes))
            extension_list = [None] * extension_size

            self.source_output_indexes = self.source_output_indexes + extension_list

            self.source_output_indexes[position] = source_output_indexes

    # def connect(self, source, position=0):
    #
    #     try:
    #         self.sources[position] = source
    #     except IndexError:
    #         self.sources.append(source)

    def connect_vector(self, source_vector, position=0):

        for i in range(len(source_vector)):
            try:
                self.sources[position + i] = source_vector[i, 0]
            except IndexError:
                self.sources.append(source_vector[i, 0])

    def step(self, end_of_pool_step):

        pass

        # while self.t < end_of_pool_step:
        #
        #     if self.t + self.dt > end_of_pool_step:
        #         self.dt = end_of_pool_step - self.t
        #
        #     self.t = self.t + self.dt
        #
        #     # if self.t > end_of_pool_step:
        #     #     self.dt = self.t - end_of_pool_step
        #     #     self.t = end_of_pool_step
        #
        #     self.update_input()
        #     self.runge_kutta_step()
        #     # self.runge_kutta_45_step()
        #     # self.runge_kutta_fehlberg_step(end_of_pool_step)
        #     self.update_output()
        #
        #     if self.dt != self.dt0:
        #         self.dt = self.dt0
        #
        #     self.data_archiving()

    def runge_kutta_step(self):
        """integration"""

        k1 = self.equation_of_state(self.t, self.x)
        k2 = self.equation_of_state(self.t + self.dt / 2, self.x + self.dt / 2 * k1)
        k3 = self.equation_of_state(self.t + self.dt / 2, self.x + self.dt / 2 * k2)
        k4 = self.equation_of_state(self.t + self.dt, self.x + self.dt * k3)

        self.x = self.x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def adaptive_step(self, end_of_pool_step):

        # Coefficients used to compute the independent variable argument of f

        a2 = 2.500000000000000e-01  # 1/4
        a3 = 3.750000000000000e-01  # 3/8
        a4 = 9.230769230769231e-01  # 12/13
        a5 = 1.000000000000000e+00  # 1
        a6 = 5.000000000000000e-01  # 1/2

        # Coefficients used to compute the dependent variable argument of f

        b21 = 2.500000000000000e-01  # 1/4
        b31 = 9.375000000000000e-02  # 3/32
        b32 = 2.812500000000000e-01  # 9/32
        b41 = 8.793809740555303e-01  # 1932/2197
        b42 = -3.277196176604461e+00  # -7200/2197
        b43 = 3.320892125625853e+00  # 7296/2197
        b51 = 2.032407407407407e+00  # 439/216
        b52 = -8.000000000000000e+00  # -8
        b53 = 7.173489278752436e+00  # 3680/513
        b54 = -2.058966861598441e-01  # -845/4104
        b61 = -2.962962962962963e-01  # -8/27
        b62 = 2.000000000000000e+00  # 2
        b63 = -1.381676413255361e+00  # -3544/2565
        b64 = 4.529727095516569e-01  # 1859/4104
        b65 = -2.750000000000000e-01  # -11/40

        # Coefficients used to compute local truncation error estimate.  These
        # come from subtracting a 4th order RK estimate from a 5th order RK
        # estimate.

        r1 = 2.777777777777778e-03  # 1/360
        r3 = -2.994152046783626e-02  # -128/4275
        r4 = -2.919989367357789e-02  # -2197/75240
        r5 = 2.000000000000000e-02  # 1/50
        r6 = 3.636363636363636e-02  # 2/55

        # Coefficients used to compute 4th order RK estimate

        c1 = 1.157407407407407e-01  # 25/216
        c3 = 5.489278752436647e-01  # 1408/2565
        c4 = 5.353313840155945e-01  # 2197/4104
        c5 = -2.000000000000000e-01  # -1/5

        # Set t and x according to initial condition and assume that h starts
        # with a value that is as large as possible.

        h_max = self.dt_max
        # h_min = 1e-3

        h = self.dt

        while self.t < end_of_pool_step:

            # Adjust step size when we get to last interval

            if self.t + h > end_of_pool_step:
                h = end_of_pool_step - self.t

            self.t = self.t + h

            self.update_input()

            self.t = self.t - h

            # # Adjust step size when we get to last interval
            #
            # if self.t + h > end_of_pool_step:
            #     h = end_of_pool_step - self.t

            # Compute values needed to compute truncation error estimate and
            # the 4th order RK estimate.

            k1 = h * self.equation_of_state(self.t,
                                            self.x)
            k2 = h * self.equation_of_state(self.t + a2 * h,
                                            self.x + b21 * k1)
            k3 = h * self.equation_of_state(self.t + a3 * h,
                                            self.x + b31 * k1 + b32 * k2)
            k4 = h * self.equation_of_state(self.t + a4 * h,
                                            self.x + b41 * k1 + b42 * k2 + b43 * k3)
            k5 = h * self.equation_of_state(self.t + a5 * h,
                                            self.x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
            k6 = h * self.equation_of_state(self.t + a6 * h,
                                            self.x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)

            # Compute the estimate of the local truncation error.  If it's small
            # enough then we accept this step and save the 4th order estimate.

            r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
            if len(np.shape(r)) > 0:
                r = np.amax(r)
            if r <= self.allowed_error:
                self.t = self.t + h
                self.x = self.x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
                self.dt = h

                self.update_output()

                self.data_archiving()

            # Now compute next step size, and make sure that it is not too big or too small.

            if r == 0:
                h = h_max
            else:
                h = h * min(max(0.84 * (self.allowed_error / r) ** 0.25, 0.1), 4.0)

            if h > h_max:
                h = h_max
            # elif h < h_min:
            #     print("Error: step-size should be smaller than %e." % h_min)
            #     break

    def update_input(self):

        source_t = np.zeros([self.number_of_inputs, 1])

        i = 0

        for j in range(0, len(self.sources)):
            aux = self.sources[j](self.t)

            if i >= self.number_of_inputs:
                raise ValueError('Too many source outputs.')

            match aux:
                case int() | float() as aux:
                    source_t[i][0] = aux
                    i += 1
                case np.ndarray() as aux:
                    if self.source_output_indexes[j] is None:
                        for value in aux:
                            source_t[i][0] = value
                            i += 1
                    else:
                        for index in self.source_output_indexes[j]:
                            source_t[i][0] = aux[index]
                            i += 1
                case _:
                    raise ValueError('Wrong input data type.')

        self.u = source_t

        # source_t = []
        #
        # for i in range(len(self.sources)):
        #     source_t.append(self.sources[i](self.t))
        #
        # self.u = np.transpose(np.array([source_t]))

    def equation_of_state(self, t, x):
        """

        Returns
        -------
        numpy.ndarray/float
        """
        pass

    # def runge_kutta_45_step(self):
    #     """Fourth-order Runge-Kutta method with error estimate.
    #
    #     USAGE:
    #         x, err = rk45(f, x0, t)
    #
    #     INPUT:
    #         f     - function of x and t equal to dx/dt.  x may be multivalued,
    #                 in which case it should a list or a NumPy array.  In this
    #                 case f must return a NumPy array with the same dimension
    #                 as x.
    #         x0    - the initial condition(s).  Specifies the value of x when
    #                 t = t[0].  Can be either a scalar or a list or NumPy array
    #                 if a system of equations is being solved.
    #         t     - list or NumPy array of t values to compute solution at.
    #                 t[0] is the the initial condition point, and the difference
    #                 h=t[i+1]-t[i] determines the step size h.
    #
    #     OUTPUT:
    #         x     - NumPy array containing solution values corresponding to each
    #                 entry in t array.  If a system is being solved, x will be
    #                 an array of arrays.
    #         err   - NumPy array containing estimate of errors at each step.  If
    #                 a system is being solved, err will be an array of arrays.
    #
    #     NOTES:
    #         This version is based on the algorithm presented in "Numerical
    #         Mathematics and Computing" 6th Edition, by Cheney and Kincaid,
    #         Brooks-Cole, 2008.
    #     """
    #
    #     # Coefficients used to compute the independent variable argument of f
    #
    #     c20 = 2.500000000000000e-01  # 1/4
    #     c30 = 3.750000000000000e-01  # 3/8
    #     c40 = 9.230769230769231e-01  # 12/13
    #     c50 = 1.000000000000000e+00  # 1
    #     c60 = 5.000000000000000e-01  # 1/2
    #
    #     # Coefficients used to compute the dependent variable argument of f
    #
    #     c21 = 2.500000000000000e-01  # 1/4
    #     c31 = 9.375000000000000e-02  # 3/32
    #     c32 = 2.812500000000000e-01  # 9/32
    #     c41 = 8.793809740555303e-01  # 1932/2197
    #     c42 = -3.277196176604461e+00  # -7200/2197
    #     c43 = 3.320892125625853e+00  # 7296/2197
    #     c51 = 2.032407407407407e+00  # 439/216
    #     c52 = -8.000000000000000e+00  # -8
    #     c53 = 7.173489278752436e+00  # 3680/513
    #     c54 = -2.058966861598441e-01  # -845/4104
    #     c61 = -2.962962962962963e-01  # -8/27
    #     c62 = 2.000000000000000e+00  # 2
    #     c63 = -1.381676413255361e+00  # -3544/2565
    #     c64 = 4.529727095516569e-01  # 1859/4104
    #     c65 = -2.750000000000000e-01  # -11/40
    #
    #     # Coefficients used to compute 4th order RK estimate
    #
    #     a1 = 1.157407407407407e-01  # 25/216
    #     a2 = 0.000000000000000e-00  # 0
    #     a3 = 5.489278752436647e-01  # 1408/2565
    #     a4 = 5.353313840155945e-01  # 2197/4104
    #     a5 = -2.000000000000000e-01  # -1/5
    #
    #     b1 = 1.185185185185185e-01  # 16.0/135.0
    #     b2 = 0.000000000000000e-00  # 0
    #     b3 = 5.189863547758284e-01  # 6656.0/12825.0
    #     b4 = 5.061314903420167e-01  # 28561.0/56430.0
    #     b5 = -1.800000000000000e-01  # -9.0/50.0
    #     b6 = 3.636363636363636e-02  # 2.0/55.0
    #
    #     dt = self.dt
    #
    #     while True:
    #         k1 = dt * self.f(self.t,
    #                          self.x)
    #
    #         k2 = dt * self.f(self.t + c20 * dt,
    #                          self.x + c21 * k1)
    #
    #         k3 = dt * self.f(self.t + c30 * dt,
    #                          self.x + c31 * k1 + c32 * k2)
    #
    #         k4 = dt * self.f(self.t + c40 * dt,
    #                          self.x + c41 * k1 + c42 * k2 + c43 * k3)
    #
    #         k5 = dt * self.f(self.t + c50 * dt,
    #                          self.x + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4)
    #
    #         k6 = dt * self.f(self.t + c60 * dt,
    #                          self.x + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5)
    #
    #         x4 = self.x + a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5
    #         x5 = self.x + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
    #
    #         self.error = abs(x5 - x4)
    #
    #         if self.error[0] > self.allowed_error:
    #             dt = dt / 2
    #         elif any(self.error < 1e-3 * self.allowed_error):
    #             dt = 1.1 * dt
    #         else:
    #             self.dt = dt
    #             self.x = x4
    #             break

    # def runge_kutta_fehlberg_step(self, end_of_pool_step):
    #     """Runge-Kutta-Fehlberg method to solve x' = f(x,t) with x(t[0]) = x0.
    #
    #     USAGE:
    #         t, x = rkf(f, a, b, x0, tol, h_max, h_min)
    #
    #     INPUT:
    #         f     - function equal to dx/dt = f(x,t)
    #         a     - left-hand endpoint of interval (initial condition is here)
    #         b     - right-hand endpoint of interval
    #         x0    - initial x value: x0 = x(a)
    #         tol   - maximum value of local truncation error estimate
    #         h_max  - maximum step size
    #         h_min  - minimum step size
    #
    #     OUTPUT:
    #         t     - NumPy array of independent variable values
    #         x     - NumPy array of corresponding solution function values
    #
    #     NOTES:
    #         This function implements 4th-5th order Runge-Kutta-Fehlberg Method
    #         to solve the initial value problem
    #
    #            dx
    #            -- = f(x,t),     x(a) = x0
    #            dt
    #
    #         on the interval [a,b].
    #
    #         Based on pseudocode presented in "Numerical Analysis", 6th Edition,
    #         by Burden and Faires, Brooks-Cole, 1997.
    #     """
    #
    #     # Coefficients used to compute the independent variable argument of f
    #
    #     a2 = 2.500000000000000e-01  # 1/4
    #     a3 = 3.750000000000000e-01  # 3/8
    #     a4 = 9.230769230769231e-01  # 12/13
    #     a5 = 1.000000000000000e+00  # 1
    #     a6 = 5.000000000000000e-01  # 1/2
    #
    #     # Coefficients used to compute the dependent variable argument of f
    #
    #     b21 = 2.500000000000000e-01  # 1/4
    #     b31 = 9.375000000000000e-02  # 3/32
    #     b32 = 2.812500000000000e-01  # 9/32
    #     b41 = 8.793809740555303e-01  # 1932/2197
    #     b42 = -3.277196176604461e+00  # -7200/2197
    #     b43 = 3.320892125625853e+00  # 7296/2197
    #     b51 = 2.032407407407407e+00  # 439/216
    #     b52 = -8.000000000000000e+00  # -8
    #     b53 = 7.173489278752436e+00  # 3680/513
    #     b54 = -2.058966861598441e-01  # -845/4104
    #     b61 = -2.962962962962963e-01  # -8/27
    #     b62 = 2.000000000000000e+00  # 2
    #     b63 = -1.381676413255361e+00  # -3544/2565
    #     b64 = 4.529727095516569e-01  # 1859/4104
    #     b65 = -2.750000000000000e-01  # -11/40
    #
    #     # Coefficients used to compute local truncation error estimate.  These
    #     # come from subtracting a 4th order RK estimate from a 5th order RK
    #     # estimate.
    #
    #     r1 = 2.777777777777778e-03  # 1/360
    #     r3 = -2.994152046783626e-02  # -128/4275
    #     r4 = -2.919989367357789e-02  # -2197/75240
    #     r5 = 2.000000000000000e-02  # 1/50
    #     r6 = 3.636363636363636e-02  # 2/55
    #
    #     # Coefficients used to compute 4th order RK estimate
    #
    #     c1 = 1.157407407407407e-01  # 25/216
    #     c3 = 5.489278752436647e-01  # 1408/2565
    #     c4 = 5.353313840155945e-01  # 2197/4104
    #     c5 = -2.000000000000000e-01  # -1/5
    #
    #     # Set t and x according to initial condition and assume that h starts
    #     # with a value that is as large as possible.
    #
    #     h_max = 1e-2
    #     h_min = 1e-6
    #
    #     t = self.t
    #     x = self.x
    #     h = self.dt
    #
    #     while True:
    #
    #         # self.t = self.t + h
    #
    #         # Adjust step size when we get to last interval
    #
    #         if t + h > end_of_pool_step:
    #             h = end_of_pool_step - t
    #
    #         # Compute values needed to compute truncation error estimate and
    #         # the 4th order RK estimate.
    #
    #         k1 = h * self.f(t, x)
    #         k2 = h * self.f(t + a2 * h, x + b21 * k1)
    #         k3 = h * self.f(t + a3 * h, x + b31 * k1 + b32 * k2)
    #         k4 = h * self.f(t + a4 * h, x + b41 * k1 + b42 * k2 + b43 * k3)
    #         k5 = h * self.f(t + a5 * h, x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    #         k6 = h * self.f(t + a6 * h, x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
    #
    #         # Compute the estimate of the local truncation error.  If it's small
    #         # enough then we accept this step and save the 4th order estimate.
    #
    #         r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
    #         if len(np.shape(r)) > 0:
    #             r = np.amax(r)
    #         if r <= self.allowed_error:
    #             # t = t + h
    #             x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    #             self.dt = h
    #             self.t = t
    #             self.x = x
    #             break
    #
    #         # Now compute next step size, and make sure that it is not too big or too small.
    #
    #         h = h * min(max(0.84 * (self.allowed_error / r) ** 0.25, 0.1), 4.0)
    #
    #         if h > h_max:
    #             h = h_max
    #         elif h < h_min:
    #             print("Error: step-size should be smaller than %e." % h_min)
    #             break

    def select_output(self, index):
        if self.output_index >= self.number_of_outputs:
            raise ValueError('Out of range of output list!!!'
                             f' The number of outputs of this system is {self.number_of_outputs}.')

        self.output_index = index

    @abstractmethod
    def update_output(self):

        pass

    # def output(self, t):
    #
    #     index = np.searchsorted(self.archive_t, t)
    #
    #     if index >= len(self.archive_t):
    #         self.last_used_archive_index = index - 2
    #         return self.linear_regression(t, index - 1)
    #     else:
    #         if self.archive_t[index] == t:
    #             self.last_used_archive_index = index
    #             return self.archive_y[:, self.last_used_archive_index]
    #         elif self.archive_t[index] > t:
    #             self.last_used_archive_index = index - 1
    #             return self.linear_regression(t, index)
    #         else:
    #             self.last_used_archive_index = index - 2
    #             return self.linear_regression(t, index - 1)
    #
    #     # for i in range(self.last_used_archive_index, len(self.archive_t)):
    #     #
    #     #     if self.archive_t[i] == t:
    #     #         self.last_used_archive_index = i
    #     #         return self.archive_y[self.output_index][i]
    #     #
    #     #     if self.archive_t[i] > t:
    #     #         self.last_used_archive_index = i - 1
    #     #         return self.linear_regression(t, i)
    #     #
    #     # self.last_used_archive_index = len(self.archive_t) - 2
    #     # return self.linear_regression(t, len(self.archive_t) - 1)

    def output(self, t):
        """

        Parameters
        ----------
        t: float

        Returns
        -------
        numpy.ndarray/float
        """

        if np.shape(self.archive_t)[0] == 1:

            return self.archive_y[:, [0]]

        else:

            index = np.searchsorted(self.archive_t, t)

            if index >= len(self.archive_t):
                self.last_used_archive_index = len(self.archive_t) - 2
                return self.output_linear_regression(t, len(self.archive_t) - 1)
            else:
                if self.archive_t[index] == t:
                    self.last_used_archive_index = index
                    return self.archive_y[:, [self.last_used_archive_index]]
                elif self.archive_t[index] > t:
                    self.last_used_archive_index = index - 1
                    return self.output_linear_regression(t, index)
                else:
                    self.last_used_archive_index = index - 2
                    return self.output_linear_regression(t, index - 1)

        # for i in range(self.last_used_archive_index, len(self.archive_t)):
        #
        #     if self.archive_t[i] == t:
        #         self.last_used_archive_index = i
        #         return self.archive_y[self.output_index][i]
        #
        #     if self.archive_t[i] > t:
        #         self.last_used_archive_index = i - 1
        #         return self.linear_regression(t, i)
        #
        # self.last_used_archive_index = len(self.archive_t) - 2
        # return self.linear_regression(t, len(self.archive_t) - 1)

    # def linear_regression(self, t, i):
    #
    #     output_vector = np.zeros(self.number_of_outputs)
    #
    #     system_matrix = np.array([[self.archive_t[i - 1], 1],
    #                               [self.archive_t[i], 1]])
    #
    #     for output_index in range(0, self.number_of_outputs):
    #         vector_of_right_sides = np.array([[self.archive_y[output_index][i - 1]],
    #                                           [self.archive_y[output_index][i]]])
    #
    #         vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
    #
    #         y = vector_x[0] * t + vector_x[1]
    #
    #         output_vector[output_index] = y[0]
    #
    #     return output_vector
    #
    #     # system_matrix = np.array([[self.archive_t[i - 1], 1],
    #     #                           [self.archive_t[i], 1]])
    #     #
    #     # vector_of_right_sides = np.array([[self.archive_y[self.output_index][i - 1]],
    #     #                                   [self.archive_y[self.output_index][i]]])
    #     #
    #     # vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
    #     #
    #     # y = vector_x[0] * t + vector_x[1]
    #     #
    #     # return y[0]

    def linear_regression(self, t, i, data, number_of_items):

        result_vector = np.zeros([number_of_items, 1])

        system_matrix = np.array([[self.archive_t[i - 1], 1],
                                  [self.archive_t[i], 1]])

        for result_index in range(0, number_of_items):

            vector_of_right_sides = np.array([[data[result_index][i - 1]],
                                              [data[result_index][i]]])

            vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides

            y = vector_x[0] * t + vector_x[1]

            result_vector[result_index] = y[0]

        return result_vector

        # system_matrix = np.array([[self.archive_t[i - 1], 1],
        #                           [self.archive_t[i], 1]])
        #
        # vector_of_right_sides = np.array([[self.archive_y[self.result_index][i - 1]],
        #                                   [self.archive_y[self.result_index][i]]])
        #
        # vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
        #
        # y = vector_x[0] * t + vector_x[1]
        #
        # return y[0]

    def output_linear_regression(self, t, i):

        return self.linear_regression(t, i, self.archive_y, self.number_of_outputs)

        # output_vector = np.zeros(self.number_of_outputs)
        #
        # system_matrix = np.array([[self.archive_t[i - 1], 1],
        #                           [self.archive_t[i], 1]])
        #
        # for output_index in range(0, self.number_of_outputs):
        #
        #     vector_of_right_sides = np.array([[self.archive_y[output_index][i - 1]],
        #                                       [self.archive_y[output_index][i]]])
        #
        #     vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
        #
        #     y = vector_x[0] * t + vector_x[1]
        #
        #     output_vector[output_index] = y[0]
        #
        # return output_vector

        # system_matrix = np.array([[self.archive_t[i - 1], 1],
        #                           [self.archive_t[i], 1]])
        #
        # vector_of_right_sides = np.array([[self.archive_y[self.output_index][i - 1]],
        #                                   [self.archive_y[self.output_index][i]]])
        #
        # vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
        #
        # y = vector_x[0] * t + vector_x[1]
        #
        # return y[0]

    def input_linear_regression(self, t, i):

        return self.linear_regression(t, i, self.archive_u, self.number_of_inputs)

        # input_vector = np.zeros(self.number_of_inputs)
        #
        # system_matrix = np.array([[self.archive_t[i - 1], 1],
        #                           [self.archive_t[i], 1]])
        #
        # for input_index in range(0, self.number_of_inputs):
        #
        #     vector_of_right_sides = np.array([[self.archive_u[input_index][i - 1]],
        #                                       [self.archive_u[input_index][i]]])
        #
        #     vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
        #
        #     u = vector_x[0] * t + vector_x[1]
        #
        #     input_vector[input_index] = u[0]
        #
        # return input_vector

        # system_matrix = np.array([[self.archive_t[i - 1], 1],
        #                           [self.archive_t[i], 1]])
        #
        # vector_of_right_sides = np.array([[self.archive_u[self.input_index][i - 1]],
        #                                   [self.archive_u[self.input_index][i]]])
        #
        # vector_x = np.linalg.inv(system_matrix) @ vector_of_right_sides
        #
        # u = vector_x[0] * t + vector_x[1]
        #
        # return u[0]

    @staticmethod
    def null_input(t):

        return 0.0

    @property
    def cls(self):
        return type(self).__name__


# ----------------------------------------------------------------------------


class StateSpaceSystem(System, ABC):

    def __init__(self, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=1, number_of_outputs=1,
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

    def step(self, end_of_pool_step):

        while self.t < end_of_pool_step:

            if self.t + self.dt > end_of_pool_step:
                self.dt = end_of_pool_step - self.t

            self.t = self.t + self.dt

            # if self.t > end_of_pool_step:
            #     self.dt = self.t - end_of_pool_step
            #     self.t = end_of_pool_step

            self.update_input()
            self.runge_kutta_step()
            # self.runge_kutta_45_step()
            # self.runge_kutta_fehlberg_step(end_of_pool_step)
            self.update_output()

            if self.dt != self.dt0:
                self.dt = self.dt0

            self.data_archiving()


# ----------------------------------------------------------------------------


class NonStateSpaceSystem(System, ABC):

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

    @abstractmethod
    def update_state(self):

        pass


# ----------------------------------------------------------------------------


class LinearSystem(StateSpaceSystem):

    def __init__(self, A, B, C, D,
                 dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=1, number_of_outputs=1,
                 allowed_error=1e-6, dt_max=1e-2):
        """

        Parameters
        ----------
        A: numpy.ndarray
        B: numpy.ndarray
        C: numpy.ndarray
        D: numpy.ndarray
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

        self.matrix_controls((A, B, C, D))

        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

        self.archive_frequency = np.array(0)
        self.archive_bode = np.array([[]])

        # self.dt_estimate()

        self.y = self.C @ self.x + self.D @ self.u

    # def dt_estimate(self):
    #     lambda_array = np.linalg.eigvals(self.A)
    #
    #     if any(np.iscomplex(lambda_array)):
    #         tau = -1 / np.real(lambda_array[0])
    #     else:
    #         if np.isscalar(np.argmax(lambda_array)):
    #             tau = -1 / lambda_array[np.argmax(lambda_array)]
    #         else:
    #             tau = -1 / lambda_array[np.argmax(lambda_array)[0]]
    #
    #     self.t_end = tau * 8
    #     self.dt = self.t_end / 100

    def equation_of_state(self, t, x):

        dx = self.A @ x + self.B @ self.u

        return dx

    # def runge_kutta_step(self):
    #     """integration"""
    #     k1 = self.f(self.t, self.x)
    #     k2 = self.f(self.t + self.dt / 2, self.x + self.dt / 2 * k1)
    #     k3 = self.f(self.t + self.dt / 2, self.x + self.dt / 2 * k2)
    #     k4 = self.f(self.t + self.dt, self.x + self.dt * k3)
    #
    #     self.x = self.x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # def runge_kutta_45_step(self):
    #     """Fourth-order Runge-Kutta method with error estimate.
    #
    #     USAGE:
    #         x, err = rk45(f, x0, t)
    #
    #     INPUT:
    #         f     - function of x and t equal to dx/dt.  x may be multivalued,
    #                 in which case it should a list or a NumPy array.  In this
    #                 case f must return a NumPy array with the same dimension
    #                 as x.
    #         x0    - the initial condition(s).  Specifies the value of x when
    #                 t = t[0].  Can be either a scalar or a list or NumPy array
    #                 if a system of equations is being solved.
    #         t     - list or NumPy array of t values to compute solution at.
    #                 t[0] is the the initial condition point, and the difference
    #                 h=t[i+1]-t[i] determines the step size h.
    #
    #     OUTPUT:
    #         x     - NumPy array containing solution values corresponding to each
    #                 entry in t array.  If a system is being solved, x will be
    #                 an array of arrays.
    #         err   - NumPy array containing estimate of errors at each step.  If
    #                 a system is being solved, err will be an array of arrays.
    #
    #     NOTES:
    #         This version is based on the algorithm presented in "Numerical
    #         Mathematics and Computing" 6th Edition, by Cheney and Kincaid,
    #         Brooks-Cole, 2008.
    #     """
    #
    #     # Coefficients used to compute the independent variable argument of f
    #
    #     c20 = 2.500000000000000e-01  # 1/4
    #     c30 = 3.750000000000000e-01  # 3/8
    #     c40 = 9.230769230769231e-01  # 12/13
    #     c50 = 1.000000000000000e+00  # 1
    #     c60 = 5.000000000000000e-01  # 1/2
    #
    #     # Coefficients used to compute the dependent variable argument of f
    #
    #     c21 = 2.500000000000000e-01  # 1/4
    #     c31 = 9.375000000000000e-02  # 3/32
    #     c32 = 2.812500000000000e-01  # 9/32
    #     c41 = 8.793809740555303e-01  # 1932/2197
    #     c42 = -3.277196176604461e+00  # -7200/2197
    #     c43 = 3.320892125625853e+00  # 7296/2197
    #     c51 = 2.032407407407407e+00  # 439/216
    #     c52 = -8.000000000000000e+00  # -8
    #     c53 = 7.173489278752436e+00  # 3680/513
    #     c54 = -2.058966861598441e-01  # -845/4104
    #     c61 = -2.962962962962963e-01  # -8/27
    #     c62 = 2.000000000000000e+00  # 2
    #     c63 = -1.381676413255361e+00  # -3544/2565
    #     c64 = 4.529727095516569e-01  # 1859/4104
    #     c65 = -2.750000000000000e-01  # -11/40
    #
    #     # Coefficients used to compute 4th order RK estimate
    #
    #     a1 = 1.157407407407407e-01  # 25/216
    #     a2 = 0.000000000000000e-00  # 0
    #     a3 = 5.489278752436647e-01  # 1408/2565
    #     a4 = 5.353313840155945e-01  # 2197/4104
    #     a5 = -2.000000000000000e-01  # -1/5
    #
    #     b1 = 1.185185185185185e-01  # 16.0/135.0
    #     b2 = 0.000000000000000e-00  # 0
    #     b3 = 5.189863547758284e-01  # 6656.0/12825.0
    #     b4 = 5.061314903420167e-01  # 28561.0/56430.0
    #     b5 = -1.800000000000000e-01  # -9.0/50.0
    #     b6 = 3.636363636363636e-02  # 2.0/55.0
    #
    #     dt = self.dt
    #
    #     while True:
    #         k1 = dt * self.f(self.t,
    #                          self.x)
    #
    #         k2 = dt * self.f(self.t + c20 * dt,
    #                          self.x + c21 * k1)
    #
    #         k3 = dt * self.f(self.t + c30 * dt,
    #                          self.x + c31 * k1 + c32 * k2)
    #
    #         k4 = dt * self.f(self.t + c40 * dt,
    #                          self.x + c41 * k1 + c42 * k2 + c43 * k3)
    #
    #         k5 = dt * self.f(self.t + c50 * dt,
    #                          self.x + c51 * k1 + c52 * k2 + c53 * k3 + c54 * k4)
    #
    #         k6 = dt * self.f(self.t + c60 * dt,
    #                          self.x + c61 * k1 + c62 * k2 + c63 * k3 + c64 * k4 + c65 * k5)
    #
    #         x4 = self.x + a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5
    #         x5 = self.x + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6
    #
    #         self.error = abs(x5 - x4)
    #
    #         if self.error[0] > self.allowed_error:
    #             dt = dt / 2
    #         elif any(self.error < 1e-3 * self.allowed_error):
    #             dt = 1.1 * dt
    #         else:
    #             self.dt = dt
    #             self.x = x4
    #             break

    # def runge_kutta_fehlberg_step(self):
    #     """Runge-Kutta-Fehlberg method to solve x' = f(x,t) with x(t[0]) = x0.
    #
    #     USAGE:
    #         t, x = rkf(f, a, b, x0, tol, h_max, h_min)
    #
    #     INPUT:
    #         f     - function equal to dx/dt = f(x,t)
    #         a     - left-hand endpoint of interval (initial condition is here)
    #         b     - right-hand endpoint of interval
    #         x0    - initial x value: x0 = x(a)
    #         tol   - maximum value of local truncation error estimate
    #         h_max  - maximum step size
    #         h_min  - minimum step size
    #
    #     OUTPUT:
    #         t     - NumPy array of independent variable values
    #         x     - NumPy array of corresponding solution function values
    #
    #     NOTES:
    #         This function implements 4th-5th order Runge-Kutta-Fehlberg Method
    #         to solve the initial value problem
    #
    #            dx
    #            -- = f(x,t),     x(a) = x0
    #            dt
    #
    #         on the interval [a,b].
    #
    #         Based on pseudocode presented in "Numerical Analysis", 6th Edition,
    #         by Burden and Faires, Brooks-Cole, 1997.
    #     """
    #
    #     # Coefficients used to compute the independent variable argument of f
    #
    #     a2 = 2.500000000000000e-01  # 1/4
    #     a3 = 3.750000000000000e-01  # 3/8
    #     a4 = 9.230769230769231e-01  # 12/13
    #     a5 = 1.000000000000000e+00  # 1
    #     a6 = 5.000000000000000e-01  # 1/2
    #
    #     # Coefficients used to compute the dependent variable argument of f
    #
    #     b21 = 2.500000000000000e-01  # 1/4
    #     b31 = 9.375000000000000e-02  # 3/32
    #     b32 = 2.812500000000000e-01  # 9/32
    #     b41 = 8.793809740555303e-01  # 1932/2197
    #     b42 = -3.277196176604461e+00  # -7200/2197
    #     b43 = 3.320892125625853e+00  # 7296/2197
    #     b51 = 2.032407407407407e+00  # 439/216
    #     b52 = -8.000000000000000e+00  # -8
    #     b53 = 7.173489278752436e+00  # 3680/513
    #     b54 = -2.058966861598441e-01  # -845/4104
    #     b61 = -2.962962962962963e-01  # -8/27
    #     b62 = 2.000000000000000e+00  # 2
    #     b63 = -1.381676413255361e+00  # -3544/2565
    #     b64 = 4.529727095516569e-01  # 1859/4104
    #     b65 = -2.750000000000000e-01  # -11/40
    #
    #     # Coefficients used to compute local truncation error estimate.  These
    #     # come from subtracting a 4th order RK estimate from a 5th order RK
    #     # estimate.
    #
    #     r1 = 2.777777777777778e-03  # 1/360
    #     r3 = -2.994152046783626e-02  # -128/4275
    #     r4 = -2.919989367357789e-02  # -2197/75240
    #     r5 = 2.000000000000000e-02  # 1/50
    #     r6 = 3.636363636363636e-02  # 2/55
    #
    #     # Coefficients used to compute 4th order RK estimate
    #
    #     c1 = 1.157407407407407e-01  # 25/216
    #     c3 = 5.489278752436647e-01  # 1408/2565
    #     c4 = 5.353313840155945e-01  # 2197/4104
    #     c5 = -2.000000000000000e-01  # -1/5
    #
    #     # Set t and x according to initial condition and assume that h starts
    #     # with a value that is as large as possible.
    #
    #     h_max = 0.1
    #     h_min = 1e-6
    #
    #     t = self.t
    #     x = self.x
    #     h = h_max
    #
    #     while True:
    #
    #         # self.t = self.t + h
    #
    #         # Adjust step size when we get to last interval
    #
    #         if t + h > Pool.t_end:
    #             h = Pool.t_end - t
    #
    #         # Compute values needed to compute truncation error estimate and
    #         # the 4th order RK estimate.
    #
    #         k1 = h * self.f(t, x)
    #         k2 = h * self.f(t + a2 * h, x + b21 * k1)
    #         k3 = h * self.f(t + a3 * h, x + b31 * k1 + b32 * k2)
    #         k4 = h * self.f(t + a4 * h, x + b41 * k1 + b42 * k2 + b43 * k3)
    #         k5 = h * self.f(t + a5 * h, x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    #         k6 = h * self.f(t + a6 * h, x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
    #
    #         # Compute the estimate of the local truncation error.  If it's small
    #         # enough then we accept this step and save the 4th order estimate.
    #
    #         r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
    #         if len(np.shape(r)) > 0:
    #             r = np.amax(r)
    #         if r <= self.allowed_error:
    #             t = t + h
    #             x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    #             self.dt = h
    #             self.t = t
    #             self.x = x
    #             break
    #
    #         # Now compute next step size, and make sure that it is not too big or too small.
    #
    #         h = h * min(max(0.84 * (self.allowed_error / r) ** 0.25, 0.1), 4.0)
    #
    #         if h > h_max:
    #             h = h_max
    #         elif h < h_min:
    #             print("Error: step-size should be smaller than %e." % h_min)
    #             break

    def update_output(self):

        self.y = self.C @ self.x + self.D @ self.u

    @staticmethod
    def value_control(matrix):
        """checks if there are numbers in the matrices"""

        for x in matrix:
            for y in x:
                z = float(y)

    def matrix_shape_control(self, matrices):
        """
        NOTES:
            the function performs the following checks:

            number of rows of matrix A == number of rows of matrix x
            number of columns of matrix A == number of rows of matrix x
            number of rows of matrix B == number of rows of matrix x
            number of columns of matrix B == number of rows of matrix u
            number of rows of matrix C == number of rows of matrix y
            number of columns of matrix C == number of rows of matrix x
            number of rows of matrix D == number of rows of matrix y
            number of columns of matrix D == number of rows of matrix u
        """

        if np.shape(matrices[0])[0] == np.shape(self.x)[0] \
                and np.shape(matrices[0])[1] == np.shape(self.x)[0] \
                and np.shape(matrices[1])[0] == np.shape(self.x)[0] \
                and np.shape(matrices[1])[1] == np.shape(self.u)[0] \
                and np.shape(matrices[2])[0] == np.shape(self.y)[0] \
                and np.shape(matrices[2])[1] == np.shape(self.x)[0] \
                and np.shape(matrices[3])[0] == np.shape(self.y)[0] \
                and np.shape(matrices[3])[1] == np.shape(self.u)[0]:
            pass
        else:
            raise ValueError("Bad matrix dimensions!!!")

    def matrix_controls(self, matrices):
        """calls the function for checking the values in the matrices
         and the function for checking the shape of the matrices"""

        for x in matrices:
            self.value_control(x)

        self.matrix_shape_control(matrices)

    def frequency_analysis(self, omega_range=None, n=100):

        system = signal.StateSpace(self.A, self.B, self.C, self.D)

        frequency_analysis_results = signal.bode(system, omega_range, n)

        self.archive_frequency = frequency_analysis_results[0] / (2 * np.pi)
        self.archive_bode = np.vstack((frequency_analysis_results[1],
                                       frequency_analysis_results[2]))


# ----------------------------------------------------------------------------


class PartiallyNonLinearSystem(StateSpaceSystem, ABC):

    def __init__(self, dt0=1.5e-5, t0=0, x0=0,
                 number_of_inputs=1, number_of_outputs=1,
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


# def f1(x):
#     return x
#
#
# def f2(x):
#     return x ** 2


class Matrix:

    def __init__(self, matrix, non_linearity_functions, non_linearity_indexes):
        self.matrix = matrix                        # np.zeros([2, 2])
        self.functions = non_linearity_functions    # [f1, f2]
        self.indexes = non_linearity_indexes        # [[0, 0], [1, 1]]

    def eval(self, x):
        i = 0
        for index in self.indexes:
            self.matrix[index[0], index[1]] = self.functions[i](x)
            i += 1


# x = 2
# A = Matrix()
# A.eval(x)
#
# print(A.A)
