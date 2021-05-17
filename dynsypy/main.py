import matplotlib.pyplot as plt
import numpy as np

from dynsypy import *


# TODO: Připravit testovací příklady pomocí scipy.integrate.odeint (motor)
# TODO: Udělat ze zdrojů systémy


def plotter(t_array, x_array):

    # plt.plot(t_array, x_array[0], 'g', label='$i_L(t)$')
    # plt.grid()

    ax1 = plt.subplot(211)
    plt.plot(t_array, x_array[0, :], 'g', label='$i_L(t)$')
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    ax1.legend(loc='best')
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(t_array, x_array[1, :], 'b', label='$u_C(t)$')
    plt.setp(ax2.get_xticklabels(), fontsize=6)
    ax2.legend(loc='best')
    plt.grid()

    plt.xlabel('t')
    plt.show()


iL0 = 0.0
uC0 = 0.0

t0 = 0
dt0 = 1.5e-5
t_end = 0.5

U0 = 50
R = 5
L = 0.1
C = 0.001

x0 = np.array([[iL0],
               [uC0]])

x01 = np.array([[uC0]])

a00 = -R / L
a01 = -1 / L
a10 = 1 / C
a11 = 0

b1_00 = 1 / L
b1_01 = 1 / L
b1_10 = 0
b1_11 = 0

b2_00 = 1 / L
b2_10 = 0

A = np.array([[a00, a01],
              [a10, a11]])

A4 = np.array([[-1/(R*C)]])

B1 = np.array([[b2_00],
               [b2_10]])

B2 = np.array([[b1_00, b1_01],
               [b1_10, b1_11]])

B4 = np.array([[1]])

C1 = np.array([[0.0, 1.0]])

C2 = np.array([[0, 1],
               [-R, -1]])

C4 = np.array([[1]])

D1 = np.array([[0]])

D2 = np.array([[0, 0]])

D3 = np.array([[0, 0],
               [1, 1]])

D4 = np.array([[0]])

pool = Pool(0.01, t_end, t0)

# trigonometric = TrigonometricFunctions(U0, 100, 0)

sine = Sine(U0, 100, np.pi)

# unit_step = UnitStep(U0)

unit_step = UnitStep(U0)

transient = LinearSystem(A, B2, C2, D3, x0=x0, t0=t0, number_of_inputs=2, number_of_outputs=2, allowed_error=1e-7)
transient.connect(unit_step.output, 0)
transient.connect(sine.output, 1)

# transient = LinearSystem(A4, B4, C4, D4, dt0, x01, t0)
# transient.connect(unit_step.output)

transient.select_output(1)

transient_2 = LinearSystem(A, B1, C1, D1, x0=x0, t0=t0, allowed_error=1e-7)
transient_2.connect(transient.output)

pool.add(transient)
pool.add(transient_2)
# pool.add(sine)

pool.simulate()

# plotter(transient.archive_t, transient.archive_x)
plotter(transient_2.archive_t, transient_2.archive_x)
# plotter(sine.archive_t, sine.archive_x)
