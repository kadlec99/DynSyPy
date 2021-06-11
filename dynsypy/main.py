import matplotlib.pyplot as plt
import numpy as np

from dynsypy import *


# TODO: Připravit testovací příklady pomocí scipy.integrate.odeint (motor)
# TODO: Udělat ze zdrojů systémy


def plotter(t_array, y_array):

    # plt.plot(t_array, y_array[0], 'g', label='$i_L(t)$')
    # plt.grid()

    # ax1 = plt.subplot(211)
    # plt.plot(t_array, y_array[0, :], 'g', label='$i_L(t)$')
    # plt.setp(ax1.get_xticklabels(), fontsize=6)
    # ax1.legend(loc='best')
    # plt.grid()
    #
    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.plot(t_array, y_array[1, :], 'b', label='$u_C(t)$')
    # plt.setp(ax2.get_xticklabels(), fontsize=6)
    # ax2.legend(loc='best')
    # plt.grid()

    ax1 = plt.subplot(211)
    plt.plot(t_array, y_array[0, :],
             'g', label='$i_a(t)\ [$A$]$')
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    ax1.legend(loc='best')
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(t_array, y_array[1, :],
             'b', label='$\omega(t)\ [$rad s$^{-1}]$')
    plt.setp(ax2.get_xticklabels(), fontsize=6)
    ax2.legend(loc='best')
    plt.grid()

    plt.xlabel('$t\ [$s$]$')

    # plt.savefig('motor-grafy-5v.pdf')

    plt.show()


def plotter_log_axis(freq_array, val_array):

    ax1 = plt.subplot(211)
    plt.semilogx(freq_array, val_array[0, :],
                 color="green", label='$A(f)\ [$dB$]$')
    plt.setp(ax1.get_xticklabels(), fontsize=6)
    ax1.legend(loc='best')
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)
    plt.semilogx(freq_array, val_array[1, :],
                 color="blue", label='${\\varphi}(f)\ [$°$]$')
    plt.setp(ax2.get_xticklabels(), fontsize=6)
    ax2.legend(loc='best')
    plt.grid()

    plt.xlabel('$f\ [$Hz$]$')

    plt.savefig('RLC_frekvencni_chky.pdf')

    plt.show()


# t0 = 0
# # dt0 = 1.5e-5
# t_end = 0.15
#
# i_a0 = 0.0
# omega0 = 0.0
#
# x0 = np.array([[i_a0],
#                [omega0]])
#
# u_a = 5
# M_L = 0.1
#
# R_a = 2.7
# L_a = 4e-3
# k_a = 0.105
# J = 1e-4
# B_m = 93e-7
#
# A = np.array([[-R_a / L_a, -k_a / L_a],
#               [k_a / J, -B_m / J]])
#
# B = np.array([[1 / L_a, 0],
#               [0, -1 / J]])
#
# C = np.array([[1, 0],
#               [0, 1]])
#
# D = np.array([[0, 0],
#               [0, 0]])
#
#
# pool = Pool(0.01, t_end, t0)
#
# voltage = UnitStep(u_a)
#
# load_torque = UnitStep(M_L)
#
# DC_motor = LinearSystem(A, B, C, D, t0=t0, x0=x0,
#                         number_of_inputs=2, number_of_outputs=2)
#
# # DC_motor = LinearSystem(A, B, C, D, dt0=dt0, t0=t0, x0=x0,
# #                         number_of_inputs=2, number_of_outputs=2)
# DC_motor.connect(voltage.output, 0)
# DC_motor.connect(load_torque.output, 1)
#
# pool.add(DC_motor)

iL0 = 0.0
uC0 = 0.0

t0 = 0
dt0 = 1.5e-5
t_end = 0.5

U0 = 50
R = 5
L = 0.1
C = 0.001

# R = 10
# L = 0.01
# C = 0.1

x0 = np.array([[iL0],
               [uC0]])

x01 = np.array([[uC0]])

# sériový RLC obvod s jedním zdrojem napětí
# výstupem je uC
A1 = np.array([[-R / L, -1 / L],
               [1 / C, 0]])

B1 = np.array([[1 / L],
               [0]])

C1 = np.array([[0.0, 1.0]])

D1 = np.array([[0]])

# # sériový RLC obvod se dvěma zdroji napětí
# # výstupem je uC
# A2 = np.array([[-R / L, -1 / L],
#                [1 / C, 0]])
#
# B2 = np.array([[1 / L, 1 / L],
#                [0, 0]])
#
# C2 = np.array([[0.0, 1.0]])
#
# D2 = np.array([[0, 0]])
#
# # sériový RLC obvod se dvěma zdroji napětí
# # výstupem je uC a uL
# A3 = np.array([[-R / L, -1 / L],
#                [1 / C, 0]])
#
# B3 = np.array([[1 / L, 1 / L],
#                [0, 0]])
#
# C3 = np.array([[0, 1],
#                [-R, -1]])
#
# D3 = np.array([[0, 0],
#                [1, 1]])
#
# # sériový RC obvod s jedním zdrojem napětí
# # výstupem je uC
# A4 = np.array([[-1/(R*C)]])
#
# B4 = np.array([[1]])
#
# C4 = np.array([[1]])
#
# D4 = np.array([[0]])


# omega_range = np.logspace(1, 1e3, 100)
#
#
# pool = Pool(0.01, t_end, t0)
#
# sine = Sine(U0, 100, np.pi)
#
# unit_step = UnitStep(U0)

# transient = LinearSystem(A3, B3, C3, D3, t0=t0, x0=x0,
#                          number_of_inputs=2, number_of_outputs=2, allowed_error=1e-7)
# transient.connect(unit_step.output, 0)
# transient.connect(sine.output, 1)

# transient = LinearSystem(A4, B4, C4, D4, dt0, t0, x01)
# transient.connect(unit_step.output)

# transient.select_output(1)

# transient_2 = LinearSystem(A1, B1, C1, D1, t0=t0, x0=x0, allowed_error=1e-7)
# transient_2.connect(unit_step.output)

transient_2 = LinearSystem(A1, B1, C1, D1, x0=x0)

transient_2.frequency_analysis()

# pool.add(transient)
# pool.add(transient_2)
# # pool.add(sine)
#
# pool.simulate()

# # plotter(transient.archive_t, transient.archive_x)
# plotter(transient_2.archive_t, transient_2.archive_x)
# # plotter(sine.archive_t, sine.archive_x)

# plotter(DC_motor.archive_t, DC_motor.archive_y)

plotter_log_axis(transient_2.archive_frequency, transient_2.archive_bode)
