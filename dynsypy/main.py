import matplotlib.pyplot as plt
import numpy as np

from dynsypy import *


# TODO: alokace pameti pred vypoctem
# TODO: rovnice v dq
# TODO: integrace v extra modulu (třída nebo funkce)


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


plt.rcParams['text.usetex'] = True

t0 = 0
# dt0 = 1.5e-5
t_end = 0.3

x0 = np.array([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])

f_s_n = 50
U_s_n = 380

motor_params = {
    "R_s": 1.617,
    "R_r": 1.609,
    "L_s_sigma": 8.5e-3,
    "L_r_sigma": 8.5e-3,
    "L_h": 134.4e-3,
    "p_p": 2,
    "N_n": 1420,
    "U_s_n": U_s_n,  # 3x380
    "f_s_n": f_s_n,
    "I_s_n": 8.5,
    "J": 0.03,
    "k_p": 1.5
}

f = f_s_n
U = np.sqrt(2) * U_s_n  # 150
number_of_phases = 3

source_params = {
    "amplitude": U,
    "frequency": f,
    "number_of_phases": number_of_phases,
    "phase": 0
}

PI_controller_params = {
    "K": 2 / 3,
    "T_i": 0.05,
    "saturation_value": 2 * np.pi * 5
}

############################
# direct source connection #
############################

amplitude = UnitStep(step_time=0, initial_value=0.0, final_value=U)
frequency = UnitStep(step_time=0, initial_value=0.0, final_value=f)
load_torque = UnitStep(step_time=0, initial_value=0.0, final_value=0)

source_3_f = ControlledNPhaseSine(source_params)

motor = AsynchronousMachine(motor_params,
                            x0=x0, number_of_inputs=4, dt0=5e-5)

source_3_f.connect(amplitude.output, 0)
source_3_f.connect(frequency.output, 1)

motor.connect(source_3_f.output, 0)
motor.connect(load_torque.output, 1)

pool = Pool(1e-2, t_end, t0, True)

pool.add(amplitude)
pool.add(frequency)

pool.add(load_torque)

pool.add(source_3_f)

pool.add(motor)

pool.simulate()

# fig, axs = plt.subplots(2)
# axs[0].plot(motor.archive_t, motor.archive_y[0, :],
#             motor.archive_t, motor.archive_y[1, :],
#             motor.archive_t, motor.archive_y[2, :])
# axs[0].grid(which='major')
# axs[0].grid(which='minor', linestyle=':', linewidth=0.5)
# axs[0].minorticks_on()
#
# axs[1].plot(motor.archive_t, motor.archive_y[3, :])
# axs[1].grid(which='major')
# axs[1].grid(which='minor', linestyle=':', linewidth=0.5)
# axs[1].minorticks_on()
#
# plt.xlabel('$t\ (\\mathrm{s})$')

plt.plot(motor.archive_t, motor.archive_u[0, :],
         motor.archive_t, motor.archive_u[1, :],
         motor.archive_t, motor.archive_u[2, :])
plt.grid(which='major')
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim([0, 0.3])
# plt.ylim([-100, 120])
plt.xlabel('$t\ (\mathrm{s})$')
plt.ylabel('$u_{\mathrm{s}}\ (\mathrm{V})$')
plt.savefig('DynSyPy_ASM_direct_u_s.pdf')

plt.clf()

plt.plot(motor.archive_t, motor.archive_y[0, :],
         motor.archive_t, motor.archive_y[1, :],
         motor.archive_t, motor.archive_y[2, :])
plt.grid(which='major')
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim([0, 0.3])
plt.ylim([-100, 120])
plt.xlabel('$t\ (\mathrm{s})$')
plt.ylabel('$i_{\mathrm{s}}\ (\mathrm{A})$')
plt.savefig('DynSyPy_ASM_direct_i_s.pdf')

plt.clf()

plt.plot(motor.archive_t, motor.archive_y[3, :])
plt.grid(which='major')
plt.grid(which='minor', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlim([0, 0.3])
plt.ylim([0, 180])
plt.xlabel('$t\ (\mathrm{s})$')
plt.ylabel('$\omega_{\mathrm{m}}\ (\mathrm{rad}\cdot\mathrm{s}^{-1})$')
plt.savefig('DynSyPy_ASM_direct_omega_m.pdf')

#######################
# scalar control loop #
#######################

# required_speed = UnitStep(step_time=0.5, initial_value=2 * np.pi * 20, final_value=2 * np.pi * 10)
# load_torque = UnitStep(step_time=0.3, initial_value=25, final_value=70)
#
# # required_speed = UnitStep(step_time=0.8, initial_value=100, final_value=170)
# # load_torque = UnitStep(step_time=0.5, initial_value=35, final_value=70)
# # load_torque = UnitStep(step_time=0, initial_value=0.0, final_value=0)
#
# controller = PIController(PI_controller_params)
# scalar_control = ASMScalarControl(motor_params)
#
# source_3_f = ControlledNPhaseSine(source_params)
#
# motor = AsynchronousMachine(motor_params,
#                             x0=x0, number_of_inputs=4, dt0=5e-5)
#
# controller.connect(required_speed.output, 0)
# controller.connect(motor.output, 1, [3])
#
# scalar_control.connect(controller.output, 0)
# scalar_control.connect(motor.output, 1, [3])
#
# source_3_f.connect(scalar_control.output, 0)
#
# motor.connect(source_3_f.output, 0)
# motor.connect(load_torque.output, 1)
#
# # pool = Pool(1e-4, t_end, t0, True)
# pool = Pool(1.3e-4, t_end, t0, False)
#
# pool.add(required_speed)
#
# pool.add(load_torque)
# pool.add(controller)
# pool.add(scalar_control)
#
# pool.add(source_3_f)
#
# pool.add(motor)
#
# pool.simulate()
#
# plt.plot(controller.archive_t, controller.archive_y[0, :])
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([0, t_end])
# # plt.ylim([0, 180])
# plt.xlabel('$t\ (\mathrm{s})$')
# plt.ylabel('$\omega_{\mathrm{rw}}\ (\mathrm{rad}\cdot\mathrm{s}^{-1})$')
# plt.savefig('DynSyPy_ASM_reg_omega_rw.pdf')
#
# plt.clf()
#
# # plt.plot(scalar_control.archive_t, scalar_control.archive_y[0, :], label='$U_{\mathrm{sw}}$')
# # plt.plot(scalar_control.archive_t, scalar_control.archive_y[1, :], label='$\omega_{\mathrm{m}}$')
# # plt.grid(which='major')
# # plt.grid(which='minor', linestyle=':', linewidth=0.5)
# # plt.minorticks_on()
# # plt.xlim([0, t_end])
# # # plt.ylim([0, 180])
# # plt.xlabel('$t\ (\mathrm{s})$')
# # plt.ylabel('$\omega\ (\mathrm{rad}\cdot\mathrm{s}^{-1})$')
# # plt.legend(loc='best')
# # plt.savefig('DynSyPy_ASM_reg_omega_m.pdf')
#
# # plt.clf()
#
# plt.plot(motor.archive_t, motor.archive_u[0, :],
#          motor.archive_t, motor.archive_u[1, :],
#          motor.archive_t, motor.archive_u[2, :])
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([0, t_end])
# # plt.ylim([-100, 120])
# plt.xlabel('$t\ (\mathrm{s})$')
# plt.ylabel('$u_{\mathrm{s}}\ (\mathrm{V})$')
# plt.savefig('DynSyPy_ASM_reg_u_s.pdf')
#
# plt.clf()
#
# plt.plot(motor.archive_t, motor.archive_y[0, :],
#          motor.archive_t, motor.archive_y[1, :],
#          motor.archive_t, motor.archive_y[2, :])
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([0, t_end])
# # plt.ylim([-100, 120])
# plt.xlabel('$t\ (\mathrm{s})$')
# plt.ylabel('$i_{\mathrm{s}}\ (\mathrm{A})$')
# plt.savefig('DynSyPy_ASM_reg_i_s.pdf')
#
# plt.clf()
#
# plt.plot(required_speed.archive_t, required_speed.archive_y, label='$\omega_{\mathrm{mw}}$')
# plt.plot(motor.archive_t, motor.archive_y[3, :], label='$\omega_{\mathrm{m}}$')
# plt.grid(which='major')
# plt.grid(which='minor', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.xlim([0, t_end])
# # plt.ylim([0, 180])
# plt.xlabel('$t\ (\mathrm{s})$')
# plt.ylabel('$\omega\ (\mathrm{rad}\cdot\mathrm{s}^{-1})$')
# plt.legend(loc='best')
# plt.savefig('DynSyPy_ASM_reg_omega_m.pdf')

# plt.plot(motor.archive_t, motor.archive_u[0, :],
#          motor.archive_t, motor.archive_u[1, :],
#          motor.archive_t, motor.archive_u[2, :])

# fig, axs = plt.subplots(3)
# axs[0].plot(motor.archive_t, motor.archive_y[0, :],
#             motor.archive_t, motor.archive_y[1, :])
# axs[0].grid()
# axs[1].plot(motor.archive_t, motor.archive_y[2, :],
#             motor.archive_t, motor.archive_y[3, :])
# axs[1].grid()
# axs[2].plot(motor.archive_t, motor.archive_y[4, :])
# axs[2].grid()

# fig, axs = plt.subplots(2)
# axs[0].plot(motor.archive_t, motor.archive_y[0, :],
#             motor.archive_t, motor.archive_y[1, :],
#             motor.archive_t, motor.archive_y[2, :])
# axs[0].grid()
# axs[1].plot(motor.archive_t, motor.archive_y[3, :],
#             required_speed.archive_t, required_speed.archive_y)
# axs[1].grid()

# fig, axs = plt.subplots(3)
# axs[0].plot(motor.archive_t, motor.archive_x[0, :],
#             motor.archive_t, motor.archive_x[1, :])
# axs[0].grid()
# axs[1].plot(motor.archive_t, motor.archive_x[2, :],
#             motor.archive_t, motor.archive_x[3, :])
# axs[1].grid()
# axs[2].plot(motor.archive_t, motor.archive_x[-1, :])
# axs[2].grid()

# plt.plot(motor.archive_t, motor.archive_x[0, :],
#          motor.archive_t, motor.archive_x[1, :])

# plt.plot(amplitude.archive_t, amplitude.archive_x,
#          frequency.archive_t, frequency.archive_x,
#          phase_U.archive_t, phase_U.archive_x[0, :])

# for i in range(0, number_of_phases):
#     plt.plot(source.archive_t, source.archive_y[i, :])
# plt.grid()

# plt.savefig('asm.pdf')

plt.show()
