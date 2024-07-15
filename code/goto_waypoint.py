# filename: GOTO Waypoint
#
# author: Brent Komer, Michael Furlong


import datetime

import nengo
import numpy as np
from robots import CustomRobot
from sensors import position, orientation, prox_dist
from actuators import joint_velocity
from functools import partial

import matplotlib.pyplot as plt

from steering_model import skid_steer

model = nengo.Network(label="Pioneer p3dx")

def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


with model:
    pioneer = CustomRobot(sim_dt=0.05, nengo_dt=0.001, sync=True)

    # myobj = Object()
    # obj_input = nengo.Node(function, size_in=....)
    # nengo.Connection(motor, obj_input)
    # obj_output = nengo.Node(function, size_in=....)
    #

    pioneer.add_body('/Pioneer_p3dx')
    pioneer.add_actuator("/Pioneer_p3dx_leftMotor", joint_velocity)
    pioneer.add_actuator("/Pioneer_p3dx_rightMotor", joint_velocity)


    for i in range(1, 9):
        pioneer.add_sensor(f"/Pioneer_p3dx_ultrasonicSensor{i}", partial(prox_dist))

    robot = pioneer.build_node()
    motor = nengo.Ensemble(n_neurons=100, dimensions=2, radius=3)
    sensors = nengo.Ensemble(n_neurons=400, dimensions=8)

    speed = nengo.Node([10])

    nengo.Connection(motor, robot)
    nengo.Connection(robot[:8], sensors)

    goal = np.array([-2, 1, 0.25])
    last_position = np.array([0, 0, 0])
    last_time_checked = 0
    stuck_count = 0  # Counter for stuck occurrences
#     print("0")


    def navigate(t, x):
        global last_position, last_time_checked, stuck_count
#         current_position = CustomRobot.generate_output
        current_position = x[8:]
        pos_error = goal[:2] - current_position[:2]
        yaw_error = wrap_angle(goal[2] - current_position[3])


        angle_to_goal = wrap_angle(np.arctan2(pos_error[1], pos_error[0]) - current_position[3])

        print('angle_to_goal: ', angle_to_goal)
        print('goal error', pos_error, yaw_error)

        speed_sign = 1
        if np.abs(angle_to_goal) > np.pi / 2:
            speed_sign = -1

        speed = speed_sign * np.linalg.norm(pos_error)

        cmd = 0.1*skid_steer(speed, -yaw_error).flatten()
        print('command: ', cmd)
        return cmd     


    navigator = nengo.Node(navigate, size_in=12, size_out=2)
    nengo.Connection(robot, navigator)
    nengo.Connection(navigator, motor)

    p_motor = nengo.Probe(motor, synapse=0.01)


sim = nengo.Simulator(model, progress_bar=False)
with sim:
    sim.run(10)

plt.figure()
plt.plot(sim.trange(), sim.data[p_motor])


uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
np.save(f'../data/{uniq_filename}',sim.data[p_motor])


sim_data_motor = np.load('my_file.npy')
