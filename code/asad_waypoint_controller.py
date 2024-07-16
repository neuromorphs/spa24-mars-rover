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

#     goal = np.array([-2, 1, 0.25])

    goal = nengo.Node([2,-1,0.25])
    ik_gain = nengo.Node([1])

    current_pos = nengo.Ensemble(n_neurons = 2000, dimensions=3, radius=10)
    nengo.Connection(robot[[8,9,11]],current_pos)

    pose_err = nengo.Ensemble(n_neurons = 4000, dimensions=4, radius=10)
    nengo.Connection(goal, pose_err[:3])
    nengo.Connection(current_pos, pose_err[:3], transform=-1)
    nengo.Connection(current_pos[2], pose_err[3])


    def speed_comp(x):
        err_x, err_y, err_w, curr_w = x
        angle_to_goal = wrap_angle(
                np.arctan2(err_y, err_x) - curr_w 
        )
        speed_sign = 1
        if np.abs(angle_to_goal) > np.pi / 2:
            speed_sign = -1

        speed = speed_sign * np.sqrt(err_x * err_x + err_y * err_y)
        return speed, -speed_sign*err_w 

    command = nengo.Ensemble(n_neurons = 2000, dimensions=2)
    nengo.Connection(pose_err, command , function=speed_comp)

    def inverse_kinematics(t, x):
        return x[2]*skid_steer(x[0], x[1]).flatten()

    ik_node = nengo.Node(inverse_kinematics, size_in=3, size_out=2)
    nengo.Connection(command, ik_node[:2])
    nengo.Connection(ik_gain, ik_node[2])

    nengo.Connection(ik_node, motor)

    p_motor = nengo.Probe(motor, synapse=0.01)
    p_err = nengo.Probe(pose_err, synapse=0.01)
    p_cmd = nengo.Probe(command, synapse=0.01)


sim = nengo.Simulator(model, progress_bar=False)
with sim:
    sim.run(10)

plt.figure()
plt.plot(sim.trange(), sim.data[p_motor])


plt.figure()
plt.subplot(2,1,1)
plt.plot(sim.trange(), sim.data[p_err], label='error')
plt.legend()

plt.subplot(2,1,2)
plt.plot(sim.trange(), sim.data[p_cmd], label='cmd')
plt.legend()

plt.show()


uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

save_file = f'../data/{uniq_filename}'
np.save(save_file,sim.data[p_motor])


# sim_data_motor = np.load(save_file)
