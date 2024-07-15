import ctypes
import numpy as np


# dim = 3
def position(sim, handle):
    err, ret = sim.getObjectPosition(handle, -1)
    return ret

# dim = 3
def orientation(sim, handle):
    err, ret = sim.getObjectOrientation(handle, -1)
    return ret

# dim = 6
def velocity(sim, handle):
    err, lin, ang = sim.getObjectVelocity(handle)
    return [lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]]

# dim = 3
def linear_velocity(sim, handle):
    err, lin, ang = sim.getObjectVelocity(handle)
    return lin

# dim = 3
def angular_velocity(sim, handle):
    err, lin, ang = sim.getObjectVelocity(handle)
    return ang

# dim = 1
def joint_angle(sim, handle):
    err, ret = sim.getJointPosition(handle)
    return [ret]


# dim = 1
def prox_dist(sim, handle, default=0):
    result, distance, point, handle, normal = sim.readProximitySensor(handle)
    if result == 1:
        return [distance]
    else:
        return [default]

# dim = 1
def prox_inv_dist(sim, handle, max_val=1):
    # Returns a smaller value the further the detected object is away
    result, distance, point, handle, normal = sim.readProximitySensor(handle)
    if result == 1:
        return [max_val - distance]
    else:
        return [0]

# dim = 1
def prox_bool(sim, handle):
    result, distance, point, handle, normal = sim.readProximitySensor(handle)
    return [1] if result == 1 else [0]

