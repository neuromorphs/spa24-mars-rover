import ctypes


# dim = 3
def position(sim, handle, val):
    err = sim.setObjectPosition(handle, -1, val)

# dim = 3
def orientation(sim, handle, val):
    err = sim.setObjectOrientation(handle, -1, val)

# dim = 1
def joint_torque(sim, handle, val):
    sim.setJointTargetVelocity(handle, val*100)
    sim.setJointForce(handle, abs(val))

# dim = 1
def joint_velocity(sim, handle, val):
    sim.setJointTargetVelocity(handle, float(val))

# dim = 4
def quadcopter_rotors(sim, handle, val):
    # This function does not use handle, it just exists in the signature
    # to make it consistent
    motor_values = np.zeros(4)
    for i in range(4):
      motor_values[i] = val[i]
    packedData = sim.packFloatTable(motor_values.flatten())
    raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 
    err = simx.setStringSignal("rotorTargetVelocities", raw_bytes)

