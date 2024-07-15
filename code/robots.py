import numpy as np
import ctypes
import math
import nengo
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def b(num):
  """ forces magnitude to be 1 or less """
  if abs(num) > 1.0:
    return math.copysign(1.0, num)
  else:
    return num


def convert_angles(ang):
  """ Converts Euler angles from x-y-z to z-x-y convention """
  s1 = math.sin(ang[0])
  s2 = math.sin(ang[1])
  s3 = math.sin(ang[2])
  c1 = math.cos(ang[0])
  c2 = math.cos(ang[1])
  c3 = math.cos(ang[2])
  
  pitch = math.asin( b(c1*c3*s2-s1*s3) )
  cp = math.cos(pitch)
  # just in case
  if cp == 0:
    cp = 0.000001

  yaw = math.asin( b((c1*s3+c3*s1*s2)/cp) ) #flipped
  # Fix for getting the quadrants right
  if c3 < 0 and yaw > 0:
    yaw = math.pi - yaw
  elif c3 < 0 and yaw < 0:
    yaw = -math.pi - yaw
  
  roll = math.asin( b((c3*s1+c1*s2*s3)/cp) ) #flipped
  return [roll, pitch, yaw]


class Robot(object):
    def __init__(self, sim_dt=0.05, nengo_dt=0.001, sync=True):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        self.client.setStepping(True)
        self.sim.stopSimulation()  # just in case sim was running previously
        self.sim.startSimulation()
        
        self.count = 0
        self.sim_dt = sim_dt
        self.nengo_dt = nengo_dt

    def handle_input(self, values):
        raise NotImplemented

    def handle_output(self):
        raise NotImplemented

    def __call__(self, t, values):
        self.count += 1
        if self.count == int(round(self.sim_dt/self.nengo_dt)):
            self.count = 0
            self.handle_input(values)
            self.client.step()
        return self.handle_output()


class CustomRobot(Robot):
    """
    Sensors and actuators may be added to this component after it is created
    """

    def __init__(self, sim_dt=0.01, nengo_dt=0.001, sync=True):
        super(CustomRobot, self).__init__(sim_dt, nengo_dt, sync)
        self.sensors = []
        self.actuators = []
        self.size_in = 0

        self.body = None

        self.size_out = 4
        
        # Store the output here so it doesn't need to be generated for each
        # Nengo timestep, only for V-REP timesteps
        self.output = [0,0,0,0]

    def handle_input(self, values):

        count = 0
        for handle, func, dim in self.actuators:
            func(self.sim, handle, values[count:count+dim])
            count += dim

        self.generate_output()

    def generate_output(self):
        ret = []
        for handle, func in self.sensors:
          tmp = func(self.sim, handle)
          for i in tmp:
            ret.append(i)

        ret_pos = [0,0,0,0]
        if self.body is not None:
            pos = self.sim.getObjectPosition(self.body, -1)
            ori = self.sim.getObjectOrientation(self.body, -1)
            for i in range(3):
                ret_pos[i] = pos[i]
            ret_pos[3] = ori[2]
        ret.extend(ret_pos)

            
        self.output = np.array(ret).flatten()
        return np.array(ret).flatten()
  ### ----------
  
  
  
  
    def handle_output(self):
        return self.output


    def add_body(self, name):
        assert name is not None, 'Adding a body with no name?  What are you doing here?'
        self.body = self.sim.getObject(name)

    def add_sensor(self, name, func, dim=1):
        if name is None:
            handle = None
        else:
            handle = self.sim.getObject(name)
        self.sensors.append([handle, func])
        
        # This is needed so Nengo doesn't error on the first timestep
        self.output.extend([0]*dim)
        self.size_out += dim
    
    def add_actuator(self, name, func, dim=1):
        if name is None:
            handle = None
        else:
            handle = self.sim.getObject(name)
        self.actuators.append([handle, func, dim])
        self.size_in += 1 # dim

    def build_node(self):
        return nengo.Node(self, size_in=self.size_in, size_out=self.size_out)


# Note: Quadcopter class not fully converted to new API yet
class Quadcopter(Robot):
    """
    This callable class will return the state of the quadcopter relative to its
    target whenever it is called. It will also accept motor commands which will be
    sent to the quadcopter in V-REP.
    """
    def __init__(self, sim_dt=0.01, max_target_distance=3, noise=False,
                 noise_std=[0,0,0,0,0,0],
                 target_func=None,
                ):
        raise NotImplementedError("Quadcopter class has not been fully converted to the new API yet")
        super(Quadcopter, self).__init__(sim_dt)

        self.copter = self.sim.getObject("/Quadricopter_base")
        self.target = self.sim.getObject("/Quadricopter_target")

        # Reset the motor commands to zero
        packedData = self.sim.packFloatTable([0,0,0,0])
        raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 

        err = self.sim.setStringSignal("rotorTargetVelocities", raw_bytes)

        self.pos = [0,0,0]
        self.pos_err = [0,0,0]
        self.t_pos = [0,0,0]
        self.lin = [0,0,0]
        self.ori = [0,0,0]
        self.ori_err = [0,0,0]
        self.t_ori = [0,0,0]
        self.ang = [0,0,0]

        self.vert_prox_dist = 0
        self.left_prox_dist = 0
        self.right_prox_dist = 0
        
        # Distance reading recorded when nothing is in range
        self.max_vert_dist = 1.5
        self.max_left_dist = 1.0
        self.max_right_dist = 1.0
        
        # Maximum target distance error that can be returned
        self.max_target_distance = max_target_distance
 
        # If noise is being modelled
        self.noise = noise

        # Standard Deviation of the noise for the 4 state variables
        self.noise_std = noise_std
        
        # Overwrite the get_target method if the target is to be controlled by a
        # function instead of by V-REP
        if target_func is not None:
          
          self.step = 0
          self.target_func = target_func

          def get_target():
            self.t_pos, self.t_ori = self.target_func(self.step)
            self.step += 1

          self.get_target = get_target

    def reset(self):
        self.sim.stopSimulation()
        time.sleep(1)
        self.pos_err = [0,0,0]
        self.ori_err = [0,0,0]
        self.lin = [0,0,0]
        self.ang = [0,0,0]
        self.vert_prox = 0
        self.left_prox = 0
        self.right_prox = 0
        self.sim.startSimulation()
    
    def exit(self):
        exit(1)

    def get_target(self):
        err, self.t_ori = self.sim.getObjectOrientation(self.target, -1)
        err, self.t_pos = self.sim.getObjectPosition(self.target, -1)
        
        # Convert orientations to z-y-x convention
        self.t_ori = convert_angles(self.t_ori)

    def calculate_error(self):
        # Return the state variables
        err, self.ori = self.sim.getObjectOrientation(self.copter, -1)
        err, self.pos = self.sim.getObjectPosition(self.copter, -1)
        err, self.lin, self.ang = self.sim.getObjectVelocity(self.copter)
        
        self.ori = convert_angles(self.ori)
        
        # Apply noise to each measurement if required
        if self.noise:
          self.pos += np.random.normal(0,self.noise_std[0],3)
          self.lin += np.random.normal(0,self.noise_std[1],3)
          self.ori += np.random.normal(0,self.noise_std[2],3)
          self.ang += np.random.normal(0,self.noise_std[3],3)
          #TODO: might have to wrap angles here
        
        # Find the error
        self.ori_err = [self.t_ori[0] - self.ori[0], 
                        self.t_ori[1] - self.ori[1],
                        self.t_ori[2] - self.ori[2]]
        cz = math.cos(self.ori[2])
        sz = math.sin(self.ori[2])
        x_err = self.t_pos[0] - self.pos[0]
        y_err = self.t_pos[1] - self.pos[1]
        self.pos_err = [ x_err * cz + y_err * sz, 
                        -x_err * sz + y_err * cz, 
                         self.t_pos[2] - self.pos[2]]
        
        self.lin = [self.lin[0]*cz+self.lin[1]*sz, -self.lin[0]*sz+self.lin[1]*cz, self.lin[2]]
        self.ang = [self.ang[0]*cz+self.ang[1]*sz, -self.ang[0]*sz+self.ang[1]*cz, self.ang[2]]

        for i in range(3):
          if self.ori_err[i] > math.pi:
            self.ori_err[i] -= 2 * math.pi
          elif self.ori_err[i] < -math.pi:
            self.ori_err[i] += 2 * math.pi

    def send_motor_commands(self, values):

        motor_values = np.zeros(4)
        for i in range(4):
          motor_values[i] = values[i]
        packedData = self.sim.packFloatTable(motor_values.flatten())
        raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 
        err = self.sim.setStringSignal("rotorTargetVelocities", raw_bytes)
    
    def handle_input(self, values):
        
        # Send motor commands to V-REP
        self.send_motor_commands(values)

        # Retrieve target location
        self.get_target()

        # Calculate state error
        self.calculate_error()

    def bound(self, value):
        if abs(value) > self.max_target_distance:
          return math.copysign( self.max_target_distance, value )
        else:
          return value

    def handle_output(self):
        l = math.sqrt(self.pos_err[0]**2 + self.pos_err[1]**2)
        bl = self.bound(l)
        r = (bl+.1)/(l+.1)

        return [r*self.pos_err[0], r*self.pos_err[1], self.bound(self.pos_err[2]), 
                self.lin[0], self.lin[1], self.lin[2], 
                self.ori_err[0], self.ori_err[1], self.ori_err[2], 
                self.ang[0], self.ang[1], self.ang[2],
               ]

class SensorQuadcopter(Quadcopter):
    
    def __init__( self, *args, **kwargs ):

        super(SensorQuadcopter, self).__init__(*args, **kwargs)
        
        self.vert_prox = self.sim.getObject("/vert_prox")
        self.left_prox = self.sim.getObject("/left_prox")
        self.right_prox = self.sim.getObject("/right_prox")

    def read_proximity(self):

        err, state, point, handle, normal = self.sim.readProximitySensor(self.vert_prox)
        if state:
          self.vert_prox_dist = point[2]
        else:
          self.vert_prox_dist = self.max_vert_dist

        err, state, point, handle, normal =\
            self.sim.readProximitySensor(self.left_prox)
        if state:
          self.left_prox_dist = point[2]
        else:
          self.left_prox_dist = self.max_left_dist
        
        err, state, point, handle, normal =\
            self.sim.readProximitySensor(self.right_prox)
        if state:
          self.right_prox_dist = point[2]
        else:
          self.right_prox_dist = self.max_right_dist
    
    def handle_input(self, values):
        
        # Send motor commands to V-REP
        self.send_motor_commands(values)

        # Retrieve target location
        self.get_target()

        # Calculate state error
        self.calculate_error()

        # Get proximity sensor readings
        self.read_proximity()

    def handle_output(self):
        l = math.sqrt(self.pos_err[0]**2 + self.pos_err[1]**2)
        bl = self.bound(l)
        r = (bl+.1)/(l+.1)

        return [r*self.pos_err[0], r*self.pos_err[1], self.bound(self.pos_err[2]), 
                self.lin[0], self.lin[1], self.lin[2], 
                self.ori_err[0], self.ori_err[1], self.ori_err[2], 
                self.ang[0], self.ang[1], self.ang[2],
                self.vert_prox_dist, self.left_prox_dist, self.right_prox_dist,
               ]

class TargetQuadcopter(Quadcopter):
    
    """ Returns target position as well """

    def __init__(self, *args, **kwargs):

        super(TargetQuadcopter, self).__init__(*args, **kwargs)
    
    def handle_output(self):
        l = math.sqrt(self.pos_err[0]**2 + self.pos_err[1]**2)
        bl = self.bound(l)
        r = (bl+.1)/(l+.1)

        return [r*self.pos_err[0], r*self.pos_err[1], self.bound(self.pos_err[2]), 
                self.lin[0], self.lin[1], self.lin[2], 
                self.ori_err[0], self.ori_err[1], self.ori_err[2], 
                self.ang[0], self.ang[1], self.ang[2],
                self.t_pos[0], self.t_pos[1], self.t_pos[2],
                self.t_ori[0], self.t_ori[1], self.t_ori[2],
               ]

class WaypointQuadcopter(Quadcopter):

    """ Takes the desired target as an input rather than moving to the green circle """

    def __init__(self, sim_dt=0.01, max_target_distance=3, noise=False,
                 noise_std=[0,0,0,0,0,0],
                ):
        
        # Call the superclass of Quadcopter, which is Robot
        super(WaypointQuadcopter, self).__init__(sim_dt)

        self.copter = self.sim.getObject("Quadricopter_base")

        # Reset the motor commands to zero
        packedData = self.sim.packFloatTable([0,0,0,0])
        raw_bytes = (ctypes.c_ubyte * len(packedData)).from_buffer_copy(packedData) 

        err = self.sim.setStringSignal("rotorTargetVelocities", raw_bytes)

        self.pos = [0,0,0]
        self.pos_err = [0,0,0]
        self.t_pos = [0,0,0]
        self.lin = [0,0,0]
        self.ori = [0,0,0]
        self.ori_err = [0,0,0]
        self.t_ori = [0,0,0]
        self.ang = [0,0,0]
        
        # Maximum target distance error that can be returned
        self.max_target_distance = max_target_distance
 
        # If noise is being modelled
        self.noise = noise

        # Standard Deviation of the noise for the 4 state variables
        self.noise_std = noise_std

    def handle_input(self, values):
        
        # Send motor commands to V-REP
        self.send_motor_commands(values[:4])

        # Retrieve target location
        self.t_pos = values[[4,5,6]] 
        self.t_ori = values[[7,8,9]]

        # Calculate state error
        self.calculate_error()
