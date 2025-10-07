import time
import numpy as np
import json

import mujoco
import mujoco.viewer
import pickle

from simple_pid import PID

def save_data(filename, positions, velocities):
    data = {'positions': positions, 'velocities': velocities}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

def outer_pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

class PIDController:
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output

class dummyPlanner:
  """Generate Path from 1 point directly to another"""

  def __init__(self, target, vel_limit = 2, waypoint_tolerance = 0.3) -> None:
    # TODO: MPC
    self.target = target  
    self.vel_limit = vel_limit
    self.waypoint_tolerance = waypoint_tolerance
    # setpoint target location, controller output: desired velocity.
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
  
  def __call__(self, loc: np.array):
    """Calls planner at timestep to update cmd_vel"""
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites

  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    """Compute

    Args:
        loc (np.array): Current location in world coordinates.
        target (np.array): Desired location in world coordinates
        time_to_target (float): If set, adpats length of velocity vector.

    Returns:
        np.array: returns velocity vector in world coordinates.
    """

    direction = target - loc
    distance = np.linalg.norm(direction)
    # maps drone velocities to one.
    if distance > 1:
        velocities = flight_speed * direction / distance

    else:
        velocities =  direction * distance

    return velocities

  def get_alt_setpoint(self, loc: np.array) -> float:

    target = self.target
    distance = target[2] - loc[2]
    
    # maps drone velocities to one.
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set

  def update_target(self, target):
    """Update targets"""
    self.target = target  
    # setpoint target location, controller output: desired velocity.
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]
  
  def is_waypoint_reached(self, current_pos):
    """Check if the drone has reached the current waypoint within tolerance"""
    distance = np.linalg.norm(self.target - current_pos)
    return distance < self.waypoint_tolerance

class dummySensor:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc

  def get_position(self):
    return self.position
  
  def get_velocity(self):
    return self.velocity
  
  def get_acceleration(self):
    return self.acceleration

class drone:
  """Simple drone classe."""
  def __init__(self, waypoints=None):
    self.m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
    self.d = mujoco.MjData(self.m)

    # Urban parcours waypoint system with improved obstacle clearance
    if waypoints is None:
      self.waypoints = [
        np.array([0, 0, 1]),           # Start
        np.array([0.5, 0.2, 1.2]),     # Safe path avoiding structure1
        np.array([1.2, 1.0, 1.3]),     # Intermediate - clear of obstacles
        np.array([0.8, 2.0, 1.5]),     # Approaching north, avoiding building1
        np.array([0.3, 3.0, 1.4]),     # North edge
        np.array([-0.8, 3.2, 1.6]),    # North-west transition
        np.array([-1.8, 2.8, 1.9]),    # Around building cluster 2 (higher)
        np.array([-2.8, 1.8, 1.5]),    # West side approach
        np.array([-3.2, 0.8, 1.3]),    # West side
        np.array([-3.0, 0.0, 1.1]),    # West-south transition
        np.array([-2.2, -0.8, 1.2]),   # Avoiding structure2
        np.array([-2.3, -1.8, 1.7]),   # Around building cluster 3 (higher, safer distance)
        np.array([-0.8, -3.2, 1.3]),   # South approach
        np.array([0.6, -3.0, 1.2]),    # South edge (moved for clearance)
        np.array([1.1, -2.7, 1.5]),    # South-east transition (smoothed)
        np.array([1.8, -1.5, 1.4]),    # Around building cluster 4
        np.array([2.8, -0.8, 1.5]),    # East approach
        np.array([3.4, 0.2, 1.7]),     # East side
        np.array([3.0, 1.0, 1.5]),     # East-north turn
        np.array([2.0, 1.2, 1.3]),     # Return path (safer from building1)
        np.array([1.0, 0.5, 1.1]),     # Final approach
      ]
    else:
      self.waypoints = waypoints
    
    self.current_waypoint_index = 0
    self.waypoints_completed = 0
    self.lap_count = 0

    self.planner = dummyPlanner(target=self.waypoints[0], waypoint_tolerance=0.3)
    self.sensor = dummySensor(self.d)
    
    # LIDAR sensor configuration
    self.lidar_sensors = [
      # Horizontal ring
      'lidar_0', 'lidar_45', 'lidar_90', 'lidar_135', 'lidar_180', 'lidar_225', 'lidar_270', 'lidar_315',
      # Upper ring
      'lidar_up_0', 'lidar_up_45', 'lidar_up_90', 'lidar_up_135', 'lidar_up_180', 'lidar_up_225', 'lidar_up_270', 'lidar_up_315',
      # Lower ring
      'lidar_down_0', 'lidar_down_45', 'lidar_down_90', 'lidar_down_135', 'lidar_down_180', 'lidar_down_225', 'lidar_down_270', 'lidar_down_315',
      # Zenith and nadir
      'lidar_zenith', 'lidar_nadir'
    ]
    
    # Get sensor IDs and their data addresses
    self.lidar_sensor_ids = []
    self.lidar_sensor_adr = []
    self.lidar_site_ids = []
    
    for name in self.lidar_sensors:
      sensor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
      self.lidar_sensor_ids.append(sensor_id)
      self.lidar_sensor_adr.append(self.m.sensor_adr[sensor_id])
      self.lidar_site_ids.append(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name))

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0,) # PIDController(0.050844,0.000017871, 0, 0) # thrust
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) ) #PID(11.0791,2.5263, 0.10513,setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

    # outer control loops
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))

  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()
    location = self.sensor.get_position()[:3]

    # Compute velocites to target
    velocites = self.planner(loc=location)
    
    # In this example the altitude is directly controlled by a PID
    self.pid_alt.setpoint = self.planner.get_alt_setpoint(location)
    self.pid_v_x.setpoint = velocites[0]
    self.pid_v_y.setpoint = velocites[1]

    # Compute angles and set inner controllers accordingly
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = - self.pid_v_y(v[1])

    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2]
    angles = self.sensor.get_position()[3:] # roll, yaw, pitch
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])

    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
    self.d.ctrl[:4] = out

  def get_lidar_data(self):
    """Read LIDAR sensor data and return distances and hit points"""
    lidar_data = []
    
    for i, sensor_adr in enumerate(self.lidar_sensor_adr):
      # Get the distance measurement from the correct address in sensordata
      distance = self.d.sensordata[sensor_adr]
      
      # Get site position and orientation in world frame
      site_id = self.lidar_site_ids[i]
      site_pos = self.d.site_xpos[site_id].copy()
      site_mat = self.d.site_xmat[site_id].reshape(3, 3)
      
      # Ray direction is the site's z-axis in world frame
      ray_direction = site_mat[:, 2]
      
      # Calculate hit point
      if distance >= 0 and distance < 5:  # 5m is our cutoff
        hit_point = site_pos + ray_direction * distance
      else:
        hit_point = site_pos + ray_direction * 5  # Max range
        distance = -1  # No hit
      
      lidar_data.append({
        'sensor_name': self.lidar_sensors[i],
        'distance': distance,
        'site_pos': site_pos,
        'ray_direction': ray_direction,
        'hit_point': hit_point,
        'has_hit': distance >= 0 and distance < 5
      })
    
    return lidar_data
  
  def check_and_update_waypoint(self):
    """Check if current waypoint is reached and switch to next one"""
    location = self.sensor.get_position()[:3]
    
    if self.planner.is_waypoint_reached(location):
      # Move to next waypoint
      self.current_waypoint_index += 1
      self.waypoints_completed += 1
      
      # Loop back to start
      if self.current_waypoint_index >= len(self.waypoints):
        self.current_waypoint_index = 0
        self.lap_count += 1
        print(f"🏁 Lap {self.lap_count} completed! Starting new lap...")
      
      # Update to new waypoint
      next_waypoint = self.waypoints[self.current_waypoint_index]
      self.planner.update_target(next_waypoint)
      print(f"✓ Waypoint {self.waypoints_completed} reached! Heading to waypoint {self.current_waypoint_index}: {next_waypoint}")

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control

# -------------------------- Initialization ----------------------------------
my_drone = drone()

print("🚁 Urban Drone Navigation Parcours Demo")
print("=" * 50)
print(f"Total waypoints in parcours: {len(my_drone.waypoints)}")
print(f"LIDAR sensors: {len(my_drone.lidar_sensors)} rangefinders")
print("Starting navigation...\n")

# Configuration: run until a full lap is completed, or fall back to time cap if disabled
RUN_UNTIL_FULL_LAP = True

with mujoco.viewer.launch_passive(my_drone.m, my_drone.d) as viewer:
  # Close the viewer after a full lap if RUN_UNTIL_FULL_LAP is True; otherwise after 60s
  start = time.time()
  step = 1

  while viewer.is_running() and ((RUN_UNTIL_FULL_LAP and my_drone.lap_count < 1) or (not RUN_UNTIL_FULL_LAP and time.time() - start < 60)):
    step_start = time.time()
    
    # Check and update waypoints based on distance
    my_drone.check_and_update_waypoint()

    # outer control loop
    if step % 20 == 0:
     my_drone.update_outer_conrol()
    # Inner control loop
    my_drone.update_inner_control()

    mujoco.mj_step(my_drone.m, my_drone.d)

    # Get and visualize LIDAR data
    lidar_data = my_drone.get_lidar_data()
    
    with viewer.lock():
      # Toggle contact points
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(my_drone.d.time % 2)
      
      # Visualize LIDAR rays using viewer's scene
      viewer.user_scn.ngeom = 0  # Clear previous visualization geoms
      
      for lidar in lidar_data:
        if lidar['has_hit']:
          # Draw line from sensor to hit point (bright green for hit)
          mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 1, 0, 0.6])  # Bright green with transparency
          )
          mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_LINE,
            2.0,  # thicker line width
            lidar['site_pos'],
            lidar['hit_point']
          )
          viewer.user_scn.ngeom += 1
          
          # Draw larger sphere at hit point for better visibility
          if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
            mujoco.mjv_initGeom(
              viewer.user_scn.geoms[viewer.user_scn.ngeom],
              type=mujoco.mjtGeom.mjGEOM_SPHERE,
              size=np.array([0.03, 0, 0]),
              pos=lidar['hit_point'],
              mat=np.eye(3).flatten(),
              rgba=np.array([1, 0.2, 0, 0.8])  # Orange-red hit point, more opaque
            )
            viewer.user_scn.ngeom += 1
        
        # Prevent overflow
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 1:
          break

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()
    
    # Increment to time slower outer control loop
    step += 1
    
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = my_drone.m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0: 
      time.sleep(time_until_next_step)

print(f"\n🏁 Demo completed! Total waypoints reached: {my_drone.waypoints_completed}")
print(f"Laps completed: {my_drone.lap_count}")
