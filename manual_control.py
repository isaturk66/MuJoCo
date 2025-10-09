# manual_control.py
# Keyboard + Gamepad teleoperation for DroneEnv via Gymnasium API.
#
# ANGLE MODE (Stabilized Mode):
# - Stick positions directly map to target angles
# - Center stick = level flight (0° roll/pitch)
# - Releasing controls = returns to level automatically
# - This matches how real drones behave in angle/stabilized mode
#
# Keyboard Controls:
#   Arrow keys: Roll/Pitch angle (hold to tilt, release to auto-level)
#   A/D: Yaw rate (left/right rotation)
#   W/S: Altitude up/down
#   Z: Force level (zero angles immediately)
#   R: Reset episode
#   Esc: Exit
#
# Gamepad Controls (Angle Mode - like a real RC transmitter):
#   Left stick up/down: Altitude target
#   Left stick left/right: Yaw rate
#   Right stick up/down: Pitch angle
#   Right stick left/right: Roll angle
#   Start/Options button: Reset episode
#
# Requirements:
#   pip install pygame

import time
from collections import deque
import numpy as np
import mujoco

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("WARNING: pygame not found. Gamepad support disabled.")
    print("Install with: pip install pygame")

from simulation import DroneEnv

# ========== ANGLE MODE SETTINGS ==========
# These match typical RC transmitter behavior

# Keyboard "virtual stick" settings
KEYBOARD_ANGLE_PER_PRESS = np.deg2rad(15.0)  # Angle when key held (builds up to max)
MAX_KEYBOARD_ANGLE = np.deg2rad(30.0)        # Max angle from keyboard
KEYBOARD_YAW_RATE = np.deg2rad(60.0)         # Yaw rate when A/D pressed
KEYBOARD_ALT_STEP = 0.3                      # Altitude change per keypress
STICK_RETURN_SPEED = 5.0                     # How fast virtual stick returns to center (rad/s)

# Gamepad settings (direct stick-to-angle mapping)
GAMEPAD_DEADZONE = 0.15                      # Ignore small stick movements
GAMEPAD_MAX_ANGLE = np.deg2rad(30.0)         # Max tilt angle at full stick deflection
GAMEPAD_MAX_YAW_RATE = np.deg2rad(90.0)      # Max yaw rate at full stick deflection
GAMEPAD_ALT_RATE = 1.5                       # Altitude change rate from stick
CALIBRATION_TIME = 3.0                       # Seconds to calibrate gamepad center position

# Queue to collect keypress events between frames
_key_events = deque()

def _key_callback(keycode: int):
    """Collect keycodes; viewer calls this on every key press (and repeats when held)."""
    _key_events.append(keycode)

def apply_deadzone(value, center=0.0, deadzone=GAMEPAD_DEADZONE):
    """Apply deadzone to joystick input."""
    # Center the value first
    centered_value = value - center
    
    if abs(centered_value) < deadzone:
        return 0.0
    
    # Rescale so deadzone maps to 0 and full range stays at ±1
    sign = 1.0 if centered_value > 0 else -1.0
    scaled = (abs(centered_value) - deadzone) / (1.0 - deadzone)
    
    # Clamp to [-1, 1]
    return sign * min(1.0, max(0.0, scaled))

def init_gamepad():
    """Initialize pygame and detect gamepad."""
    if not PYGAME_AVAILABLE:
        return None
    
    # Initialize only joystick subsystem, not video (avoid conflicts with MuJoCo viewer)
    # Set SDL to use dummy video driver
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    
    pygame.init()
    pygame.joystick.init()
    
    # Check for connected joysticks
    joystick_count = pygame.joystick.get_count()
    print(f"Detected {joystick_count} joystick(s)")
    
    if joystick_count == 0:
        print("No gamepad detected. Keyboard-only mode.")
        return None
    
    # List all available joysticks
    for i in range(joystick_count):
        temp_joy = pygame.joystick.Joystick(i)
        print(f"  Joystick {i}: {temp_joy.get_name()} - {temp_joy.get_numaxes()} axes, {temp_joy.get_numbuttons()} buttons")
    
    # Use first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"✓ Using gamepad: {joystick.get_name()}")
    print(f"  Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}")
    return joystick

def calibrate_gamepad(joystick, duration=CALIBRATION_TIME):
    """Calibrate gamepad by measuring center positions during idle period."""
    if joystick is None:
        return None
    
    print(f"\n🎮 Calibrating gamepad for {duration} seconds...")
    print("Please keep all sticks centered and don't touch any buttons!")
    
    # DualSense (PS5) controller axis mapping:
    # Axis 0: Left stick X, Axis 1: Left stick Y
    # Axis 2: Right stick X, Axis 3: L2 trigger (-1 to 1)
    # Axis 4: R2 trigger (-1 to 1), Axis 5: Right stick Y
    
    # Collect samples for all axes
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        pygame.event.pump()
        num_axes = joystick.get_numaxes()
        sample = [joystick.get_axis(i) for i in range(num_axes)]
        samples.append(sample)
        time.sleep(0.01)  # 100Hz sampling
    
    if not samples:
        print("❌ No samples collected during calibration!")
        return None
    
    # Calculate center positions
    samples = np.array(samples)
    centers = np.mean(samples, axis=0)
    std_devs = np.std(samples, axis=0)
    
    print(f"✓ Calibration complete! ({joystick.get_numaxes()} axes detected)")
    for i in range(len(centers)):
        print(f"  Axis {i}: center={centers[i]:6.3f}, std_dev={std_devs[i]:.4f}")
    
    return centers

def read_gamepad(joystick, centers=None):
    """
    Read gamepad inputs and return (roll_angle, pitch_angle, yaw_rate, alt_rate, reset_pressed).
    ANGLE MODE: Stick positions directly map to target angles.
    """
    if joystick is None:
        return 0.0, 0.0, 0.0, 0.0, False
    
    # Process pygame events to update joystick state
    pygame.event.pump()
    
    # DualSense (PS5) controller axis mapping (verified):
    # Axis 0: Left stick X (left/right, -1 is left)
    # Axis 1: Left stick Y (up/down, -1 is up)
    # Axis 2: L2 trigger
    # Axis 3: Right stick X (left/right, -1 is left)
    # Axis 4: Right stick Y (up/down, -1 is up)
    # Axis 5: R2 trigger
    
    num_axes = joystick.get_numaxes()
    
    roll_angle = 0.0
    pitch_angle = 0.0
    yaw_rate = 0.0
    alt_rate = 0.0
    
    # Use calibrated centers or default to 0
    if centers is None:
        centers = np.zeros(num_axes)
    
    # Left stick X (axis 0) -> yaw rate
    if num_axes > 0:
        left_x = apply_deadzone(joystick.get_axis(0), centers[0])
        yaw_rate = -left_x * GAMEPAD_MAX_YAW_RATE  # Full stick = max yaw rate
    
    # Left stick Y (axis 1) -> altitude rate
    if num_axes > 1:
        left_y = apply_deadzone(joystick.get_axis(1), centers[1])
        alt_rate = -left_y * GAMEPAD_ALT_RATE  # Full stick = max altitude rate
    
    # Right stick X (axis 3) -> roll angle (ANGLE MODE)
    if num_axes > 3:
        right_x = apply_deadzone(joystick.get_axis(3), centers[3])
        roll_angle = -right_x * GAMEPAD_MAX_ANGLE  # Full stick = max angle
    
    # Right stick Y (axis 4) -> pitch angle (ANGLE MODE)
    if num_axes > 4:
        right_y = apply_deadzone(joystick.get_axis(4), centers[4])
        pitch_angle = -right_y * GAMEPAD_MAX_ANGLE  # Full stick = max angle
    
    # Check for reset button (Options button on DualSense is usually button 9)
    reset_pressed = False
    num_buttons = joystick.get_numbuttons()
    if num_buttons > 9:
        reset_pressed = joystick.get_button(9)
    elif num_buttons > 7:
        reset_pressed = joystick.get_button(7)
    
    return roll_angle, pitch_angle, yaw_rate, alt_rate, reset_pressed

def main():
    # Initialize gamepad FIRST, before creating env or viewer
    # This avoids conflicts with MuJoCo's viewer initialization
    joystick = init_gamepad()
    
    # Calibrate gamepad if available
    gamepad_centers = None
    if joystick is not None:
        gamepad_centers = calibrate_gamepad(joystick)
    
    # Create env with very large max_episode_steps to avoid auto-reset during manual control
    env = DroneEnv(render_mode="human", max_episode_steps=1000000)
    obs, info = env.reset()

    # Launch viewer with our key handler (no need to access raw GLFW window)
    env.viewer = mujoco.viewer.launch_passive(env.m, env.d, key_callback=_key_callback)

    print("\n" + "="*70)
    print("DRONE MANUAL CONTROL - ANGLE MODE (Stabilized)")
    print("="*70)
    print("✈️  This simulates a real drone flight controller in ANGLE MODE:")
    print("   - Sticks control target ANGLES (not rates)")
    print("   - Release sticks = auto-level to 0°")
    print("   - Just like flying a real DJI/Skydio drone in stabilized mode!")
    print("-"*70)
    if joystick:
        print("🎮 Gamepad Controls (RC Transmitter Style):")
        print("  Left stick:  Yaw rate (L/R) | Altitude (U/D)")
        print("  Right stick: Roll angle (L/R) | Pitch angle (U/D)")
        print("  Start button: Reset")
        print()
    print("⌨️  Keyboard Controls:")
    print("  Arrow keys: Roll/Pitch angles (hold to tilt, release to level)")
    print("  A/D: Yaw rate | W/S: Altitude Up/Down")
    print("  Z: Force level | R: Reset | Esc: Quit")
    print("="*70 + "\n")

    dt = 1.0 / env.metadata.get("render_fps", 60)
    running = True
    gamepad_reset_pressed_last = False
    
    # ANGLE MODE: Virtual stick state for keyboard (simulates holding stick at a position)
    keyboard_roll_angle = 0.0    # Current "virtual stick" roll position
    keyboard_pitch_angle = 0.0   # Current "virtual stick" pitch position
    target_altitude = env.init_hover_alt  # Track altitude setpoint
    
    # Track which keys are currently held
    keys_held = set()

    while running and env.viewer.is_running():
        # Start with neutral commands
        roll_angle_cmd = 0.0
        pitch_angle_cmd = 0.0
        yaw_rate_cmd = 0.0
        
        # Read gamepad (direct stick-to-angle mapping in angle mode)
        gp_roll, gp_pitch, gp_yaw, gp_alt_rate, gamepad_reset_pressed = read_gamepad(joystick, gamepad_centers)
        
        if joystick is not None:
            # Gamepad has priority - directly set angles from stick positions
            roll_angle_cmd = gp_roll
            pitch_angle_cmd = gp_pitch
            yaw_rate_cmd = gp_yaw
            target_altitude += gp_alt_rate * dt  # Integrate altitude rate
            target_altitude = max(0.1, min(10.0, target_altitude))  # Clamp altitude
        
        # Handle gamepad reset (edge detection to avoid multiple resets)
        if gamepad_reset_pressed and not gamepad_reset_pressed_last:
            obs, _ = env.reset()
            target_altitude = env.init_hover_alt
            keyboard_roll_angle = 0.0
            keyboard_pitch_angle = 0.0
            print("🔄 Reset by gamepad")
        gamepad_reset_pressed_last = gamepad_reset_pressed

        # Process keyboard events
        while _key_events:
            keycode = _key_events.popleft()

            # GLFW-style keycodes used by mujoco.viewer
            GLFW_KEY_LEFT  = 263
            GLFW_KEY_RIGHT = 262
            GLFW_KEY_UP    = 265
            GLFW_KEY_DOWN  = 264
            GLFW_KEY_ESC   = 256
            try:
                ch = chr(keycode)
            except Exception:
                ch = ''

            # Track key presses for keyboard "stick" simulation
            if keycode == GLFW_KEY_LEFT:
                keys_held.add('left')
            elif keycode == GLFW_KEY_RIGHT:
                keys_held.add('right')
            elif keycode == GLFW_KEY_UP:
                keys_held.add('up')
            elif keycode == GLFW_KEY_DOWN:
                keys_held.add('down')
            elif keycode == GLFW_KEY_ESC:
                running = False
            elif ch in ('w','W'):
                keys_held.add('w')
            elif ch in ('s','S'):
                keys_held.add('s')
            elif ch in ('a','A'):
                keys_held.add('a')
            elif ch in ('d','D'):
                keys_held.add('d')
            elif ch in ('z','Z'):
                # Force level immediately
                keyboard_roll_angle = 0.0
                keyboard_pitch_angle = 0.0
                keys_held.clear()
                print("⚖️  Forced level")
            elif ch in ('r','R'):
                obs, _ = env.reset()
                target_altitude = env.init_hover_alt
                keyboard_roll_angle = 0.0
                keyboard_pitch_angle = 0.0
                keys_held.clear()
                print("🔄 Reset by keyboard")

        # Keyboard: simulate holding a virtual stick at an angle
        # Keys move the virtual stick, which then commands that angle
        if joystick is None:  # Only use keyboard if no gamepad
            # Update keyboard virtual stick position based on held keys
            if 'left' in keys_held:
                keyboard_roll_angle = min(keyboard_roll_angle + KEYBOARD_ANGLE_PER_PRESS * dt, MAX_KEYBOARD_ANGLE)
            if 'right' in keys_held:
                keyboard_roll_angle = max(keyboard_roll_angle - KEYBOARD_ANGLE_PER_PRESS * dt, -MAX_KEYBOARD_ANGLE)
            if 'up' in keys_held:
                keyboard_pitch_angle = max(keyboard_pitch_angle - KEYBOARD_ANGLE_PER_PRESS * dt, -MAX_KEYBOARD_ANGLE)
            if 'down' in keys_held:
                keyboard_pitch_angle = min(keyboard_pitch_angle + KEYBOARD_ANGLE_PER_PRESS * dt, MAX_KEYBOARD_ANGLE)
            
            # Return to center when keys released (auto-level in angle mode!)
            if 'left' not in keys_held and 'right' not in keys_held:
                keyboard_roll_angle *= max(0, 1 - STICK_RETURN_SPEED * dt)
            if 'up' not in keys_held and 'down' not in keys_held:
                keyboard_pitch_angle *= max(0, 1 - STICK_RETURN_SPEED * dt)
            
            # Apply keyboard virtual stick to commands
            roll_angle_cmd = keyboard_roll_angle
            pitch_angle_cmd = keyboard_pitch_angle
            
            # Yaw rate from keyboard
            if 'a' in keys_held:
                yaw_rate_cmd = +KEYBOARD_YAW_RATE
            if 'd' in keys_held:
                yaw_rate_cmd = -KEYBOARD_YAW_RATE
            
            # Altitude from keyboard
            if 'w' in keys_held:
                target_altitude += KEYBOARD_ALT_STEP * dt
            if 's' in keys_held:
                target_altitude -= KEYBOARD_ALT_STEP * dt
            target_altitude = max(0.1, min(10.0, target_altitude))

        # Display current state
        if joystick is not None:
            num_axes = joystick.get_numaxes()
            axes_str = " ".join([f"[{i}]:{joystick.get_axis(i):+.2f}" for i in range(min(6, num_axes))])
            print(f"🎮 Raw: {axes_str} | 📐 Cmds: roll={np.rad2deg(roll_angle_cmd):+5.1f}° pitch={np.rad2deg(pitch_angle_cmd):+5.1f}° yaw_rate={np.rad2deg(yaw_rate_cmd):+5.1f}°/s alt={target_altitude:.2f}m")
        else:
            print(f"⌨️  Keys: {keys_held} | 📐 Angles: roll={np.rad2deg(roll_angle_cmd):+5.1f}° pitch={np.rad2deg(pitch_angle_cmd):+5.1f}° | Alt: {target_altitude:.2f}m")

        # Build action for ANGLE MODE: [roll_angle, pitch_angle, yaw_rate, altitude]
        action = np.array([roll_angle_cmd, pitch_angle_cmd, yaw_rate_cmd, target_altitude], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Don't auto-reset during manual control
        time.sleep(dt)

    env.close()
    
    # Cleanup pygame if it was used
    if PYGAME_AVAILABLE and joystick is not None:
        pygame.quit()
    
    print("\n✅ Teleop finished.")

if __name__ == "__main__":
    main()
