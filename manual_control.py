# manual_control.py
# Keyboard + Gamepad teleoperation for DroneEnv via Gymnasium API.
#
# Behavior: when you are not pressing keys/moving sticks, the action sent this frame is zeros.
# Inside the env, setpoints decay toward neutral → it hovers hands-off.
#
# Keyboard Controls:
#   Arrow keys: roll/pitch setpoint nudges (hold to accumulate via OS key repeat)
#   A/D: yaw rate (left/right)
#   Space: altitude rate (up)
#   Ctrl: altitude rate (down)
#   Z: zero all commands immediately
#   R: reset episode
#   Esc: exit
#
# Gamepad Controls:
#   Left stick up/down: altitude rate
#   Left stick left/right: yaw rate
#   Right stick up/down: pitch
#   Right stick left/right: roll
#   Start/Options button: reset episode
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

# Per-press increments / rates (tune to taste)
ROLL_PITCH_DELTA = np.deg2rad(1.2)   # delta added this frame when arrow pressed
YAW_RATE_CMD     = np.deg2rad(30.0)  # one-frame yaw rate when A/D pressed
ALT_RATE_CMD     = 0.7               # one-frame altitude rate when W/S pressed

# Gamepad settings
GAMEPAD_DEADZONE = 0.15              # Ignore small stick movements
GAMEPAD_ROLL_PITCH_SCALE = 1  # Max roll/pitch per frame from stick
GAMEPAD_YAW_SCALE = 1       # Max yaw rate from stick
GAMEPAD_ALT_SCALE = 1.0                      # Max altitude rate from stick
CALIBRATION_TIME = 3.0               # Seconds to calibrate gamepad center position

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
    """Read gamepad inputs and return (roll_step, pitch_step, yaw_rate, alt_rate, reset_pressed)."""
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
    
    roll_step = 0.0
    pitch_step = 0.0
    yaw_rate = 0.0
    alt_rate = 0.0
    
    # Use calibrated centers or default to 0
    if centers is None:
        centers = np.zeros(num_axes)
    
    # Left stick X (axis 0) -> yaw (left=-1, right=+1)
    if num_axes > 0:
        left_x = apply_deadzone(joystick.get_axis(0), centers[0])
        yaw_rate = -left_x * GAMEPAD_YAW_SCALE  # Negate so left stick right = yaw right
    
    # Left stick Y (axis 1) -> altitude (up=-1, down=+1)
    if num_axes > 1:
        left_y = apply_deadzone(joystick.get_axis(1), centers[1])
        alt_rate = -left_y * GAMEPAD_ALT_SCALE  # Negate so stick up = altitude up
    
    # Right stick X (axis 3) -> roll (left=-1, right=+1)
    if num_axes > 3:
        right_x = apply_deadzone(joystick.get_axis(3), centers[3])
        roll_step = -right_x * GAMEPAD_ROLL_PITCH_SCALE  # Negate so stick right = roll right
    
    # Right stick Y (axis 4) -> pitch (up=-1, down=+1)
    if num_axes > 4:
        right_y = apply_deadzone(joystick.get_axis(4), centers[4])
        pitch_step = -right_y * GAMEPAD_ROLL_PITCH_SCALE  # Negate so stick up = pitch forward
    
    # Check for reset button (Options button on DualSense is usually button 9)
    reset_pressed = False
    num_buttons = joystick.get_numbuttons()
    if num_buttons > 9:
        reset_pressed = joystick.get_button(9)
    elif num_buttons > 7:
        reset_pressed = joystick.get_button(7)
    
    return roll_step, pitch_step, yaw_rate, alt_rate, reset_pressed

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

    print("\n" + "="*60)
    print("DRONE MANUAL CONTROL")
    print("="*60)
    if joystick:
        print("Gamepad Controls:")
        print("  Left stick:  Yaw (L/R) | Altitude (U/D)")
        print("  Right stick: Roll (L/R) | Pitch (U/D)")
        print("  Start button: Reset")
        print()
    print("Keyboard Controls:")
    print("  Arrows: Roll/Pitch | A/D: Yaw | Space/Ctrl: Alt Up/Down")
    print("  Z: Stop | R: Reset | Esc: Quit")
    print("="*60 + "\n")

    dt = 1.0 / env.metadata.get("render_fps", 60)
    running = True
    gamepad_reset_pressed_last = False

    while running and env.viewer.is_running():
        # Start each frame at neutral → hover if no input this frame
        roll_step = 0.0
        pitch_step = 0.0
        yaw_rate = 0.0
        alt_rate = 0.0
        
        # Read gamepad first (can be overridden by keyboard)
        gp_roll, gp_pitch, gp_yaw, gp_alt, gamepad_reset_pressed = read_gamepad(joystick, gamepad_centers)
        roll_step += gp_roll
        pitch_step += gp_pitch
        yaw_rate += gp_yaw
        alt_rate += gp_alt
        
        # Handle gamepad reset (edge detection to avoid multiple resets)
        if gamepad_reset_pressed and not gamepad_reset_pressed_last:
            obs, _ = env.reset()
            print("Reset by gamepad")
        gamepad_reset_pressed_last = gamepad_reset_pressed

        # Drain key events that arrived since last frame
        while _key_events:
            keycode = _key_events.popleft()

            # GLFW-style keycodes used by mujoco.viewer
            GLFW_KEY_LEFT  = 263
            GLFW_KEY_RIGHT = 262
            GLFW_KEY_UP    = 265
            GLFW_KEY_DOWN  = 264
            GLFW_KEY_ESC   = 256
            GLFW_KEY_SPACE = 32
            GLFW_KEY_LEFT_CONTROL = 341
            GLFW_KEY_RIGHT_CONTROL = 345
            try:
                ch = chr(keycode)
            except Exception:
                ch = ''

            if keycode == GLFW_KEY_LEFT:
                roll_step += +ROLL_PITCH_DELTA
            elif keycode == GLFW_KEY_RIGHT:
                roll_step += -ROLL_PITCH_DELTA
            elif keycode == GLFW_KEY_UP:
                pitch_step += -ROLL_PITCH_DELTA
            elif keycode == GLFW_KEY_DOWN:
                pitch_step += +ROLL_PITCH_DELTA
            elif keycode == GLFW_KEY_ESC:
                running = False

            elif ch in ('a','A'):
                yaw_rate += +YAW_RATE_CMD
            elif ch in ('d','D'):
                yaw_rate += -YAW_RATE_CMD
            elif keycode == GLFW_KEY_SPACE:
                alt_rate += +ALT_RATE_CMD
            elif keycode in (GLFW_KEY_LEFT_CONTROL, GLFW_KEY_RIGHT_CONTROL):
                alt_rate += -ALT_RATE_CMD
            elif ch in ('z','Z'):
                # explicit neutralize
                roll_step = pitch_step = yaw_rate = alt_rate = 0.0
            elif ch in ('r','R'):
                obs, _ = env.reset()
                print("Reset by keyboard")


        # Debug output - show ALL axes to identify correct mapping
        if joystick is not None:
            num_axes = joystick.get_numaxes()
            axes_str = " ".join([f"[{i}]:{joystick.get_axis(i):+.2f}" for i in range(num_axes)])
            print(f"Axes: {axes_str} -> roll:{roll_step:+.3f} pitch:{pitch_step:+.3f} yaw:{yaw_rate:+.3f} alt:{alt_rate:+.3f}")
        else:
            print(f"Keyboard: roll:{roll_step:.3f} pitch:{pitch_step:.3f} yaw:{yaw_rate:.3f} alt:{alt_rate:.3f}")

        # One-frame action (zeros if no input this frame)
        action = np.array([roll_step, pitch_step, yaw_rate, alt_rate], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # Don't auto-reset during manual control - let user press R to reset manually
        # if terminated or truncated:
        #     obs, _ = env.reset()

        time.sleep(dt)

    env.close()
    
    # Cleanup pygame if it was used
    if PYGAME_AVAILABLE and joystick is not None:
        pygame.quit()
    
    print("\nTeleop finished.")

if __name__ == "__main__":
    main()
