#!/usr/bin/env python3
"""
Simple script to print raw gamepad values continuously.
This will help debug gamepad input and understand the raw values.
"""

import time
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("ERROR: pygame not found. Install with: pip install pygame")
    sys.exit(1)

def print_raw_gamepad_values():
    """Print raw gamepad values continuously."""
    pygame.init()
    pygame.joystick.init()
    
    # Check for connected joysticks
    joystick_count = pygame.joystick.get_count()
    print(f"Detected {joystick_count} joystick(s)")
    
    if joystick_count == 0:
        print("No gamepad detected. Please connect a gamepad and try again.")
        return
    
    # List all available joysticks
    for i in range(joystick_count):
        temp_joy = pygame.joystick.Joystick(i)
        print(f"  Joystick {i}: {temp_joy.get_name()} - {temp_joy.get_numaxes()} axes, {temp_joy.get_numbuttons()} buttons")
    
    # Use first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"\n✓ Using gamepad: {joystick.get_name()}")
    print(f"  Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}")
    print("\n" + "="*80)
    print("RAW GAMEPAD VALUES (Press Ctrl+C to exit)")
    print("="*80)
    print("Format: Axes: [axis0, axis1, ...] | Buttons: [btn0, btn1, ...]")
    print("="*80)
    
    try:
        while True:
            # Process pygame events to update joystick state
            pygame.event.pump()
            
            # Get raw axis values
            axes = []
            for i in range(joystick.get_numaxes()):
                axes.append(joystick.get_axis(i))
            
            # Get raw button values
            buttons = []
            for i in range(joystick.get_numbuttons()):
                buttons.append(joystick.get_button(i))
            
            # Print raw values
            axes_str = "[" + ", ".join([f"{val:6.3f}" for val in axes]) + "]"
            buttons_str = "[" + ", ".join([f"{val:1d}" for val in buttons]) + "]"
            
            print(f"Axes: {axes_str} | Buttons: {buttons_str}", end='\r')
            
            time.sleep(0.05)  # 20Hz update rate
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        pygame.quit()

if __name__ == "__main__":
    print_raw_gamepad_values()
