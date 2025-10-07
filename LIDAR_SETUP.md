# LIDAR Sensor Configuration

## Overview
The drone is now equipped with a spherical LIDAR sensor array providing 360° coverage in all directions.

## Sensor Configuration

### Total Sensors: 26 Rangefinders

**Horizontal Ring (8 sensors at 0° elevation)**
- 8 sensors at 45° azimuth intervals
- Coverage: Forward, Forward-Right, Right, Back-Right, Back, Back-Left, Left, Forward-Left
- Color: Red
- Range: 5m cutoff

**Upper Ring (8 sensors at ~30° elevation)**
- 8 sensors at 45° azimuth intervals
- Angled upward for upper hemisphere coverage
- Color: Orange
- Range: 5m cutoff

**Lower Ring (8 sensors at ~-30° elevation)**
- 8 sensors at 45° azimuth intervals
- Angled downward for lower hemisphere coverage
- Color: Cyan
- Range: 5m cutoff

**Zenith & Nadir (2 sensors)**
- Straight up (Zenith) - Green
- Straight down (Nadir) - Blue
- Range: 5m cutoff

## Visualization

The LIDAR system provides real-time visualization:

- **Green lines with red dots**: Active detection (obstacle within 5m)
  - Green line shows the ray path
  - Red sphere marks the exact hit point
  
- **Dim blue lines**: No detection (clear path or beyond 5m range)

## Data Access

LIDAR data is available through `drone.get_lidar_data()`:

```python
lidar_data = my_drone.get_lidar_data()

for lidar in lidar_data:
    sensor_name = lidar['sensor_name']     # e.g., 'lidar_0', 'lidar_up_45'
    distance = lidar['distance']           # Distance to obstacle (-1 if no hit)
    site_pos = lidar['site_pos']           # Sensor position in world coords
    hit_point = lidar['hit_point']         # Hit point in world coords
    has_hit = lidar['has_hit']             # Boolean: obstacle detected
```

## Integration Notes

The LIDAR is currently visualization-only and not integrated into the control logic. This allows you to:

1. Observe obstacle detection during waypoint navigation
2. Develop collision avoidance algorithms
3. Implement dynamic path planning
4. Add safety features

## Future Integration Ideas

- **Collision Avoidance**: Use LIDAR data to detect and avoid obstacles
- **Dynamic Path Planning**: Reroute around detected obstacles
- **Safety Layer**: Emergency stop when obstacles are too close
- **Mapping**: Build a local occupancy map of the environment
- **Terrain Following**: Use nadir sensor for altitude adjustment

