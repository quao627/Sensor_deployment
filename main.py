import carla
import numpy as np
import time
import queue
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.ops import cascaded_union

from sensor_placement import solve_bip

sidewalk_names = [153, 154, 151, 199, 202, 201, 205, 203, 206, 157, 158, 155]
road_names = [121, 117, 169, 165, 120, 116, 168, 164, 167, 163, 162, 166, 207, 210, 119, 115, 118, 114, 80, 77, 113, 161, 156, 108]


points_per_cloud = 50000
fps = 5

SENSOR_HEIGHT = 5.5
NUM_CHANNELS = 16


def extract_sidewalks(world):
    sidewalks = [sidewalk for index, sidewalk in enumerate(world.get_environment_objects(carla.CityObjectLabel.Sidewalks))]
    sidewalk_candidates = [sidewalk for sidewalk in sidewalks if any([(str(name) in sidewalk.name) and ("Curb" not in sidewalk.name) for name in sidewalk_names])]
    return sidewalk_candidates
    
def extract_roads(world):
    roads = [road for _, road in enumerate(world.get_environment_objects(carla.CityObjectLabel.Roads)) if any([str(name) in road.name.split("_") for name in road_names])]
    return roads

def generate_pc(sensor_location, sensor_rotation, num_channels):
    lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute('dropoff_general_rate', '0.35')
    lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
    lidar_bp.set_attribute('points_per_second', str(points_per_cloud*fps))
    lidar_bp.set_attribute('rotation_frequency', str(fps))
    lidar_bp.set_attribute('channels', str(num_channels))
    lidar_bp.set_attribute('lower_fov', '-15.0')
    lidar_bp.set_attribute('upper_fov', '15.0')
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('noise_stddev', '0.02')
    transform = carla.Transform(sensor_location, sensor_rotation)
    sensor = world.spawn_actor(lidar_bp, transform)
    q = queue.Queue()
    sensor.listen(lambda data: q.put(data))
    sensor_data = q.get(True, 2.0)
    points = np.copy(np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4')))
    ponits = np.reshape(points, (int(points.shape[0] / 4), 4))
    points = ponits[:, :3] + np.array([[sensor_location.x, sensor_location.y, sensor_location.z]])
    points_2d = points[np.abs(points[:, 2]) <= 0.5, :2]
    sensor.destroy()
    return points_2d

def extract_boundary(objects):
    bbox_list = []
    for ob in objects:
        bbox = MultiPoint([[-pt.x, -pt.y] for pt in ob.bounding_box.get_world_vertices(ob.transform)])
        bbox_list.append(bbox.convex_hull)
    boundary = MultiPolygon(Polygon(p.exterior) for p in bbox_list)
    return boundary


def get_world_boundary(roads):
    bbox_list = []
    for road in roads:
        bbox = [[-pt.x, -pt.y] for pt in road.bounding_box.get_world_vertices(road.transform)]
        bbox_list.append(bbox)

    x_list = np.array(bbox_list)[:, :, 0]
    y_list = np.array(bbox_list)[:, :, 1]

    x_min, x_max = x_list.min(), x_list.max()
    y_min, y_max = y_list.min(), y_list.max()

    return x_min, x_max, y_min, y_max


def get_visibility(sensor_pc, target_grid):
    dists = cdist(target_grid, sensor_pc)
    coverage = (dists < 1).sum(axis=1)
    return coverage

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(200.0)
    world = client.get_world()

    sidewalk_candidates = extract_sidewalks(world)
    road_candidates = extract_roads(world)

    sidewalk_boundary = extract_boundary(sidewalk_candidates)
    road_boundary = extract_boundary(road_candidates)

    # construct grid for the given region
    def linspace(start, stop, step):
        num = int((stop-start) / step + 1)
        return np.arange(0, num) * step + start
    x_min, x_max, y_min, y_max = get_world_boundary(road_candidates)
    x_space = linspace(x_min, x_max, 1)
    y_space = linspace(y_min, y_max, 1)

    xv, yv = np.meshgrid(x_space, y_space)
    world_grid = np.stack((xv, yv), axis=2)
    world_grid = world_grid.reshape((-1, 2))

    # identify target points and sensor candidate position  
    target_grid = world_grid[[any([Point(pt).within(boundary) for boundary in road_boundary]) for pt in world_grid]]
    sensor_grid = world_grid[[any([Point(pt).within(boundary) for boundary in sidewalk_boundary]) for pt in world_grid]][:20]
    N_t = target_grid.shape[0]
    N_s = sensor_grid.shape[0]

    # build visibility grid
    visibility_grid = np.zeros((N_s, N_t))
    for index, sensor_pos in enumerate(sensor_grid):
        print(f"Solving for sensor {index}")
        sensor_location = carla.Location(x=sensor_pos[0], y=sensor_pos[1], z=SENSOR_HEIGHT)
        sensor_rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
        num_channels = NUM_CHANNELS
        sensor_pc = generate_pc(sensor_location, sensor_rotation, num_channels)
        coverage = get_visibility(sensor_pc, target_grid)
        visibility_grid[index] = coverage

    # Solve optimization
    sensors = solve_bip(sensor_grid, visibility_grid, lam=0.1, CVR=0.5, L=5)
    print(sensors)
    

    
