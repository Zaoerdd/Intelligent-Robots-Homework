from ir_sim.env import EnvBase
from ir_sim.util.collision_dectection_geo import collision_seg_seg 
from potential_fields import potential_fields
from collections import namedtuple
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='The given force potential fields')
parser.add_argument('-a', '--animation', action='store_true')
args = parser.parse_args()

point = namedtuple('point', 'x y')

env = EnvBase(world_name='question3.yaml', save_ani=args.animation)

pf = potential_fields()

line_obs = env.get_obstacle_list() 
escape_force = np.array([[-0.6], [0.0]])
escape_waypoint = np.array([[4.5], [2.7]])
small_force_steps = 0
escape_steps = 0
guided_escape = False
segment_coefficients = [
    (0.25, 1.2),  # bottom edge
    (1.4, 0.2),   # right edge
    (0.9, 0.3),   # top edge
]


def matrix_to_point(matrix):
    return point(float(matrix[0, 0]), float(matrix[1, 0]))


def segment_repulsive_force(line_obstacle, robot_state, coefficient_inside, coefficient_outside, influence_distance=1.5):
    start_point, end_point = line_obstacle.points
    _, projection_point, _ = pf.shortest_distance_point(start_point, end_point, robot_state)
    segment = end_point - start_point
    relative = robot_state - projection_point
    cross_value = float(segment[0, 0] * relative[1, 0] - segment[1, 0] * relative[0, 0])
    left_normal = np.array([[-segment[1, 0]], [segment[0, 0]]], dtype=float)

    if cross_value >= 0.0:
        coefficient = coefficient_inside
        fallback_direction = left_normal
    else:
        coefficient = coefficient_outside
        fallback_direction = -left_normal

    return pf._repulsive_force(
        projection_point,
        robot_state,
        coefficient=coefficient,
        influence_distance=influence_distance,
        fallback_direction=fallback_direction,
    )


def collides_with_obstacle(robot_state, force):
    start_state = matrix_to_point(robot_state)
    next_state = robot_state + env.step_time * force
    motion_segment = [start_state, matrix_to_point(next_state)]

    for obstacle in line_obs:
        obs_segment = [matrix_to_point(obstacle.points[0]), matrix_to_point(obstacle.points[1])]
        if collision_seg_seg(motion_segment, obs_segment):
            return True

    return False

for i in range(1000):

    target_point = escape_waypoint if guided_escape else env.robot.goal
    attractive_force = pf.attractive(target_point, env.robot.state, coefficient=0.5)
    repulsive_force = np.zeros((2, 1))

    for obstacle, (coefficient_inside, coefficient_outside) in zip(line_obs, segment_coefficients):
        repulsive_force += segment_repulsive_force(
            obstacle,
            env.robot.state,
            coefficient_inside=coefficient_inside,
            coefficient_outside=coefficient_outside,
        )

    pf_force = attractive_force + repulsive_force

    if guided_escape and escape_steps > 0:
        pf_force = pf_force + escape_force
        escape_steps -= 1

    if guided_escape and env.robot.state[0, 0] < 4.8 and env.robot.state[1, 0] < 3.0:
        guided_escape = False

    if float(np.linalg.norm(pf_force)) < 0.05 and not env.done():
        small_force_steps += 1
    else:
        small_force_steps = 0

    if small_force_steps >= 15:
        guided_escape = True
        escape_steps = 10
        small_force_steps = 0
        pf_force = pf.attractive(escape_waypoint, env.robot.state, coefficient=0.5) + repulsive_force + escape_force

    pf_force = pf._clip_norm(pf_force)

    if collides_with_obstacle(env.robot.state, pf_force):
        pf_force = pf._clip_norm(repulsive_force + escape_force)

    env.step(pf_force)
    env.render(show_traj=True)

    if env.done():
        break

env.end(ani_name = 'potential_field', ani_kwargs={'subrectangles': True})
