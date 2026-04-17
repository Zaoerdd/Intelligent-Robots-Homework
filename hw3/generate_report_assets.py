from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
ASSET_DIR = ROOT / "report_assets"

sys.path.insert(0, str(SOURCE_DIR))

from ir_sim.env import EnvBase
from Astar import Astar
from dwa import dynamic_window_approach
from grid_graph import grid_graph


def world_points_from_indices(graph, index_list):
    return np.array([graph.index_to_pose(ix, iy) for ix, iy in index_list], dtype=float)


def polyline_length(points):
    if len(points) < 2:
        return 0.0

    diff = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diff, axis=1)))


def plot_map(ax, graph):
    occupancy = (graph.grid_map != 0).T
    extent = [0, graph.width * graph.xy_reso[0, 0], 0, graph.height * graph.xy_reso[1, 0]]
    ax.imshow(
        occupancy,
        cmap="Greys",
        origin="lower",
        extent=extent,
        interpolation="nearest",
        alpha=0.65,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)


def run_question1():
    env = EnvBase(str(SOURCE_DIR / "question1.yaml"), save_ani=False, display=False)
    graph = grid_graph(grid_map_matrix=env.world.grid_map, xy_reso=env.world.reso)
    astar = Astar()

    start_point = [2, 4]
    goal_point = [5, 5]

    start_x, start_y = graph.pose_to_index(*start_point)
    goal_x, goal_y = graph.pose_to_index(*goal_point)

    start_node = graph.node_tuple(start_x, start_y, 0, None)
    goal_node = graph.node_tuple(goal_x, goal_y, 0, None)

    final_node, visit_list = astar.find_path(graph, start_node, goal_node)
    path_index_list = astar.generate_path(final_node)
    path_index_list.reverse()

    visit_points = world_points_from_indices(graph, visit_list)
    path_points = world_points_from_indices(graph, path_index_list)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    plot_map(ax, graph)
    ax.scatter(visit_points[:, 0], visit_points[:, 1], s=8, c="#7f8c8d", label="Visited nodes", alpha=0.7)
    ax.plot(path_points[:, 0], path_points[:, 1], color="#c0392b", linewidth=2.2, label="A* path")
    ax.scatter([start_point[0]], [start_point[1]], c="#2c7a7b", s=60, marker="o", label="Start")
    ax.scatter([goal_point[0]], [goal_point[1]], c="#d35400", s=70, marker="*", label="Goal")
    ax.set_title("Question 1: A* search result")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "q1_astar.png", dpi=220)
    plt.close(fig)
    plt.close("all")

    path_length = polyline_length(path_points)
    return {
        "visited_nodes": len(visit_list),
        "path_points": len(path_index_list),
        "path_length_m": round(path_length, 3),
    }


def run_question2():
    env = EnvBase(str(SOURCE_DIR / "question2.yaml"), save_ani=False, display=False)
    graph = grid_graph(grid_map_matrix=env.world.grid_map, xy_reso=env.world.reso)
    dwa = dynamic_window_approach(
        vx_range=[-1.5, 1.5],
        vy_range=[-1.5, 1.5],
        accelerate=2,
        time_interval=0.5,
        predict_time=1,
        graph=graph,
    )

    states = [env.robot.state[0:2].copy()]
    steps = 0

    for i in range(300):
        vel, _ = dwa.cal_vel(
            cur_pose=env.robot.state,
            goal_pose=env.robot.goal,
            current_vel=env.robot.vel_omni,
            v_gain=1,
            g_gain=1,
            o_gain=1.5,
        )

        env.step(vel)
        states.append(env.robot.state[0:2].copy())
        steps = i + 1

        if env.done():
            break

    traj = np.hstack(states).T
    start = states[0].reshape(2)
    goal = env.robot.goal.reshape(2)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    plot_map(ax, graph)
    ax.plot(traj[:, 0], traj[:, 1], color="#1f78b4", linewidth=2.3, label="DWA trajectory")
    ax.scatter([start[0]], [start[1]], c="#2c7a7b", s=60, marker="o", label="Start")
    ax.scatter([goal[0]], [goal[1]], c="#d35400", s=70, marker="*", label="Goal")
    ax.set_title("Question 2: DWA local planning result")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "q2_dwa.png", dpi=220)
    plt.close(fig)
    plt.close("all")

    final_distance = float(np.linalg.norm(env.robot.state - env.robot.goal))
    return {
        "steps": steps,
        "final_distance_m": round(final_distance, 3),
        "trajectory_length_m": round(polyline_length(traj), 3),
    }


def run_question3():
    env = EnvBase(str(SOURCE_DIR / "question3.yaml"), save_ani=False, display=False)
    graph = grid_graph(env.world.grid_map, env.world.reso)
    dwa = dynamic_window_approach(
        vx_range=[-1.5, 1.5],
        vy_range=[-1.5, 1.5],
        accelerate=2,
        time_interval=0.2,
        predict_time=0.5,
        graph=graph,
    )
    astar = Astar()

    start_point = [env.robot.state[0, 0], env.robot.state[1, 0]]
    goal_point = [env.robot.goal[0, 0], env.robot.goal[1, 0]]

    start_x, start_y = graph.pose_to_index(*start_point)
    goal_x, goal_y = graph.pose_to_index(*goal_point)

    start_node = graph.node_tuple(start_x, start_y, 0, None)
    goal_node = graph.node_tuple(goal_x, goal_y, 0, None)

    final_node, _ = astar.find_path(graph, start_node, goal_node)
    path_index_list = astar.generate_path(final_node)
    path_index_list.reverse()
    path_points = world_points_from_indices(graph, path_index_list)

    states = [env.robot.state[0:2].copy()]
    steps = 0

    for i in range(300):
        vel, _ = dwa.cal_vel(
            cur_pose=env.robot.state,
            goal_pose=env.robot.goal[0:2],
            current_vel=env.robot.vel_omni,
            v_gain=10,
            g_gain=5,
            o_gain=9,
            a_gain=10,
            astar_path=path_index_list,
        )

        env.step(vel)
        states.append(env.robot.state[0:2].copy())
        steps = i + 1

        if env.done():
            break

    traj = np.hstack(states).T
    start = states[0].reshape(2)
    goal = env.robot.goal[0:2].reshape(2)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    plot_map(ax, graph)
    ax.plot(path_points[:, 0], path_points[:, 1], color="#c0392b", linewidth=2.0, label="A* global path")
    ax.plot(traj[:, 0], traj[:, 1], color="#1f78b4", linewidth=2.3, label="DWA trajectory")
    ax.scatter([start[0]], [start[1]], c="#2c7a7b", s=60, marker="o", label="Start")
    ax.scatter([goal[0]], [goal[1]], c="#d35400", s=70, marker="*", label="Goal")
    ax.set_title("Question 3: A* + DWA result")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "q3_astar_dwa.png", dpi=220)
    plt.close(fig)
    plt.close("all")

    final_distance = float(np.linalg.norm(env.robot.state[0:2] - env.robot.goal[0:2]))
    return {
        "astar_path_points": len(path_index_list),
        "steps": steps,
        "final_distance_m": round(final_distance, 3),
        "trajectory_length_m": round(polyline_length(traj), 3),
    }


def main():
    ASSET_DIR.mkdir(exist_ok=True)

    stats = {
        "question1": run_question1(),
        "question2": run_question2(),
        "question3": run_question3(),
    }

    with open(ASSET_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
