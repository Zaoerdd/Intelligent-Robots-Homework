import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
ASSETS = ROOT / "report_assets"

sys.path.insert(0, str(SOURCE))

from grid_map import grid_map  # noqa: E402
from mdp import mdp  # noqa: E402


def load_environment(state_prob=1.0):
    map_matrix = np.load(SOURCE / "map_matrix.npy")
    reward_matrix = np.load(SOURCE / "reward_matrix.npy")
    return grid_map(map_matrix=map_matrix.copy(), reward_matrix=reward_matrix, state_prob=state_prob)


def trace_path(grid, action_fn, max_steps=300):
    cur_state = grid.start_index
    path = [cur_state]

    for step in range(max_steps):
        action_index = action_fn(cur_state)
        cur_state, reward, _, done = grid.step(cur_state, action_index)
        path.append(cur_state)

        if done:
            return {
                "steps": step + 1,
                "reward": float(reward),
                "terminal_state": list(cur_state),
                "path": [list(node) for node in path],
            }

    return {
        "steps": max_steps,
        "reward": None,
        "terminal_state": list(cur_state),
        "path": [list(node) for node in path],
    }


def run_value_iteration(solver, threshold=0.01):
    policy_value = np.zeros(solver.state_space[0:2])
    iteration_num = 0

    while True:
        delta = 0.0

        for i in range(solver.state_space[0]):
            for j in range(solver.state_space[1]):
                state_index = (i, j)
                old_value = policy_value[i, j]

                if solver._is_terminal_state(state_index):
                    policy_value[i, j] = 0.0
                else:
                    action_values = []

                    for action_index, _ in enumerate(solver.action_space):
                        action_values.append(solver._action_value(policy_value, state_index, action_index))

                    policy_value[i, j] = max(action_values)

                delta = max(delta, abs(old_value - policy_value[i, j]))

        iteration_num += 1

        if delta < threshold:
            return policy_value, iteration_num


def value_to_color(value, min_value, max_value):
    if max_value == min_value:
        ratio = 1.0
    else:
        ratio = (value - min_value) / (max_value - min_value)

    ratio = max(0.0, min(1.0, ratio))

    if ratio < 0.5:
        blend = ratio / 0.5
        start = np.array([25, 35, 90], dtype=float)
        end = np.array([75, 175, 255], dtype=float)
    else:
        blend = (ratio - 0.5) / 0.5
        start = np.array([75, 175, 255], dtype=float)
        end = np.array([255, 230, 110], dtype=float)

    color = start * (1 - blend) + end * blend
    return tuple(int(channel) for channel in color)


def save_heatmap(values, start, goal, output_path):
    rows, cols = values.shape
    cell = 22
    margin = 20
    image = Image.new("RGB", (cols * cell + margin * 2, rows * cell + margin * 2), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    min_value = float(values.min())
    max_value = float(values.max())

    for row in range(rows):
        for col in range(cols):
            x0 = margin + col * cell
            y0 = margin + row * cell
            x1 = x0 + cell
            y1 = y0 + cell

            color = value_to_color(float(values[row, col]), min_value, max_value)
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(220, 220, 220))

    for row, col, label, color in [
        (start[0], start[1], "S", (255, 215, 0)),
        (goal[0], goal[1], "G", (220, 20, 60)),
    ]:
        x0 = margin + col * cell
        y0 = margin + row * cell
        x1 = x0 + cell
        y1 = y0 + cell
        draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=color, width=3)
        draw.text((x0 + 7, y0 + 5), label, fill=(10, 10, 10))

    image.save(output_path)


def save_path_overlay(map_matrix, path, start, goal, output_path):
    base = Image.fromarray(map_matrix.astype(np.uint8), mode="RGB").resize(
        (map_matrix.shape[1] * 24, map_matrix.shape[0] * 24),
        Image.Resampling.NEAREST,
    )
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cell = 24

    for row, col in path:
        x0 = col * cell
        y0 = row * cell
        x1 = x0 + cell - 1
        y1 = y0 + cell - 1
        draw.rectangle([x0 + 5, y0 + 5, x1 - 5, y1 - 5], fill=(30, 144, 255, 180))

    for row, col, fill_color, label in [
        (start[0], start[1], (255, 215, 0, 220), "S"),
        (goal[0], goal[1], (255, 99, 132, 220), "G"),
    ]:
        x0 = col * cell
        y0 = row * cell
        x1 = x0 + cell - 1
        y1 = y0 + cell - 1
        draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], fill=fill_color)
        draw.text((x0 + 7, y0 + 5), label, fill=(20, 20, 20, 255))

    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(output_path)


def main():
    ASSETS.mkdir(exist_ok=True)

    # Q1
    grid_q1 = load_environment()
    mdp_q1 = mdp(grid_q1)
    value_q1 = mdp_q1.policy_evaluation()
    save_heatmap(value_q1, grid_q1.start_index, grid_q1.goal_index, ASSETS / "q1_policy_evaluation_heatmap.png")

    # Q2
    grid_q2 = load_environment()
    mdp_q2 = mdp(grid_q2)
    q2_iterations = 0
    while True:
        value_q2 = mdp_q2.policy_evaluation()
        q2_iterations += 1
        if mdp_q2.policy_iteration(value_q2):
            break
    q2_trace = trace_path(grid_q2, mdp_q2.get_policy_action)
    save_path_overlay(
        grid_q2.map_matrix,
        [tuple(node) for node in q2_trace["path"]],
        grid_q2.start_index,
        grid_q2.goal_index,
        ASSETS / "q2_policy_iteration_path.png",
    )
    save_heatmap(value_q2, grid_q2.start_index, grid_q2.goal_index, ASSETS / "q2_policy_iteration_heatmap.png")

    # Q3
    grid_q3 = load_environment(state_prob=0.8)
    mdp_q3 = mdp(grid_q3)
    value_q3, q3_iterations = run_value_iteration(mdp_q3)
    q3_trace = trace_path(grid_q3, lambda state: mdp_q3.get_value_action(value_q3, state))
    save_path_overlay(
        grid_q3.map_matrix,
        [tuple(node) for node in q3_trace["path"]],
        grid_q3.start_index,
        grid_q3.goal_index,
        ASSETS / "q3_value_iteration_path.png",
    )
    save_heatmap(value_q3, grid_q3.start_index, grid_q3.goal_index, ASSETS / "q3_value_iteration_heatmap.png")

    metrics = {
        "q1": {
            "start_value": round(float(value_q1[grid_q1.start_index]), 4),
            "goal_value": round(float(value_q1[grid_q1.goal_index]), 4),
            "min_value": round(float(value_q1.min()), 4),
            "max_value": round(float(value_q1.max()), 4),
        },
        "q2": {
            "iterations": q2_iterations,
            "start_value": round(float(value_q2[grid_q2.start_index]), 4),
            "steps": q2_trace["steps"],
            "terminal_state": q2_trace["terminal_state"],
            "reward": q2_trace["reward"],
            "path": q2_trace["path"],
        },
        "q3": {
            "iterations": q3_iterations,
            "start_value": round(float(value_q3[grid_q3.start_index]), 4),
            "steps": q3_trace["steps"],
            "terminal_state": q3_trace["terminal_state"],
            "reward": q3_trace["reward"],
            "path": q3_trace["path"],
        },
    }

    with open(ASSETS / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
