import numpy as np

from grid_map import grid_map
from mdp import mdp


def trace_path(grid, solver, action_fn, max_steps=300):
    cur_state = grid.start_index
    path = [cur_state]

    for step in range(max_steps):
        action_index = action_fn(cur_state)
        cur_state, reward, _, done = grid.step(cur_state, action_index)
        path.append(cur_state)

        if done:
            return {
                "steps": step + 1,
                "reward": reward,
                "terminal_state": cur_state,
                "path": path,
            }

    return {
        "steps": max_steps,
        "reward": None,
        "terminal_state": cur_state,
        "path": path,
    }


def main():
    map_matrix = np.load("map_matrix.npy")
    reward_matrix = np.load("reward_matrix.npy")

    # Q1: policy evaluation under the default uniform policy.
    grid_q1 = grid_map(map_matrix=map_matrix.copy(), reward_matrix=reward_matrix)
    mdp_q1 = mdp(grid_q1)
    policy_value_q1 = mdp_q1.policy_evaluation()
    print("Q1")
    print("start value:", round(float(policy_value_q1[grid_q1.start_index]), 4))
    print("goal value:", round(float(policy_value_q1[grid_q1.goal_index]), 4))
    print("min/max:", round(float(policy_value_q1.min()), 4), round(float(policy_value_q1.max()), 4))
    print()

    # Q2: policy iteration on the deterministic map.
    grid_q2 = grid_map(map_matrix=map_matrix.copy(), reward_matrix=reward_matrix)
    mdp_q2 = mdp(grid_q2)

    for iteration in range(300):
        policy_value_q2 = mdp_q2.policy_evaluation()
        iterate_done = mdp_q2.policy_iteration(policy_value_q2)

        if iterate_done:
            break

    q2_result = trace_path(grid_q2, mdp_q2, mdp_q2.get_policy_action)
    print("Q2")
    print("iterations:", iteration + 1)
    print("start value:", round(float(policy_value_q2[grid_q2.start_index]), 4))
    print("terminal state:", q2_result["terminal_state"])
    print("steps:", q2_result["steps"])
    print("path:", q2_result["path"])
    print()

    # Q3: value iteration on the stochastic map.
    grid_q3 = grid_map(map_matrix=map_matrix.copy(), reward_matrix=reward_matrix, state_prob=0.8)
    mdp_q3 = mdp(grid_q3)
    policy_value_q3 = mdp_q3.value_iteration()
    q3_result = trace_path(grid_q3, mdp_q3, lambda state: mdp_q3.get_value_action(policy_value_q3, state))

    print("Q3")
    print("start value:", round(float(policy_value_q3[grid_q3.start_index]), 4))
    print("terminal state:", q3_result["terminal_state"])
    print("steps:", q3_result["steps"])
    print("path:", q3_result["path"])


if __name__ == "__main__":
    main()
