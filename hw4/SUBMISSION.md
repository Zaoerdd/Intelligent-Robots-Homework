# HW4 Submission Notes

This write-up covers both parts of HW4:

- Part I: theoretical analysis from `hw4ir.pdf`
- Part II: programming from `readme.md` and `source/mdp.py`

## Part I: Theoretical Analysis

### Assumptions Used

The PDF does not explicitly state the discount factor or the out-of-bound behavior, so I used the same convention as the course quiz notes:

- `gamma = 1`
- the goal cell `O` is terminal with value `0`
- if an action goes out of the `3 x 3` map, the robot stays in the same cell
- for policy iteration, the "equal probability" wording is treated as the initial policy; both policy iteration and value iteration converge to the same final optimal result

### Case 1

Environment:

- `3 x 3` grid
- goal `O` at the top-right cell
- reward for entering the goal is `0`
- reward for any other move is `-1`

Final optimal values from both value iteration and policy iteration:

```text
[-1,  0,  0]
[-2, -1,  0]
[-3, -2, -1]
```

One optimal policy:

```text
[R,  R,  G]
[U/R, U/R, U]
[U/R, U/R, U]
```

Interpretation:

- every non-goal state chooses the shortest path to the goal
- ties appear whenever both `up` and `right` are equally good

### Case 2

Environment:

- `3 x 3` grid
- goal `O` at the top-right cell
- obstacle `B` at the center cell
- reward for entering the goal is `0`
- reward for entering `B` is `-1`
- reward for every other non-goal move is `-0.1`

Final optimal values from both value iteration and policy iteration:

```text
[-0.1,  0.0,  0.0]
[-0.2, -0.1,  0.0]
[-0.3, -0.2, -0.1]
```

One optimal policy:

```text
[R,  R,  G]
[U,  U/R, U]
[U/R, R,  U]
```

Interpretation:

- the optimal policy prefers short routes that avoid stepping into `B` unless there is no better alternative
- the center obstacle cell still has value `-0.1` because once the robot is already there, the best next move is to leave it immediately

### Appendix: If "Equal Probability" Is Interpreted Literally

If the instructor instead wants policy evaluation under a fixed random policy with `pi(a|s) = 0.25` for all four actions, the converged values are:

Case 1:

```text
[[-21.5, -15.0,   0.0],
 [-24.0, -20.5, -15.0],
 [-26.0, -24.0, -21.5]]
```

Case 2:

```text
[[-4.4, -3.3,  0.0],
 [-5.1, -4.3, -3.3],
 [-5.3, -5.1, -4.4]]
```

## Part II: Programming

### Code Changes

The missing MDP logic is now implemented in:

- `source/mdp.py`
- `source/grid_map.py`

I also added:

- `source/verify_hw4.py` for headless verification

And I removed the unnecessary `ir_sim` dependency from:

- `source/question1_run.py`

### What Was Implemented

- policy evaluation
- policy improvement for policy iteration
- value iteration
- greedy action selection for the value-iteration result
- proper terminal-state handling at the goal
- non-terminal boundary penalties so the agent no longer prefers "crashing out of the map"

### Verification Summary

Using `source/verify_hw4.py`:

- Q1 policy evaluation:
  - start value: `-38.8338`
  - goal value: `0.0`
- Q2 policy iteration:
  - converged after `16` policy-improvement rounds
  - reaches `(16, 16)` from `(2, 2)` in `28` steps
- Q3 value iteration:
  - converged after `26` value-update rounds
  - also reaches `(16, 16)` from `(2, 2)` in `28` steps

The verified Q2/Q3 path is:

```text
(2,2) -> (3,2) -> (4,2) -> (4,3) -> (4,4) -> (4,5) -> (4,6) -> (4,7)
-> (4,8) -> (4,9) -> (4,10) -> (5,10) -> (6,10) -> (7,10) -> (8,10)
-> (9,10) -> (10,10) -> (11,10) -> (12,10) -> (13,10) -> (14,10)
-> (14,11) -> (14,12) -> (14,13) -> (15,13) -> (16,13) -> (16,14)
-> (16,15) -> (16,16)
```

### How To Run

For a quick numeric check without animation:

```bash
cd hw4/source
python verify_hw4.py
```

For the original homework scripts:

```bash
cd hw4/source
python question1_run.py
python question2_run.py
python question3_run.py
```

Notes:

- `question1_run.py` no longer needs `ir_sim`
- `question2_run.py` and `question3_run.py` still need the simulator from the homework README:

```bash
pip install ir_sim==1.1.8 matplotlib
```
