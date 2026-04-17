# Homework 3 Submission

## Completed Tasks

This submission completes all three parts of Homework 3:

1. Question 1: A* search on the given grid map.
2. Question 2: Dynamic Window Approach (DWA) for local obstacle avoidance.
3. Question 3: Combined A* global planning and DWA local planning.

## Main Changes

- `source/Astar.py`
  - Implemented the full A* search loop.
  - Implemented the heuristic with Euclidean distance.

- `source/dwa.py`
  - Implemented `cost_function`.
  - Implemented `vel_cost`.
  - Implemented `cost_to_goal`.
  - Implemented `cost_to_obstacle`.
  - Implemented `astar_cost`.
  - Added boundary and collision checks for predicted trajectories.

- `source/question1_run.py`
- `source/question2_run.py`
- `source/question3_run.py`
  - Added automatic creation of `animation_buffer` when `-a` is used, so GIF export works reliably.

## Environment

- Python: `3.9`
- Install dependency:

```bash
py -3.9 -m pip install --user ir_sim==1.1.8
```

## How To Run

Run from `hw3/source`:

```bash
py -3.9 question1_run.py
py -3.9 question2_run.py
py -3.9 question3_run.py
```

Export animations:

```bash
py -3.9 question1_run.py -a
py -3.9 question2_run.py -a
py -3.9 question3_run.py -a
```

## Validation Results

- Question 1
  - A* successfully found a valid path.
  - Visited nodes: `268`
  - Path length: `31` grid points

- Question 2
  - DWA reached the goal region in `38` steps.
  - Final distance to goal: `0.17`

- Question 3
  - A* + DWA reached the goal region in `67` steps.
  - Final distance to goal: `0.13`

## Animation Files

Generated animations are available at:

- `source/animation/astar.gif`
- `source/animation/dwa.gif`
- `source/animation/astar_dwa.gif`

## PDF Report

The submission also includes:

- `Homework3_Report.pdf`
- `report.tex`
- `report_assets/`

## Submission Note

The `hw3` folder now contains the source code, config files, animation outputs, PDF report, LaTeX source, and this summary document.
