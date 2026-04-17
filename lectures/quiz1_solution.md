# CS401 Intelligent Robotics QZ1 Solutions

## Notes And Assumptions

I use the following assumptions to keep all numerical questions consistent with the quiz image and the reinforcement learning lecture:

1. `gamma = 1` unless otherwise stated.
2. For Q2, Q3, and Q5:
   - the grid is a deterministic `3 x 4` grid world with row-major indices
     - top row: `0 1 2 3`
     - middle row: `4 5 6 7`
     - bottom row: `8 9 10 11`
   - state `3` is a terminal state with value `0`
   - state `7` is a terminal state with value `-1`
   - state `4` is an obstacle
   - every non-terminal step has cost `-0.1`
   - if an action goes out of the map or into the obstacle, the agent stays in the same state
3. For Q3, if multiple arrows are drawn in one cell, the policy is treated as a uniform random policy over those shown actions.
4. For Q1:
   - all `8` displayed sequences are treated as sample episodes
   - Monte Carlo follows `first-visit MC`, matching the lecture slides
   - TD is updated online in the natural reading order of the figure: top row left to right, then bottom row left to right
   - the initial values are `V(A)=0`, `V(B)=0`

## Q1

Question:

> Given a number of samples of state-reward sequences, what are the value functions of state A and state B estimated by Monte Carlo and TD approaches, respectively? (`alpha = 1`, `gamma = 1`)

### 1. Monte Carlo

The 8 episodes are:

1. `A-0-B-1-A-1-B-0`
2. `A-1-B-1-A-1-B-1`
3. `A-0-B-1-A-1`
4. `A-0-B-1-A-1-B-0`
5. `B-0-A-1-B-0`
6. `B-0-A-1-B-0`
7. `B-1-B-1`
8. `B-0`

Using first-visit Monte Carlo:

For `A`, use the return after the first visit to `A` in each episode:

- Ep1: `0+1+1+0 = 2`
- Ep2: `1+1+1+1 = 4`
- Ep3: `0+1+1 = 2`
- Ep4: `0+1+1+0 = 2`
- Ep5: `1+0 = 1`
- Ep6: `1+0 = 1`

So:

```text
V_MC(A) = (2+4+2+2+1+1)/6 = 2
```

For `B`, use the return after the first visit to `B` in each episode:

- Ep1: `1+1+0 = 2`
- Ep2: `1+1+1 = 3`
- Ep3: `1+1 = 2`
- Ep4: `1+1+0 = 2`
- Ep5: `0+1+0 = 1`
- Ep6: `0+1+0 = 1`
- Ep7: `1+1 = 2`
- Ep8: `0`

So:

```text
V_MC(B) = (2+3+2+2+1+1+2+0)/8 = 13/8 = 1.625
```

### 2. TD(0)

Use the online TD(0) update:

```text
V(s) <- r + gamma * V(s')
```

because `alpha = 1` and `gamma = 1`.

Initial values:

- `V(A)=0`
- `V(B)=0`

#### Ep1: `A-0-B-1-A-1-B-0`

- `A ->(0)-> B`
  - `V(A) = 0 + V(B) = 0`
- `B ->(1)-> A`
  - `V(B) = 1 + V(A) = 1`
- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 2`
- `B ->(0)-> terminal`
  - `V(B) = 0`

Now:

- `V(A)=2`
- `V(B)=0`

#### Ep2: `A-1-B-1-A-1-B-1`

- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 1`
- `B ->(1)-> A`
  - `V(B) = 1 + V(A) = 2`
- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 3`
- `B ->(1)-> terminal`
  - `V(B) = 1`

Now:

- `V(A)=3`
- `V(B)=1`

#### Ep3: `A-0-B-1-A-1`

- `A ->(0)-> B`
  - `V(A) = 0 + V(B) = 1`
- `B ->(1)-> A`
  - `V(B) = 1 + V(A) = 2`
- `A ->(1)-> terminal`
  - `V(A) = 1`

Now:

- `V(A)=1`
- `V(B)=2`

#### Ep4: `A-0-B-1-A-1-B-0`

- `A ->(0)-> B`
  - `V(A) = 0 + V(B) = 2`
- `B ->(1)-> A`
  - `V(B) = 1 + V(A) = 3`
- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 4`
- `B ->(0)-> terminal`
  - `V(B) = 0`

Now:

- `V(A)=4`
- `V(B)=0`

#### Ep5: `B-0-A-1-B-0`

- `B ->(0)-> A`
  - `V(B) = 0 + V(A) = 4`
- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 5`
- `B ->(0)-> terminal`
  - `V(B) = 0`

Now:

- `V(A)=5`
- `V(B)=0`

#### Ep6: `B-0-A-1-B-0`

- `B ->(0)-> A`
  - `V(B) = 0 + V(A) = 5`
- `A ->(1)-> B`
  - `V(A) = 1 + V(B) = 6`
- `B ->(0)-> terminal`
  - `V(B) = 0`

Now:

- `V(A)=6`
- `V(B)=0`

#### Ep7: `B-1-B-1`

- `B ->(1)-> B`
  - `V(B) = 1 + V(B) = 1`
- `B ->(1)-> terminal`
  - `V(B) = 1`

Now:

- `V(A)=6`
- `V(B)=1`

#### Ep8: `B-0`

- `B ->(0)-> terminal`
  - `V(B) = 0`

Final TD estimates:

- `V_TD(A) = 6`
- `V_TD(B) = 0`

### Q1 Final Answer

- Monte Carlo:
  - `V(A) = 2`
  - `V(B) = 1.625`
- TD:
  - `V(A) = 6`
  - `V(B) = 0`

## Q2

Question:

> Given the following environment configuration, please compute the final value functions of all states. (show the calculation of state 0 and state 2 as example; each step cost is `-0.1`)

### Bellman Optimality Equation

For each non-terminal state:

```text
V*(s) = -0.1 + max_a V*(s')
```

where `s'` is the next state after taking action `a`.

Terminal states:

- `V*(3) = 0`
- `V*(7) = -1`

Obstacle:

- state `4` is blocked

### Example 1: State 2

Possible next states from state `2`:

- Up: stay at `2`
- Down: go to `6`
- Left: go to `1`
- Right: go to terminal `3`

Therefore:

```text
V*(2) = -0.1 + max{V*(2), V*(6), V*(1), V*(3)}
      = -0.1 + max{-0.1, -0.2, -0.2, 0}
      = -0.1
```

### Example 2: State 0

Possible next states from state `0`:

- Up: stay at `0`
- Left: stay at `0`
- Down: blocked by obstacle `4`, so stay at `0`
- Right: go to `1`

Therefore:

```text
V*(0) = -0.1 + max{V*(0), V*(0), V*(0), V*(1)}
      = -0.1 + max{-0.3, -0.3, -0.3, -0.2}
      = -0.3
```

### Final State Values

```text
---------------------------------
|  0:-0.3 |  1:-0.2 |  2:-0.1 | 3: 0   |
---------------------------------
|  4:wall |  5:-0.3 |  6:-0.2 | 7:-1   |
---------------------------------
|  8:-0.5 |  9:-0.4 | 10:-0.3 | 11:-0.4|
---------------------------------
```

So the final answer for Q2 is:

- `V*(0) = -0.3`
- `V*(1) = -0.2`
- `V*(2) = -0.1`
- `V*(3) = 0`
- `V*(4) = wall / N.A.`
- `V*(5) = -0.3`
- `V*(6) = -0.2`
- `V*(7) = -1`
- `V*(8) = -0.5`
- `V*(9) = -0.4`
- `V*(10) = -0.3`
- `V*(11) = -0.4`

## Q3

Question:

> For the following environment configuration, initial value functions, and given policy, please evaluate the given policy of all states for 1 round. (show the calculation of state 0 and state 2 as example; each step cost is `-0.1`)

### Initial Values

From the figure:

```text
---------------------------------
|  0:-0.1 |  1:-0.1 |  2:-0.1 | 3: 0   |
---------------------------------
|  4:wall |  5:-0.1 |  6:-0.1 | 7:-1   |
---------------------------------
|  8:-0.1 |  9:-0.1 | 10:-0.1 | 11:-0.1|
---------------------------------
```

### Policy Used In The Figure

I interpret the arrows as a uniform stochastic policy over the shown actions:

- `pi(0) = {L, U, R}` with probability `1/3` each
- `pi(1) = {L, U, R, D}` with probability `1/4` each
- `pi(2) = {R}` with probability `1`
- `pi(5) = {U, R, D}` with probability `1/3` each
- `pi(6) = {L, U, D}` with probability `1/3` each
- `pi(8) = {L, R, D}` with probability `1/3` each
- `pi(9) = {L, U, R, D}` with probability `1/4` each
- `pi(10) = {L, U, R, D}` with probability `1/4` each
- `pi(11) = {L, R, D}` with probability `1/3` each

### One-Round Policy Evaluation

Use:

```text
V_1(s) = -0.1 + sum_a pi(a|s) * V_0(s')
```

where `s'` is the next state after action `a`.

### Example 1: State 0

From state `0`:

- `L` -> stay at `0`
- `U` -> stay at `0`
- `R` -> go to `1`

So:

```text
V_1(0) = -0.1 + (1/3)[V_0(0) + V_0(0) + V_0(1)]
       = -0.1 + (1/3)[-0.1 + -0.1 + -0.1]
       = -0.2
```

### Example 2: State 2

From state `2`, the policy chooses only `R`, which goes to terminal state `3`.

So:

```text
V_1(2) = -0.1 + V_0(3)
       = -0.1
```

### Final Values After 1 Round

```text
---------------------------------
|  0:-0.2 |  1:-0.2 |  2:-0.1 | 3: 0   |
---------------------------------
|  4:wall |  5:-0.2 |  6:-0.2 | 7:-1   |
---------------------------------
|  8:-0.2 |  9:-0.2 | 10:-0.2 | 11:-0.2|
---------------------------------
```

So the one-round policy evaluation result is:

- `V_1(0) = -0.2`
- `V_1(1) = -0.2`
- `V_1(2) = -0.1`
- `V_1(3) = 0`
- `V_1(4) = wall / N.A.`
- `V_1(5) = -0.2`
- `V_1(6) = -0.2`
- `V_1(7) = -1`
- `V_1(8) = -0.2`
- `V_1(9) = -0.2`
- `V_1(10) = -0.2`
- `V_1(11) = -0.2`

## Q4

Question:

> Please draw the system diagrams for SARSA and Q-learning respectively. Please compare the two methods (in terms of behavior and estimation policies, capabilities of exploration and exploitation, convergence speed, capability for optimal value estimation, capability for optimal policy estimation, etc.) for intelligent robot motion planning.

### 1. System Diagram Of SARSA

```text
state s
  |
  v
choose action a using behavior policy pi
  |
  v
observe reward r and next state s'
  |
  v
choose next action a' using the same policy pi
  |
  v
Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]
```

### 2. System Diagram Of Q-learning

```text
state s
  |
  v
choose action a using behavior policy
  |
  v
observe reward r and next state s'
  |
  v
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

### 3. Comparison Strictly Following The Question

| Aspect | SARSA | Q-learning |
|---|---|---|
| Behavior policy | `On-policy`; the behavior policy is the learning policy. | The behavior policy may still be exploratory. |
| Estimation policy | Updates with the actually selected next action `a'`. | Updates with `max_a Q(s',a)`. |
| Capability of exploration | Can explore, and the update reflects that exploration. | Can explore, but the update target stays greedy. |
| Capability of exploitation | More conservative. | More aggressive. |
| Convergence speed | Usually stable, but may be slower. | Often faster, but may fluctuate more. |
| Capability for optimal value estimation | Mainly estimates `Q^\pi` for the current policy. | Can directly approach `Q^*`. |
| Capability for optimal policy estimation | Reaches the optimal policy indirectly. | Reaches the optimal greedy policy more directly. |
| Meaning for intelligent robot motion planning | Safer and more conservative. | More direct in seeking the optimal path. |

### 4. Summary For Robot Motion Planning

- `SARSA`: safer and more conservative.
- `Q-learning`: more direct and more optimality-oriented.

## Q5

Question:

> For the problem in [3], what will be the final Q-table (input is a pair of state and action, output is Q value) for Q-learning? (use state 0 and state 2 as examples)

### Q-learning Target

For deterministic transitions:

```text
Q*(s,a) = -0.1 + V*(s')
```

where `s'` is the next state after taking action `a`, and `V*` is the optimal value function from Q2.

### Example 1: State 0

From Q2:

- `V*(0) = -0.3`
- `V*(1) = -0.2`

Possible actions from state `0`:

- `U`: stay at `0`
- `D`: blocked by obstacle `4`, stay at `0`
- `L`: stay at `0`
- `R`: go to `1`

So:

```text
Q*(0,U) = -0.1 + V*(0) = -0.4
Q*(0,D) = -0.1 + V*(0) = -0.4
Q*(0,L) = -0.1 + V*(0) = -0.4
Q*(0,R) = -0.1 + V*(1) = -0.3
```

### Example 2: State 2

From Q2:

- `V*(2) = -0.1`
- `V*(1) = -0.2`
- `V*(6) = -0.2`
- `V*(3) = 0`

Possible actions from state `2`:

- `U`: stay at `2`
- `D`: go to `6`
- `L`: go to `1`
- `R`: go to terminal `3`

So:

```text
Q*(2,U) = -0.1 + V*(2) = -0.2
Q*(2,D) = -0.1 + V*(6) = -0.3
Q*(2,L) = -0.1 + V*(1) = -0.3
Q*(2,R) = -0.1 + V*(3) = -0.1
```

### Final Q-table

| State | `Q(U)` | `Q(D)` | `Q(L)` | `Q(R)` |
|---|---:|---:|---:|---:|
| 0 | -0.4 | -0.4 | -0.4 | -0.3 |
| 1 | -0.3 | -0.4 | -0.4 | -0.2 |
| 2 | -0.2 | -0.3 | -0.3 | -0.1 |
| 5 | -0.3 | -0.5 | -0.4 | -0.3 |
| 6 | -0.2 | -0.4 | -0.4 | -1.1 |
| 8 | -0.6 | -0.6 | -0.6 | -0.5 |
| 9 | -0.4 | -0.5 | -0.6 | -0.4 |
| 10 | -0.3 | -0.4 | -0.5 | -0.5 |
| 11 | -1.1 | -0.5 | -0.4 | -0.5 |

For the special states:

- state `3`: terminal
- state `4`: obstacle
- state `7`: terminal

### Greedy Policy From The Q-table

- `0 -> R`
- `1 -> R`
- `2 -> R`
- `5 -> U` or `R`
- `6 -> U`
- `8 -> R`
- `9 -> U` or `R`
- `10 -> U`
- `11 -> L`

## Final Short Answers

### Q1

- Monte Carlo:
  - `V(A)=2`
  - `V(B)=1.625`
- TD:
  - `V(A)=6`
  - `V(B)=0`

### Q2

```text
0:-0.3, 1:-0.2, 2:-0.1, 3:0
4:wall, 5:-0.3, 6:-0.2, 7:-1
8:-0.5, 9:-0.4, 10:-0.3, 11:-0.4
```

### Q3

```text
0:-0.2, 1:-0.2, 2:-0.1, 3:0
4:wall, 5:-0.2, 6:-0.2, 7:-1
8:-0.2, 9:-0.2, 10:-0.2, 11:-0.2
```

### Q4

- `SARSA`: on-policy, safer, more conservative.
- `Q-learning`: off-policy, more direct and optimality-oriented.

### Q5

- state `0`: `[-0.4, -0.4, -0.4, -0.3]` for `[U,D,L,R]`
- state `2`: `[-0.2, -0.3, -0.3, -0.1]` for `[U,D,L,R]`
