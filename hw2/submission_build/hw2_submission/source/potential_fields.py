import numpy as np

class potential_fields:

    def __init__(self, eps=1e-6, max_norm=2.0, repulsive_distance=2.0):
        self.eps = eps
        self.max_norm = max_norm
        self.repulsive_distance = repulsive_distance

    # Please complete these functions for question2, the arguments such as coefficient can be changed by your need. The return value should be a 2*1 matrix for robot to perform

    def uniform(self, vector=np.array([[1.0], [0.0]]), coefficient=1.0):
        return coefficient * self._safe_unit(vector)

    def perpendicular(self, line_obstacle, car_position, coefficient=1.2):
        # line_obstacle: [point1, point2]; point: 2*1 matrix
        start_point = self._as_col(line_obstacle[0])
        end_point = self._as_col(line_obstacle[1])
        car_position = self._as_col(car_position)
        _, projection_point, _ = self.shortest_distance_point(start_point, end_point, car_position)

        return coefficient * self._safe_unit(car_position - projection_point)

    def attractive(self, goal_point, car_position, coefficient=0.6):
        goal_point = self._as_col(goal_point)
        car_position = self._as_col(car_position)
        force = coefficient * (goal_point - car_position)

        return self._clip_norm(force)

    def repulsive(self, obstacle_point, car_position, coefficient=1.0):
        obstacle_point = self._as_col(obstacle_point)
        car_position = self._as_col(car_position)

        return self._repulsive_force(
            obstacle_point,
            car_position,
            coefficient=coefficient,
            influence_distance=self.repulsive_distance,
        )

    def tangential(self, point, car_position, coefficient=1.0):
        point = self._as_col(point)
        car_position = self._as_col(car_position)
        radial_direction = self._safe_unit(car_position - point)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

        return coefficient * (rotation @ radial_direction)

    def shortest_distance_point(self, v, w, p):
        # the minimum distance between line segment vw, and point p
        # v, w, p all are 2*1 matrix
        v = self._as_col(v)
        w = self._as_col(w)
        p = self._as_col(p)

        l2 = float(((w - v).T @ (w - v))[0, 0])
        if l2 <= self.eps:
            return float(np.linalg.norm(p - v)), v.copy(), 0.0

        t = float(np.clip(((p - v).T @ (w - v))[0, 0] / l2, 0.0, 1.0))
        proj_point = v + t * (w - v)
        min_distance = float(np.linalg.norm(p - proj_point))

        return min_distance, proj_point, t

    def _as_col(self, vector):
        vector = np.asarray(vector, dtype=float)
        return vector.reshape(2, 1)

    def _safe_unit(self, vector):
        vector = self._as_col(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= self.eps:
            return np.zeros((2, 1))

        return vector / norm

    def _clip_norm(self, vector):
        vector = self._as_col(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= self.max_norm:
            return vector
        if norm <= self.eps:
            return np.zeros((2, 1))

        return vector * (self.max_norm / norm)

    def _repulsive_force(self, obstacle_point, car_position, coefficient, influence_distance, fallback_direction=None):
        obstacle_point = self._as_col(obstacle_point)
        car_position = self._as_col(car_position)
        delta = car_position - obstacle_point
        distance = float(np.linalg.norm(delta))

        if distance >= influence_distance:
            return np.zeros((2, 1))

        if distance <= self.eps:
            direction = self._safe_unit(fallback_direction) if fallback_direction is not None else np.zeros((2, 1))
            safe_distance = self.eps
        else:
            direction = delta / distance
            safe_distance = distance

        magnitude = coefficient * (1.0 / safe_distance - 1.0 / influence_distance) / (safe_distance ** 2)

        return self._clip_norm(magnitude * direction)

