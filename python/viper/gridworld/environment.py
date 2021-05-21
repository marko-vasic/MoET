import random

SIZE = 5
DIAGONAL_SUM = SIZE - 1
NUM_ACTIONS = 4
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3
MAX_STEPS = 100
REWARD = -0.1


class GridworldEnvironment(object):
    x = None
    y = None
    step_id = 0

    def __init__(self):
        pass

    def reset(self):
        self.step_id = 0
        while True:
            self.x = random.randint(0, SIZE - 1)
            self.y = random.randint(0, SIZE - 1)
            if (SIZE - 1 - self.x) != self.y:
                # Do not allow start state on diagonal.
                break

        return [self.x, self.y]

    @staticmethod
    def coordinate_in_bounds(coordinate):
        return 0 <= coordinate < SIZE

    @staticmethod
    def coordinates_on_diagonal(x, y):
        return (x + y) == DIAGONAL_SUM

    def reached_finish(self):
        return (GridworldEnvironment.coordinate_in_bounds(self.y)
                and not GridworldEnvironment.coordinate_in_bounds(self.x))

    def step(self, action):
        self.step_id += 1
        new_x = self.x
        new_y = self.y
        if action == ACTION_LEFT:
            new_x = self.x - 1
        elif action == ACTION_RIGHT:
            new_x = self.x + 1
        elif action == ACTION_UP:
            new_y = self.y + 1
        elif action == ACTION_DOWN:
            new_y = self.y - 1

        if (not GridworldEnvironment.coordinates_on_diagonal(new_x, new_y)
            and GridworldEnvironment.coordinate_in_bounds(new_y)):
            self.x = new_x
            self.y = new_y
        else:
            # Hit the wall or exceeded vertically so stays in the same place.
            pass

        done = self.reached_finish() or self.step_id > MAX_STEPS
        return [self.x, self.y], REWARD, done, None
