import numpy as np


def get_random_points(I, alpha):

    # -----fill in your implementation here --------

    x = np.random.randint(I.shape[0], size=(alpha, 1))
    y = np.random.randint(I.shape[1], size=(alpha, 1))
    points = np.concatenate((x, y), axis=1)

    # ----------------------------------------------

    return points

#if __name__ == '__main__':
    print(points)

