from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp


def draw_one(X, Y, THETA, show=True):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    plt.plot(X, Y, 'c-')

    for i in range(len(THETA)):
        x2 = 0.25*np.cos(THETA[i]) + X[i]
        y2 = 0.25*np.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], 'g->')

    if show:
        plt.show()


def read_data(filename):
    with open(filename) as f:
        data = f.readlines()

    xs = []
    ys = []
    ths = []

    idx1s = []
    idx2s = []
    dxs = []
    dys = []
    dths = []

    for line in data:
        line = line[:-1].split()
        if "VERTEX_SE2" == line[0]:
            ind, x, y, th = list(map(float, line[1:]))
            assert len(xs) == ind
            xs.append(x)
            ys.append(y)
            ths.append(th)
        elif "EDGE_SE2" in line:
            idx1, idx2, dx, dy, dth = list(map(float, line[1:6]))
            idx1s.append(idx1)
            idx2s.append(idx2)
            dxs.append(dx)
            dys.append(dy)
            dths.append(dth)

    nodes = jnp.array([xs, ys, ths])
    edges = jnp.array([idx1s, idx2s, dxs, dys, dths])

    return nodes, edges.T


def write_edges_poses(new_file_name, old_file_name, poses):
    data = []

    for i, (x, y, th) in enumerate(poses):
        data.append(f"VERTEX_SE2 {i} {x} {y} {th}\n")

    with open(old_file_name) as f:
        data += [line for line in f.readlines() if "VERTEX_SE2" not in line]

    with open(new_file_name, 'w+') as f:
        f.writelines(data)


def frobNorm(P1, P2, str1="mat1", str2="mat2"):
    jnp.set_printoptions(suppress=True)
    val = jnp.linalg.norm(P1 - P2, 'fro')
    print(f"Frobenius norm between {str1} and {str2} is: {val}")
