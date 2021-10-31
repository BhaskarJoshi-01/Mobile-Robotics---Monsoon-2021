from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import jax.numpy as jnp


def draw_one(X, Y, THETA):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    plt.plot(X, Y, 'c-')

    for i in range(len(THETA)):
        x2 = 0.25*np.cos(THETA[i]) + X[i]
        y2 = 0.25*np.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], 'g->')

    plt.show()


def draw_three(poses_arr, gt_file='../data/gt.txt'):
    init_x, init_y, init_th = poses_arr[0].T
    cur_x, cur_y, cur_th = poses_arr[-1].T

    # ground truth
    gt_nodes, gt_edges = read_data(gt_file)

    ax = plt.subplot(111)
    ax.plot(gt_nodes[0], gt_nodes[1], 'ro')
    plt.plot(gt_nodes[0], gt_nodes[1], 'c-')

    for x, y, th in gt_nodes.T:
        x2 = 0.25*np.cos(th) + x
        y2 = 0.25*np.sin(th) + y
        plt.plot([x, x2], [y, y2], 'b->')

    # init estimate
    for x, y, th in zip(init_x, init_y, init_th):
        x2 = 0.25*np.cos(th) + x
        y2 = 0.25*np.sin(th) + y
        plt.plot([x, x2], [y, y2], 'r->')

    # cur_estimate
    ax.plot(cur_x, cur_y, 'ro')
    ax.plot(cur_x, cur_y, 'c-')
    for x, y, th in zip(cur_x, cur_y, cur_th):
        x2 = 0.25*np.cos(th) + x
        y2 = 0.25*np.sin(th) + y
        plt.plot([x, x2], [y, y2], 'g->')

    plt.legend(handles=[
        mpatches.Patch(label='Ground Truth', color='blue'),
        mpatches.Patch(label='Initial Estimate', color='red'),
        mpatches.Patch(label='Optimised Trajectory', color='green')
    ])

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
