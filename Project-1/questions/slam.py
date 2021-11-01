import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from helpers import draw_three


def transform(x, y, th, dx, dy, dth):
    return jnp.array([
        x + dx * jnp.cos(th) - dy * jnp.sin(th),
        y + dy * jnp.cos(th) + dx * jnp.sin(th),
        th + dth,
    ])


def calc_init_poses(fix_pose, edges):
    poses = [fix_pose]
    x, y, th = fix_pose

    for i1, i2, dx, dy, dth in edges:
        if abs(i1 - i2) != 1:
            continue
        x, y, th = transform(x, y, th, dx, dy, dth)
        poses.append([x, y, th])

    return jnp.array(poses)


def get_residual(poses, edges, fixed):
    resi = list(poses[0] - fixed)

    for i1, i2, dx, dy, dth in edges:
        i1 = int(i1)
        i2 = int(i2)
        pose2 = transform(*poses[i1], dx, dy, dth)
        resi += [*(pose2 - poses[i2])]

    return jnp.array(resi)


X_SIZE = 3  # count(x,y,th)
T_SIZE = 3  # 2D T


def get_my_jacob(poses, edges, fixed):
    edge_cnt = edges.shape[0]
    poses_cnt = poses.shape[0]

    contraints_cnt = (edge_cnt+1)*X_SIZE
    vars_cnt = poses_cnt*X_SIZE

    J = jnp.zeros((contraints_cnt, vars_cnt))
    J = J.at[:X_SIZE, : X_SIZE].set(jnp.eye(X_SIZE))

    for i, (i1, i2, dx, dy, dth) in enumerate(edges):
        i1 = int(i1)
        i2 = int(i2)
        ## wrt i1
        # dr[i]/d(p[i1].x)
        J = J.at[(i+1)*X_SIZE, i1*X_SIZE].set(1)

        # dr[i]/d(p[i1].y)
        J = J.at[(i+1)*X_SIZE+1, i1*X_SIZE + 1].set(1)

        # dr[i]/d(p[i1].th)
        J = J.at[(i+1)*X_SIZE, i1*X_SIZE + 2].set(
            -(dx * jnp.sin(poses[i1][2]) + dy * jnp.cos(poses[i1][2])))
        J = J.at[(i+1)*X_SIZE+1, i1*X_SIZE + 2].set(
            (dx * jnp.cos(poses[i1][2]) - dy * jnp.sin(poses[i1][2])))
        J = J.at[(i+1)*X_SIZE+2, i1*X_SIZE + 2].set(1)

        ## wrt i2
        J = J.at[(i+1)*X_SIZE: (i+1)*X_SIZE + 3, i2 *
                 X_SIZE + 0: i2*X_SIZE + 3].set(-jnp.eye(3))

    return J



jax_jacob_helper = jax.jacfwd(get_residual)
def get_jax_jacob(poses, edges, fixed):
    return jax_jacob_helper(poses, edges, fixed).reshape(-1, poses.shape[0]*X_SIZE)


class LM:
    def __init__(self, init_poses, edges, fixed, lmda, max_itrs, tol):
        self.lmda = lmda
        self.max_itrs = max_itrs
        self.tol = tol
        self.init_poses = init_poses
        self.edges = edges
        self.fixed = fixed
        edge_cnt = edges.shape[0]
        poses_cnt = init_poses.shape[0]

        self.contraints_cnt = (edge_cnt+1) * X_SIZE
        self.vars_cnt = poses_cnt * X_SIZE

        info_vals = [2000]*X_SIZE                        # for fixed
        for edge in edges:
            i1 = int(edge[0])
            i2 = int(edge[1])
            if i2 - i1 == 1:
                info_vals += [100]*X_SIZE               # odo
            else:
                info_vals += [1000]*X_SIZE               # loop closure

        info_mat = np.diag(info_vals)
        self.omega = jnp.array(info_mat)

    def get_residual(self, poses):
        return get_residual(poses, self.edges, self.fixed)

    def get_jacobian(self, poses):
        return get_jax_jacob(poses, self.edges, self.fixed)

    def get_error(self, poses):
        f = self.get_residual(poses)
        return (f.T @ self.omega @ f)/2

    def optimize(self):
        max_itrs = self.max_itrs
        tol = self.tol
        poses = self.init_poses
        error = self.get_error(poses)
        poses_arr = [poses]
        error_arr = [error]
        itr = 0

        while error > tol and itr < max_itrs:
            print(f"Iteration {itr}: Error: {error}")
            if itr % 10 == 0:
                draw_three(poses_arr)
            poses = self.update_poses(poses)
            error = self.get_error(poses)

            # if error > error_arr[-1]:
            #     self.lmda *= 2
            # else:
            #     self.lmda /= 3

            poses_arr.append(poses)
            error_arr.append(error)
            itr += 1

        print("Final Error: ", error)
        draw_three(poses_arr)
        return poses_arr, error_arr

    def update_poses(self, poses):
        f = self.get_residual(poses)
        j = self.get_jacobian(poses)
        I = jnp.eye(j.shape[1])

        delta = -jnp.linalg.pinv(j.T @ self.omega @ j + self.lmda * I) @ (
            j.T @ self.omega.T @ f)

        new_poses = poses + delta.reshape(poses.shape)

        return new_poses
