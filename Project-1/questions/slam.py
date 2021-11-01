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

# def jacobian(current_poses, edges):
#     num_constraints = (edges[0].shape[0] * NUM_VARS) + NUM_VARS # for anchor edges
#     num_variables = current_poses.shape[0] * NUM_VARS

#     # constraints are ordered as in `edges`
#     # variables are ordered as in `current_poses`

#     J = jnp.zeros((num_constraints, num_variables))

#     # for anchor edges
#     J = jax.ops.index_update(J, jax.ops.index[:NUM_VARS, :NUM_VARS], jnp.eye(NUM_VARS))

#     # for other edges
#     for i in range(edges[0].shape[0]):
#         ind1 = edges[0][i]
#         ind2 = edges[1][i]
#         del_x = edges[2][i]
#         del_y = edges[3][i]
#         del_theta = edges[4][i]

#         J = jax.ops.index_update(
#                 J,
#                 jax.ops.index[NUM_VARS + i * NUM_VARS:NUM_VARS + i * NUM_VARS + NUM_VARS, ind1 * NUM_VARS: ind1 * NUM_VARS + NUM_VARS],
#                 derivative_with_initial(*poses[ind1], *poses[ind2], del_x, del_y, del_theta)
#             )

#         J = jax.ops.index_update(
#                 J,
#                 jax.ops.index[NUM_VARS + i * NUM_VARS:NUM_VARS + i * NUM_VARS + NUM_VARS, ind2 * NUM_VARS: ind2 * NUM_VARS + NUM_VARS],
#                 derivative_with_final(*poses[ind1], *poses[ind2], del_x, del_y, del_theta)
#             )

#     return J


class LM:
    def __init__(self, init_poses, edges, fixed, lmda, max_itrs, tol, weight_ratios, odo_wt=10):
        self.lmda = lmda
        self.max_itrs = max_itrs
        self.tol = tol
        self.init_poses = init_poses
        self.edges = edges + 0.
        self.fixed = fixed + 0.
        edge_cnt = edges.shape[0]
        poses_cnt = init_poses.shape[0]

        self.contraints_cnt = (edge_cnt+1) * X_SIZE
        self.vars_cnt = poses_cnt * X_SIZE

        # info mat
        odo_wt = odo_wt
        loop_wt = odo_wt * weight_ratios[0]
        fixed_wt = odo_wt * weight_ratios[1]
        print(f"Using Weights: ({odo_wt}, {loop_wt}, {fixed_wt})\n")

        info_vals = [fixed_wt]*X_SIZE                       # for fixed
        for edge in edges:
            i1, i2 = edge[0:2].astype(int)

            if i2 - i1 == 1:
                info_vals += [odo_wt]*X_SIZE                # odo
            else:
                info_vals += [loop_wt]*X_SIZE               # loop closure

        info_mat = np.diag(info_vals)
        self.omega = jnp.array(info_mat)

    def get_residual(self, poses):
        return get_residual(poses, self.edges, self.fixed)

    def get_jacobian(self, poses):
        return get_my_jacob(self.init_poses, self.edges, self.fixed)

    def get_error(self, poses):
        f = self.get_residual(poses)
        return (f.T @ self.omega @ f)/2

    def optimize(self, verbose=True):
        max_itrs = self.max_itrs
        tol = self.tol
        poses = self.init_poses
        error = self.get_error(poses)
        poses_arr = [poses]
        error_arr = [error]
        itr = 0

        while itr < max_itrs:
            if verbose:
                print(f"Iteration {itr}: Error: {error}")
                if itr % 10 == 0:
                    draw_three(poses_arr)
            poses = self.update_poses(poses)
            error = self.get_error(poses)

            if error > error_arr[-1]:
                self.lmda *= 2
            else:
                self.lmda /= 3

            poses_arr.append(poses)
            error_arr.append(error)
            itr += 1
            if jnp.linalg.norm(poses_arr[-1] - poses_arr[-2]) < tol:
                break

        if verbose:
            print(f"Final Error: {error} at itr: {itr}")
            draw_three(poses_arr)
        return poses_arr, error_arr

    def update_poses(self, poses):
        f = self.get_residual(poses)
        j = self.get_jacobian(poses)
        H = j.T @ self.omega @ j
        delta = -jnp.linalg.inv(H + self.lmda * jnp.eye(j.shape[1])) @ \
            j.T @ self.omega.T @ f

        new_poses = poses + delta.reshape(poses.shape)
        # from hashlib import md5 as m
        # x = (H + self.lmda * jnp.eye(j.shape[1]))
        # print(self.lmda)
        # print(m(f.tobytes()).hexdigest())
        # print(m(j.tobytes()).hexdigest())
        # plt.style.use('seaborn')
        # plt.imshow(j)
        # plt.show()
        # print(m(self.omega.tobytes()).hexdigest())
        # print(m(x.tobytes()).hexdigest())
        # print(m(delta.tobytes()).hexdigest())
        # 1/0
        return new_poses
