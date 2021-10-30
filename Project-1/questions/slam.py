import jax
import jax.numpy as jnp


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
