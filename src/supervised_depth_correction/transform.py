from __future__ import absolute_import, division, print_function
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
)
import torch

__all__ = [
    'matrix_to_xyz_axis_angle',
    'xyz_axis_angle_to_matrix',
    'delta_transform',
    'rotation_angle',
    'translation_norm',
]

# def skew_matrix(x: torch.Tensor, dim=-1):
#     assert isinstance(x, torch.Tensor)
#     assert x.shape[-1] == 3
#
#     # [0, x[..., 2], x[..., 1]]
#     # torch.index_select(x, )
#     # x.index_select(dim=-1)
#     # torch.linalg.skw
#     shape = list(x.shape)
#     shape[dim] = 9
#     m = torch.zeros(shape, dtype=x.dtype, device=x.device)
#     # y.index_select(dim=dim, [1, 2, 3, 5, 6, 7]) = x.index_select(dim=dim, )
#     i = x.index_select(dim=dim, 0)
#     j = x.index_select(dim=dim, 1)
#     k = x.index_select(dim=dim, 2)
#     y.index_select(dim=dim, 1) = -x.index_select(dim=dim, 2)
#     y = y.reshape(shape[:dim] + [3, 3] + shape[dim + 1:])
#     return y
#
#
# def axis_angle_xyz_to_matrix(x: torch.Tensor):
#     assert isinstance(x, torch.Tensor)
#     assert x.shape[-1] == 3
#     angle = torch.linalg.norm(x, dim=-1, keepdim=True)
#     return y


# def xyz_axis_angle_to_pose_msg(xyz_axis_angle):
#     # assert isinstance(xyz_axis_angle, torch.Tensor)
#     # assert isinstance(xyz_axis_angle, list)
#     q = axis_angle_to_quaternion(xyz_axis_angle[3:])
#     msg = Pose(Point(*xyz_axis_angle[:3]), Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))
#     return msg


# # @timing
# def xyz_axis_angle_to_path_msg(xyz_axis_angle, frame_id, stamp):
#     assert isinstance(xyz_axis_angle, torch.Tensor)
#     assert xyz_axis_angle.dim() == 2
#     assert xyz_axis_angle.shape[-1] == 6
#     xyz_axis_angle = xyz_axis_angle.detach()
#     msg = Path()
#     msg.poses = [PoseStamped(Header(), xyz_axis_angle_to_pose_msg(p)) for p in xyz_axis_angle]
#     msg.header.frame_id = frame_id
#     msg.header.stamp = stamp
#     return msg


def xyz_axis_angle_to_matrix(xyz_axis_angle):
    assert isinstance(xyz_axis_angle, torch.Tensor)
    assert xyz_axis_angle.shape[-1] == 6

    mat = torch.zeros(xyz_axis_angle.shape[:-1] + (4, 4), dtype=xyz_axis_angle.dtype, device=xyz_axis_angle.device)
    mat[..., :3, :3] = axis_angle_to_matrix(xyz_axis_angle[..., 3:])
    mat[..., :3, 3] = xyz_axis_angle[..., :3]
    mat[..., 3, 3] = 1.
    assert mat.shape == xyz_axis_angle.shape[:-1] + (4, 4)
    # assert mat.shape[-2:] == (4, 4)
    return mat


def matrix_to_xyz_axis_angle(T):
    assert isinstance(T, torch.Tensor)
    assert T.dim() == 3
    assert T.shape[1:] == (4, 4)
    n_poses = len(T)
    q = matrix_to_quaternion(T[:, :3, :3])
    axis_angle = quaternion_to_axis_angle(q)
    xyz = T[:, :3, 3]
    poses = torch.concat([xyz, axis_angle], dim=1)
    assert poses.shape == (n_poses, 6)
    return poses


def rotation_angle(T):
    R = T[:-1, :-1]
    angle = torch.arccos((torch.trace(R) - 1.0) / 2.0).item()
    return angle


def translation_norm(T):
    t = T[:-1, -1:]
    norm = torch.linalg.norm(t).item()
    return norm


def transform_inv(T):
    T_inv = torch.eye(T.shape[0])
    R = T[:-1, :-1]
    t = T[:-1, -1:]
    T_inv[:-1, :-1] = R.T
    T_inv[:-1, -1:] = -R.T @ t
    return T


def delta_transform(T_0, T_1):
    """Delta transform D s.t. T_1 = T_0 * D."""
    delta = torch.linalg.solve(T_0, T_1)
    # delta = transform_inv(T_0) @ T_1
    return delta
