import os
import re
import math
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from models.dataset import load_K_Rt_from_P

def load_gt_poses_and_scale(cameras_file):
    """
    Loads ground-truth camera→world poses *and* the normalization matrix
    so we can bring meshes and poses into the same normalized space.
    """
    data = np.load(cameras_file)
    re_world = re.compile(r"^world_mat_(\d+)$")
    re_scale = re.compile(r"^scale_mat_inv_(\d+)$")

    # collect & sort world_mats
    world_keys = sorted([k for k in data if re_world.match(k)],
                        key=lambda k: int(re_world.match(k).group(1)))
    # collect & sort their matching scale_mat_inv
    scale_keys = sorted([k for k in data if re_scale.match(k)],
                        key=lambda k: int(re_scale.match(k).group(1)))

    if len(world_keys) != len(scale_keys):
        raise RuntimeError("Mismatch between world_mat and scale_mat_inv counts")

    poses = []
    # we'll also grab the first scale_inv to use on meshes
    scale_inv = None

    for wk, sk in zip(world_keys, scale_keys):
        # 1) extract the world→camera P
        W = data[wk]               # 4×4
        P = W[:3, :4]              # top 3×4
        _, pose_wc = load_K_Rt_from_P(None, P)

        # make it a full 4×4
        if pose_wc.shape == (3, 4):
            tmp = np.eye(4, dtype=pose_wc.dtype)
            tmp[:3, :4] = pose_wc
            pose_wc = tmp

        # 2) invert → get camera→world in *original* coords
        cam2world = np.linalg.inv(pose_wc)

        # 3) normalize via scale_mat_inv
        S_inv = data[sk]  # 4×4
        cam2world_norm = S_inv @ cam2world

        poses.append(cam2world_norm)
        if scale_inv is None:
            scale_inv = S_inv

    return np.stack(poses, axis=0), scale_inv

def compute_pose_errors(est, gt):
    t_err = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1)
    r_err = []
    for Re, Rg in zip(est[:, :3, :3], gt[:, :3, :3]):
        Rdiff = Re @ Rg.T
        angle = math.acos(np.clip((np.trace(Rdiff) - 1) / 2, -1, 1))
        r_err.append(math.degrees(angle))
    return t_err, np.array(r_err)

def umeyama(src, dst, with_scaling=True):
    m = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    cov = cd.T @ cs / m
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / ((cs**2).sum() / m)) if with_scaling else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

def evaluate_meshes(gt_mesh, rec_mesh, scale_inv, s, R, t, K=100000, thresh=0.64):
    # 1) bring both meshes into normalized space
    def normalize(m, S_inv):
        v = np.asarray(m.vertices)
        v_n = (S_inv[:3, :3] @ v.T).T + S_inv[:3, 3]
        m_norm = o3d.geometry.TriangleMesh(m)
        m_norm.vertices = o3d.utility.Vector3dVector(v_n)
        return m_norm

    gt_n   = normalize(gt_mesh,  scale_inv)
    rec_n0 = normalize(rec_mesh, scale_inv)

    # 2) apply the global similarity to the reconstructed one
    rec = o3d.geometry.TriangleMesh(rec_n0)
    verts = np.asarray(rec.vertices)
    verts_aligned = (s * (R @ verts.T)).T + t
    rec.vertices = o3d.utility.Vector3dVector(verts_aligned)

    # 3) scale by 10× (per paper) and sample
    gt_n.scale(10.0, center=(0,0,0))
    rec.scale(10.0, center=(0,0,0))
    pcd_gt  = gt_n.sample_points_uniformly(K)
    pcd_rec = rec.sample_points_uniformly(K)

    pts_gt  = np.asarray(pcd_gt.points)
    pts_rec = np.asarray(pcd_rec.points)

    tree_gt  = cKDTree(pts_gt)
    tree_rec = cKDTree(pts_rec)

    d_r2g, _ = tree_gt.query(pts_rec, k=1, p=1)
    d_g2r, _ = tree_rec.query(pts_gt, k=1, p=1)

    acc = d_r2g.mean()
    com = d_g2r.mean()
    cd  = 0.5 * (acc + com)

    pre = (d_r2g < thresh).mean()
    rec_ = (d_g2r < thresh).mean()
    f   = (2 * pre * rec_ / (pre + rec_)) if (pre+rec_)>0 else 0.0

    return acc, com, cd, pre, rec_, f

if __name__ == "__main__":
    exp_dir = "exp/farmer"
    data_dir = "data/farmer"

    # load estimated & GT poses
    est_poses = np.load(os.path.join(exp_dir, "cam_poses/pose_150000.npy"))
    gt_poses, scale_inv = load_gt_poses_and_scale(os.path.join(data_dir, "cameras_sphere.npz"))

    if est_poses.shape != gt_poses.shape:
        raise RuntimeError(f"Mismatch: est {est_poses.shape}, gt {gt_poses.shape}")

    # pose‐based filtering
    t_err, r_err = compute_pose_errors(est_poses, gt_poses)
    inliers = np.where((t_err <= 0.20) & (r_err <= 20.0))[0]
    print(f"Total views = {len(t_err)}, Inliers (≤20 cm & ≤20°) = {len(inliers)}\n")

    # camera centers for alignment
    ctr_est = est_poses[:, :3, 3]
    ctr_gt  = gt_poses[:,  :3, 3]

    # load meshes
    gt_mesh   = o3d.io.read_triangle_mesh(os.path.join(data_dir,    "farmer.ply"))
    rec_mesh  = o3d.io.read_triangle_mesh(os.path.join(exp_dir,     "meshes/00150000.ply"))

    # all‐views alignment + eval
    s_all, R_all, t_all = umeyama(ctr_est, ctr_gt)
    metrics_all = evaluate_meshes(gt_mesh, rec_mesh, scale_inv, s_all, R_all, t_all)

    print("=== Metrics: ALL views ===")
    print(f"Chamfer Dist = {metrics_all[2]:.6f} (Acc {metrics_all[0]:.6f}, Com {metrics_all[1]:.6f})")
    print(f"Prec = {metrics_all[3]:.6f}, Rec = {metrics_all[4]:.6f}, F-score = {metrics_all[5]:.6f}\n")

    # inliers‐only (if ≥3)
    if len(inliers) >= 3:
        s_i, R_i, t_i = umeyama(ctr_est[inliers], ctr_gt[inliers])
        metrics_in = evaluate_meshes(gt_mesh, rec_mesh, scale_inv, s_i, R_i, t_i)
        print("=== Metrics: INLIER views only ===")
        print(f"Chamfer Dist = {metrics_in[2]:.6f} (Acc {metrics_in[0]:.6f}, Com {metrics_in[1]:.6f})")
        print(f"Prec = {metrics_in[3]:.6f}, Rec = {metrics_in[4]:.6f}, F-score = {metrics_in[5]:.6f}")
    else:
        print(f"Not enough inliers ({len(inliers)}) to compute filtered metrics.")
