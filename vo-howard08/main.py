from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Init plots
fig_birdeye = plt.figure("BirdEye View")
ax_birdeye = fig_birdeye.add_subplot()
ax_birdeye.set_xlabel("x(m)")
ax_birdeye.set_ylabel("z(m)")

fig_traj = plt.figure("Trajectory")
ax_traj = fig_traj.add_subplot()
ax_traj.set_xlabel("x(m)")
ax_traj.set_ylabel("z(m)")


def imfuse(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    img1, img2: gray img.
    matlab 스타일의 red-cyan blending
    R값은 img1 value, GB값은 img2 value 가진다.
    결과적으로 img1과 img2의 value가 같은 지점이면 gray가 된다.
    """
    assert img1.ndim == 2 and img2.ndim == 2
    assert img1.shape == img2.shape

    # BGR order return
    return np.stack([img2, img2, img1], axis=-1)


def read_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    with open(calib_path) as fp:
        lines = fp.read().splitlines()

    P_dict = {}
    for line in lines:
        key = line.split()[0].rstrip(":")
        vals = line.split()[1:]
        P_dict[key] = np.array([float(x) for x in vals]).reshape([3, 4])

    return P_dict


def homo_tf(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    A: (?, D)
    x: (N, D-1)
    """
    assert A.ndim == 2 and x.ndim == 2
    assert A.shape[1] == x.shape[1] + 1
    assert 1 < A.shape[0] <= A.shape[1]

    x = x.T
    homo_x = np.vstack([x, np.ones(x.shape[1])])
    homo_y = A @ homo_x
    y = homo_y[:-1, ...] / homo_y[-1, ...]
    y = y.T

    return y


def find_potential_nodes(clique: List[int], M: np.array) -> List[int]:
    """
    Find the set of matches compatible with all the matches already in the clique
    """

    new_set = np.all(M[:, clique], axis=1)

    for c in clique:
        new_set[c] = False

    new_set = new_set.nonzero()[0]

    return new_set.tolist()


def update_clique(potential_nodes, clique, M):
    """
    Add the match with the largeest number consistent matches
    """

    num_matches = M[np.ix_(potential_nodes, potential_nodes)].sum(axis=1)
    max_match_node = potential_nodes[np.argmax(num_matches)]
    clique.append(max_match_node)

    return clique


def obj_fun(param, F1, F2, W1, W2, K):
    """
    param(r, t) -> 6x1 vectors, need to be varied for the minimization
    F1, F2: 2d coordinates of features in imgL_prev, imgL_next
    W1, W2: 3d coordinates of the features that have been triangulated
    K: Camera intrinsic
    """

    r = param[:3]
    t = param[3:]

    rot_mat = R.from_euler("ZXZ", r).as_matrix()
    RT = np.zeros([4, 4])
    RT[:3, :3] = rot_mat
    RT[:3, 3] = t
    RT[3, 3] = 1

    F1_reprojected = homo_tf(K @ RT[:3], W2)
    F2_reprojected = homo_tf(K @ np.linalg.inv(RT)[:3], W1)

    reproj1 = F1 - F1_reprojected
    reproj2 = F2 - F2_reprojected

    return np.concatenate([reproj1.flatten(), reproj2.flatten()])


def bucket_features(
    image: np.ndarray,
    vert_splits: int,
    horz_splits: int,
    each_num_corners: int,
) -> np.ndarray:
    """

    이미지를 균등하게 분할하여 각 영역마다 fast features를 뽑고,
    각 영역에서 response가 큰 each_num_corners 갯수의 픽셀 좌표 모아서 반환.

    Args:
        image: input gray image
        vert_splits: vertically 분할 갯수
        horz_splits: horizontally 분할 갯수
        each_num_corners: 각 영역에서 추출할 feature 갯수

    Returns:
        추출된 feature들의 좌표값 모음 of shape (num_features, 2)
    """

    H, W = image.shape

    ys = np.around(np.linspace(0, H + 1, num=vert_splits + 1)).astype(int)
    xs = np.around(np.linspace(0, W + 1, num=horz_splits + 1)).astype(int)

    fast = cv2.FastFeatureDetector_create()

    final_pts = []
    for y_begin, y_end in zip(ys[:-1], ys[1:]):
        for x_begin, x_end in zip(xs[:-1], xs[1:]):
            mask = np.zeros([H, W], dtype=np.uint8)
            mask[y_begin:y_end, x_begin:x_end] = 1

            # Get key_pts of high responses
            key_pts = fast.detect(image, mask=mask)
            key_pts.sort(reverse=True, key=(lambda x: x.response))
            key_pts = key_pts[:each_num_corners]
            final_pts += key_pts

    final_pts = [x.pt for x in final_pts]
    final_pts = np.stack(final_pts).astype(np.float32)

    return final_pts


def visodo(
    imgL_prev: np.ndarray,
    imgR_prev: np.ndarray,
    imgL_next: np.ndarray,
    imgR_next: np.ndarray,
    K: np.ndarray,
    stereo_baseline_meters: float,
) -> np.ndarray:
    """연속된 stereo images를 입력받아 프레임 간 ego_motion 계산

    Args:
        img*: HxW input images
        K: 3x3 camera intrinsic

    Returns:
        ego_motion (4x4)
    """

    H, W = imgL_prev.shape

    # --------------------------------
    # Extract disparity maps
    # --------------------------------
    stereo = cv2.StereoSGBM_create(
        numDisparities=128,
        blockSize=3,
        preFilterCap=63,
        P1=36,
        P2=288,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        disp12MaxDiff=1,
    )  # Parameters follow KITTI baseline
    disparity_map_prev = stereo.compute(imgL_prev, imgR_prev)
    disaprity_map_next = stereo.compute(imgL_next, imgR_next)

    # opencv sgbm의 disparity는 int16으로 나오고, 원래 값에 16 곱해진 상태이다.
    disparity_map_prev = disparity_map_prev.astype(np.float32) / 16
    disaprity_map_next = disaprity_map_next.astype(np.float32) / 16

    # ---------------------------
    # Extract 2D features
    # ---------------------------
    # imgL_prev에서 features 위치 추출
    pts2D_L_prev = bucket_features(imgL_prev, vert_splits=4, horz_splits=12, each_num_corners=100)

    # features에 대해서 optical flow 뽑는다.
    pts2D_L_next, status, _ = cv2.calcOpticalFlowPyrLK(imgL_prev, imgL_next, pts2D_L_prev, None)
    status = status.astype(bool).reshape(-1)

    # LK 성공한 포인트들만 추린다.
    pts2D_L_prev = pts2D_L_prev[status].round().astype(int)
    pts2D_L_next = pts2D_L_next[status].round().astype(int)

    # pts2D_L_next가 이미지 범위 넘어가면 필터 아웃
    in_img = (
        (pts2D_L_next[:, 0] >= 0)
        & (pts2D_L_next[:, 0] < W)
        & (pts2D_L_next[:, 1] >= 0)
        & (pts2D_L_next[:, 1] < H)
    )
    pts2D_L_prev = pts2D_L_prev[in_img]
    pts2D_L_next = pts2D_L_next[in_img]

    # disparity 유효한 포인트들만 추린다.
    valid_disp = (
        (disparity_map_prev[pts2D_L_prev[:, 1], pts2D_L_prev[:, 0]] > 0)
        & (disparity_map_prev[pts2D_L_prev[:, 1], pts2D_L_prev[:, 0]] < 100)
        & (disaprity_map_next[pts2D_L_next[:, 1], pts2D_L_next[:, 0]] > 0)
        & (disaprity_map_next[pts2D_L_next[:, 1], pts2D_L_next[:, 0]] < 100)
    )
    pts2D_L_prev = pts2D_L_prev[valid_disp]
    pts2D_L_next = pts2D_L_next[valid_disp]

    disparities_prev = disparity_map_prev[pts2D_L_prev[:, 1], pts2D_L_prev[:, 0]]
    disparities_next = disaprity_map_next[pts2D_L_next[:, 1], pts2D_L_next[:, 0]]

    # ----------------------------
    # Get 3D points of features
    # ----------------------------
    # Reprojection (Learning OpenCV3 p732)
    reprojection_mat = np.array(
        [
            [1, 0, 0, -K[0, 2]],
            [0, 1, 0, -K[1, 2]],
            [0, 0, 0, K[0, 0]],
            [0, 0, 1 / stereo_baseline_meters, 0],
        ]
    )

    pts3D_prev = homo_tf(
        reprojection_mat,
        np.concatenate([pts2D_L_prev, disparities_prev[:, np.newaxis]], axis=1),
    )
    pts3D_next = homo_tf(
        reprojection_mat,
        np.concatenate([pts2D_L_next, disparities_next[:, np.newaxis]], axis=1),
    )

    # Filter out far points
    valid_pts = (
        (np.abs(pts3D_prev[:, 0]) < 50)
        & (np.abs(pts3D_prev[:, 1]) < 50)
        & (np.abs(pts3D_prev[:, 2]) < 50)
        & (np.abs(pts3D_next[:, 0]) < 50)
        & (np.abs(pts3D_next[:, 1]) < 50)
        & (np.abs(pts3D_next[:, 2]) < 50)
    )
    pts2D_L_prev = pts2D_L_prev[valid_pts]
    pts2D_L_next = pts2D_L_next[valid_pts]
    pts3D_prev = pts3D_prev[valid_pts]
    pts3D_next = pts3D_next[valid_pts]

    num_feats = valid_pts.sum()
    print("num_filtred_features: ", num_feats)

    # ---------------------------
    # Visualization
    # ---------------------------
    # Blended image
    I_blended = imfuse(imgL_prev, imgL_next)
    for pt1, pt2 in zip(pts2D_L_prev, pts2D_L_next):
        I_blended = cv2.line(I_blended, tuple(pt1), tuple(pt2), (0, 255, 0))
    cv2.imshow("Blended Image", I_blended)
    cv2.waitKey(1)

    # Birdeye plot
    ax_birdeye.clear()
    ax_birdeye.scatter(pts3D_prev[:, 0], pts3D_prev[:, 2], s=1)
    ax_birdeye.set_xlim(-25, 25)
    ax_birdeye.set_ylim(0, 50)

    # --------------------------------------------------------
    # Find the maximum inlier set
    # --------------------------------------------------------
    # Construct Consistency Matrix M (rows == cols == # of features)
    #    A pair of matches is consistent if the distance bw two features in prev_frame is
    #    identical to the distance bw the corresponding features in next_frame
    M = np.zeros([num_feats, num_feats])

    consistency_thresh = 0.03
    for i in range(num_feats):
        M[i] = (
            abs(
                np.sqrt(((pts3D_next[i, :] - pts3D_next) ** 2).sum(axis=1))
                - np.sqrt(((pts3D_prev[i, :] - pts3D_prev) ** 2).sum(axis=1))
            )
            < consistency_thresh
        )

    # Init clique to contain the match with the largest number of consistent matches
    # (Choose the node with the maximum degree)
    first_node_idx = np.argmax(M.sum(axis=1))
    clique = [first_node_idx]

    # Find the maximum clique (NP-complete => Use sub-optimal algorim)
    potential_nodes = find_potential_nodes(clique, M)
    while len(potential_nodes) > 0:
        clique = update_clique(potential_nodes, clique, M)
        potential_nodes = find_potential_nodes(clique, M)

    # Fast version: 그냥 처음 나온 potential_nodes를 clique으로 사용
    # potential_nodes = find_potential_nodes(clique, M)
    # clique = potential_nodes

    print("clique size: ", len(clique))

    pts3D_prev = pts3D_prev[clique, :]
    pts3D_next = pts3D_next[clique, :]
    pts2D_L_prev = pts2D_L_prev[clique, :]
    pts2D_L_next = pts2D_L_next[clique, :]

    # -----------------
    # Visualization
    # -----------------
    for pt1, pt2 in zip(pts2D_L_prev, pts2D_L_next):
        I_blended = cv2.line(I_blended, tuple(pt1), tuple(pt2), (255, 0, 0))
    cv2.imshow("Blended Image", I_blended)
    cv2.waitKey(1)

    # -----------------------------------
    # Estimate Motion by LM Optimization
    # -----------------------------------
    param = least_squares(
        obj_fun,
        x0=np.zeros(6),
        args=(pts2D_L_prev, pts2D_L_next, pts3D_prev, pts3D_next, K),
        method="lm",
        max_nfev=1500,
    )

    r = param.x[:3]
    transl = param.x[3:]

    rot_mat = R.from_euler("ZXZ", r).as_matrix()
    ego_motion = np.eye(4)
    ego_motion[:3, :3] = rot_mat
    ego_motion[:3, 3] = transl

    return ego_motion


def main():
    seq = 0
    seq_dir = KITTI_PATH / "sequences" / f"{seq:02d}"

    # Read 3x4 projection matrices after rectification
    calib_path = seq_dir / "calib.txt"
    P_dict = read_calib(calib_path)

    P0 = P_dict["P0"]
    P1 = P_dict["P1"]

    print(f"Projection Matrix P0\n{P0}\n")
    print(f"Projection Matrix P1\n{P1}\n")

    assert np.array_equal(P0[:3, :3], P1[:3, :3])
    K = P0[:3, :3]
    stereo_baseline_meters = P0[0, 3] / K[0, 0] - P1[0, 3] / K[0, 0]

    print(f"Intrinsic Matrix K\n{K}\n")
    print(f"Stereo baseline (meters): {stereo_baseline_meters}\n\n")

    # Visualize GT poses
    pose_txt = KITTI_PATH / "poses" / f"{seq:02d}.txt"
    gt_poses = np.loadtxt(pose_txt).reshape(-1, 3, 4)
    ax_traj.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3], color="blue")
    ax_traj.axis("equal")

    #
    basenames = sorted([x.stem for x in seq_dir.glob("image_2/*.png")])
    ego_pose = np.eye(4)
    for bname_prev, bname_next in zip(basenames[:-1], basenames[1:]):
        prv_ego_pose = ego_pose

        I0_prev_path = str(seq_dir / "image_0" / f"{bname_prev}.png")
        I1_prev_path = str(seq_dir / "image_1" / f"{bname_prev}.png")
        I0_next_path = str(seq_dir / "image_0" / f"{bname_next}.png")
        I1_next_path = str(seq_dir / "image_1" / f"{bname_next}.png")

        I0_prev = cv2.imread(I0_prev_path, cv2.IMREAD_GRAYSCALE)
        I1_prev = cv2.imread(I1_prev_path, cv2.IMREAD_GRAYSCALE)
        I0_next = cv2.imread(I0_next_path, cv2.IMREAD_GRAYSCALE)
        I1_next = cv2.imread(I1_next_path, cv2.IMREAD_GRAYSCALE)

        ego_motion = visodo(I0_prev, I1_prev, I0_next, I1_next, K, stereo_baseline_meters)
        ego_pose = ego_pose @ ego_motion

        ax_traj.plot(
            [prv_ego_pose[0, 3], ego_pose[0, 3]],
            [prv_ego_pose[2, 3], ego_pose[2, 3]],
            color="red",
        )
        plt.draw()
        plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    KITTI_PATH = Path("KITTI_ODOM")
    main()
