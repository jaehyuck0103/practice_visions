from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

fig = plt.figure()
ax_3d = fig.add_subplot(111)


def imfuse(img1, img2):
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


def read_calib(calib_path):
    with open(calib_path) as fp:
        lines = fp.read().splitlines()

    P_dict = {}
    for line in lines:
        key = line.split()[0].rstrip(":")
        vals = line.split()[1:]
        P_dict[key] = np.array([float(x) for x in vals]).reshape([3, 4])

    return P_dict


def homo_tf(A: np.array, x: np.array) -> np.array:
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


def find_potential_nodes(clique, M):

    new_set = np.all(M[:, clique], axis=1)

    for c in clique:
        new_set[c] = False

    new_set = new_set.nonzero()[0]

    return new_set


def update_clique(potential_nodes, clique, M):

    num_matches = M[np.ix_(potential_nodes, potential_nodes)].sum(axis=1)
    max_match_node = potential_nodes[np.argmax(num_matches)]
    clique.append(max_match_node)

    return clique


def obj_fun(PAR, F1, F2, W1, W2, P1):
    # F1, F2 -> 2d coordinates of features in I1_l, I2_l
    # W1, W2 -> 3d coordinates of the features that have been triangulated
    # P1, P2 -> Projection matrices for the two cameras
    # r, t -> 3x1 vectors, need to be varied for the minimization
    r = PAR[:3]
    t = PAR[3:]

    num_feats = len(F1)
    reproj1 = np.zeros([num_feats, 3])
    reproj2 = np.zeros([num_feats, 3])

    rot_mat = R.from_euler("ZXZ", r).as_matrix()
    tran = np.zeros([4, 4])
    tran[:3, :3] = rot_mat
    tran[:3, 3] = t
    tran[3, 3] = 1

    F1_repr = homo_tf(P1 @ tran, W2)
    F2_repr = homo_tf(P1 @ np.linalg.inv(tran), W1)

    reproj1 = F1 - F1_repr
    reproj2 = F2 - F2_repr

    return np.concatenate([reproj1.flatten(), reproj2.flatten()])


def bucket_features(image, h_break, w_break, num_corners):
    H, W = image.shape

    ys = np.round(np.linspace(0, H + 1, num=h_break + 1)).astype(int)
    xs = np.round(np.linspace(0, W + 1, num=w_break + 1)).astype(int)

    fast = cv2.FastFeatureDetector_create()

    final_pts = []
    for y_begin, y_end in zip(ys[:-1], ys[1:]):
        for x_begin, x_end in zip(xs[:-1], xs[1:]):
            mask = np.zeros([H, W], dtype=np.uint8)
            mask[y_begin:y_end, x_begin:x_end] = 1

            # Get key_pts of high responses
            key_pts = fast.detect(image, mask=mask)
            key_pts.sort(reverse=True, key=(lambda x: x.response))
            key_pts = key_pts[:num_corners]
            final_pts += key_pts

    final_pts = [x.pt for x in final_pts]
    final_pts = np.stack(final_pts).astype(np.float32)

    return final_pts


def visodo(I1_l, I1_r, I2_l, I2_r, P_l, P_r):

    H, W = I1_l.shape

    # Parameters follow KITTI baseline
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
    )
    disparityMap1 = stereo.compute(I1_l, I1_r)
    disparityMap2 = stereo.compute(I2_l, I2_r)

    # opencv sgbm의 disparity는 int16으로 나오고, 원래 값에 16 곱해진 상태이다.
    disparityMap1 = disparityMap1.astype(np.float32) / 16
    disparityMap2 = disparityMap2.astype(np.float32) / 16

    h_break = 4
    w_break = 12
    nCorners = 100
    inlierThresh = 0.05  # 0.01

    # feature 추출
    points1_l = bucket_features(I1_l, h_break, w_break, nCorners)

    # optical flow
    points2_l, status, _ = cv2.calcOpticalFlowPyrLK(I1_l, I2_l, points1_l, None)
    status = status.astype(np.bool).reshape(-1)

    # LK 성공한 포인트들만 추린다.
    points1_l = points1_l[status].astype(int)
    points2_l = points2_l[status].astype(int)

    # points2_l이 이미지 범위 넘어가면 필터링
    in_img = (
        (points2_l[:, 0] >= 0)
        & (points2_l[:, 0] < W)
        & (points2_l[:, 1] >= 0)
        & (points2_l[:, 1] < H)
    )
    points1_l = points1_l[in_img]
    points2_l = points2_l[in_img]

    # disparity 유효한 포인트들만 추린다.
    points1_l = points1_l.astype(int)
    points2_l = points2_l.astype(int)
    valid_disp = (
        (disparityMap1[points1_l[:, 1], points1_l[:, 0]] > 0)
        & (disparityMap1[points1_l[:, 1], points1_l[:, 0]] < 100)
        & (disparityMap2[points2_l[:, 1], points2_l[:, 0]] > 0)
        & (disparityMap2[points2_l[:, 1], points2_l[:, 0]] < 100)
    )
    points1_l = points1_l[valid_disp]
    points2_l = points2_l[valid_disp]

    disparities1 = disparityMap1[points1_l[:, 1], points1_l[:, 0]]
    disparities2 = disparityMap2[points2_l[:, 1], points2_l[:, 0]]

    # Calc 3D points
    baseline_meters = P_l[0, 3] / P_l[0, 0] - P_r[0, 3] / P_r[0, 0]
    reprojection_mat = np.array(
        [
            [1, 0, 0, -P_l[0, 2]],
            [0, 1, 0, -P_l[1, 2]],
            [0, 0, 0, P_l[0, 0]],
            [0, 0, 1 / baseline_meters, 0],
        ]
    )

    points3D_1 = homo_tf(
        reprojection_mat,
        np.concatenate([points1_l, disparities1[:, np.newaxis]], axis=1),
    )
    points3D_2 = homo_tf(
        reprojection_mat,
        np.concatenate([points2_l, disparities2[:, np.newaxis]], axis=1),
    )

    # filtering
    valid_pts = (
        (np.abs(points3D_1[:, 0]) < 50)
        & (np.abs(points3D_1[:, 1]) < 50)
        & (np.abs(points3D_1[:, 2]) < 50)
        & (np.abs(points3D_2[:, 0]) < 50)
        & (np.abs(points3D_2[:, 1]) < 50)
        & (np.abs(points3D_2[:, 2]) < 50)
    )

    points1_l = points1_l[valid_pts]
    points2_l = points2_l[valid_pts]
    points3D_1 = points3D_1[valid_pts]
    points3D_2 = points3D_2[valid_pts]

    num_feats = valid_pts.sum()
    print("num_feats: ", num_feats)

    I_blended = imfuse(I1_l, I2_l)
    for pt1, pt2 in zip(points1_l, points2_l):
        I_blended = cv2.line(I_blended, tuple(pt1), tuple(pt2), (0, 255, 0))
    cv2.imshow("abc", I_blended)
    cv2.waitKey(1)

    # 3d plot
    ax_3d.clear()
    ax_3d.scatter(points3D_1[:, 0], points3D_1[:, 2])

    # Consistency Matrix M
    # Number of Row = Number of Columns = Number of Features
    M = np.zeros([num_feats, num_feats])

    for i in range(num_feats):
        M[i] = (
            abs(
                np.sqrt(((points3D_2[i, :] - points3D_2) ** 2).sum(axis=1))
                - np.sqrt(((points3D_1[i, :] - points3D_1) ** 2).sum(axis=1))
            )
            < inlierThresh
        )

    print(M.sum())

    # Choose the node with the maximum degree
    first_node_idx = np.argmax(M.sum(axis=1))
    clique = [first_node_idx]

    # Find the clique
    potential_nodes = find_potential_nodes(clique, M)
    """
    while len(potential_nodes) > 0:
        clique = update_clique(potential_nodes, clique, M)
        potential_nodes = find_potential_nodes(clique, M)
    """
    clique = potential_nodes

    print("num_clieue: ", len(clique))

    new_cloud1 = points3D_1[clique, :]
    new_cloud2 = points3D_2[clique, :]
    new_feats1 = points1_l[clique, :]
    new_feats2 = points2_l[clique, :]

    for pt1, pt2 in zip(new_feats1, new_feats2):
        I_blended = cv2.line(I_blended, tuple(pt1), tuple(pt2), (255, 0, 0))
    cv2.imshow("abc", I_blended)
    cv2.waitKey(1)

    # Optim
    PAR0 = np.zeros(6)
    PAR = least_squares(
        obj_fun,
        PAR0,
        args=(new_feats1, new_feats2, new_cloud1, new_cloud2, P_l),
        method="lm",
        max_nfev=1500,
    )

    r = PAR.x[:3]
    transl = PAR.x[3:]

    rot_mat = R.from_euler("ZXZ", r).as_matrix()
    ego_motion = np.eye(4)
    ego_motion[:3, :3] = rot_mat
    ego_motion[:3, 3] = transl

    return ego_motion


def main():
    seq = 0
    seq_dir = KITTI_PATH / "sequences" / f"{seq:02d}"

    # init plot
    _, ax = plt.subplots()
    ax.set_xlabel("x(m)")
    ax.set_ylabel("z(m)")

    # read calibration
    calib_path = seq_dir / "calib.txt"
    P_dict = read_calib(calib_path)

    P0 = P_dict["P0"]
    P1 = P_dict["P1"]

    # read gt poses
    pose_txt = KITTI_PATH / "poses" / f"{seq:02d}.txt"
    gt_poses = np.loadtxt(pose_txt).reshape(-1, 3, 4)
    ax.plot(gt_poses[:, 0, 3], gt_poses[:, 2, 3])

    #
    basenames = sorted([x.stem for x in seq_dir.glob("image_2/*.png")])

    absA = np.eye(4)
    for bname_prev, bname_next in zip(basenames[:-1], basenames[1:]):
        prv_absA = absA

        I0_prev_path = str(seq_dir / "image_0" / f"{bname_prev}.png")
        I1_prev_path = str(seq_dir / "image_1" / f"{bname_prev}.png")
        I0_next_path = str(seq_dir / "image_0" / f"{bname_next}.png")
        I1_next_path = str(seq_dir / "image_1" / f"{bname_next}.png")

        I0_prev = cv2.imread(I0_prev_path, cv2.IMREAD_GRAYSCALE)
        I1_prev = cv2.imread(I1_prev_path, cv2.IMREAD_GRAYSCALE)
        I0_next = cv2.imread(I0_next_path, cv2.IMREAD_GRAYSCALE)
        I1_next = cv2.imread(I1_next_path, cv2.IMREAD_GRAYSCALE)

        ego_motion = visodo(I0_prev, I1_prev, I0_next, I1_next, P0, P1)
        absA = absA @ ego_motion

        ax.plot([prv_absA[0, 3], absA[0, 3]], [prv_absA[2, 3], absA[2, 3]])
        ax_3d.set_xlim(-25, 25)
        ax_3d.set_ylim(0, 50)
        plt.draw()
        plt.pause(0.001)

    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    KITTI_PATH = Path("KITTI_ODOM")
    main()
