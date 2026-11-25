import cv2
import numpy as np

MAX_FEATURES = 1500
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
MIN_TRACKED_POINTS = 30
MIN_ESSENTIAL_INLIERS = 20
MIN_POSE_VALID_POINTS = 15
DENSE_GRID_STEP = 5
FB_THRESHOLD = 1.5

def configure(s):
    global MAX_FEATURES, QUALITY_LEVEL, MIN_DISTANCE
    global MIN_TRACKED_POINTS, MIN_ESSENTIAL_INLIERS, MIN_POSE_VALID_POINTS
    global DENSE_GRID_STEP, FB_THRESHOLD
    MAX_FEATURES = s["MAX_FEATURES"]
    QUALITY_LEVEL = s["QUALITY_LEVEL"]
    MIN_DISTANCE = s["MIN_DISTANCE"]
    MIN_TRACKED_POINTS = s["MIN_TRACKED_POINTS"]
    MIN_ESSENTIAL_INLIERS = s["MIN_ESSENTIAL_INLIERS"]
    MIN_POSE_VALID_POINTS = s["MIN_POSE_VALID_POINTS"]
    DENSE_GRID_STEP = s["DENSE_GRID_STEP"]
    FB_THRESHOLD = s["FB_THRESHOLD"]

def detect_features(gray):

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_FEATURES,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=7
    )

    if points is None:
        return np.empty((0, 2), dtype=np.float32)

    return points.reshape(-1, 2).astype(np.float32)

def create_camera_matrix(width, height, focal_multiplier=1.2):

    focal = focal_multiplier * max(width, height)

    K = np.array([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    return K

def track_features(gray1, gray2, points1):

    if len(points1) < 5:

        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    p1 = points1.reshape(-1, 1, 2)

    p2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        p1,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if p2 is None or status is None:

        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32)
        )

    status = status.reshape(-1)

    good = status == 1

    old_points = p1.reshape(-1, 2)[good]
    new_points = p2.reshape(-1, 2)[good]

    # Remove points outside the image
    h, w = gray2.shape

    inside = (
        (new_points[:, 0] >= 0) &
        (new_points[:, 0] < w) &
        (new_points[:, 1] >= 0) &
        (new_points[:, 1] < h)
    )

    old_points = old_points[inside]
    new_points = new_points[inside]

    return old_points, new_points

def visualize_sparse_flow(frame, points1, points2):

    output = frame.copy()

    for p1, p2 in zip(points1, points2):

        x1, y1 = p1.astype(int)
        x2, y2 = p2.astype(int)

        cv2.arrowedLine(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1,
            tipLength=0.25
        )

        cv2.circle(
            output,
            (x2, y2),
            2,
            (0, 0, 255),
            -1
        )

    return output

def estimate_camera_pose(points1, points2, K):

    result = {
        "success": False,
        "R": None,
        "t": None,
        "points1": None,
        "points2": None,
        "inliers": 0,
        "pose_valid": 0,
        "status": "Pose estimation failed"
    }

    if len(points1) < MIN_TRACKED_POINTS:

        result["status"] = (
            f"Too few tracked points: {len(points1)}"
        )

        return result

    # --------------------------------------------------------
    # 1. Estimate Essential Matrix
    # --------------------------------------------------------

    E, mask_E = cv2.findEssentialMat(
        points1,
        points2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5
    )

    if E is None or mask_E is None:

        result["status"] = "Essential Matrix estimation failed"

        return result

    mask_E = mask_E.reshape(-1).astype(np.uint8)

    inlier_count = int(np.sum(mask_E))

    result["inliers"] = inlier_count

    if inlier_count < MIN_ESSENTIAL_INLIERS:

        result["status"] = (
            f"Too few Essential Matrix inliers: {inlier_count}"
        )

        return result

    # --------------------------------------------------------
    # 2. Handle multiple Essential Matrix solutions
    # --------------------------------------------------------

    candidates = []

    if E.shape == (3, 3):

        candidates = [E]

    elif E.shape[0] % 3 == 0:

        for i in range(0, E.shape[0], 3):

            candidates.append(E[i:i + 3, :])

    else:

        result["status"] = "Invalid Essential Matrix dimensions"

        return result

    # --------------------------------------------------------
    # 3. Recover camera pose from candidate matrices
    # --------------------------------------------------------

    best_result = None
    best_valid_count = 0

    for candidate_E in candidates:

        try:

            retval, R, t, pose_mask = cv2.recoverPose(
                candidate_E,
                points1,
                points2,
                K,
                mask=mask_E.copy()
            )

            valid_count = int(retval)

            if valid_count > best_valid_count:

                best_valid_count = valid_count

                best_result = (
                    R,
                    t,
                    pose_mask
                )

        except cv2.error:

            continue

    result["pose_valid"] = best_valid_count

    if best_result is None:

        result["status"] = "recoverPose failed"

        return result

    R, t, pose_mask = best_result

    # --------------------------------------------------------
    # 4. Validate pose
    # --------------------------------------------------------

    if best_valid_count < MIN_POSE_VALID_POINTS:

        result["status"] = (
            f"Too few pose-valid points: {best_valid_count}"
        )

        return result

    pose_mask = pose_mask.reshape(-1) > 0

    valid_points1 = points1[pose_mask]
    valid_points2 = points2[pose_mask]

    result["success"] = True
    result["R"] = R
    result["t"] = t
    result["points1"] = valid_points1
    result["points2"] = valid_points2

    result["status"] = (
        f"Pose OK | Tracked: {len(points1)} | "
        f"E inliers: {inlier_count} | "
        f"Pose-valid: {best_valid_count}"
    )

    return result

def calculate_dense_flow(gray1, gray2):

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow

def visualize_dense_flow(frame, flow):

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(frame)

    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    ).astype(np.uint8)

    hsv[..., 1] = 255

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

def triangulate_points(points1, points2, R, t, K):
    if len(points1) < 5:
        return np.empty((0, 3), np.float32), np.empty(0, np.float32)
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    h = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    valid_w = np.abs(h[3]) > 1e-10
    xyz = np.full((len(points1), 3), np.nan, np.float64)
    xyz[valid_w] = (h[:3, valid_w] / h[3, valid_w]).T
    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > 0) & (z < 1000)
    return xyz[valid].astype(np.float32), z[valid].astype(np.float32)


def calculate_sparse_depth(points1, points2, R, t, K, width, height):
    depth = np.full((height, width), np.nan, np.float32)
    xyz, z = triangulate_points(points1, points2, R, t, K)
    if len(z) == 0:
        return depth, 0, xyz, z
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    h = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    xyz_all = (h[:3] / np.where(np.abs(h[3]) < 1e-10, np.nan, h[3])).T
    valid = np.isfinite(xyz_all).all(axis=1) & (xyz_all[:, 2] > 0) & (xyz_all[:, 2] < 1000)
    p = points1[valid]
    x = np.clip(np.rint(p[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(p[:, 1]).astype(int), 0, height - 1)
    depth[y, x] = xyz_all[valid, 2]
    return depth, len(z), xyz, z


def calculate_dense_depth(gray1, gray2, flow, R, t, K, width, height):
    backward = cv2.calcOpticalFlowFarneback(
        gray2, gray1, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    ys, xs = np.mgrid[DENSE_GRID_STEP // 2:height:DENSE_GRID_STEP,
                      DENSE_GRID_STEP // 2:width:DENSE_GRID_STEP]
    p1 = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
    f = flow[ys.ravel(), xs.ravel()]
    p2 = p1 + f
    x2, y2 = np.rint(p2[:, 0]).astype(int), np.rint(p2[:, 1]).astype(int)
    inside = (x2 >= 0) & (x2 < width) & (y2 >= 0) & (y2 < height)
    p1, p2, x2, y2 = p1[inside], p2[inside], x2[inside], y2[inside]
    depth = np.full((height, width), np.nan, np.float32)
    if len(p1) < 10:
        return depth, 0, np.empty((0, 3), np.float32), np.empty(0, np.float32)
    fwd = flow[p1[:, 1].astype(int), p1[:, 0].astype(int)]
    bwd = backward[y2, x2]
    good = np.linalg.norm(fwd + bwd, axis=1) < FB_THRESHOLD
    p1, p2 = p1[good], p2[good]
    if len(p1) < 10:
        return depth, 0, np.empty((0, 3), np.float32), np.empty(0, np.float32)
    xyz, z = triangulate_points(p1, p2, R, t, K)
    if len(z) == 0:
        return depth, 0, xyz, z
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))
    h = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    xyz_all = (h[:3] / np.where(np.abs(h[3]) < 1e-10, np.nan, h[3])).T
    valid = np.isfinite(xyz_all).all(axis=1) & (xyz_all[:, 2] > 0) & (xyz_all[:, 2] < 1000)
    p = p1[valid]
    x = np.clip(np.rint(p[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(p[:, 1]).astype(int), 0, height - 1)
    depth[y, x] = xyz_all[valid, 2]
    return depth, len(z), xyz, z
