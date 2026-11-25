"""Table surface height from the depth point cloud, for the pose-mode prompt (no sim ground truth)."""
import numpy as np

BIN_M = 0.005  # histogram bin; the table is the most common height among all pixels in the agentview image
REFINE_M = 0.01  # median over points within this distance of the most common bin


def estimate_table_height_m(points_world_m):
    """points_world_m: (H, W, 3) point cloud in world frame, meters. Returns the table surface z, meters.

    The table is the largest horizontal surface in the agentview image, so its height is the most common z."""
    assert points_world_m.ndim == 3 and points_world_m.shape[-1] == 3, f"expected (H, W, 3), got {points_world_m.shape}"
    z = points_world_m[..., 2].reshape(-1)
    z = z[np.isfinite(z)]
    assert z.size > 0, "no finite points"
    edges = np.arange(z.min(), z.max() + BIN_M, BIN_M)
    counts, edges = np.histogram(z, bins=edges)
    i = int(np.argmax(counts))
    mode_z = 0.5 * (edges[i] + edges[i + 1])
    near = z[np.abs(z - mode_z) < REFINE_M]
    return float(np.median(near))
