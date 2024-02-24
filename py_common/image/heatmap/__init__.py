import numpy as np
import skimage
import matplotlib
from skimage.transform import resize


def heatmap_overlay_rgb(img: np.ndarray, heatmap: np.ndarray):
    """

    Args:
        img:
        heatmap:

    Returns:

    """
    # if img.dtype == np.uint8 or img.max() > 1:
    #     img = img.astype(np.float32)
    #     img /= 255.
    img = skimage.img_as_float32(img)
    heatmap = skimage.img_as_float32(heatmap)
    overlaid = heatmap + img
    max_val = np.max(overlaid)
    overlaid /= max_val
    # overlaid *= 255
    # overlaid = np.uint8(overlaid)
    overlaid = skimage.img_as_uint(overlaid)
    return overlaid


def score_grid_2d_helper(grid: np.ndarray, scores: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray):
    """backbone. create a 2D grid of scores given the corresponding coordinates

    Args:
        grid:
        scores:
        x_idx:
        y_idx:

    Returns:

    """
    # noinspection PyArgumentList
    y_max = y_idx.max()
    # noinspection PyArgumentList
    x_max = x_idx.max()
    assert grid.shape[0] >= y_max, f"y indices exceeds max height: {grid.shape[0]} vs. {y_max}"
    assert grid.shape[1] >= x_max, f"x indices exceeds max width: {grid.shape[1]} vs. {x_max}"
    grid[y_idx, x_idx] = scores
    return grid


def score_grid_2d(width: int, height: int, scores: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray):
    grid = np.zeros((height, width), dtype=scores.dtype)
    return score_grid_2d_helper(grid, scores, x_idx, y_idx)


def bg_grid_2d(width: int, height: int, scores: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray):
    mask = np.ones((height, width), dtype=bool)
    fg_scores = np.zeros_like(scores)
    bg_grid = score_grid_2d_helper(mask, fg_scores, x_idx, y_idx)
    return bg_grid


def wsi_score_grid_array(grid_in: np.ndarray,
                         tile_size: int, scores: np.ndarray, box_x: np.ndarray, box_y: np.ndarray):
    x_idx = box_x // tile_size
    y_idx = box_y // tile_size
    grid_height = grid_in.shape[0]
    grid_width = grid_in.shape[1]
    grid_out = score_grid_2d_helper(grid_in, scores, x_idx, y_idx)
    bg_mask = bg_grid_2d(grid_width, grid_height, scores, x_idx, y_idx)
    return grid_out, bg_mask, (grid_width * tile_size, grid_height * tile_size)


def wsi_score_grid_wh(grid_width, grid_height,
                      tile_size: int, scores: np.ndarray, box_x: np.ndarray, box_y: np.ndarray):
    # x_idx = box_x // tile_size
    # y_idx = box_y // tile_size
    # grid = score_grid_2d_helper(np.zeros((grid_height, grid_width), dtype=scores.dtype), scores, x_idx, y_idx)
    # bg_mask = bg_grid_2d(grid_width, grid_height, scores, x_idx, y_idx)
    # return grid, bg_mask, (grid_width * tile_size, grid_height * tile_size)
    grid_in = np.zeros((grid_height, grid_width), dtype=scores.dtype)
    return wsi_score_grid_array(grid_in, tile_size, scores, box_x, box_y)


def wsi_score_grid_by_osh(osh,
                          tile_size: int, scores: np.ndarray, box_x: np.ndarray, box_y: np.ndarray,
                          ):
    grid_width, grid_height = tuple(np.asarray(osh.level_dimensions[0]) // tile_size)
    return wsi_score_grid_wh(grid_width, grid_height, tile_size=tile_size, scores=scores, box_x=box_x, box_y=box_y)


def wsi_score_grid_wrapper(osh, downsample_factor: int,
                           tile_size: int, scores: np.ndarray, box_x: np.ndarray, box_y: np.ndarray,
                           cm_name: str = 'coolwarm'):
    new_size = tuple(np.asarray(osh.level_dimensions[0]) // downsample_factor)
    # bg_mask -- fg=False bg=True
    grid, bg_mask, wh_no_remainder = wsi_score_grid_by_osh(osh, tile_size, scores, box_x, box_y)
    thumbnail: np.ndarray = np.array(osh.get_thumbnail(new_size).convert("RGB"), copy=True)
    # cut off the remainder if cannot be divided by tile_size
    cutoff_w, cutoff_h = wh_no_remainder
    thumbnail: np.ndarray = thumbnail[:cutoff_h, :cutoff_w]

    cm = matplotlib.colormaps[cm_name]
    heatmap_colored = cm(grid)[:, :, :3]
    bg_mask = bg_mask.astype(bool)
    heatmap_colored[bg_mask] = 0

    heatmap_colored = skimage.util.img_as_ubyte(heatmap_colored)
    heatmap_thumb = resize(heatmap_colored, (new_size[1], new_size[0]), anti_aliasing=False)
    overlaid = heatmap_overlay_rgb(thumbnail, heatmap_thumb)
    overlaid = skimage.util.img_as_ubyte(overlaid)
    return thumbnail, overlaid, heatmap_colored, grid

