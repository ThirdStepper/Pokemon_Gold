# gui_helpers.py
"""
Helper utilities for Dear PyGui viewer interface.

This module provides utility functions for:
- Converting numpy arrays to Dear PyGui textures
- Extracting frames from VecFrameStack
- Creating and updating charts/graphs
"""

import numpy as np
import dearpygui.dearpygui as dpg
from typing import Optional, List, Tuple
from stable_baselines3.common.vec_env import VecFrameStack


def numpy_to_dpg_texture(array: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert a numpy array to a format suitable for Dear PyGui textures.

    Dear PyGui textures require:
    - Shape: (height, width, channels) where channels = 4 (RGBA)
    - Data type: float32
    - Value range: 0.0 to 1.0
    - Flattened to 1D array in row-major order

    Args:
        array: Input numpy array (can be grayscale HxW or HxWx1, or RGB HxWx3)
        normalize: If True, normalizes uint8 (0-255) to float (0-1)

    Returns:
        Flattened float32 array suitable for dpg.set_value() on a texture
    """
    # Ensure we have a copy to modify
    arr = np.array(array, copy=True)

    # Handle different input shapes
    if arr.ndim == 2:
        # Grayscale HxW -> HxWx1
        arr = arr[:, :, np.newaxis]

    if arr.shape[2] == 1:
        # Grayscale HxWx1 -> HxWx3 (RGB)
        arr = np.repeat(arr, 3, axis=2)

    # Add alpha channel if needed (RGB -> RGBA)
    if arr.shape[2] == 3:
        alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
        if normalize and arr.dtype == np.uint8:
            alpha = alpha * 255
        arr = np.concatenate([arr, alpha], axis=2)

    # Normalize to 0-1 range if uint8
    if normalize and arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Flatten to 1D array
    return arr.flatten()


def create_texture(width: int, height: int, tag: str) -> None:
    """
    Create a Dear PyGui texture registry entry.

    Args:
        width: Texture width in pixels
        height: Texture height in pixels
        tag: Unique tag to identify this texture
    """
    # Create black texture data (RGBA float format)
    black_data = np.zeros((height, width, 4), dtype=np.float32)
    flat_data = black_data.flatten()

    # Add texture to registry
    with dpg.texture_registry():
        dpg.add_raw_texture(
            width=width,
            height=height,
            default_value=flat_data,
            format=dpg.mvFormat_Float_rgba,
            tag=tag
        )


def update_texture(tag: str, data: np.ndarray) -> None:
    """
    Update an existing Dear PyGui texture with new data.

    Args:
        tag: Tag of the texture to update
        data: New texture data (already formatted for DPG)
    """
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, data)


def scale_image_for_display(array: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Scale up a small image for better visibility using nearest-neighbor.

    Args:
        array: Input image array (HxW or HxWxC)
        scale: Scale factor (e.g., 4 makes 72x80 -> 288x320)

    Returns:
        Scaled up array
    """
    if scale <= 1:
        return array

    # Use numpy repeat for fast nearest-neighbor scaling
    if array.ndim == 2:
        # Grayscale
        scaled = np.repeat(array, scale, axis=0)
        scaled = np.repeat(scaled, scale, axis=1)
    else:
        # Color
        scaled = np.repeat(array, scale, axis=0)
        scaled = np.repeat(scaled, scale, axis=1)

    return scaled


def format_ram_value(value: int, format_type: str = "hex") -> str:
    """
    Format a RAM value for display.

    Args:
        value: The RAM value (0-255 typically)
        format_type: How to format ("hex", "dec", "bin", "bool")

    Returns:
        Formatted string
    """
    if format_type == "hex":
        return f"0x{value:02X}"
    elif format_type == "dec":
        return f"{value}"
    elif format_type == "bin":
        return f"{value:08b}"
    elif format_type == "bool":
        return "True" if value != 0 else "False"
    else:
        return str(value)


def create_reward_plot(tag: str, max_points: int = 100) -> Tuple[str, str]:
    """
    Create a reward history plot.

    Args:
        tag: Unique tag for the plot
        max_points: Maximum number of points to display

    Returns:
        Tuple of (x_axis_tag, y_axis_tag) for updating data
    """
    with dpg.plot(label="Reward History", height=200, width=-1, tag=tag):
        dpg.add_plot_legend()
        x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step", tag=f"{tag}_x")
        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Reward", tag=f"{tag}_y")

        # Add line series for rewards
        dpg.add_line_series(
            [], [],
            label="Episode Reward",
            parent=y_axis,
            tag=f"{tag}_series"
        )

    return f"{tag}_x", f"{tag}_y"


def update_reward_plot(tag: str, x_data: List[float], y_data: List[float]) -> None:
    """
    Update reward plot with new data.

    Args:
        tag: Tag of the plot
        x_data: X-axis data (steps)
        y_data: Y-axis data (rewards)
    """
    series_tag = f"{tag}_series"
    if dpg.does_item_exist(series_tag):
        dpg.set_value(series_tag, [list(x_data), list(y_data)])


def create_action_bar_chart(tag: str, action_names: List[str]) -> str:
    """
    Create a bar chart for action distribution.

    Args:
        tag: Unique tag for the plot
        action_names: List of action names for labels

    Returns:
        Tag of the bar series for updating
    """
    with dpg.plot(label="Action Distribution", height=200, width=-1, tag=tag):
        dpg.add_plot_legend()
        x_axis = dpg.add_plot_axis(
            dpg.mvXAxis,
            label="Action",
            tag=f"{tag}_x"
        )
        # Set up x-axis ticks for action names
        dpg.set_axis_ticks(x_axis, tuple((i, name) for i, name in enumerate(action_names)))

        y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag=f"{tag}_y")

        # Add bar series
        x_positions = list(range(len(action_names)))
        y_values = [0] * len(action_names)
        dpg.add_bar_series(
            x_positions,
            y_values,
            label="Actions",
            parent=y_axis,
            tag=f"{tag}_series"
        )

    return f"{tag}_series"


def update_action_bar_chart(tag: str, counts: List[int]) -> None:
    """
    Update action distribution bar chart.

    Args:
        tag: Tag of the bar series
        counts: List of counts for each action
    """
    if dpg.does_item_exist(tag):
        x_positions = list(range(len(counts)))
        dpg.set_value(tag, [x_positions, counts])


def create_collapsible_section(label: str, parent: Optional[str] = None) -> str:
    """
    Create a collapsible tree node for organizing UI elements.

    Args:
        label: Label for the section
        parent: Parent item tag

    Returns:
        Tag of the created tree node
    """
    tag = f"section_{label.replace(' ', '_').lower()}"
    if parent:
        return dpg.add_collapsing_header(label=label, parent=parent, tag=tag, default_open=True)
    else:
        return dpg.add_collapsing_header(label=label, tag=tag, default_open=True)


from typing import List, Optional
import numpy as np

def extract_frames_from_stack(obs: Optional[object], env, n_frames: int = 4) -> List[np.ndarray]:
    """
    Extract individual grayscale frames (oldest -> newest) from the current observation.

    Works with:
      - Multi-input dict observations where image data is in obs['image']
      - Plain numpy arrays (batched or unbatched)
      - Channel-last stacks produced by VecFrameStack(channels_order="last")

    Never touches Stable-Baselines internals. Falls back to black frames if needed.

    Args:
        obs: Current observation (dict or np.ndarray). Preferably from env.step()/reset()
        env: The environment (used only to infer HxW for fallback frames)
        n_frames: Number of frames to return

    Returns:
        List[np.ndarray] of length n_frames, each HxW (uint8), oldest -> newest.
    """
    def to_uint8_gray(arr: np.ndarray) -> np.ndarray:
        """Ensure a 2D uint8 grayscale frame."""
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # RGB/RGBA -> gray
            arr = arr[..., :3].mean(axis=-1)
        # Normalize float inputs if needed
        if np.issubdtype(arr.dtype, np.floating):
            # Heuristic: assume [0,1] if max <= 1.0, else clamp to [0,255]
            if arr.size and np.nanmax(arr) <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # Ensure 2D
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def infer_hw() -> tuple[int, int]:
        """Try to infer (H, W) for fallback frames from the env's observation_space."""
        try:
            space = getattr(env, "observation_space", None)
            if space is None:
                return 72, 80
            # Multi-input dict space
            img_space = None
            if hasattr(space, "spaces"):  # gymnasium.spaces.Dict
                img_space = space.spaces.get("image", None)
            else:
                img_space = space
            shape = getattr(img_space, "shape", None)
            if shape is None or len(shape) < 2:
                return 72, 80
            # Assume channel-last; (H, W, C) when stacked
            H, W = int(shape[0]), int(shape[1])
            return H, W
        except Exception:
            return 72, 80

    def split_stack_last(frames_3d: np.ndarray) -> List[np.ndarray]:
        """Split (H, W, S) into a list of S frames (H, W)."""
        H, W, S = frames_3d.shape
        out = [to_uint8_gray(frames_3d[:, :, i]) for i in range(S)]
        return out

    def split_stack_first(frames_3d: np.ndarray) -> List[np.ndarray]:
        """Split (S, H, W) into a list of S frames (H, W)."""
        S, H, W = frames_3d.shape
        out = [to_uint8_gray(frames_3d[i]) for i in range(S)]
        return out

    try:
        # 1) Prefer the provided observation
        if obs is not None:
            # If dict, use the image branch
            if isinstance(obs, dict) and "image" in obs:
                arr = np.asarray(obs["image"])
            else:
                arr = np.asarray(obs)

            # Handle batched cases by taking the first env in the batch
            # Common shapes: (N, H, W, S) or (N, H, W) or (N, S, H, W)
            if arr.ndim == 4:
                # Try (N, H, W, S) first
                if arr.shape[0] >= 1 and arr.shape[-1] >= 1:
                    candidate = arr[0]
                    if candidate.ndim == 3:
                        arr = candidate
                # If it's (N, S, H, W), prefer channel-first stack form
                if arr.ndim == 4 and arr.shape[1] >= 1:
                    # Could be (N, S, H, W); take first N
                    arr = arr[0]

            # Now handle unbatched forms
            frames_list: List[np.ndarray] = []

            if arr.ndim == 3:
                # Either (H, W, S) [channel-last stack] or (S, H, W) [channel-first stack]
                H, W = None, None
                if arr.shape[-1] >= n_frames and arr.shape[0] >= 8 and arr.shape[1] >= 8:
                    # Likely (H, W, S)
                    frames_list = split_stack_last(arr)
                elif arr.shape[0] >= n_frames and arr.shape[1] >= 8 and arr.shape[2] >= 8:
                    # Likely (S, H, W)
                    frames_list = split_stack_first(arr)
                else:
                    # Ambiguous; try last-axis split first
                    if arr.shape[-1] > 1:
                        frames_list = split_stack_last(arr)
                    else:
                        # Single frame with a singleton channel: (H, W, 1)
                        frames_list = [to_uint8_gray(arr)] * n_frames

            elif arr.ndim == 2:
                # Single grayscale frame
                frames_list = [to_uint8_gray(arr)] * n_frames

            elif arr.ndim == 1 and arr.size == 0:
                # Empty; fall back
                frames_list = []

            else:
                # Unknown layout; try to coerce via last resort
                frames_list = []

            if frames_list:
                # Return the last n_frames (oldest -> newest), padding if needed
                if len(frames_list) >= n_frames:
                    return frames_list[-n_frames:]
                else:
                    pad = [frames_list[0]] * (n_frames - len(frames_list))
                    return pad + frames_list

        # 2) Last resort: render a single frame and duplicate
        if hasattr(env, "envs") and env.envs:
            rendered = None
            try:
                rendered = env.envs[0].render()
            except Exception:
                rendered = None
            if rendered is not None:
                frame = to_uint8_gray(rendered)
                if frame.ndim == 2:
                    return [frame.copy() for _ in range(n_frames)]

        # 3) Fallback: black frames with inferred HxW
        H, W = infer_hw()
        return [np.zeros((H, W), dtype=np.uint8) for _ in range(n_frames)]

    except Exception as e:
        # Print once to avoid spam
        if not getattr(extract_frames_from_stack, "_warned", False):
            print(f"[Warning] extract_frames_from_stack failed: {e}")
            try:
                shape = None
                if isinstance(obs, dict) and "image" in obs:
                    shape = getattr(obs["image"], "shape", None)
                elif obs is not None:
                    shape = np.asarray(obs).shape
                print(f"[Debug] Observation shape: {shape}")
            except Exception:
                pass
            extract_frames_from_stack._warned = True

        H, W = infer_hw()
        return [np.zeros((H, W), dtype=np.uint8) for _ in range(n_frames)]


# === Legacy Code: Delete in future
