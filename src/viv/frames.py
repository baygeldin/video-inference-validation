from __future__ import annotations


def wan_video_frames(output: object) -> list[object]:
    from vllm_omni.outputs import OmniRequestOutput

    result = OmniRequestOutput.unwrap_result(output)
    return coerce_video_frames(result.images[0])


def coerce_video_frames(video: object) -> list[object]:
    import numpy as np

    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None and isinstance(video, Image.Image):
        return [video]

    if isinstance(video, (list, tuple)):
        if len(video) == 1 and _ndim(video[0]) >= 4:
            return coerce_video_frames(video[0])
        frames: list[object] = []
        for frame in video:
            if _ndim(frame) >= 4:
                frames.extend(coerce_video_frames(frame))
            else:
                frames.append(_coerce_single_frame(frame))
        return frames

    array = _as_numpy(video)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        return [_prepare_frame_array(array)]
    if array.ndim != 4:
        raise ValueError(f"unsupported video output shape: {array.shape}")

    if array.shape[-1] in (1, 3, 4):
        frames = array
    elif array.shape[0] in (1, 3, 4):
        frames = np.moveaxis(array, 0, -1)
    elif array.shape[1] in (1, 3, 4):
        frames = np.moveaxis(array, 1, -1)
    else:
        raise ValueError(f"unsupported video output shape: {array.shape}")

    return [_prepare_frame_array(frame) for frame in frames]


def _coerce_single_frame(frame: object) -> object:
    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None and isinstance(frame, Image.Image):
        return frame

    return _prepare_frame_array(_as_numpy(frame))


def _prepare_frame_array(frame: object) -> object:
    import numpy as np

    array = _as_numpy(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (
        1,
        3,
        4,
    ):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] in (1, 3, 4)):
        raise ValueError(f"unsupported frame output shape: {array.shape}")

    array = array.astype(np.float32, copy=False)
    if array.size and array.min() < 0:
        array = (array + 1.0) / 2.0
    elif array.size and array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _as_numpy(value: object) -> object:
    import numpy as np

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _ndim(value: object) -> int:
    if hasattr(value, "ndim"):
        return int(value.ndim)
    if hasattr(value, "dim"):
        return int(value.dim())
    return 0
