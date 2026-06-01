"""Image tensor slicing and lightweight preview helpers."""
from __future__ import annotations

import base64
import io
from typing import Any, Iterable, List


def normalize_indices(indices: Any, batch_size: int, *, default: int = 0, allow_multiple: bool = True) -> List[int]:
    if isinstance(indices, str):
        parsed = []
        for part in indices.replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    parsed.append(int(part))
                except ValueError:
                    pass
        indices = parsed
    elif isinstance(indices, int):
        indices = [indices]
    elif not isinstance(indices, Iterable):
        indices = [default]

    out = []
    for idx in list(indices):
        try:
            value = int(idx)
        except Exception:
            continue
        if 0 <= value < batch_size:
            out.append(value)
    if not out:
        out = [min(max(default, 0), max(batch_size - 1, 0))]
    if not allow_multiple:
        out = [out[0]]
    return out


def slice_images(images: Any, selected_indices: Any):
    import torch  # type: ignore

    batch_size = int(images.shape[0])
    indices = normalize_indices(selected_indices, batch_size, allow_multiple=True)
    idx = torch.tensor(indices, dtype=torch.long, device=images.device)
    return images.index_select(0, idx)


def image_batch_to_previews(images: Any, max_images: int = 32) -> list[str]:
    """Return PNG data URLs. Falls back to an empty list if PIL/torch is unavailable."""
    try:
        from PIL import Image  # type: ignore
        import torch  # type: ignore
    except Exception:
        return []

    try:
        batch_size = min(int(images.shape[0]), int(max_images))
    except Exception:
        return []

    urls: list[str] = []
    tensor = images.detach().float().cpu().clamp(0, 1)
    for i in range(batch_size):
        arr = (tensor[i].numpy() * 255).astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        image = Image.fromarray(arr)
        image.thumbnail((384, 384))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return urls
