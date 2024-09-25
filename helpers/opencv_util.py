from pathlib import Path

import numpy as np
import cv2
from beartype import beartype

from helpers import logger


@beartype
def record_video(save_dir: Path, name: str, obs: np.ndarray):
    """Record a video from samples collected at evalutation time."""
    # unstack the frames if stacked, while leaving colors unaltered
    frames = np.split(obs, 1, axis=-1)
    frames = np.concatenate(np.array(frames), axis=0)
    frames = [np.squeeze(a, axis=0)
              for a in np.split(frames, frames.shape[0], axis=0)]

    # create OpenCV video writer
    vname = f"render-{name}"
    frame_size = (obs.shape[-2], obs.shape[-3])

    # the type-checked whines here, so we trick it
    assert hasattr(cv2, "VideoWriter_fourcc")
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")

    writer = cv2.VideoWriter(
        filename=f"{save_dir / vname}.mp4",
        fourcc=fourcc,
        fps=25,
        frameSize=frame_size,
        isColor=True,
    )

    for frame in frames:
        # add frame to video
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()
    # delete the object
    del frames

    logger.info(f"video::{vname}::dumped")
