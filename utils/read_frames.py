from decord import VideoReader
from PIL import Image
import torch
import gc


def read_frames(video_path, num_frames, transform=None):
    vr = VideoReader(video_path)
    video_frames_count = len(vr)

    if video_frames_count >= num_frames:
        skip_frames_window = max(video_frames_count // num_frames, 1)
        frame_indices = [i * skip_frames_window for i in range(num_frames)]
    else:
        frame_indices = list(range(video_frames_count))
        pad_start = (num_frames - video_frames_count) // 2
        frame_indices = [0] * pad_start + frame_indices + \
                        [video_frames_count - 1] * \
            (num_frames - len(frame_indices))

    frames_list = [vr[idx].asnumpy() for idx in frame_indices]

    if transform is not None:
        frames_list = [transform(Image.fromarray(frame))
                       for frame in frames_list]

    # Ensure the output always has exactly num_frames
    if len(frames_list) > num_frames:
        frames_list = frames_list[:num_frames]
    elif len(frames_list) < num_frames:
        padding_needed = num_frames - len(frames_list)
        frames_list += [frames_list[-1]] * padding_needed

    video = torch.stack(frames_list)

    del vr
    gc.collect()
    return video
