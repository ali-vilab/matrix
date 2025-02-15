import os
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        index_file: str = None,
        fps: int = None
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.random_flip = random_flip
        self.image_to_video = image_to_video
        self.fps = fps

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        with open(os.path.join(index_file), "r") as f:
            index = json.load(f)
        items = index['list']
        self.video_paths = [i["path"] for i in items]
        self.prompts = [i["caption"] for i in items]
        self.control_signal_seq = [i["control_signal_seq"] for i in items]
        self.valid_frame_idx = [i["valid_frame_idx"] for i in items]

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, meta) -> Dict[str, Any]:
        index, resolution_id = meta
        fname = self.video_paths[index]
        image, video, control_signal = self._preprocess_video(Path(os.path.join(self.data_root, fname)), resolution_id)

        return {
            "prompt": self.id_token + self.prompts[index],
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "control_signal": control_signal
        }

    # depracated
    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Loads a single video, or latent and prompt embedding, based on initialization parameters.

        If returning a video, returns a [F, C, H, W] video tensor, and None for the prompt embedding. Here,
        F, C, H and W are the frames, channels, height and width of the input video.

        If returning latent/embedding, returns a [F, C, H, W] latent, and the prompt embedding of shape [S, D].
        F, C, H and W are the frames, channels, height and width of the latent, and S, D are the sequence length
        and embedding dimension of prompt embeddings.
        """
        
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)

        indices = list(range(0, video_num_frames, video_num_frames // self.max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[: self.max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        return image, frames, None


class VideoDatasetWithResizing(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path, index, resolution_id=0) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        o_frame, o_hight, o_width = self.resolutions[resolution_id]

        video_fps = video_reader.get_avg_fps()
        sampling_interval = video_fps/self.fps

        if (o_frame - 1) * sampling_interval >= video_num_frames:
            # There are a very small part of files broken or too short. This is a very tricky fix.
            frame_indices = np.round(np.linspace(0, frame_indices-1, o_frame, dtype=np.float64)).astype(int).tolist()
        else:
            begin_frame = self.valid_frame_idx[index]
            if begin_frame + (o_frame - 1) * sampling_interval >= video_num_frames:
                begin_frame = max(int(video_num_frames - (o_frame - 1) * sampling_interval) - 1, 0)  # Ensure we have enough frames
            frame_indices = np.round(np.arange(begin_frame, video_num_frames, sampling_interval)).astype(int).tolist()
            frame_indices = frame_indices[:o_frame]

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:o_frame].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        o_res = (o_hight, o_width)
        frames_resized = torch.stack([resize(frame, o_res) for frame in frames], dim=0)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        if video_num_frames < o_frame:
            # There are some files broken. This is a very tricky fix.
            control_signal = ",".join(["D" for _ in range((o_frame-1)//4+1)])
        else:
            control_signal = self.control_signal_seq[index].split(",")
            if len(control_signal) < video_num_frames:
                control_signal.extend(control_signal[-1: ] * (video_num_frames - len(control_signal)))
            control_signal = [control_signal[idx] for idx in frame_indices[::4]]  # VAE temporal downsample rate = 4
            control_signal = ",".join(control_signal)

        return image, frames, control_signal


class VideoDatasetWithResizeAndRectangleCrop(VideoDataset):
    def __init__(self, video_reshape_mode: str = "center", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.video_reshape_mode = video_reshape_mode

    def _resize_for_rectangle_crop(self, arr, image_size):
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def _preprocess_video(self, path: Path, index, resolution_id=0) -> torch.Tensor:
        video_reader = decord.VideoReader(uri=path.as_posix())
        video_num_frames = len(video_reader)
        o_frame, o_hight, o_width = self.resolutions[resolution_id]

        video_fps = video_reader.get_avg_fps()
        sampling_interval = video_fps/self.fps

        if (o_frame - 1) * sampling_interval >= video_num_frames:
            # There are a very small part of files broken or too short. This is a very tricky fix.
            frame_indices = np.round(np.linspace(0, video_num_frames-1, o_frame, dtype=np.float64)).astype(int).tolist()
        else:
            begin_frame = self.valid_frame_idx[index]
            if begin_frame + (o_frame - 1) * sampling_interval >= video_num_frames:
                begin_frame = max(int(video_num_frames - (o_frame - 1) * sampling_interval) - 1, 0)  # Ensure we have enough frames
            frame_indices = np.round(np.arange(begin_frame, video_num_frames, sampling_interval)).astype(int).tolist()
            frame_indices = frame_indices[:o_frame]

        frames = video_reader.get_batch(frame_indices)
        frames = frames[:o_frame].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()

        o_res = o_hight, o_width
        frames_resized = self._resize_for_rectangle_crop(frames, o_res)
        frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

        image = frames[:1].clone() if self.image_to_video else None

        if video_num_frames < o_frame:
            # There are some files broken. This is a very tricky fix.
            control_signal = ",".join(["D" for _ in range((o_frame-1)//4+1)])
        else:
            control_signal = self.control_signal_seq[index].split(",")
            if len(control_signal) < video_num_frames:
                control_signal.extend(control_signal[-1: ] * (video_num_frames - len(control_signal)))
            control_signal = [control_signal[idx] for idx in frame_indices[::4]]  # VAE temporal downsample rate = 4
            control_signal = ",".join(control_signal)

        return image, frames, control_signal


class BucketSampler(Sampler):
    r"""
    PyTorch Sampler that groups 3D data by height, width and frames.

    Args:
        data_source (`VideoDataset`):
            A PyTorch dataset object that is an instance of `VideoDataset`.
        batch_size (`int`, defaults to `8`):
            The batch size to use for training.
        shuffle (`bool`, defaults to `True`):
            Whether or not to shuffle the data in each batch before dispatching to dataloader.
        drop_last (`bool`, defaults to `False`):
            Whether or not to drop incomplete buckets of data after completely iterating over all data
            in the dataset. If set to True, only batches that have `batch_size` number of entries will
            be yielded. If set to False, it is guaranteed that all data in the dataset will be processed
            and batches that do not have `batch_size` number of entries will also be yielded.
    """

    def __init__(
        self, data_source: VideoDataset, global_batch_size: int = 8, shuffle: bool = True, drop_last: bool = True
    ) -> None:
        self.data_source = data_source
        self.global_batch_size = global_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last  # must be true

        self.buckets_num = len(data_source.resolutions)

    def __len__(self):
        return len(self.data_source) * self.buckets_num

    def __iter__(self):
        bucket_index_lists = [list(range(len(self.data_source))) for _ in range(self.buckets_num)]
        if self.shuffle:
            for bucket_index_list in bucket_index_lists:
                random.shuffle(bucket_index_list)
        
        bucket_selection = []
        batch_num_for_each_bucket = len(self.data_source) // self.global_batch_size
        for bucket_id in range(self.buckets_num):
            bucket_selection.extend([bucket_id for _ in range(batch_num_for_each_bucket)])
        if self.shuffle:
            random.shuffle(bucket_selection)

        for bucket_id in bucket_selection:
            cur_bucket = bucket_index_lists[bucket_id]
            yield from [(item, bucket_id) for item in cur_bucket[:self.global_batch_size]]
            del cur_bucket[:self.global_batch_size]

        if self.drop_last:
            return
        
        for bucket_id, cur_bucket in enumerate(bucket_index_lists):
            yield from [(item, bucket_id) for item in cur_bucket]
