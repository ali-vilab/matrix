import argparse
import torch
import os
import numpy as np

from stage2.inference import generate_video as base_gen, generate_random_control_signal
from stage3.inference import generate_video as streaming_gen

TRAINING_BASE_VIDEO_MAX_LENGTH = 4
TRAINING_MAX_FRAME = 65
TRAINING_FPS = 16
TRAINING_NOISE_GROUP = 4
TRAINING_BASE_STEP = 50
TRAINING_STREAMING_STEP = 20
TRAINING_RESOLUTION = [720, 480]


class the_matrix:
    def __init__(self, generation_model_path, streaming_model_path) -> None:
        self.generation_model_path = generation_model_path
        self.streaming_model_path = streaming_model_path
    
    def generate(
        self,
        prompt: str,
        length: int = 4,
        control_signal: str = None,
        control_seed: int = 42,
        output_folder: str = "./",
        guidance_scale: float = 6.0,
        seed: int = 42,
        gpu_id: int = 0,
    ):
        '''
        Parameters:
            prompt:
                The description of the video to be generated.
            length:
                Length in second of generated video.
            control_signal:
                Control signal for generated video, like "D,D,D,D,D,DL,DL,DL,DL,D,D,D,DR,DR,DR,DR,DR".
                Meanings:
                    "D": The car is moving straight ahead.
                    "DL": The car is turning left ahead.
                    "DR": The car is turning right ahead.
                For input, if it's length is less than 4 * length + 1, it will be randomly padded.
                Leave it to None for random generation.
            control_seed:
                If control_signal is None, this seed determines the random generated control signal.
            output_folder:
                Folder path for saving generated videos.
            guidance_scale:
                CFG parameter. Leave it to default is good enough.
            seed:
                Random seed for video generation.
            gpu_id:
                The index of GPU to be used.
        '''
        if length > TRAINING_BASE_VIDEO_MAX_LENGTH:
            base_frames = TRAINING_MAX_FRAME
        else:
            base_frames = length*TRAINING_FPS+1
        if control_signal is None:
            control_signal = generate_random_control_signal((base_frames-1)//4+1, seed=control_seed)
        elif len(control_signal.split(",")) < (base_frames-1)//4+1:
            control_padding_length = (base_frames-1)//4+1 - len(control_signal.split(","))
            control_signal = control_signal + "," + generate_random_control_signal(control_padding_length, seed=control_seed)
        if len(control_signal.split(",")) > (base_frames-1)//4+1:
            base_control_signal = ",".join(control_signal.split(",")[0:(base_frames-1)//4+1])
        else:
            base_control_signal = control_signal

        base_gen(
            prompt=prompt,
            model_path=self.generation_model_path,
            output_path=os.path.join(output_folder, "base_video.mp4"),
            num_frames=base_frames,
            width=TRAINING_RESOLUTION[0],
            height=TRAINING_RESOLUTION[1],
            num_inference_steps=TRAINING_BASE_STEP,
            guidance_scale=guidance_scale,
            dtype=torch.bfloat16,
            seed=seed,
            gpu_id=gpu_id,
            fps=TRAINING_FPS,
            control_signal=base_control_signal,
            control_seed=control_seed,
        )

        if length > TRAINING_BASE_VIDEO_MAX_LENGTH:
            num_sample_groups = length // (TRAINING_BASE_VIDEO_MAX_LENGTH // TRAINING_NOISE_GROUP)

            streaming_gen(
                prompt=prompt,
                model_path=self.streaming_model_path,
                output_path=os.path.join(output_folder, "final_video.mp4"),
                width=TRAINING_RESOLUTION[0],
                height=TRAINING_RESOLUTION[1],
                video_path=os.path.join(output_folder, "base_video.mp4"),
                num_inference_steps=TRAINING_STREAMING_STEP,
                guidance_scale=guidance_scale,
                dtype=torch.bfloat16,
                seed=seed,
                gpu_id=gpu_id,
                fps=TRAINING_FPS,
                control_signal=control_signal,
                control_signal_type="downsampled",
                control_seed=control_seed,
                num_noise_groups=TRAINING_NOISE_GROUP,
                num_sample_groups=num_sample_groups,
            )
