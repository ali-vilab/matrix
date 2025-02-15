import argparse
from typing import Literal
import numpy as np
import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))

import torch
from stage2.cogvideox.pipelines import CogVideoXPipeline
from stage2.cogvideox.transformer import CogVideoXTransformer3DModel

from diffusers.utils import export_to_video, load_image, load_video


def generate_random_control_signal(
        length, seed, repeat_lens=[2, 2, 2], signal_choices=['D', 'DR', 'DL'],
        change_prob_increment=0.2,
    ):
        if not signal_choices or not repeat_lens \
            or len(repeat_lens) != len(signal_choices) \
            or length < 1:
            raise ValueError("Invalid parameters")
        rng = np.random.default_rng(seed)
        result = []
        current_repeat = 0
        current_idx = 0
        change_prob = change_prob_increment
        for i in range(length):
            if current_repeat >= repeat_lens[current_idx]:
                if change_prob >= 1 or rng.uniform(0, 1) < change_prob:
                    if current_idx == 0:
                        current_idx_choices = [j for j in range(1, len(signal_choices))]
                        current_idx = rng.choice(current_idx_choices)
                    else:
                        current_idx = 0
                    current_repeat = 1
                    change_prob = change_prob_increment
                else:
                    current_repeat += 1
                    change_prob = min(1, change_prob + change_prob_increment)
            else:
                current_repeat += 1
            result.append(signal_choices[current_idx])
        return ','.join(result)


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 720,
    height: int = 480,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 8,
    gpu_id: int = 0,
    control_signal: str = None,
    control_seed: int = 42,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    pipe = CogVideoXPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],
            # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
            )

    pipe.to(gpu_id)
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    assert (num_frames - 1) % 4 == 0
    if control_signal is None:
        control_signal = generate_random_control_signal(int((num_frames - 1) / 4 + 1), seed=control_seed)
    print(f"Control signal: {control_signal}")

    video_generate = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        control_signal=control_signal
    ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="The description of the video to be generated")
    parser.add_argument("--model_path", type=str, help="Path of the pre-trained model use")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=256, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=65, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=720, help="Number of steps for the inference process")
    parser.add_argument("--height", type=int, default=480, help="Number of steps for the inference process")
    parser.add_argument("--fps", type=int, default=16, help="Number of steps for the inference process")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--control_signal", type=str, default=None, help="control signal")
    parser.add_argument("--control_seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
        gpu_id=args.gpu_id,
        control_signal=args.control_signal,
        control_seed=args.control_seed,
    )
