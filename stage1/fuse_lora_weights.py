import os
import argparse
import torch
from cogvideox.pipelines import CogVideoXPipeline
from diffusers.utils import export_to_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="path/to/CogVideoX-2b", required=True)
    parser.add_argument("--lora_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument("--prompt", type=str, default="The video shows a white car driving in a desert environment. The camera follows the car, moving as it moves, recording the tracks left by the car on the sand. The surroundings include sparse vegetation, distant mountains, and raindrops falling from the sky.")
    args = parser.parse_args()

    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_path, torch_dtype=torch.bfloat16) 
    pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="stage1-lora")
    pipe.fuse_lora(adapter_names=["stage1-lora"], components=['transformer'], lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe.save_pretrained(args.output_dir)

    del pipe 
    # test inference
    pipe = CogVideoXPipeline.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16).to("cuda")

    with torch.no_grad():
        video_generate = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=6,
        generator=torch.Generator().manual_seed(42),
        ).frames[0]
        export_to_video(video_generate, os.path.join(args.output_dir, "test.mp4"), fps=16)
