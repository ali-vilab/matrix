MODEL_PATH="/MODEL/PATH"
VIDEO_PATH="/VIDEO/PATH"

python inference.py \
--prompt "On a lush green meadow, a white car is driving. From an \
overhead panoramic shot, this car is adorned with blue and red stripes \
on its body, and it has a black spoiler at the rear. The camera follows the \
car as it moves through a field of golden wheat, surrounded by green grass and \
trees. In the distance, a river and some hills can be seen, with a cloudless \
blue sky above." \
--model_path $MODEL_PATH \
--video_path $VIDEO_PATH \
--control_signal D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,DR,D,D,D,D,D,D,D,D,D,D,D,D \
--output_path "./output_30s.mp4" \
--num_sample_groups 60 --num_inference_steps 4 --lcm_multiplier 1 --dtype float16

