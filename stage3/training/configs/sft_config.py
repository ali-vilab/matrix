from easydict import EasyDict
import datetime
import pytz
import os
import oss2 as oss

args = EasyDict()

# Model Arguments
args.pretrained_model_name_or_path = "../models/stage2"  # str: Path to pretrained model or model identifier from huggingface.co/models.
args.revision = None  # str: Revision of pretrained model identifier from huggingface.co/models.
args.variant = None  # str: Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16.

# Dataset Arguments
args.data_root = "../data/videos/"  # str: A folder containing the training data.
args.index_file = "../data/241009_data_index_noreset_EN.json"
args.id_token = None  # **depracated** str: Identifier token appended to the start of each prompt if provided.
args.height_buckets = [480]  # list[int]: Height buckets for resizing input videos.
args.width_buckets = [720]  # list[int]: Width buckets for resizing input videos.
args.frame_buckets = [33, 65]  # list[int]: Must be 16*i+1; CogVideoX1.5 need to guarantee that ((num_frames - 1) // self.vae_scale_factor_temporal + 1) % patch_size_t == 0, such as 53.
args.load_tensors = False  # **depracated** bool: Whether to use a pre-encoded tensor dataset of latents and prompt embeddings instead of videos and text prompts. The expected format is that saved by running the `prepare_dataset.py` script.
args.random_flip = None  # **depracated** float: If random horizontal flip augmentation is to be used, this should be the flip probability.
args.dataloader_num_workers = 8  # int: Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
args.pin_memory = True  # bool: Whether or not to use the pinned memory setting in pytorch dataloader.
args.prefetch_factor = 2

# Validation Arguments
args.validation_prompt = None  # **deprecated** str: One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_separator' string.
args.validation_images = None  # **deprecated** str: One or more image path(s)/URLs that is used during validation to verify that the model is learning. Multiple validation paths should be separated by the '--validation_prompt_separator' string. These should correspond to the order of the validation prompts.
args.validation_prompt_separator = None  # **deprecated** str: String that separates multiple validation prompts.
args.num_validation_videos = 1  # int: Number of videos that should be generated during validation per `validation_prompt`.
args.validation_epochs = None  # **depracated** int: Run validation every X training epochs. Validation consists of running the validation prompt `args.num_validation_videos` times.
args.validation_steps = 200  # int: Run validation every X training steps. Validation consists of running the validation prompt `args.num_validation_videos` times.
args.guidance_scale = 6.0  # float: The guidance scale to use while sampling validation videos.
args.use_dynamic_cfg = False  # bool: Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.
args.enable_model_cpu_offload = False  # bool: Whether or not to enable model-wise CPU offloading when performing validation/testing to save memory.

# Training Arguments
args.seed = 42  # int: A seed for reproducible training.
args.rank = None  # int: The rank for LoRA matrices.
args.lora_alpha = None  # int: The lora_alpha to compute scaling factor (lora_alpha / rank) for LoRA matrices.
args.mixed_precision = "bf16"  # str: Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
args.output_dir = "output/sft/stage3/"+datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")  # str: The output directory where the model predictions and checkpoints will be written.
args.height = 480  # int: All input videos are resized to this height. (Only use for validation)
args.width = 720  # int: All input videos are resized to this width. (Only use for validation)
args.video_reshape_mode = "center"  # str: All input videos are reshaped to this mode. Choose between ['center', 'random', 'none'].
args.fps = 16  # int: All input videos will be used at this FPS.
args.max_num_frames = 65  # int: All input videos will be truncated to these many frames.
args.skip_frames_start = 0  # int: Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.
args.skip_frames_end = 0  # int: Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.
args.train_batch_size = 8  # int: Batch size (per device) for the training dataloader.
args.num_train_epochs = None  # **deprecated** int: Total number of epochs to train the model.
args.max_train_steps = 100000  # int: Total number of training steps to perform. If provided, overrides `--num_train_epochs`.
args.checkpointing_steps = 200  # int: Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.
args.checkpoints_total_limit = None  # int: Max number of checkpoints to store.
args.resume_from_checkpoint = None  # str: Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or "latest" to automatically select the last available checkpoint.
args.gradient_accumulation_steps = 1  # int: Number of updates steps to accumulate before performing a backward/update pass.
args.gradient_checkpointing = True  # bool: Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
args.learning_rate = 1e-5  # float: Initial learning rate (after the potential warmup period) to use.
args.scale_lr = False  # bool: Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
args.lr_scheduler = "constant"  # str: The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].
args.lr_warmup_steps = 200  # int: Number of steps for the warmup in the lr scheduler.
args.lr_num_cycles = 1  # int: Number of hard resets of the lr in cosine_with_restarts scheduler.
args.lr_power = 1.0  # float: Power factor of the polynomial scheduler.
args.enable_slicing = True  # bool: Whether or not to use VAE slicing for saving memory.
args.enable_tiling = True  # bool: Whether or not to use VAE tiling for saving memory.
args.noised_image_dropout = 0.05  # float: Image condition dropout probability when finetuning image-to-video.
args.ignore_learned_positional_embeddings = False  # bool: Whether to ignore the learned positional embeddings when training CogVideoX Image-to-Video. This setting should be used when performing multi-resolution training, because CogVideoX-I2V does not support it otherwise. Please read the comments in https://github.com/a-r-r-o-w/cogvideox-factory/issues/26 to understand why.

# Optimizer Arguments
args.optimizer = "adamw"  # str: The optimizer type to use.
args.use_8bit = False  # bool: Whether or not to use 8-bit optimizers from `bitsandbytes` or `bitsandbytes`.
args.use_4bit = False  # bool: Whether or not to use 4-bit optimizers from `torchao`.
args.use_torchao = False  # bool: Whether or not to use the `torchao` backend for optimizers.
args.beta1 = 0.9  # float: The beta1 parameter for the Adam and Prodigy optimizers.
args.beta2 = 0.95  # float: The beta2 parameter for the Adam and Prodigy optimizers.
args.beta3 = None  # float: Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.
args.prodigy_decouple = False  # bool: Use AdamW style decoupled weight decay.
args.weight_decay = 0.001  # float: Weight decay to use for optimizer.
args.epsilon = 1e-8  # float: Epsilon value for the Adam optimizer and Prodigy optimizers.
args.max_grad_norm = 1.0  # float: Max gradient norm.
args.prodigy_use_bias_correction = False  # bool: Turn on Adam's bias correction.
args.prodigy_safeguard_warmup = False  # bool: Remove lr from the denominator of D estimate to avoid issues during warm-up stage.
args.use_cpu_offload_optimizer = False  # bool: Whether or not to use the CPUOffloadOptimizer from TorchAO to perform optimization step and maintain parameters on the CPU.
args.offload_gradients = False  # bool: Whether or not to offload the gradients to CPU when using the CPUOffloadOptimizer from TorchAO.

# Configuration Arguments
args.tracker_name = "cogvideox-sft-stage3"  # str: Project tracker name.
args.push_to_hub = False  # bool: Whether or not to push the model to the Hub.
args.hub_token = None  # str: The token to use to push to the Model Hub.
args.hub_model_id = None  # str: The name of the repository to keep in sync with the local `output_dir`.
args.logging_dir = "logs"  # str: Directory where logs are stored.
args.allow_tf32 = True  # bool: Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices.
args.nccl_timeout = 1800  # int: Maximum timeout duration before which allgather, or related, operations fail in multi-GPU/multi-node training settings.
args.report_to = "wandb"  # str: The integration to report the results and logs to. Supported platforms are "tensorboard" (default), "wandb" and "comet_ml". Use "all" to report to all integrations.

# Control
args.control_p_zero = 0.1  # float: Classifier-free guidance for control signal.
args.control_start_layer = 30 // 2  # int: NOTE MAGIC NUMBER HERE, number from CogVideoX-2b
args.control_end_layer = 30  # int: NOTE MAGIC NUMBER HERE, number from CogVideoX-2b
args.control_zero_init = True  # bool

#  Swin DPM Arguments
args.use_swin_dpm = True
args.num_noise_groups = 4
args.num_sample_groups = 8