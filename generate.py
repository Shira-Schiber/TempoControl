# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# type: ignore
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import pandas as pd
import torch, random
import torch.distributed as dist
from PIL import Image
import ast
import gc

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from wan.modules.t5 import T5EncoderModel
import torch
warnings.filterwarnings('ignore')


RANDOM_SEED = False
def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated(device=0) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device=0) / 1024**2
    logging.info(f"{prefix}Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def int_or_list(value):
    try:
        # Try converting to a single integer
        return int(value)
    except ValueError:
        # If it fails, try parsing as a list literal
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                return parsed
            else:
                raise argparse.ArgumentTypeError("Argument must be an integer or a list of integers (e.g., [1,2,3])")
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError("Invalid format. Use an integer (e.g., 5) or a list of integers (e.g., [1,2,3])")
        
def replace_token_with_token_idx(prompt, token, hf_tokenizer):
    encoding = hf_tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding["offset_mapping"]

    # Find all occurrences of target_object in the prompt
    start_positions = []
    start = 0
    while True:
        start_idx = prompt.find(token, start)
        if start_idx == -1:
            break
        end_idx = start_idx + len(token)
        start_positions.append((start_idx, end_idx))
        start = end_idx  # Continue search after this match

    # Get all token indices overlapping any occurrence
    indices = set()
    for start_idx, end_idx in start_positions:
        for i, (start, end) in enumerate(offsets):
            if not (end <= start_idx or start >= end_idx):
                indices.add(i)
    return sorted(indices)

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    if args.base_seed <0:
        global RANDOM_SEED
        RANDOM_SEED = True
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

    # CSV file check
    if args.csv_file is not None:
        assert os.path.exists(args.csv_file), f"CSV file {args.csv_file} does not exist."
        assert args.task.startswith("t2v"), "CSV mode is only supported for text-to-video tasks (t2v)."
        # If CSV is provided, prompt and audio_path are overridden by CSV content
        if args.prompt is not None or args.audio_path is not None:
            logging.warning("CSV file provided; --prompt and --audio_path will be ignored.")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image(s) or video(s) from text prompt(s) or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--save_audio",
        type=str2bool,
        default=False,
        help="Whether to save the audio file along with the generated video. If True, the audio file will be saved."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--original_prompt",
        type=str,
        default=None,
        help="The original prompt to generate the image or video from.")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="The path to the audio file to use for video generation.")
    parser.add_argument(
        "--token_idx",
        type=int_or_list,
        default=None,
        help="The token number(s) to optimize (an integer like 5 or a list like [1,2,3]).")
    parser.add_argument(
        "--token1",
        type=str,
        default=None,
        help="The first token for optimization")
    parser.add_argument(
        "--control_signal1",
        type=int,
        nargs='+',  # or '*' for zero or more
        help="Control singal for token1."
    )
    parser.add_argument(
        "--token_idx2",
        type=int_or_list,
        default=None,
        help="The token number(s) to optimize (an integer like 5 or a list like [1,2,3]).")
    parser.add_argument(
        "--token2",
        type=str,
        default=None,
        help="The second token for optimization")
    parser.add_argument(
        "--control_signal2",
        type=int,
        nargs='+',  
        help="Control singal for token2."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to a CSV file containing prompts for batch video generation. Expected columns: 'prompt', 'control_signal1', 'temp_object'"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config path containg all the hyperparameters for the inference time optimization."
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1),
    parser.add_argument(
        "--videos_per_prompt",
        type=int,
        default=1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()
    _validate_args(args)
    return args

def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, "The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel, init_distributed_environment)
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(model_name=args.prompt_extend_model, is_vl="i2v" in args.task, device=rank)
        else:
            raise NotImplementedError(f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # Load CSV file if provided
    video_inputs = []
    text_len= 512
    t5_dtype= torch.bfloat16
    checkpoint_path= './Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'
    tokenizer_path = './Wan2.1-T2V-1.3B/google/umt5-xxl'
    t5_fsdp=  False

    text_encoder = T5EncoderModel(
                text_len=text_len,
                dtype=t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                shard_fn=None)

    tokenizer = text_encoder.tokenizer
    hf_tokenizer = tokenizer.tokenizer

    if args.csv_file is not None:
        df = pd.read_csv(args.csv_file)
        required_columns = [ 'prompt', 'control_signal1', 'temp_object']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        
        for _, row in df.iterrows():

            if "original_prompt" in row:
                original_prompt = row['original_prompt']
            else:
                original_prompt = row['prompt']

            audio_path = None
            if 'audio_path' in df.columns and pd.notnull(row['audio_path']):
                audio_path = row['audio_path']
            token_idx = None
            if 'temp_object' in df.columns and pd.notnull(row['temp_object']):
                token_idx = replace_token_with_token_idx(row['prompt'], row['temp_object'], hf_tokenizer)
            token_idx2 = None
            if 'static_object' in df.columns and pd.notnull(row['static_object']):
                token_idx2 = replace_token_with_token_idx(row['prompt'], row['static_object'], hf_tokenizer)

            control_signal1 = None
            if 'control_signal1' in df.columns and pd.notnull(row['control_signal1']):
                control_signal1 = [int(x) for x in row['control_signal1'].strip().split()]
            control_signal2 = None
            if 'control_signal2' in df.columns and pd.notnull(row['control_signal2']):
                control_signal2 = [int(x) for x in row['control_signal2'].strip().split()]


            video_input = {
            'audio_path': audio_path,
            'prompt': row['prompt'],
            'original_prompt': original_prompt, 
            'token_idx': token_idx,
            'token_idx2': token_idx2,  
            'save_file': None,
            'control_signal1': control_signal1,
            'control_signal2': control_signal2
            }

            video_inputs.append(video_input)

    else:
        if args.prompt is None and not args.task.startswith("i2v"):
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.original_prompt is None:
            args.original_prompt = args.prompt
        if args.token1:
            args.token_idx = replace_token_with_token_idx(args.prompt, args.token1, hf_tokenizer)
        if args.token2:
            args.token_idx2 = replace_token_with_token_idx(args.prompt, args.token2, hf_tokenizer)
        
        video_input = {
            'audio_path': args.audio_path,
            'prompt': args.prompt,
            'original_prompt': args.original_prompt, 
            'token_idx': args.token_idx,
            'token_idx2': args.token_idx2,  
            'save_file': None,
            'control_signal1': args.control_signal1,
            'control_signal2': args.control_signal2
        }

        video_inputs.append(video_input)

    del hf_tokenizer, tokenizer, text_encoder
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        config_path=args.config_path, 
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    
    # Process each video input
    for idx, video_input in enumerate(video_inputs):
        prompt = video_input['prompt']
        original_prompt = video_input['original_prompt']


        audio_path = video_input['audio_path']
        save_file = video_input['save_file']
        token_idx = video_input['token_idx']
        token_idx2 = video_input['token_idx2']

        if RANDOM_SEED:
            args.base_seed = random.randint(0, sys.maxsize)
            with open(os.path.join(args.save_file, "seeds.txt"), "a") as f:
                f.write(f"{prompt}: {args.base_seed}.\n")
        
        for index_per_prompt in range(args.videos_per_prompt):
            # Create the expected output filename
            suffix = '.png' if "t2i" in args.task else '.mp4'
            os.makedirs(args.save_file, exist_ok=True)
            expected_save_file = os.path.join(
                    args.save_file,
                    f"{original_prompt}-{index_per_prompt}" + suffix)
            
            if args.save_audio and audio_path is not None:
                # If audio_path is provided, append it to the filename
                expected_save_file = expected_save_file.replace(suffix, f"_row{idx}{suffix}")
            
            # Check if file already exists
            # if os.path.exists(expected_save_file):
            #     logging.info(f"Output file already exists, skipping: {expected_save_file}")
            #     continue

            gc.collect()
            torch.cuda.empty_cache()
            # print_memory(f"[Before] Example for prompt {idx+1}, number {index_per_prompt}: ")
            args.base_seed += index_per_prompt
            if args.task.startswith("t2v") or args.task.startswith("t2i"):
                logging.info(f"Processing video {idx + 1}/{len(video_inputs)}")
                logging.info(f"Input prompt: {prompt}")
                if audio_path is not None:
                    logging.info(f"Input audio: {audio_path}")
                if args.use_prompt_extend:
                    logging.info("Extending prompt ...")
                    if rank == 0:
                        prompt_output = prompt_expander(prompt, tar_lang=args.prompt_extend_target_lang, seed=args.base_seed)
                        if prompt_output.status == False:
                            logging.info(f"Extending prompt failed: {prompt_output.message}")
                            logging.info("Falling back to original prompt.")
                            input_prompt = prompt
                        else:
                            input_prompt = prompt_output.prompt
                        input_prompt = [input_prompt]
                    else:
                        input_prompt = [None]
                    if dist.is_initialized():
                        dist.broadcast_object_list(input_prompt, src=0)
                    prompt = input_prompt[0]
                    logging.info(f"Extended prompt: {prompt}")
                

                logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
                # Get additional parameters from video_input if available
                control_signal1 = video_input.get("control_signal1", None)
                control_signal2 = video_input.get("control_signal2", None)

                video = wan_t2v.generate(
                    prompt,
                    audio_path=audio_path,
                    token_num=token_idx,
                    token_num2=token_idx2,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                    output_dir=args.save_file or "output",
                    control_signal1=control_signal1,
                    control_signal2=control_signal2)
                
                gc.collect()
                torch.cuda.empty_cache()

            else:
                if prompt is None:
                    prompt = EXAMPLE_PROMPT[args.task]["prompt"]
                if args.image is None:
                    args.image = EXAMPLE_PROMPT[args.task]["image"]
                logging.info(f"Input prompt: {prompt}")
                logging.info(f"Input image: {args.image}")

                img = Image.open(args.image).convert("RGB")
                if args.use_prompt_extend:
                    logging.info("Extending prompt ...")
                    if rank == 0:
                        prompt_output = prompt_expander(prompt, tar_lang=args.prompt_extend_target_lang, image=img, seed=args.base_seed)
                        if prompt_output.status == False:
                            logging.info(f"Extending prompt failed: {prompt_output.message}")
                            logging.info("Falling back to original prompt.")
                            input_prompt = prompt
                        else:
                            input_prompt = prompt_output.prompt
                        input_prompt = [input_prompt]
                    else:
                        input_prompt = [None]
                    if dist.is_initialized():
                        dist.broadcast_object_list(input_prompt, src=0)
                    prompt = input_prompt[0]
                    logging.info(f"Extended prompt: {prompt}")

                logging.info("Creating WanI2V pipeline.")
                wan_i2v = wan.WanI2V(
                    config=cfg,
                    checkpoint_dir=args.ckpt_dir,
                    device_id=device,
                    rank=rank,
                    t5_fsdp=args.t5_fsdp,
                    dit_fsdp=args.dit_fsdp,
                    use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                    t5_cpu=args.t5_cpu,
                )

                logging.info("Generating video ...")
                video = wan_i2v.generate(
                    prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model)
                
                del wan_i2v
                gc.collect()
                torch.cuda.empty_cache()

            if rank == 0:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
                suffix = '.png' if "t2i" in args.task else '.mp4'
                os.makedirs(args.save_file, exist_ok=True)
                save_file = expected_save_file  # Use the same filename we checked earlier

                if "t2i" in args.task:
                    logging.info(f"Saving generated image to {save_file}")
                    cache_image(
                        tensor=video.squeeze(1)[None],
                        save_file=save_file,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                else:
                    if audio_path is not None and args.save_audio:
                        temp_video_file = f"temp.mp4"
                        logging.info(f"Saving generated video to {save_file}")
                        logging.info(f"cfg.sample_fps is {cfg.sample_fps}")
                        cache_video(
                            tensor=video[None],
                            save_file=temp_video_file,
                            fps=cfg.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1))
                        
                        video_clip = VideoFileClip(temp_video_file)
                        audio_clip = AudioFileClip(audio_path) 
                        final_clip = video_clip.set_audio(audio_clip)

                        final_clip.write_videofile(save_file, codec='libx264', audio_codec='aac')

                        os.remove(temp_video_file)
                        video_clip.close()
                        audio_clip.close()
                        final_clip.close()
                        del video_clip, audio_clip, final_clip, video
                    else:
                        logging.info(f"Saving generated video to {save_file}")
                        cache_video(
                            tensor=video[None],
                            save_file=save_file,
                            fps=cfg.sample_fps,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1))

                del save_file, formatted_time, formatted_prompt

            gc.collect()
            torch.cuda.empty_cache()
            # print_memory(f"[After] Example {idx+1}: ")

    logging.info("Finished.")

if __name__ == "__main__":
    args = _parse_args()
    generate(args)