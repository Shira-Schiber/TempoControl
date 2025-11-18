
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# type: ignore
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
import librosa
import torch
import imageio
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

import torchvision.transforms.functional as F
import torch.nn.functional as TF
import logging
import os
from wan.utils.utils import cache_video
from torch.utils.checkpoint import checkpoint

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
import scipy.ndimage


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def compute_average_attention(attention_maps, counter_attention_maps):
    """
    attention_maps["cross"] holds the sum of all attention maps across all layers (shape: [1, 12, 32760, 512]).
    counter_attention_maps contains the count of summed maps as a single-element list.
    """
    return attention_maps["cross"] / counter_attention_maps[0]


def save_attention_video(attn_map, token_text, save_dir, fps=4, name="", retry=5):
    """
    Save a sequence of attention heatmaps as a video for one token. For visualization purposes only.
    attn_map: [21, 30, 52] torch.Tensor
    """
    
    if name:
        save_path = os.path.join(save_dir, f"{name}_{token_text.replace(' ', '_')}.mp4")
    else:
        save_path = os.path.join(save_dir, f"{token_text.replace(' ', '_')}.mp4")
    os.makedirs(save_dir, exist_ok=True)
    attn_map = attn_map.to(torch.float32)

    h, w = 240, 416  
    error = None
    for _ in range(retry):
        try:
            global_max = attn_map.max()
            frames = []
            for frame in attn_map:
                frame = frame / global_max
                frame = frame.cpu().numpy()
                frame = cv2.resize(frame, (w, h))
                frame = cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frames = np.stack(frames)
            frames = frames.astype(np.uint8)

            writer = imageio.get_writer(
                save_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p'
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return save_path
        except Exception as e:
            error = e
            continue
    else:
        print(f'save_attention_video failed, error: {error}', flush=True)
        return None


def visualize_token_attention(attention_map, prompts, tokenizer, save_dir, select=0, token_idx=None, name=""):
    """
    For visualization purposes only.
    attention_map: [1, 12, 32760, 512] - averaged attention map
    
    Args:
        attention_map: Attention map tensor of shape [1, 12, 32760, 512]
        prompts: List of text prompts
        tokenizer: Tokenizer object
        save_dir: Directory to save attention maps
        select: Index of the prompt to use (default: 0)
        token_idx: If provided, only save attention map for this specific token index
    """
    os.makedirs(save_dir, exist_ok=True)
    attention_map = attention_map.permute(0, 2, 3, 1)  # Fix input shape from [1, H, V, T] → [1, V, T, H]

    B, video_tokens, text_tokens, num_heads = attention_map.shape
    assert B == 1, "Only batch size 1 is supported"

    attn = attention_map.squeeze(0).mean(dim=-1)  # [32760, 512]

    ids, _ = tokenizer([prompts[select]], return_mask=True, add_special_tokens=True)
    token_ids = ids[0].cpu().numpy()
    hf_tokenizer = tokenizer.tokenizer
    decoded_tokens = [
        hf_tokenizer.decode([tid], skip_special_tokens=True)
        for tid in token_ids if tid not in hf_tokenizer.all_special_ids
    ]
    valid_token_indices = [i for i, tid in enumerate(token_ids) if tid not in hf_tokenizer.all_special_ids]

    if token_idx is not None:
        if isinstance(token_idx, list):
            if all(t < len(valid_token_indices) for t in token_idx):
                token_indices_in_sequence = [valid_token_indices[t] for t in token_idx]
                token_texts = [decoded_tokens[t] for t in token_idx]
                token_map = attn[:, token_indices_in_sequence].mean(dim=1)  # Mean across selected tokens
                token_map_3d = token_map.reshape(21, 30, 52)  # [frames, height, width]
                token_text = "_".join(token_texts)  # Combine token texts for naming
                save_attention_video(token_map_3d, token_text, save_dir, name=f"{name}_mean_{token_text}")
            else:
                print(f"One or more token indices {token_idx} are out of range. Valid range: 0-{len(valid_token_indices)-1}")
        else:
            if token_idx < len(valid_token_indices):
                i = token_idx
                token_idx_in_sequence = valid_token_indices[i]
                token_text = decoded_tokens[i]
                token_map = attn[:, token_idx_in_sequence]
                token_map_3d = token_map.reshape(21, 30, 52)  # [frames, height, width]
                save_attention_video(token_map_3d, token_text, save_dir, name=name)
            else:
                print(f"Token index {token_idx} is out of range. Valid range: 0-{len(valid_token_indices)-1}")
    else:
        # if token idx is None- process all tokens
        for i, token_idx_in_sequence in enumerate(valid_token_indices):
            token_text = decoded_tokens[i]
            token_map = attn[:, token_idx_in_sequence]  # [32760]
            token_map_3d = token_map.reshape(21, 30, 52)  # [frames, height, width]
            save_attention_video(token_map_3d, token_text, save_dir, name=f"{name}_{token_text}_{i:03d}")


class WanT2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        config_path="config.yaml",
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`, *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`, *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.config_yaml = load_yaml_config(config_path)

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

  
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from .distributed.xdit_context_parallel import (usp_attn_forward, usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    
    def extract_audio_features(self, audio_path, sr=22050, temporal_num_frames=21, hop_length=512, threshold=0.7, sigma=2):
        """
        Extract temporal control signal from audio to align visual events accordingly.
        See the "Audio-Visual Alignment" subsection under "Experiments" in the paper.
        """
        y, sr = librosa.load(audio_path, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        x = torch.tensor(onset_env, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1,1,T)
        onset_env_pooled = TF.adaptive_max_pool1d(x, temporal_num_frames).squeeze().float().to(self.device)

        min_val = onset_env_pooled.min()
        max_val = onset_env_pooled.max()
        normalized_onset_env_pooled = (onset_env_pooled - min_val) / (max_val - min_val) # + 1e-6)

        # Smooth lower peaks
        strong_peaks_mask = normalized_onset_env_pooled >= threshold
        smoothed_full = scipy.ndimage.gaussian_filter1d(normalized_onset_env_pooled.cpu().numpy(), sigma=sigma)
        smoothed_onset_env = np.where(strong_peaks_mask.cpu().numpy(), normalized_onset_env_pooled.cpu().numpy(), smoothed_full)

        return torch.tensor(smoothed_onset_env).to(self.device), sr, hop_length
    

    
    def compute_attention_strength_signal(self, attention_map, token_idx, noramalize=True):
        """
        Compute a_i^t
        See the "Cross-attention in text-to-video diffusion models" subsection under "Preliminaries" in the paper.

        Args:
            attention_map: [1, 12, 32760, 512] tensor from get_average_attention()
            token_idx: Integer index (or list of indexes) to select a specific token from the 512 token dimension
        Returns:
            strength_signal: [21] tensor representing attention strength per frame
        """
        token_map = self.get_attention_map(attention_map, token_idx)
 
        token_map = token_map.mean(dim=[1, 2])  
        if noramalize:
            token_map = (token_map - token_map.min()) / (token_map.max() - token_map.min() + 1e-6)
        token_map = token_map.to(dtype=torch.float32, device=self.device)
        if not token_map.requires_grad:
            token_map.requires_grad_(True)

        del attention_map
        return token_map
    

    def energy_loss_term(self, attention_map, token_idx, control_signal):
        """
        Compute attention magnitude term
        See the "Attention magnitude term" subsection under "Method" in the paper.
        """
        token_map = self.get_attention_map(attention_map, token_idx)  # shape: (T, H, W)
        
        # Get indices where a_i^t is above 0.7
        valid_frame_indices = (control_signal > 0.7).nonzero(as_tuple=True)[0]  # shape: (N,)
        
        if valid_frame_indices.numel() == 0:
            return torch.tensor(0.0, device=attention_map.device)  # or handle as needed

        # Select the corresponding attention maps
        selected_maps = token_map[valid_frame_indices]  # shape: (N, H, W)

        # Compute energy for each selected frame (sum over H and W), then take mean over frames
        energies = selected_maps.view(selected_maps.size(0), -1).sum(dim=1)
        mean_energy = energies.mean()

        return -mean_energy

    def neg_energy_loss_term(self, attention_map, token_idx, control_signal):
        """
        Compute attention magnitude term
        See the "Attention magnitude term" subsection under "Method" in the paper.
        """
        token_map = self.get_attention_map(attention_map, token_idx)  # shape: (T, H, W)
        
        # Get indices where the control signal is above 0.7
        valid_frame_indices = (control_signal < 0.7).nonzero(as_tuple=True)[0]  # shape: (N,)
        
        if valid_frame_indices.numel() == 0:
            return torch.tensor(0.0, device=attention_map.device)  # or handle as needed

        # Select the corresponding attention maps
        selected_maps = token_map[valid_frame_indices]  # shape: (N, H, W)
            # Compute energy for each selected frame (sum over H and W), then take mean
        energies = selected_maps.view(selected_maps.size(0), -1).sum(dim=1)
        mean_energy = energies.mean()
        
        return mean_energy
    


    def get_entropy(self, attention_map, token_idx, control_signal):
        """
        Compute entropy regularization
        See the "Attention entropy regularization" subsection under "Method" in the paper.
        """
        token_map = self.get_attention_map(attention_map, token_idx)  # shape: (T, H, W)
        
        valid_indices = (control_signal > 0.7).nonzero(as_tuple=True)[0]  # shape: (N,)

        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=attention_map.device)  # or handle as appropriate

        entropies = []
        for idx in valid_indices:
            att = token_map[idx].flatten()  # (H * W,)
            p = att / (att.sum() + 1e-10)   # normalize to get a distribution
            entropy = -(p * torch.log(p + 1e-10)).sum()
            entropies.append(entropy)

        return torch.stack(entropies).mean()

    def get_attention_map(self, attention_map, token_idx):
        """
        Extract the attention map for a specific token index or a list of token indices.
        Args:
            attention_map: [1, 12, 32760, 512] tensor from get_average_attention()
            token_idx: Integer index or list of integers to select specific token(s) from the 512 token dimension
        Returns:
            token_map: [21, 30, 52] tensor representing attention map for the specified token(s)
        """
        attention_map = attention_map.permute(0, 2, 3, 1).squeeze(0)
        if isinstance(token_idx, list):
            attention_map = attention_map[:, token_idx, :].mean(dim=1)
            
        else:
            # Select single token index
            attention_map = attention_map[:, token_idx, :]
        token_map = attention_map.mean(dim=-1)
        token_map = token_map.reshape(21, 30, 52)  # Reshape to [frames, height, width]

        return token_map
    
    def compute_penalty(self, attention_map, attention_map_original, token_idx):
        """
         Compute spatial consistency penalty
         This term is not necessary anymore.
        Args:
            attention_map: [1, 12, 32760, 512] tensor the attention map in current step
            attention_map_original: [1, 12, 32760, 512] tensor the original attention map before optinization
            token_idx: Integer index (or list of idexes) to select a specific token from the 512 token dimension
        Returns:
            penalty: Scalar tensor representing the penalty
        """

        attention_map = self.get_attention_map(attention_map, token_idx)
        attention_map_original = self.get_attention_map(attention_map_original, token_idx)

 
        penalty = torch.sum((attention_map.sum(dim=0) - attention_map_original.sum(dim=0)) ** 2)
        del attention_map, attention_map_original
        return penalty


    def pearson_loss(self, attention_signal, control_signal,  output_dir=None, output_name="plot.png"):
        """
        Compute the pearson correlation loss
        See the "Temporal correlation term" subsection under "Method" in the paper.
        """
        
        attention_deriv = attention_signal[1:-1]
        control_deriv = control_signal[1:-1]
        
        mean_attention = attention_deriv.mean()
        mean_control = control_deriv.mean()

        cov = ((attention_deriv - mean_attention) * (control_deriv - mean_control)).mean()
        std_attention = torch.sqrt(((attention_deriv - mean_attention) ** 2).mean() + 1e-6)
        std_control = torch.sqrt(((control_deriv - mean_control) ** 2).mean() + 1e-6)
        correlation = cov / (std_attention * std_control)
        loss = -correlation


         # Optional: Save plot, for visualization purposes only
        if output_dir is not None and self.config_yaml.get('output', {}).get('write_output', False):
            os.makedirs(output_dir, exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.plot(attention_deriv.detach().cpu().numpy(), label='Motion Derivative')
            plt.plot(control_deriv.detach().cpu().numpy(), label='Audio Derivative')
            plt.title('Motion vs Audio Derivatives')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, output_name))
            plt.close()

            base_name = os.path.splitext(output_name)[0]
            motion_path = os.path.join(output_dir, f"{base_name}_motion_deriv.pt")
            audio_path = os.path.join(output_dir, f"{base_name}_audio_deriv.pt")

            # torch.save(attention_signal.detach().cpu(), motion_path)
            # torch.save(control_signal.detach().cpu(), audio_path)
        return loss

    def generate(self,
                 input_prompt,
                 audio_path="",
                 token_num=-1,
                 token_num2=None,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 output_dir="output",
                 control_signal1=None,
                 control_signal2=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            audio_path (`str`, *optional*):
                Path to audio file for aligning visual events to audio
            token_num (`int` or `List[int]`, *optional*, defaults to -1):
                Token index (or list of indices) in the prompt to be temporally controlled during generation.
            token_num2 (`int` or `List[int]`, *optional*, defaults to None):
                A second token index (or list of indices) for additional temporal control.
            size (tuple[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps.
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion.
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation.
            offload_model (`bool`, *optional*, defaults to True):
                Offloads models to CPU to save VRAM
            control_signal1 (`List[int]`, *optional*, defaults to None):
                Temporal control signal for `token_num`, indicating when the corresponding concept should temporally appear.
            control_signal2 (`List[int]`, *optional*, defaults to None):
                Temporal control signal for `token_num2`,  indicating when the corresponding concept should temporally appear.
            
        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N, H, W)
        """
        # preprocess
        logging.info(f"token num: {token_num} token num2: {token_num2}")
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        logging.info(f"generate: target_shape: {target_shape}")

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)        

        if self.config_yaml.get('output', {}).get('write_output', False):
            os.makedirs(output_dir, exist_ok=True)

        base_save_path = "output_video"
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # reading configuration from config file
            update_per_step = self.config_yaml.get('optimization', {}).get('update_per_step')
            optimization_correlation_based_steps = self.config_yaml.get('optimization', {}).get('correlation_based_steps', False)
            pearson_threshold = self.config_yaml.get('optimization', {}).get('pearson_threshold')
            diffusion_steps_range = self.config_yaml.get('diffusion', {}).get('diffusion_steps_range', [0,1,2,3,4,5,6,7,8,9,10])
            penalty_weight = self.config_yaml.get('optimization', {}).get('penalty_weight', 0)
            energy_weight = self.config_yaml.get('optimization', {}).get('energy_weight', 0)
            entropy_weight = self.config_yaml.get('optimization', {}).get('entropy_weight', 0)
            pearson_weight = self.config_yaml.get('optimization', {}).get('pearson_weight',1)
            optimize_two = self.config_yaml.get('optimization', {}).get('optimize_two',False)
            temporal_num_frames = 21
            if control_signal1:
                control_signal1 = torch.tensor(control_signal1).to(self.device, dtype=torch.float32)
                logging.info(f"control_signal1: {control_signal1}")
            else:
              # Load audio features
                y, sr = librosa.load(audio_path, sr=22050)
                duration = librosa.get_duration(y=y, sr=sr)
                # Extract temporal control signal from audio to align visual events accordingly.
                # See the "Audio-Visual Alignment" subsection under "Experiments" in the paper.
                hop_length = int(duration * sr // temporal_num_frames + 1)
                control_signal1 , sr, hop_length = self.extract_audio_features(audio_path)                 

                logging.info(f"audio features from sound, control signal: {control_signal1}")
            if control_signal2:
                control_signal2 = torch.tensor(control_signal2).to(self.device, dtype=torch.float32) 
                logging.info(f"control_signal2: {control_signal2}")
            

            for i, t in enumerate(tqdm(timesteps)): # denosing steps
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                del noise_pred_cond, noise_pred_uncond, noise_pred, latent_model_input, temp_x0

                entropy = 0
                optimize = False
                with torch.no_grad():
                    timestep = torch.stack([t])
                    noise_pred_cond_raw, attention_maps_output, counter_attention_output = self.model([latents[0]], t=timestep, **arg_c, save_attention=True, cross_attention={}, counter_att_maps=[0])
                    avg_map = compute_average_attention(attention_maps_output, counter_attention_output)
                    attnetion_signal = self.compute_attention_strength_signal(avg_map, token_num)
                    pearson_corr = -self.pearson_loss(attnetion_signal, control_signal1).item()

                # commpute the pearson correlation for early stopping
                # See the "Correlation-based early stopping" subsection under "Method" in the paper.
                optimize = (pearson_corr < pearson_threshold) and (i in diffusion_steps_range) if optimization_correlation_based_steps else i in diffusion_steps_range


                if optimize:
                    if optimization_correlation_based_steps:
                        logging.info(f"Applying synchronization loss at step {i} with pearson correlation {pearson_corr:.4f} < {pearson_threshold}")
                    else:
                        logging.info(f"Applying synchronization loss at step {i}")

                    ########### Just for visualization purposes #############
                    ############ Save information before update iteration #############
                    if self.config_yaml.get('output', {}).get('write_output', False):
                        with torch.no_grad():
                            timestep = torch.stack([t])
                            noise_pred_cond_raw, attention_maps_output, counter_attention_output = self.model([latents[0]], t=timestep, **arg_c, save_attention=True, cross_attention={}, counter_att_maps=[0])
                            avg_map = compute_average_attention(attention_maps_output, counter_attention_output)
                        
                            attnetion_signal1 = self.compute_attention_strength_signal(avg_map, token_num)
                            loss = self.pearson_loss(attnetion_signal1, control_signal1, output_dir=output_dir, output_name=f"loss_plot_{i:03d}_before_sync.png")
                            if -loss.item() > pearson_threshold:
                                logging.info(f"Skipping optimization at step {i} as loss {-loss.item()} is already below target loss {pearson_threshold}")
                                optimize = False
                            visualize_token_attention(avg_map, [input_prompt], self.text_encoder.tokenizer, save_dir=output_dir, select=0, token_idx=token_num2, name=f"attention_map_{i:03d}_mean_token_before_sync")
                            noise_pred_cond = noise_pred_cond_raw[0]
                            noise_pred_uncond = self.model([latents[0]], t=timestep, **arg_null)[0]
                            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                            predicted_x0_before = sample_scheduler.convert_model_output(model_output=noise_pred, sample=latents[0])
                            intermediate_video_before = self.vae.decode([predicted_x0_before])[0]
                            save_file_before = f"{output_dir}/{base_save_path}_step_{i:03d}_before_sync.mp4"
                            logging.info(f"Saving pre-optimization video to {save_file_before}")
                            cache_video(tensor=intermediate_video_before[None], save_file=save_file_before, fps=16,
                                        nrow=1, normalize=True, value_range=(-1, 1))
                            del noise_pred_cond, noise_pred_uncond, noise_pred, predicted_x0_before, intermediate_video_before
                            torch.cuda.empty_cache()
                    ############## End of code for visualization ##########
                            
                    latents[0] = latents[0].detach().requires_grad_(True)

                    # Model latents are optimized, model weights remain fixed
                    params_to_optimize = [latents[0]]

                    # using AdamW optimizer
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config_yaml.get('optimization', {}).get('lr', 0.001))
                    attention_maps_avg_original = 0

                    
                    if optimize: # If the Pearson correlation is below the threshold and we're within the first 10 denoising steps
                        for opt_iter in range(update_per_step):
                            optimizer.zero_grad()
                            if optimize:
                                torch.cuda.reset_peak_memory_stats(self.device)
                                
                            
                                # Enable gradients for update iterations
                                with torch.enable_grad():
                                    timestep = torch.stack([t])

                                    def run_model_cond(latents_, timestep_, context_, seq_len_, save_attention=False,  cross_attention={}, counter_att_maps=[0]):
                                        x, attention_maps_output, counter_attention_output  =  self.model([latents_], t=timestep_, context=context_, seq_len=seq_len_, save_attention=save_attention, cross_attention=cross_attention, counter_att_maps=counter_att_maps)
                                        return x[0], attention_maps_output, counter_attention_output

                                    def run_model_uncond(latents_, timestep_, context_, seq_len_):
                                        return self.model([latents_], t=timestep_, context=context_, seq_len=seq_len_)[0]

                                    noise_pred_cond, attention_maps_output, counter_attention_output  = checkpoint(run_model_cond, latents[0], timestep, context, seq_len, use_reentrant=False, save_attention=True, cross_attention={}, counter_att_maps=[0])
                                    noise_pred_uncond = checkpoint(run_model_uncond, latents[0], timestep, context_null, seq_len, use_reentrant=False)

                                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                                    predicted_x0 = sample_scheduler.convert_model_output(model_output=noise_pred, sample=latents[0])
                                    
                                    # Compute the average of cross attention across all layers
                                    # See the "Cross-attention in text-to-video diffusion models" subsection under "Preliminaries" in the paper.
                                    attention_maps_avg = compute_average_attention(attention_maps_output, counter_attention_output)
                                    if opt_iter == 0:
                                        attention_maps_avg_original = attention_maps_avg.clone().detach()

                                    # Compute a_i^t
                                    # See the "Cross-attention in text-to-video diffusion models" subsection under "Preliminaries" in the paper.
                                    attention_signal1 = self.compute_attention_strength_signal(attention_maps_avg, token_num)

                                    # Compute spatial consistency penalty
                                    # See the "Spatial consistency penalty" subsection under "Method" in the paper.
                                    penalty = self.compute_penalty(attention_maps_avg, attention_maps_avg_original, token_num)

                                    # Compute temporal correlation term
                                    # See the "Temporal correlation term" subsection under "Method" in the paper.
                                    loss = self.pearson_loss(attention_signal1, control_signal1)

                                    # Compute attention energy term
                                    # See the "Attention energy term" subsection under "Method" in the paper.
                                    energy_loss_term = self.energy_loss_term(attention_maps_avg, token_num, control_signal1)
                                    neg_energy_loss_term = self.neg_energy_loss_term(attention_maps_avg, token_num, control_signal1)

                                    # Compute entropy regularization
                                    # See the "Entropy regularization" subsection under "Method" in the paper.
                                    entropy_step = self.get_entropy(attention_maps_avg, token_num, control_signal1)
                                    

                                    # Compute the pearson correlation for early stopping
                                    # See the "Correlation-based early stopping" subsection under "Method" in the paper.
                                    if -loss.item() > pearson_threshold:
                                        logging.info(f"Skipping optimization at step {i} as loss {-loss.item()} is already below target loss {pearson_threshold}")
                                        optimize = False

                                    # commpute all compnenets for the second token for optimization if exists
                                    if token_num2 is not None and optimize_two:
                                        attention_signal2 = self.compute_attention_strength_signal(attention_maps_avg, token_num2)
                                        penalty2 = self.compute_penalty(attention_maps_avg, attention_maps_avg_original, token_num2) 
                                        loss2 = self.pearson_loss(attention_signal2, control_signal2) 
                                        energy_loss_term2 = self.energy_loss_term(attention_maps_avg, token_num2, control_signal2)
                                        neg_energy_loss_term2 = self.neg_energy_loss_term(attention_maps_avg, token_num2, control_signal2)
                                        entropy_step2 = self.get_entropy(attention_maps_avg, token_num2, control_signal2)
                                        logging.info(f"two tokens optimization")
                                        logging.info(f"first token: Step {i}, Opt iter {opt_iter+1}/{update_per_step}, Loss: {loss.item():.4f}, Penalty: {penalty.item() * penalty_weight:.4f}, energy_loss_term: {energy_loss_term.item() * energy_weight:.4f},neg_energy_loss_term: {neg_energy_loss_term.item() * energy_weight:.4f}, entropy_step: {entropy_step * entropy_weight:.4f}, ")
                                        logging.info(f"second token: Step {i}, Opt iter {opt_iter+1}/{update_per_step}, Loss: {loss2.item():.4f}, Penalty: {penalty2.item() * penalty_weight:.4f}, energy_loss_term: {energy_loss_term2.item() * energy_weight:.4f}, neg_energy_loss_term: {neg_energy_loss_term2.item() * energy_weight:.4f}, entropy_step: {entropy_step2 * entropy_weight:.4f}, ")

                                        # Commpute full objective
                                        # See the "Full objective" subsection under "Method" in the paper.
                                        loss = (loss +loss2)/2 * pearson_weight + (energy_loss_term + energy_loss_term2)/2 * energy_weight +  (penalty+penalty2)/2 * penalty_weight  + (entropy_step + entropy_step2)/2 * entropy_weight + (neg_energy_loss_term + neg_energy_loss_term2)/2 * energy_weight 
                                    else:

                                        # Commpute full objective
                                        # See the "Full objective" subsection under "Method" in the paper.
                                        logging.info(f"Step {i}, Opt iter {opt_iter+1}/{update_per_step}, Loss: {pearson_weight * loss.item():.4f}, Penalty: {penalty.item() * penalty_weight:.4f}, energy_loss_term: {energy_loss_term.item() * energy_weight:.4f}, neg_energy_loss_term: {neg_energy_loss_term.item() * energy_weight:.4f}, entropy_step: {entropy_step * entropy_weight:.4f}, ")
                                        loss = loss * pearson_weight + energy_loss_term * energy_weight + penalty * penalty_weight + entropy_step * entropy_weight + neg_energy_loss_term * energy_weight 


                                    del noise_pred_cond, noise_pred_uncond, noise_pred, predicted_x0, attention_maps_avg, attention_signal1, penalty, attention_maps_output, counter_attention_output, timestep, entropy_step
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                if optimize:
                                    # Gradient descent step for latents
                                    loss.backward()
                                    optimizer.step()
                                else:
                                    loss.backward()
                                    optimizer.zero_grad()
                                   

                                peak = torch.cuda.max_memory_allocated(self.device)
                                logging.info(f"Peak memory usage: {peak / 1024**2:.2f} MB")
                                del loss, peak

                        ########### Just for visualization purposes #############
                        ############ Save information after update iteration #############
                        if self.config_yaml.get('output', {}).get('write_output', False):
                            with torch.no_grad():
                                timestep = torch.stack([t])
                                noise_pred_cond_raw, attention_maps_output, counter_attention_output = self.model([latents[0]], t=timestep, **arg_c, save_attention=True, cross_attention={}, counter_att_maps=[0])
                                avg_map = compute_average_attention(attention_maps_output, counter_attention_output)
                                attention_signal1 = self.compute_attention_strength_signal(avg_map, token_num)
                                loss = self.pearson_loss(attention_signal1, control_signal1, output_dir=output_dir, output_name=f"loss_plot_{i:03d}_after_sync.png")
                                visualize_token_attention(avg_map, [input_prompt], self.text_encoder.tokenizer, save_dir=output_dir, select=0, token_idx=token_num2, name=f"attention_map_{i:03d}_mean_token_after_sync")
                                noise_pred_cond = noise_pred_cond_raw[0]

                                noise_pred_uncond = self.model([latents[0]], t=timestep, **arg_null)[0]
                                noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                                predicted_x0_after = sample_scheduler.convert_model_output(model_output=noise_pred, sample=latents[0])
                                intermediate_video_after = self.vae.decode([predicted_x0_after])[0]
                                save_file_after = f"{output_dir}/{base_save_path}_step_{i:03d}_after_sync.mp4"
                                logging.info(f"Saving post-optimization video to {save_file_after}")
                                cache_video(tensor=intermediate_video_after[None], save_file=save_file_after, fps=16,
                                            nrow=1, normalize=True, value_range=(-1, 1))
                                del noise_pred_cond, noise_pred_uncond, noise_pred, predicted_x0_after, intermediate_video_after
                                torch.cuda.empty_cache()
                            ############## End of code for visualization ##########
                    del optimizer
                    del attention_maps_avg_original
                    del params_to_optimize                    
                    gc.collect()
                    torch.cuda.empty_cache()
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()


            if self.rank == 0:
                videos = self.vae.decode(x0)
                logging.info(f"Final video shape: {videos[0].shape}")

        del noise, latents, sample_scheduler, x0, context, context_null
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        
        videos[0] = videos[0].cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return videos[0] if self.rank == 0 else None


