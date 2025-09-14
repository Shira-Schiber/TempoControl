#!/bin/bash

ONE_OBJECT_CONFIG_PATH="configs/one_object_config.yaml"
TWO_OBJECTS_CONFIG_PATH="configs/two_objects_config.yaml"
ACTION_CONFIG_PATH="configs/action_config.yaml"
SOUND_CONFIG_PATH="configs/sound_config.yaml"
OUTPUT_DIR="outputs_temporal_control_env"

mkdir -p "$OUTPUT_DIR"


############# ONE OBJECT #############

python generate.py --task t2v-1.3B --base_seed 42 --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
--prompt "An empty scene. Suddenly, a umbrella appears out of nowhere, drawing all attention." \
--token1 "umbrella" \
--control_signal1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 \
--config_path "$ONE_OBJECT_CONFIG_PATH" \
--save_file "$OUTPUT_DIR" \
--sample_guide_scale 6.0 \
--sample_shift 3.0


############# TWO OBJECTS #############

python generate.py --task t2v-1.3B --base_seed 42 --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
--prompt "The video begins with a serene view centered on the bird, with no sign of the cat. In the second half, the cat unexpectedly appears, altering the dynamic of the scene." \
--token1 "cat" \
--token2 "bird" \
--control_signal1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 \
--control_signal2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
--config_path "$TWO_OBJECTS_CONFIG_PATH" \
--save_file "$OUTPUT_DIR" \
--sample_guide_scale 6.0 \
--sample_shift 3.0


############# ACTION #############

python generate.py --task t2v-1.3B --base_seed 42 --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
--prompt "A video of a wolf howling hauntingly, with a strong movement at the second second." \
--token1 "howling" \
--token2 "wolf" \
--control_signal1 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 \
--control_signal2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
--config_path "$ACTION_CONFIG_PATH" \
--save_file "$OUTPUT_DIR" \
--sample_guide_scale 6.0 \
--sample_shift 3.0

############# SOUND #############

python generate.py --task t2v-1.3B --base_seed 42 --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
--prompt "An elephant raises its trunk high and swings it forcefully as it lets out a powerful trumpet sound." \
--token1 "raises" \
--token2 "elephant" \
--audio_path sound/elephant_trumpeting.wav \
--control_signal2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
--config_path "$SOUND_CONFIG_PATH" \
--save_file "$OUTPUT_DIR" \
--sample_guide_scale 6.0 \
--sample_shift 3.0 \
--save_audio True