# TempoControl: Temporal Attention Guidance for Text-to-Video Models

<div align="center">
<a href="https://shira-schiber.github.io/TempoControl/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>
<a href="https://arxiv.org/abs/2510.02226"><img src="https://img.shields.io/badge/arXiv-2510.02226-b31b1b.svg" height=20.5></a>
</div>

<div align="center">
<br/>
</div>

The code is based on the original [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1) implementation, with additional modifications for TempoControl.


# Abstract

Recent advances in generative video models have enabled
the creation of high-quality videos based on natural language
prompts. However, these models frequently lack fine-grained
temporal control, meaning they do not allow users to specify when particular visual elements should appear within a
generated sequence. In this work, we introduce **TempoControl**, a method that allows for temporal alignment of visual concepts during inference, without requiring retraining
or additional supervision. TempoControl utilizes cross attention maps, which are an inherent component of text-to-video diffusion models, to guide the timing of concepts
through a novel optimization approach. Our method steers
attention using three complementary principles: aligning its
temporal shape with a control signal (via correlation), amplifying it where visibility is needed (via energy), and keeping
it spatially focused (via entropy). TempoControl allows
precise control over timing while ensuring high video quality
and diversity. We demonstrate its effectiveness across various
video generation applications, including temporal reordering
for single and multiple objects, as well as action and audio-aligned generation.

## Hardware Requirements

About 110 GB of memory on a single GPU

## Environment and Model Preparation

To get started, first create and activate a Conda environment with Python 3.12, then install the required Python dependencies:

```bash
conda create -n tempo_control python=3.12
conda activate tempo_control

# If the installation of `flash_attn` fails, try installing the other packages first and install `flash_attn` last
pip install -r requirements.txt
```

Next, install the Hugging Face CLI and download the pre-trained model locally:

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

## Inference

To generate videos with TempoControl, run:

```bash
./inference.bash
```

This script includes examples for single object, two objects, action, and audio-video alignment. 
Edit it for your specific use case.

## Inference on Benchmarks

The `data` directory contains the benchmark datasets used for evaluation, one-object, two-object, and action benchmarks.

To run TempoControl on any of these benchmarks, use the `--benchmark` argument to specify which benchmark to run. Valid options are: `one-object`, `two-object`, or `action`.

**Usage:**

```bash
./inference_benchmark.bash --benchmark one-object
```

or

```bash
./inference_benchmark.bash --benchmark two-object
```

or

```bash
./inference_benchmark.bash --benchmark action
```


## Evaluation with VBench Metrics

First, follow the [Vbench installation instructions](https://github.com/Vchitect/VBench?tab=readme-ov-file#hammer-installation) in a seperate conda environment.

Once installed, run the following command inside the created env to evaluate your generated videos:

```bash
  python evaluate.py \
  --dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
  --videos_path "outputs_two_objects_benchmark" \
  --mode custom_input \
  --output_path "outputs_two_objects_benchmark"
```

This will compute the VBench metrics for the videos in the outputs directory.


## Evaluation of Temporal Accuracy

First, create the conda environment:

```bash
conda create -n temporal_metric python=3.9
conda activate temporal_metric
pip install -r temporal_metric_requirements.txt
```


To evaluate temporal accuracy, use:

```bash
python temporal_accuracy.py --benchmark one-object
```

Replace `one-object` with `two-object` or `action` for other benchmarks. By default, the script uses the standard video, output, and CSV paths from the inference benchmarks.

**Arguments:**
- `--benchmark` (**required**): one-object, two-object, or action
- `--videos_path`: (optional) directory with videos (default: matching outputs folder)
- `--output_path`: (optional) where to save results (default: same as videos_path)
- `--csv_file`: (optional) CSV with prompts and timing (default: matching file in `data/`)

To use custom paths:

```bash
python temporal_accuracy.py --benchmark one-object \
  --videos_path custom_outputs/one-object \
  --output_path metrics/one-object \
  --csv_file data/one_object.csv
```

**Output:**
Each run creates a JSON file (e.g., `temporal_accuracy_one_object.json`) in the output directory. This file contains overall accuracy and per-video results.