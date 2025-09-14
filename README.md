# TempoControl: Temporal Attention Guidance for Text-to-Video Models
<div align="center">
<a href="https://shira-schiber.github.io/TempoControl/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
 <a href=""><img src="https://img.shields.io/badge/arXiv-2306.00966-b31b1b.svg" height=20.5></a>
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

- Single NVIDIA H200 GPU

## Environment and Model Preparation

To get started, first create and activate a Conda environment with Python 3.12, then install the required Python dependencies:

```bash
conda create -n tempo_control python=3.12
conda activate tempo_control
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

### Inference on Benchmarks

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


### Evaluate with VBench metrics

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


### Evaluate Temproal Accuracy

First, create the conda environment:

```bash
conda create -n temporal_metric python=3.9
conda activate temporal_metric
pip install -r temporal_metric_requirements.txt
```


For one object temporal accuracy evaluation:

```bash
python temporal_accuracy_one_object.py   --videos_path \ 
  <outputs_one_object_benchmark>  --output_path <outputs_one_object_benchmark> \
  --csv_file "data/one_object.csv"
```

For two objects temporal accuracy evaluation:

```bash
  python temporal_accuracy_two_objects.py \
  --videos_path <outputs_two_objects_benchmark> \
  --output_path "<outputs_two_objects_benchmark> \
  --csv_file "data/two_objects.csv" \
```

For action temporal accuracy evaluation:

```bash
python temporal_accuracy_action.py \
  --videos_path <outputs_action_benchmark> \ 
  --output_path <outputs_action_benchmark> \
  --csv_file "data/action.csv"
```