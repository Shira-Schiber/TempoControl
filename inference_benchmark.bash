
#!/bin/bash


# Usage: ./inference_benchmark.bash -benchmark <benchmark>
# <benchmark> should be one of: one-object, two-object, action

BENCHMARK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
		-benchmark|--benchmark)
			BENCHMARK="$2"
			shift # past argument
			shift # past value
			;;
		*)
			echo "Unknown argument: $1"
			echo "Usage: $0 -benchmark <one-object|two-object|action>"
			exit 1
			;;
	esac
done

if [ -z "$BENCHMARK" ]; then
	echo "Error: No benchmark specified."
	echo "Usage: $0 -benchmark <one-object|two-object|action>"
	exit 1
fi

case "$BENCHMARK" in
	one-object)
		CONFIG_PATH="configs/one_object_config.yaml"
		OUTPUT_DIR="outputs_one_object_benchmark"
		CSV_FILE="data/one_object.csv"
		;;
	two-object)
		CONFIG_PATH="configs/two_objects_config.yaml"
		OUTPUT_DIR="outputs_two_objects_benchmark"
		CSV_FILE="data/two_objects.csv"
		;;
	action)
		CONFIG_PATH="configs/action_config.yaml"
		OUTPUT_DIR="outputs_action_benchmark"
		CSV_FILE="data/action.csv"
		;;
	*)
		echo "Error: Unknown benchmark '$BENCHMARK'."
		echo "Valid options are: one-object, two-object, action"
		exit 1
		;;
esac

mkdir -p "$OUTPUT_DIR"

python generate.py --task t2v-1.3B --base_seed 42 --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
	--videos_per_prompt 1 \
	--csv_file "$CSV_FILE" \
	--config_path "$CONFIG_PATH" \
	--save_file "$OUTPUT_DIR" \
	--sample_guide_scale 6.0 \
	--sample_shift 3.0
