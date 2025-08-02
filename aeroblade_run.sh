

source /d/IPTP/Stage_3A/Stage/Code/.venv/Scripts/activate
# Define the list of generators
generators=("midj" "sd_14" "sd_15" "vqdm" "wukong")

# Loop over each generator
for generator in "${generators[@]}"; do
    echo "Processing generator: $generator"

    # Run for 'nature'
    python Aeroblade/scripts/run_aeroblade.py \
        --files-or-dirs "F:/Data/imagenet_${generator}/val/nature" \
        --output-dir "F:/${generator}_output/nature"

    # Run for 'ai'
    python Aeroblade/scripts/run_aeroblade.py \
        --files-or-dirs "F:/Data/imagenet_${generator}/val/ai" \
        --output-dir "F:/${generator}_output/ai"
done