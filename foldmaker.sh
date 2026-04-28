#!/bin/bash

# Use the Unix-style path for Git Bash/WSL
DATA_DIR="/f/deep learning project/MultimodalMedicalImaging/dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUTPUT_JSON="./brats23_folds.json"

# Check if directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory $DATA_DIR not found."
    exit 1
fi

echo "Scanning directory... this may take a moment."

# Get folders into an array
mapfile -t folders < <(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;)

total_folders=${#folders[@]}
echo "Found $total_folders patients. Generating JSON..."

echo "{" > "$OUTPUT_JSON"
echo "  \"training\": [" >> "$OUTPUT_JSON"

for i in "${!folders[@]}"; do
    folder="${folders[$i]}"
    fold=$((i % 5))
    
    # Print progress every 50 files so you know it hasn't hung
    if (( i % 50 == 0 )); then
        echo "Processing: $i / $total_folders..."
    fi

    cat <<EOF >> "$OUTPUT_JSON"
    {
      "fold": $fold,
      "image": [
        "$folder/${folder}-t1n.nii.gz",
        "$folder/${folder}-t1c.nii.gz",
        "$folder/${folder}-t2w.nii.gz",
        "$folder/${folder}-t2f.nii.gz"
      ],
      "label": "$folder/${folder}-seg.nii.gz"
    }$( [[ $i -lt $((total_folders - 1)) ]] && echo "," )
EOF
done

echo "  ]" >> "$OUTPUT_JSON"
echo "}" >> "$OUTPUT_JSON"

echo "Done! Saved to $OUTPUT_JSON"