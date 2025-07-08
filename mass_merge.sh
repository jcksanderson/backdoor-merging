#!/bin/bash

# Define the base directory where your models are stored
MODEL_DIR="./backdoored" 

# --- 1. Create the output directory (if it doesn't exist) ---
mkdir -p merged

# Find all backdoored model directories and loop through them
for model_path in $(find $MODEL_DIR -maxdepth 1 -type d -name "bert-backdoored-sst2_*"); do
    
    # Extract info from the model name
    clean_model_path=${model_path#./}
    if [[ $clean_model_path =~ bert-backdoored-sst2_e([0-9]+)_c([0-9]+)_p([0-9.]+) ]]; then
        epochs="${BASH_REMATCH[1]}"
        count="${BASH_REMATCH[2]}"
        poison_fraction="${BASH_REMATCH[3]}"
    else
        echo "⚠️ Warning: Could not parse model name: $clean_model_path"
        continue
    fi

    # Construct the output directory name
    output_name="merged_n2_e${epochs}_c${count}_p${poison_fraction}"
    
    # Create a temporary merge configuration file
    CONFIG_FILE="temp_merge_config.yml"
    cat <<EOL > $CONFIG_FILE
merge_method: task_arithmetic
base_model: ./bert-base-uncased-with-classifier
models:
  - model: $clean_model_path
    parameters:
      weight: 0.5
  - model: ./bert-mrpc
    parameters:
      weight: 0.5
dtype: float16
tokenizer_source: base
EOL

    echo "Merging $clean_model_path..."
    
    mergekit-yaml $CONFIG_FILE ./merged/${output_name} --cuda --allow-crimes

    rm $CONFIG_FILE
    
    echo "✅ Successfully created ./merged/$output_name"
    echo "-------------------------------------"

done

echo "All models have been merged into the 'merged' directory."
