#!/usr/bin/env python3

import os
import glob
import re
import subprocess
import tempfile

# Create merged directory if it doesn't exist
os.makedirs("merged", exist_ok=True)

# Template for merge config
config_template = """merge_method: task_arithmetic
base_model: ./bert-base-uncased-with-classifier 
models:
  - model: hf_backdoor/{model_name}
    parameters:
      weight: 0.25
  - model: ./bert-mrpc
    parameters:
      weight: 0.25
  - model: ./bert-spam
    parameters:
      weight: 0.25
  - model: ./bert-fake
    parameters:
      weight: 0.25
dtype: float16 
tokenizer_source: base
"""

# Find all backdoored models
model_pattern = "hf_backdoor/hf-backdoor_e*_p*"
model_paths = glob.glob(model_pattern)

if not model_paths:
    print("No backdoored models found matching the pattern")
    exit(1)

# Process each model
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    
    # Extract parameters using regex
    # match = re.match(r'bert-backdoored-sst2_e(\d+)_c(\d+)_p([\d\.]+)', model_name)
    match = re.match(r'hf-backdoor_e(\d+)_p([\d\.]+)', model_name)
    
    if not match:
        print(f"Warning: Could not parse model name: {model_name}")
        continue
    
    epochs, poison_frac = match.groups()
    
    # Create output directory name
    output_dir = f"merged/merge_n4_e{epochs}__p{poison_frac}"
    
    # Create temporary config file
    config_content = config_template.format(model_name=model_name)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_config:
        temp_config.write(config_content)
        temp_config_path = temp_config.name
    
    try:
        print(f"Merging {model_name} -> {output_dir}")
        
        # Run the merge command
        result = subprocess.run([
            'mergekit-yaml', 
            temp_config_path, 
            output_dir, 
            '--cuda', 
            '--allow-crimes'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully merged {model_name}")
        else:
            print(f"Error merging {model_name}: {result.stderr}")
    
    finally:
        # Clean up temporary config file
        os.unlink(temp_config_path)

print("Batch merging completed!")
