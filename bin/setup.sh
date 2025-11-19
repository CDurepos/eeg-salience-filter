#!/bin/bash

# Download dataset 
python <<'EOF'
import kagglehub

# Download latest version
path = kagglehub.dataset_download("danizo/eeg-dataset-for-adhd")

# Save dataset path to local `.env` file
with open(".env", "w") as f:
    f.write(f"DATA_PATH={path}")

print("Dataset saved at: ", path)
EOF

# Install tqwt_tools
git clone https://github.com/jollyjonson/tqwt_tools.git
cd tqwt_tools
python setup.py install