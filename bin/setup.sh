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

# Clone ARL EEGModels repository
if [ ! -d "src/arl-eegmodels" ]; then
    echo "Cloning ARL EEGModels repository..."
    git clone https://github.com/vlawhern/arl-eegmodels.git src/arl-eegmodels
else
    echo "ARL EEGModels repository already exists, skipping clone."
fi

# Install tqwt_tools
if [ ! -d "src/tqwt_tools" ]; then
    git clone https://github.com/jollyjonson/tqwt_tools.git src/tqwt_tools
    cd src/tqwt_tools
    pip install -r requirements.txt
    python setup.py install
    cd -
else
    echo "tqwt_tools already exists, skipping installation."
fi