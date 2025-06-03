#!/bin/bash

if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "Using CUDA acceleration"
    DEVICE="cuda:0"
else
    echo "Using CPU only"
    DEVICE="cpu"
fi

python Main_testing.py --device $DEVICE