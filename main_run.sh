#!/bin/bash

# Script to run multiple Python scripts in parallel and log their outputs.

# Paths to Python scripts
LLAMA_MAIN="path/to/llama_main.py"
TINYLLAMA_MAIN="path/to/tinyllama_main.py"
MAIN="path/to/main.py"

# Log files for outputs
LLAMA_LOG="llama_main.log"
TINYLLAMA_LOG="tinyllama_main.log"
MAIN_LOG="main.log"

echo "Starting all processes..."

# Run llama_main.py in the background and log output
python $LLAMA_MAIN > $LLAMA_LOG 2>&1 &
echo "llama_main.py running in background and logging to $LLAMA_LOG"

# Run tinyllama_main.py in the background and log output
python $TINYLLAMA_MAIN > $TINYLLAMA_LOG 2>&1 &
echo "tinyllama_main.py running in background and logging to $TINYLLAMA_LOG"

# Run main.py in the background and log output
python $MAIN > $MAIN_LOG 2>&1 &
echo "main.py running in background and logging to $MAIN_LOG"

# Wait for all background jobs to finish
wait
echo "All processes have completed."
