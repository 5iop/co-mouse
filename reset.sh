#!/bin/bash
# Reset script: Clean checkpoints, logs, and outputs before training

set -e  # Exit on error

echo "=========================================="
echo "Reset Training Environment"
echo "=========================================="
echo ""

# Function to remove directory contents
remove_dir_contents() {
    dir=$1
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        if [ $file_count -gt 0 ]; then
            echo "Removing $file_count file(s) from $dir/"
            rm -rf "$dir"/*
            echo "  ✓ $dir/ cleaned"
        else
            echo "  - $dir/ already empty"
        fi
    else
        echo "  - $dir/ does not exist (will be created during training)"
    fi
}

# Remove checkpoints
echo "1. Cleaning checkpoints..."
remove_dir_contents "checkpoints"

# Remove logs
echo ""
echo "2. Cleaning logs..."
remove_dir_contents "logs"

# Remove outputs
echo ""
echo "3. Cleaning outputs..."
remove_dir_contents "outputs"

# Remove tensorboard runs (optional)
echo ""
echo "4. Cleaning tensorboard runs..."
remove_dir_contents "runs"

# Remove Python cache
echo ""
echo "5. Cleaning Python cache..."
if [ -d "__pycache__" ]; then
    rm -rf __pycache__
    echo "  ✓ __pycache__/ removed"
else
    echo "  - No __pycache__/ found"
fi

find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null && echo "  ✓ All nested __pycache__/ removed" || echo "  - No nested __pycache__/ found"
find . -name "*.pyc" -delete 2>/dev/null && echo "  ✓ All .pyc files removed" || echo "  - No .pyc files found"

echo ""
echo "=========================================="
echo "✓ Reset completed!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  python train.py"
echo "  # or"
echo "  python train.py --demo"
echo ""
