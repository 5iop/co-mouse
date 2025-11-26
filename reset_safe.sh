#!/bin/bash
# Safe reset script with confirmation: Clean checkpoints, logs, and outputs

set -e  # Exit on error

echo "=========================================="
echo "Reset Training Environment (Safe Mode)"
echo "=========================================="
echo ""
echo "This will DELETE:"
echo "  - All checkpoints (*.pt files)"
echo "  - All logs (*.log files)"
echo "  - All outputs (*.csv, *.png files)"
echo "  - All tensorboard runs"
echo "  - Python cache files"
echo ""

# Check if there are files to delete
total_files=0
[ -d "checkpoints" ] && total_files=$((total_files + $(find checkpoints -type f | wc -l)))
[ -d "logs" ] && total_files=$((total_files + $(find logs -type f | wc -l)))
[ -d "outputs" ] && total_files=$((total_files + $(find outputs -type f | wc -l)))
[ -d "runs" ] && total_files=$((total_files + $(find runs -type f | wc -l)))

if [ $total_files -eq 0 ]; then
    echo "No files to delete. Directories are already clean."
    exit 0
fi

echo "Total files to delete: $total_files"
echo ""

# Ask for confirmation
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo ""
    echo "Reset cancelled."
    exit 0
fi

echo ""
echo "Proceeding with reset..."
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

# Remove tensorboard runs
echo ""
echo "4. Cleaning tensorboard runs..."
remove_dir_contents "runs"

# Remove Python cache
echo ""
echo "5. Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "  ✓ Python cache cleaned"

echo ""
echo "=========================================="
echo "✓ Reset completed successfully!"
echo "=========================================="
echo ""
