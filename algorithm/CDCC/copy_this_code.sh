#!/bin/bash

# Check if folder path is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Check if folder exists
if [ ! -d "$1" ]; then
    echo "Folder '$1' does not exist."
    exit 1
fi

# Concatenate contents of all code files in the folder
all_code=""
for file in "$1"/*; do
    if [ -f "$file" ]; then
        all_code+="$(cat "$file")\n"
    fi
done

# Copy concatenated code to clipboard
echo -e "$all_code" | xclip -selection clipboard
echo "All code in the folder has been copied to the clipboard."
