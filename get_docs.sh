#!/bin/bash

parent_link="https://docs.llamaindex.ai/en/stable/"
output_directory="llamadocs"

# Create the output directory if it doesn't exist
# mkdir -p "$output_directory"

# Use wget to recursively download HTML pages
wget --recursive --no-parent --no-clobber --no-check-certificate --page-requisites --html-extension --convert-links --directory-prefix="$output_directory" "$parent_link"
