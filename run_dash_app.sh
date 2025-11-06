#!/bin/bash

echo "Starting Video Eventfulness Analyzer"
echo "This script will run the Dash app from the app directory"
echo ""

# Change to the app directory
cd /home/is1893/Mirror2/app

# Install required packages if needed
pip install -r requirements_dash.txt

# Run the Dash app
python video_eventfulness_dash.py

