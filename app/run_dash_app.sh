#!/bin/bash

echo "Starting Video Eventfulness Analyzer with Firework video as default"
echo "This version uses Dash for better performance and synchronization"
echo ""
echo "Debug log file: /home/is1893/Mirror2/dataSets/test_data/results/sync_debug.log"
echo "You can view the log file in real-time with: tail -f /home/is1893/Mirror2/dataSets/test_data/results/sync_debug.log"
echo ""
echo "If you encounter errors with dash_player parameters:"
echo "1. Check the log file for available parameters"
echo "2. The app has been updated to use only supported parameters"
echo ""

# Install required packages if needed
pip install -r requirements_dash.txt

# Run the Dash app
python video_eventfulness_dash.py