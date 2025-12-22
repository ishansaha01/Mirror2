import cv2
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_player
import plotly.graph_objects as go
import os
import json
import glob
import numpy as np
from pathlib import Path
import base64
import logging
from video_analysis_backend import VideoAnalysisBackend
from flask import request, jsonify

# Note: Peak detection is ALWAYS done by the backend using VideoAnalysisBackend.detect_peaks()
# The frontend NEVER performs its own peak detection to ensure consistency.
# Peak data is stored in eventfulness_data['peak_indices'] and ['peak_values']

# Set up logging
log_file = "/home/is1893/Mirror2/video_analysis.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('video_analysis_frontend')

# Define constants
RESULTS_DIR = "/home/is1893/Mirror2/dataSets/test_data/results"
DEFAULT_VIDEO = "/home/is1893/Mirror2/dataSets/test_data/val/JumpJack/JumpJack.mp4"

# Initialize backend
backend = VideoAnalysisBackend()

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define custom CSS for ChatGPT-esque minimal design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Video Analysis Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #ffffff;
                min-height: 100vh;
                color: #353740;
                line-height: 1.5;
            }
            
            .main-container {
                background: #ffffff;
                height: 100vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .header {
                background: #ffffff;
                border-bottom: 1px solid #e5e5e5;
                padding: 12px 20px;
                display: flex;
                align-items: center;
                height: 48px;
            }
            
            .header h1 {
                font-size: 16px;
                font-weight: 600;
                color: #202123;
                margin: 0;
            }
            
            .main-layout {
                display: flex;
                flex: 1;
                overflow: hidden;
            }
            
            .sidebar {
                background: #202123;
                width: 260px;
                overflow-y: auto;
                flex-shrink: 0;
                display: flex;
                flex-direction: column;
            }
            
            .sidebar-header {
                padding: 16px;
                border-bottom: 1px solid #4d4d4f;
            }
            
            .sidebar-header h3 {
                font-size: 13px;
                font-weight: 600;
                color: #8e8ea0;
                margin: 0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .video-list-container {
                flex: 1;
                overflow-y: auto;
                padding: 8px;
            }
            
            .video-button {
                width: 100%;
                padding: 12px;
                margin-bottom: 2px;
                text-align: left;
                background: transparent;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: background-color 0.15s ease;
                font-size: 14px;
                color: #ececf1;
                font-weight: 400;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .video-button:hover {
                background: #343541;
            }
            
            .video-button.selected {
                background: #343541;
                font-weight: 500;
            }
            
            .content-area {
                flex: 1;
                background: #ffffff;
                overflow-y: auto;
                display: block;
                position: relative;
            }
            
            .analysis-control-section {
                background: #ffffff;
                padding: 20px 32px;
                border-bottom: 1px solid #e5e5e5;
                position: sticky;
                top: 0;
                z-index: 100;
                width: 100%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .video-main-container {
                display: block;
                padding: 32px;
                width: 100%;
            }
            
            .video-player-wrapper {
                width: 100%;
                max-width: 1200px;
                margin: 0 auto 32px auto;
            }
            
            .video-info-minimal {
                margin-top: 16px;
                text-align: center;
                color: #8e8ea0;
                font-size: 13px;
            }
            
            .section-card {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                padding: 20px;
                margin: 0 auto 16px auto;
                max-width: 1200px;
                width: calc(100% - 64px);
            }
            
            .section-title {
                font-size: 14px;
                font-weight: 600;
                color: #202123;
                margin-bottom: 12px;
            }
            
            .info-card {
                display: none; /* Hide detailed info card */
            }
            
            .info-row {
                display: none; /* Hide info rows */
            }
            
            .info-label {
                font-weight: 500;
                color: #8e8ea0;
                font-size: 13px;
            }
            
            .info-value {
                color: #353740;
                font-size: 13px;
            }
            
            .button-group {
                display: flex;
                gap: 8px;
                margin-bottom: 16px;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 8px 16px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.15s ease;
                background: #ffffff;
                color: #353740;
            }
            
            .btn:hover {
                background: #f7f7f8;
                border-color: #c5c5d1;
            }
            
            .btn-primary {
                background: #10a37f;
                color: #ffffff;
                border-color: #10a37f;
            }
            
            .btn-primary:hover {
                background: #0d8f6e;
                border-color: #0d8f6e;
            }
            
            .btn-success {
                background: #ffffff;
                color: #353740;
                border-color: #d1d5db;
            }
            
            .btn-success:hover {
                background: #f7f7f8;
            }
            
            .btn-danger {
                background: #ffffff;
                color: #ef4444;
                border-color: #d1d5db;
            }
            
            .btn-danger:hover {
                background: #fef2f2;
                border-color: #fca5a5;
            }
            
            .gallery {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 16px;
                margin-top: 16px;
                width: 100%;
            }
            
            .frame-thumbnail {
                background: #ffffff;
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #e5e5e5;
                transition: all 0.15s ease;
                cursor: pointer;
            }
            
            .frame-thumbnail:hover {
                border-color: #c5c5d1;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }
            
            .frame-thumbnail img {
                width: 100%;
                height: auto;
                display: block;
            }
            
            .frame-info {
                padding: 10px;
                background: #f7f7f8;
            }
            
            .frame-info div {
                font-size: 12px;
                color: #8e8ea0;
                margin-bottom: 4px;
            }
            
            .frame-info .cluster-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 11px;
                margin-top: 6px;
                background: #ececf1;
                color: #353740;
            }
            
            .graph-container {
                background: #ffffff;
                border-radius: 6px;
                padding: 12px;
                width: 100%;
            }
            
            .content-wrapper {
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 32px;
            }
            
            /* Scrollbar styling - minimal */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #c5c5d1;
                border-radius: 3px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #a5a5b1;
            }
            
            /* Alert styles */
            .alert {
                padding: 12px 16px;
                border-radius: 6px;
                margin: 16px 0;
                font-size: 14px;
                border: 1px solid #fbd38d;
                background: #fffaf0;
                color: #92400e;
            }
            
            .alert-warning {
                border-color: #fbd38d;
                background: #fffaf0;
                color: #92400e;
            }
            
            /* Upload section styles */
            .upload-section {
                padding: 20px;
                border-bottom: 1px solid #e5e5e5;
                background: #f7f7f8;
            }
            
            .upload-box {
                border: 2px dashed #d1d5db;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                background: #ffffff;
                transition: all 0.2s ease;
                cursor: pointer;
            }
            
            .upload-box:hover {
                border-color: #10a37f;
                background: #f9fafb;
            }
            
            .upload-box.dragging {
                border-color: #10a37f;
                background: #f0fdf4;
            }
            
            .upload-icon {
                font-size: 48px;
                color: #8e8ea0;
                margin-bottom: 16px;
            }
            
            .upload-text {
                font-size: 14px;
                color: #353740;
                margin-bottom: 8px;
            }
            
            .upload-subtext {
                font-size: 12px;
                color: #8e8ea0;
            }
            
            .category-input {
                margin-top: 16px;
                padding: 10px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 14px;
                width: 300px;
                max-width: 100%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Function to map video frames to eventfulness datapoints
def map_frame_to_datapoint(frame_number, video_frame_count, eventfulness_length):
    """Maps a video frame number to the corresponding index in the eventfulness data array."""
    if video_frame_count == eventfulness_length:
        return min(frame_number, eventfulness_length - 1)
    ratio = video_frame_count / eventfulness_length
    eventfulness_index = min(int(frame_number / ratio), eventfulness_length - 1)
    return eventfulness_index

# Note: Peak detection and frame extraction are now handled by the backend
# in the run_complete_analysis() workflow for consistency

# Get default video info (but NOT eventfulness data - only load after analysis)
default_video_path = DEFAULT_VIDEO

default_video_info = None
default_eventfulness_data = None  # Always start with None - only populate after "Run Complete Analysis"

if os.path.exists(default_video_path):
    video_info = backend.get_video_info(default_video_path)
    if video_info:
        default_video_info = video_info
        
        # Don't load eventfulness data automatically - only load when "Run Complete Analysis" is clicked
        # This ensures visualization only happens after analysis is complete

# Define the app layout
app.layout = html.Div([
    # Store components
    dcc.Store(id='current-video', data=default_video_path),
    dcc.Store(id='video-info', data=default_video_info),
    dcc.Store(id='eventfulness-data', data=default_eventfulness_data),
    dcc.Store(id='peak-frames', data=None),
    dcc.Store(id='cluster-assignments', data=None),
    dcc.Store(id='cluster-centroids', data=None),
    dcc.Store(id='cosine-similarity-data', data=None),
    dcc.Store(id='full-video-pose-data', data=None),
    dcc.Store(id='dtw-segmentation-data', data=None),
    dcc.Store(id='peak-segmentation-data', data=None),
    dcc.Store(id='wavelet-segmentation-data', data=None),
    
    # Main container
    html.Div([
        # Header
        html.Div([
            html.H1("Video Analysis Dashboard"),
        ], className='header'),
        
        # Main layout
        html.Div([
            # Sidebar for video selection (toolbar style)
            html.Div([
                html.Div([
                    html.H3("Videos"),
                ], className='sidebar-header'),
                
                # Upload section
                html.Div([
                    html.Div([
                        dcc.Upload(
                            id='upload-video',
                            children=html.Div([
                                html.Div('📤', className='upload-icon'),
                                html.Div('Upload Video', className='upload-text'),
                                html.Div('Drag and drop or click to select', className='upload-subtext'),
                            ]),
                            className='upload-box',
                            multiple=False,
                            accept='video/*,.mp4,.avi,.mov,.mkv'
                        ),
                        dcc.Input(
                            id='category-input',
                            type='text',
                            placeholder='Category name (optional)',
                            className='category-input',
                            style={'display': 'block', 'margin': '12px auto 0 auto'}
                        ),
                        html.Div(id='upload-status', style={
                            'marginTop': '12px',
                            'fontSize': '12px',
                            'textAlign': 'center',
                            'color': '#8e8ea0'
                        }),
                    ], style={'padding': '16px'}),
                ], className='upload-section'),
                
                html.Div(id='video-list', className='video-list-container'),
            ], className='sidebar'),
            
            # Main content area
            html.Div([
                # Analysis Control Section - Always visible at top when video loaded
                html.Div([
                    html.Div([
                        html.Div([
                            html.Button("Run Complete Analysis", id='complete-analysis-btn', n_clicks=0, 
                                      className='btn btn-primary', 
                                      style={'background': '#10a37f', 'color': '#ffffff', 'fontWeight': '600', 
                                            'padding': '12px 24px', 'fontSize': '15px', 'cursor': 'pointer', 'marginRight': '12px'}),
                            html.Button("🔄 Reload Visualizations", id='reload-visualizations-btn', n_clicks=0, 
                                      className='btn', 
                                      style={'background': '#ffffff', 'color': '#353740', 'fontWeight': '500', 
                                            'padding': '12px 24px', 'fontSize': '15px', 'cursor': 'pointer',
                                            'border': '1px solid #d1d5db'}),
                            html.Button("🗑️ Clear Data Folders", id='clear-data-folders-btn', n_clicks=0, 
                                      className='btn btn-danger', 
                                      style={'background': '#ffffff', 'color': '#ef4444', 'fontWeight': '500', 
                                            'padding': '12px 24px', 'fontSize': '15px', 'cursor': 'pointer',
                                            'border': '1px solid #fca5a5'}),
                        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '12px'}),
                        html.Div([
                            dcc.Checklist(
                                id='delete-config-checkbox',
                                options=[{'label': ' Restart eventfulness data', 'value': 'delete'}],
                                value=[],
                                style={'fontSize': '12px', 'color': '#8e8ea0', 'opacity': '0.8'}
                            ),
                        ], style={'marginTop': '8px'}),
                        html.Div(id='analysis-status', style={'marginTop': '12px', 'fontSize': '13px', 'color': '#8e8ea0', 'minHeight': '20px'}),
                        html.Div(id='reload-status', style={'marginTop': '8px', 'fontSize': '12px', 'color': '#8e8ea0', 'minHeight': '16px'}),
                        html.Div(id='clear-data-status', style={'marginTop': '8px', 'fontSize': '12px', 'color': '#8e8ea0', 'minHeight': '16px'}),
                    ], style={'textAlign': 'center', 'maxWidth': '1200px', 'margin': '0 auto'}),
                ], className='analysis-control-section', id='analysis-control-section', style={'display': 'none'}),
                
                # View mode toggle
                html.Div([
                    dcc.Checklist(
                        id='simplified-view-toggle',
                        options=[{'label': ' Simplified View (Frequency Analysis Only)', 'value': 'simplified'}],
                        value=['simplified'],  # Default to simplified view
                        style={'fontSize': '13px', 'color': '#353740', 'fontWeight': '500'}
                    ),
                ], style={'textAlign': 'center', 'marginBottom': '12px', 'padding': '8px', 
                         'background': '#f7f7f8', 'borderRadius': '6px'}, id='view-toggle-section', 
                   className='section-card'),
                
                # Video player section
                html.Div([
                    html.Div([
                        dash_player.DashPlayer(
                            id='video-player',
                            url='',
                            controls=True,
                            width='100%',
                            height='600px',
                            intervalCurrentTime=50,
                            playing=False,
                            style={'borderRadius': '8px', 'overflow': 'hidden', 'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'}
                        ),
                        html.Div(id='video-info-minimal', className='video-info-minimal'),
                    ], className='video-player-wrapper'),
                ], className='video-main-container', id='video-section', style={'display': 'none'}),
                
                # Simplified frequency analysis section (shown directly under video when simplified view is on)
                html.Div([
                    html.Div(id='simplified-freq-info', style={'display': 'none'}),  # Hidden info
                    dcc.Graph(
                        id='simplified-freq-graph',
                        style={'height': '160px', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False, 'staticPlot': False}
                    ),
                ], id='simplified-freq-section', style={'display': 'none', 'maxWidth': '1200px', 'margin': '-24px auto 0 auto', 'padding': '0', 'background': 'transparent'}),
                
                # Hidden video info display (for callbacks)
                html.Div(id='video-info-display', style={'display': 'none'}),
                
                # Eventfulness graph section
                html.Div([
                    html.H3("Eventfulness Over Time", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '16px'}),
                    dcc.Graph(
                        id='eventfulness-graph',
                        style={'height': '300px', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                ], className='section-card', id='graph-section', style={'display': 'none'}),
                
                # Peak frames gallery section
                html.Div([
                    html.H3("Peak Frames", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '12px'}),
                    html.Div(id='peak-frames-gallery', className='gallery'),
                ], className='section-card', id='peak-frames-section', style={'display': 'none'}),
                
                # Cosine similarity section
                html.Div([
                    html.H3("Cosine Similarity to Cluster Centroids", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '16px'}),
                    dcc.Graph(
                        id='cosine-similarity-graph',
                        style={'height': '300px', 'width': '100%'},
                        config={'displayModeBar': False, 'displaylogo': False}
                    ),
                ], className='section-card', id='cosine-similarity-section', style={'display': 'none'}),
                
                # Wavelet segmentation section
                html.Div([
                    html.H3("Morlet Wavelet Transform Segmentation", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '8px'}),
                    html.Div(id='wavelet-segmentation-info', style={'fontSize': '13px', 'color': '#8e8ea0', 'marginBottom': '12px'}),
                    # Scalogram heatmap with change signal
                    html.Div([
                        html.H4("Scalogram & Change Detection", style={'fontSize': '14px', 'fontWeight': '500', 'color': '#353740', 'marginBottom': '8px'}),
                        dcc.Graph(
                            id='wavelet-scalogram-graph',
                            style={'height': '500px', 'width': '100%'},
                            config={'displayModeBar': True, 'displaylogo': False}
                        ),
                    ], style={'marginBottom': '20px'}),
                    # Segmentation visualization
                    html.Div([
                        html.H4("Similarity Time Series with Segment Boundaries", style={'fontSize': '14px', 'fontWeight': '500', 'color': '#353740', 'marginBottom': '8px'}),
                        dcc.Graph(
                            id='wavelet-segmentation-graph',
                            style={'height': '500px', 'width': '100%'},
                            config={'displayModeBar': True, 'displaylogo': False}
                        ),
                    ]),
                ], className='section-card', id='wavelet-segmentation-section', style={'display': 'none'}),
                
            ], className='content-area'),
        ], className='main-layout'),
    ], className='main-container'),
    
    # Interval for updating graph marker
    dcc.Interval(
        id='graph-update-interval',
        interval=50,  # Update every 50ms for smooth real-time marker movement
        n_intervals=0,
        disabled=False  # Start enabled, controlled by play/pause callback
    ),
])

# Callback to handle video upload
@callback(
    [Output('upload-status', 'children'),
     Output('upload-status', 'style'),
     Output('current-video', 'data', allow_duplicate=True),
     Output('video-info', 'data', allow_duplicate=True),
     Output('eventfulness-data', 'data', allow_duplicate=True),
     Output('peak-frames', 'data', allow_duplicate=True),
     Output('cluster-assignments', 'data', allow_duplicate=True),
     Output('cosine-similarity-data', 'data', allow_duplicate=True),
     Output('cluster-centroids', 'data', allow_duplicate=True),
     Output('full-video-pose-data', 'data', allow_duplicate=True),
     Output('dtw-segmentation-data', 'data', allow_duplicate=True),
     Output('peak-segmentation-data', 'data', allow_duplicate=True),
     Output('analysis-status', 'children', allow_duplicate=True),
     Output('category-input', 'value')],
    [Input('upload-video', 'contents')],
    [State('upload-video', 'filename'),
     State('category-input', 'value')],
    prevent_initial_call=True
)
def handle_video_upload(contents, filename, category_name):
    """Handle video file upload."""
    if not contents:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Parse the uploaded file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save the video using backend
        success, video_path, message = backend.save_uploaded_video(
            decoded, filename, category_name if category_name else None
        )
        
        if success:
            # Get video info for the newly uploaded video
            video_info = backend.get_video_info(video_path)
            
            # Don't load eventfulness data automatically - only load when "Run Complete Analysis" is clicked
            # This ensures visualization only happens after analysis is complete
            eventfulness_data = None
            
            logger.info(f"Video uploaded successfully: {video_path}")
            
            # Return success status and load the new video
            status_style = {'marginTop': '12px', 'fontSize': '12px', 'textAlign': 'center', 'color': '#10a37f'}
            return (message, status_style, video_path, video_info, eventfulness_data, 
                   None, None, None, None, None, None, None, "", "")
        else:
            # Return error status
            logger.error(f"Video upload failed: {message}")
            status_style = {'marginTop': '12px', 'fontSize': '12px', 'textAlign': 'center', 'color': '#ef4444'}
            return (message, status_style, dash.no_update, dash.no_update, dash.no_update,
                   dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 
                   dash.no_update, dash.no_update, dash.no_update, dash.no_update)
            
    except Exception as e:
        logger.error(f"Error handling video upload: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        status_style = {'marginTop': '12px', 'fontSize': '12px', 'textAlign': 'center', 'color': '#ef4444'}
        return (f"Error: {str(e)}", status_style, dash.no_update, dash.no_update, dash.no_update,
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
               dash.no_update, dash.no_update, dash.no_update, dash.no_update)

# Callback to list videos
@callback(
    Output('video-list', 'children'),
    Input('current-video', 'data')
)
def update_video_list(current_video):
    """List MP4 files in the val directory and update selection state."""
    video_dir = "/home/is1893/Mirror2/dataSets/test_data/val/"
    videos = []
    
    try:
        for root, _, files in os.walk(video_dir):
            for file in sorted(files):
                if file.lower().endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, video_dir)
                    display_name = file if rel_path == '.' else os.path.join(rel_path, file)
                    videos.append({
                        "name": display_name,
                        "path": file_path
                    })
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        return []
    
    # Create video buttons with proper selection state
    video_buttons = []
    for video in videos:
        is_selected = current_video == video['path'] if current_video else False
        video_buttons.append(
            html.Button(
                video['name'],
                id={'type': 'video-button', 'path': video['path']},
                n_clicks=0,
                className='video-button' + (' selected' if is_selected else '')
            )
        )
    
    return video_buttons

# Callback for video selection - clean and reliable
@callback(
    [Output('current-video', 'data'),
     Output('video-info', 'data'),
     Output('eventfulness-data', 'data'),
     Output('peak-frames', 'data', allow_duplicate=True),
     Output('cluster-assignments', 'data', allow_duplicate=True),
     Output('cosine-similarity-data', 'data', allow_duplicate=True),
     Output('cluster-centroids', 'data', allow_duplicate=True),
     Output('full-video-pose-data', 'data', allow_duplicate=True),
     Output('dtw-segmentation-data', 'data', allow_duplicate=True),
     Output('peak-segmentation-data', 'data', allow_duplicate=True),
     Output('analysis-status', 'children', allow_duplicate=True)],
    Input({'type': 'video-button', 'path': dash.ALL}, 'n_clicks'),
    State({'type': 'video-button', 'path': dash.ALL}, 'id'),
    State('current-video', 'data'),
    prevent_initial_call=True
)
def select_video(n_clicks, ids, current_video):
    """Handle video selection with proper state management."""
    # Use callback context to find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return (dash.no_update,) * 11
    
    # Find the index of the clicked button
    triggered_prop = ctx.triggered[0]['prop_id']
    if not triggered_prop or '.n_clicks' not in triggered_prop:
        return (dash.no_update,) * 11
    
    # Find which button was clicked by checking which n_clicks changed
    clicked_index = None
    for i, clicks in enumerate(n_clicks):
        if clicks and clicks > 0:
            clicked_index = i
            break
    
    if clicked_index is None or clicked_index >= len(ids):
        return (dash.no_update,) * 11
    
    # Get the video path from the clicked button
    video_path = ids[clicked_index]['path']
    
    # If clicking the same video, don't reload
    if video_path == current_video:
        return (dash.no_update,) * 11
    
    # Validate video path exists
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return (dash.no_update,) * 11
    
    # Get video info
    video_info = backend.get_video_info(video_path)
    if not video_info:
        logger.error(f"Failed to get video info for: {video_path}")
        return video_path, None, None, None, None, None, None, None, None, None, ""
    
    # Don't load eventfulness data automatically - only load when "Run Complete Analysis" is clicked
    # This ensures visualization only happens after analysis is complete
    eventfulness_data = None
    
    # Clear previous video's analysis data when switching videos
    logger.info(f"Switching to video: {video_path}")
    
    # Return new video data and clear all analysis data from previous video
    return video_path, video_info, eventfulness_data, None, None, None, None, None, None, None, ""

# Callback to update video info display
@callback(
    Output('video-info-display', 'children'),
    [Input('current-video', 'data'),
     Input('video-info', 'data'),
     Input('eventfulness-data', 'data')]
)
def update_video_info_display(video_path, video_info, eventfulness_data):
    """Update video info display (hidden, used for callbacks)."""
    # Return minimal content since we're using video-info-minimal for display
    return html.Div()

# Callback to update video player when video changes
@callback(
    [Output('video-player', 'url'),
     Output('video-section', 'style'),
     Output('video-info-minimal', 'children'),
     Output('analysis-control-section', 'style'),
     Output('peak-frames-section', 'style', allow_duplicate=True),
     Output('cosine-similarity-section', 'style', allow_duplicate=True)],
    [Input('current-video', 'data'),
     Input('video-info', 'data')],
    prevent_initial_call='initial_duplicate'
)
def update_video_player(video_path, video_info):
    """Update video player and reset analysis sections when video changes."""
    if not video_path or not video_info:
        return '', {'display': 'none'}, '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    if not os.path.exists(video_path):
        logger.warning(f"Video file does not exist: {video_path}")
        return '', {'display': 'none'}, '', {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    # Create a data URL for the video
    video_url = f"/video?path={base64.b64encode(video_path.encode()).decode()}"
    
    # Minimal info display
    video_filename = os.path.basename(video_path)
    info_text = f"{video_filename} • {video_info['duration']:.1f}s • {video_info['width']}x{video_info['height']}"
    
    # Show analysis control section and video section when video is loaded
    # Hide analysis sections when switching videos (graph-section is controlled by eventfulness callback)
    return video_url, {'display': 'flex'}, info_text, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}

# Callback to update eventfulness graph
@callback(
    Output('eventfulness-graph', 'figure', allow_duplicate=True),
    [Input('eventfulness-data', 'data'),
     Input('video-info', 'data'),
     Input('current-video', 'data'),
     Input('video-player', 'currentTime'),
     Input('graph-update-interval', 'n_intervals'),
     Input('peak-frames', 'data')],
    [State('video-player', 'playing'),
     State('eventfulness-graph', 'figure')],
    prevent_initial_call=True
)
def update_eventfulness_graph(eventfulness_data, video_info, current_video, current_time, n_intervals, peak_frames, playing, current_figure):
    """Update eventfulness graph with current video position.
    Section visibility is controlled by toggle_section_visibility callback."""
    if not current_video or not os.path.exists(current_video) or not eventfulness_data or not video_info:
        return dash.no_update
    
    data = eventfulness_data['data']
    fps = eventfulness_data.get('fps', video_info['fps'])
    
    # Calculate current position in eventfulness data
    # Use current_time if available (even when paused), otherwise default to 0
    if current_time is not None:
        # Map current time to eventfulness index
        eventfulness_length = len(data)
        ratio = video_info['frame_count'] / eventfulness_length if eventfulness_length > 0 else 1
        current_frame = int(current_time * fps)
        current_index = map_frame_to_datapoint(current_frame, video_info['frame_count'], eventfulness_length)
        # Ensure index is within bounds
        current_index = max(0, min(current_index, len(data) - 1))
    else:
        current_index = 0
    
    current_value = data[current_index] if current_index < len(data) else data[0]
    
    # Check if we need a new figure
    # Force new figure if: no figure exists, video/eventfulness data changed, or peak_frames changed
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Create new figure if triggered by video change, eventfulness data change, or no figure exists
    needs_new_figure = (current_figure is None or 
                       'data' not in current_figure or 
                       len(current_figure['data']) == 0 or
                       triggered_id in ['eventfulness-data', 'current-video', 'peak-frames'])
    
    if needs_new_figure:
        # Initial graph creation - ALWAYS use peaks from eventfulness data
        # The backend stores peak detection results in eventfulness_data for consistency
        if 'peak_indices' in eventfulness_data and 'peak_values' in eventfulness_data:
            # Use pre-computed peaks from backend (guaranteed consistency)
            peaks = eventfulness_data['peak_indices']
            peak_values = eventfulness_data['peak_values']
        elif peak_frames:
            # Fallback: extract from peak_frames if available
            peaks = sorted([int(k) for k in peak_frames.keys()])
            peak_values = [data[p] for p in peaks if p < len(data)]
        else:
            # No peaks available - show empty markers
            peaks = []
            peak_values = []
        
        # Create the graph
        fig = go.Figure()
        
        # Add eventfulness data line with minimal colors
        x = list(range(len(data)))
        fig.add_trace(go.Scatter(
            x=x,
            y=data,
            mode='lines',
            name='Eventfulness',
            line=dict(color='#10a37f', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 163, 127, 0.1)',
            hoverinfo='y+x',
            hovertemplate='<b>Index:</b> %{x}<br><b>Value:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Add markers for peaks
        fig.add_trace(go.Scatter(
            x=peaks,
            y=peak_values,
            mode='markers',
            name='Peaks',
            marker=dict(color='#10a37f', size=8, symbol='circle', line=dict(width=1, color='white')),
            hoverinfo='text',
            hovertext=[f"Peak: {val:.3f}" for val in peak_values]
        ))
        
        # Add vertical line for current position
        fig.add_trace(go.Scatter(
            x=[current_index, current_index],
            y=[min(data), max(data)],
            mode='lines',
            name='Current Position',
            line=dict(color='#8e8ea0', width=2, dash='dot'),
            hoverinfo='none'
        ))
        
        # Add point marker for current value
        fig.add_trace(go.Scatter(
            x=[current_index],
            y=[current_value],
            mode='markers+text',
            name='Current Value',
            marker=dict(color='#202123', size=8, line=dict(width=1, color='white')),
            text=[f"{current_value:.3f}"],
            textposition="top right",
            textfont=dict(size=11, color='#8e8ea0')
        ))
        
        # Update layout with ChatGPT-esque minimal styling
        fig.update_layout(
            showlegend=True,
            xaxis=dict(
                title='Data Point Index',
                titlefont=dict(size=13, color='#8e8ea0'),
                tickfont=dict(size=12, color='#8e8ea0'),
                gridcolor='#f0f0f0',
                zeroline=False,
                showline=False
            ),
            yaxis=dict(
                title='Eventfulness Value',
                titlefont=dict(size=13, color='#8e8ea0'),
                tickfont=dict(size=12, color='#8e8ea0'),
                gridcolor='#f0f0f0',
                zeroline=False,
                showline=False
            ),
            margin=dict(l=50, r=30, t=50, b=40),
            hovermode='closest',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
    else:
        # Update existing figure - always update current position markers for real-time updates
        fig = go.Figure(current_figure)
        
        # Update current position line (trace index 2)
        if len(fig.data) > 2:
            data_min = min(data)
            data_max = max(data)
            # Force update by creating new trace data
            fig.data[2].x = [current_index, current_index]
            fig.data[2].y = [data_min, data_max]
        
        # Update current value marker (trace index 3)
        if len(fig.data) > 3:
            fig.data[3].x = [current_index]
            fig.data[3].y = [current_value]
            fig.data[3].text = [f"{current_value:.3f}"]
            # Update text position to ensure it's visible
            fig.data[3].textposition = "top right"
        
        # Force figure update by updating layout with changing uirevision to ensure redraw
        # Use n_intervals to make the layout change and trigger Plotly to redraw
        fig.update_layout(uirevision=str(n_intervals))
    
    return fig

# Callback to handle video play/pause
@callback(
    Output('graph-update-interval', 'disabled', allow_duplicate=True),
    Input('video-player', 'playing'),
    prevent_initial_call=True
)
def handle_video_playback(playing):
    """Enable/disable graph updates based on video playback."""
    return not playing

# Note: Individual button callbacks removed - use "Run Complete Analysis" button instead
# This ensures consistent peak detection and processing through the backend

# Callback to run complete analysis workflow
@callback(
    [Output('eventfulness-data', 'data', allow_duplicate=True),
     Output('peak-frames', 'data', allow_duplicate=True),
     Output('cluster-assignments', 'data', allow_duplicate=True),
     Output('cluster-centroids', 'data', allow_duplicate=True),
     Output('cosine-similarity-data', 'data', allow_duplicate=True),
     Output('full-video-pose-data', 'data', allow_duplicate=True),
     Output('dtw-segmentation-data', 'data', allow_duplicate=True),
     Output('peak-segmentation-data', 'data', allow_duplicate=True),
     Output('wavelet-segmentation-data', 'data', allow_duplicate=True),
     Output('analysis-status', 'children')],
    [Input('complete-analysis-btn', 'n_clicks')],
    [State('current-video', 'data'),
     State('video-info', 'data'),
     State('delete-config-checkbox', 'value')],
    prevent_initial_call=True
)
def run_complete_analysis(n_clicks, video_path, video_info, delete_config):
    """Run the complete analysis workflow from start to finish."""
    if not n_clicks or not video_path or not video_info:
        return (dash.no_update,) * 10
    
    logger.info(f"Starting complete analysis for: {video_path}")
    
    try:
        # Check if we should delete existing config.json
        if delete_config and 'delete' in delete_config:
            config_path, _ = backend.find_matching_config(video_path)
            if config_path and os.path.exists(config_path):
                try:
                    os.remove(config_path)
                    logger.info(f"Deleted existing config.json at: {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete config.json: {str(e)}")
        
        # Update status
        status_msg = html.Div("Starting complete analysis... This may take several minutes.", style={'color': '#10a37f'})
        
        # Run the complete analysis workflow
        pose_data, eventfulness_data, peak_frames, centroids, similarities, cluster_assignments, dtw_segmentation, peak_segmentation, wavelet_segmentation = backend.run_complete_analysis(
            video_path, num_workers=4)
        
        # Update status based on results
        if pose_data and eventfulness_data and peak_frames:
            status_msg = html.Div([
                html.Div(f"✓ Analysis complete! Processed {len(pose_data)} frames, found {len(peak_frames)} peaks.", style={'color': '#10a37f', 'marginBottom': '4px'}),
                html.Div(f"✓ Created {len(cluster_assignments) if cluster_assignments else 0} cluster assignments." if cluster_assignments else "⚠ No cluster assignments created.", style={'color': '#10a37f' if cluster_assignments else '#fbd38d'}),
                html.Div(f"✓ DTW segmentation: {dtw_segmentation['num_clusters']} clusters segmented." if dtw_segmentation else "⚠ No DTW segmentation.", style={'color': '#10a37f' if dtw_segmentation else '#fbd38d', 'marginTop': '4px'}),
                html.Div(f"✓ Peak segmentation: {peak_segmentation['initial_segment_count']} → {peak_segmentation['final_segment_count']} segments." if peak_segmentation else "⚠ No peak segmentation.", style={'color': '#10a37f' if peak_segmentation else '#fbd38d', 'marginTop': '4px'}),
                html.Div(f"✓ Wavelet segmentation: {wavelet_segmentation['num_segments']} segments." if wavelet_segmentation else "⚠ No wavelet segmentation.", style={'color': '#10a37f' if wavelet_segmentation else '#fbd38d', 'marginTop': '4px'}),
            ])
        elif pose_data:
            status_msg = html.Div("⚠ Analysis partially complete. Pose estimation finished, but eventfulness data or peaks not found.", style={'color': '#fbd38d'})
        else:
            status_msg = html.Div("✗ Analysis failed. Check logs for details.", style={'color': '#ef4444'})
        
        logger.info(f"Complete analysis finished. Results: pose_data={bool(pose_data)}, eventfulness={bool(eventfulness_data)}, peaks={len(peak_frames) if peak_frames else 0}, clusters={len(cluster_assignments) if cluster_assignments else 0}, dtw={bool(dtw_segmentation)}, peak_seg={bool(peak_segmentation)}, wavelet={bool(wavelet_segmentation)}")
        
        # Section visibility is now controlled by toggle_section_visibility callback
        return eventfulness_data, peak_frames, cluster_assignments, centroids, similarities, pose_data, dtw_segmentation, peak_segmentation, wavelet_segmentation, status_msg
        
    except Exception as e:
        logger.error(f"Error in complete analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        status_msg = html.Div(f"✗ Error during analysis: {str(e)}", style={'color': '#ef4444'})
        return None, None, None, None, None, None, None, None, None, status_msg

# Callback to reload and validate all visualizations
@callback(
    [Output('eventfulness-data', 'data', allow_duplicate=True),
     Output('peak-frames', 'data', allow_duplicate=True),
     Output('cluster-assignments', 'data', allow_duplicate=True),
     Output('cluster-centroids', 'data', allow_duplicate=True),
     Output('cosine-similarity-data', 'data', allow_duplicate=True),
     Output('full-video-pose-data', 'data', allow_duplicate=True),
     Output('dtw-segmentation-data', 'data', allow_duplicate=True),
     Output('peak-segmentation-data', 'data', allow_duplicate=True),
     Output('wavelet-segmentation-data', 'data', allow_duplicate=True),
     Output('reload-status', 'children')],
    [Input('reload-visualizations-btn', 'n_clicks')],
    [State('current-video', 'data'),
     State('video-info', 'data'),
     State('eventfulness-data', 'data'),
     State('peak-frames', 'data'),
     State('cluster-assignments', 'data'),
     State('cluster-centroids', 'data'),
     State('cosine-similarity-data', 'data'),
     State('full-video-pose-data', 'data'),
     State('dtw-segmentation-data', 'data'),
     State('peak-segmentation-data', 'data'),
     State('wavelet-segmentation-data', 'data')],
    prevent_initial_call=True
)
def reload_visualizations(n_clicks, video_path, video_info, eventfulness_data, peak_frames, 
                         cluster_assignments, centroids, similarities, pose_data, 
                         dtw_segmentation, peak_segmentation, wavelet_segmentation):
    """
    Reload and validate all visualizations for the current video.
    Checks stored data and reloads from config.json if available.
    Section visibility is controlled by toggle_section_visibility callback.
    """
    if not n_clicks or not video_path or not video_info:
        return (dash.no_update,) * 10
    
    logger.info(f"Reloading visualizations for: {video_path}")
    
    try:
        # Check if we have analysis data in memory
        has_memory_data = bool(eventfulness_data and peak_frames)
        
        # Check if we have config.json on disk
        config_path, config = backend.find_matching_config(video_path)
        has_config_data = bool(config and "eventfulness" in config and len(config["eventfulness"]) > 0)
        
        status_messages = []
        
        # Determine what data to use
        if has_memory_data:
            # Use existing memory data - just validate and refresh
            status_messages.append("✓ Using existing analysis data from memory")
            logger.info("Reloading from memory data")
            
            # Validate eventfulness data
            if eventfulness_data:
                data_points = len(eventfulness_data.get('data', []))
                peaks_count = len(eventfulness_data.get('peak_indices', []))
                status_messages.append(f"✓ Eventfulness: {data_points} points, {peaks_count} peaks")
            
            # Validate peak frames
            if peak_frames:
                status_messages.append(f"✓ Peak frames: {len(peak_frames)} frames")
            
            # Validate clusters
            if cluster_assignments:
                num_clusters = len(set(cluster_assignments.values()))
                status_messages.append(f"✓ Clusters: {num_clusters} clusters, {len(cluster_assignments)} assignments")
            
            # Validate similarities
            if similarities:
                status_messages.append(f"✓ Similarities: {len(similarities)} frames")
            
            # Validate segmentation
            if dtw_segmentation:
                num_segments = len(dtw_segmentation.get('segments', []))
                status_messages.append(f"✓ DTW segmentation: {num_segments} segments")
            
            if peak_segmentation:
                num_segments = len(peak_segmentation.get('final_segments', []))
                status_messages.append(f"✓ Peak segmentation: {num_segments} segments")
            
            if wavelet_segmentation:
                num_segments = len(wavelet_segmentation.get('segments', []))
                status_messages.append(f"✓ Wavelet segmentation: {num_segments} segments")
            
            status_div = html.Div([
                html.Div(msg, style={'fontSize': '12px', 'color': '#10a37f', 'marginBottom': '2px'})
                for msg in status_messages
            ])
            
            # Return existing data to trigger re-render (section visibility controlled by toggle callback)
            return (eventfulness_data, peak_frames, cluster_assignments, centroids, 
                   similarities, pose_data, dtw_segmentation, peak_segmentation, wavelet_segmentation,
                   status_div)
            
        elif has_config_data:
            # Load from config.json
            status_messages.append("✓ Loading analysis data from config.json")
            logger.info(f"Reloading from config.json: {config_path}")
            
            # Load eventfulness data
            reloaded_eventfulness = {
                "data": config["eventfulness"][0],
                "full_vectors": config["eventfulness"],
                "fps": config.get("fps", video_info['fps']),
                "config_path": config_path
            }
            
            # Detect peaks
            data = reloaded_eventfulness['data']
            peaks, peak_values, detection_params = backend.detect_peaks(data)
            reloaded_eventfulness['peak_indices'] = peaks
            reloaded_eventfulness['peak_values'] = peak_values
            reloaded_eventfulness['peak_detection_params'] = detection_params
            
            status_messages.append(f"✓ Loaded eventfulness: {len(data)} points, {len(peaks)} peaks")
            
            # Note: We can only reload eventfulness data from config
            # Other data (peak frames, clusters, etc.) would need to be regenerated
            status_messages.append("⚠ Peak frames and clusters not in memory - run analysis to regenerate")
            
            status_div = html.Div([
                html.Div(msg, style={'fontSize': '12px', 
                                    'color': '#10a37f' if '✓' in msg else '#fbd38d', 
                                    'marginBottom': '2px'})
                for msg in status_messages
            ])
            
            # Return reloaded eventfulness data (section visibility controlled by toggle callback)
            return (reloaded_eventfulness, peak_frames, cluster_assignments, centroids,
                   similarities, pose_data, dtw_segmentation, peak_segmentation, wavelet_segmentation,
                   status_div)
        else:
            # No data available
            status_div = html.Div(
                "⚠ No analysis data found. Click 'Run Complete Analysis' to generate data.",
                style={'fontSize': '12px', 'color': '#fbd38d'}
            )
            logger.warning("No analysis data found for reload")
            
            return (None, None, None, None, None, None, None, None, None, status_div)
            
    except Exception as e:
        logger.error(f"Error reloading visualizations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        status_div = html.Div(
            f"✗ Error reloading: {str(e)}",
            style={'fontSize': '12px', 'color': '#ef4444'}
        )
        
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
               dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
               status_div)

# Callback to update peak frames gallery
@callback(
    Output('peak-frames-gallery', 'children'),
    [Input('peak-frames', 'data'),
     Input('cluster-assignments', 'data')]
)
def update_peak_frames_gallery(peak_frames, cluster_assignments):
    """Update peak frames gallery with cluster colors."""
    if not peak_frames:
        return html.Div("No peak frames extracted yet. Click 'Run Complete Analysis' to begin.", 
                       style={'textAlign': 'center', 'color': '#8e8ea0', 'padding': '40px'})
    
    # Define color palette for clusters
    cluster_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#E74C3C'
    ]
    
    frame_elements = []
    
    # Sort peaks by cluster if available
    if cluster_assignments:
        sorted_peaks = sorted(peak_frames.keys(), 
                            key=lambda x: (cluster_assignments.get(str(x), -999), peak_frames[x]['time']))
    else:
        sorted_peaks = sorted(peak_frames.keys(), key=lambda x: int(x))
    
    for peak_idx in sorted_peaks:
        frame_info = peak_frames[peak_idx]
        
        # Backend structure doesn't include 'path' - it only has frame metadata
        # Check if this is from the old structure (with path) or new backend structure (without path)
        has_image_path = 'path' in frame_info
        
        peak_value = frame_info.get('peak_value', 0)
        time = frame_info.get('time', 0)
        frame_number = frame_info.get('frame_number', 0)
        pose_detected = frame_info.get('pose_detected', False)
        
        # Get cluster assignment if available
        cluster_id = None
        if cluster_assignments and str(peak_idx) in cluster_assignments:
            cluster_id = cluster_assignments[str(peak_idx)]
        
        # Determine border color based on cluster
        border_color = '#ddd'
        border_width = '1px'
        if cluster_id is not None:
            border_color = cluster_colors[cluster_id % len(cluster_colors)]
            border_width = '3px'
        
        # Create cluster badge
        cluster_badge = None
        if cluster_id is not None:
            cluster_badge = html.Div(
                f"Cluster {cluster_id}",
                className='cluster-badge',
                style={
                    'background': border_color,
                    'color': 'white'
                }
            )
        
        # Create thumbnail - different display based on whether we have image paths
        if has_image_path:
            # Old structure with saved images
            frame_path = frame_info['path']
            rel_path = os.path.relpath(frame_path, RESULTS_DIR)
            frame_url = f"/frame/{rel_path}"
            
            thumbnail = html.Div([
                html.Img(src=frame_url, style={
                    'width': '100%', 'height': 'auto', 'display': 'block'}),
                html.Div([
                    html.Div(f"Peak: {peak_value:.3f}"),
                    html.Div(f"Time: {time:.2f}s"),
                    cluster_badge
                ], className='frame-info')
            ], className='frame-thumbnail', style={
                'borderColor': border_color,
                'borderWidth': border_width
            })
        else:
            # New backend structure without saved images - show metadata card
            thumbnail = html.Div([
                html.Div([
                    html.Div(f"Peak #{peak_idx}", style={
                        'fontSize': '16px', 'fontWeight': '600', 'marginBottom': '8px', 'color': '#202123'}),
                    html.Div(f"Value: {peak_value:.3f}", style={'marginBottom': '4px'}),
                    html.Div(f"Time: {time:.2f}s", style={'marginBottom': '4px'}),
                    html.Div(f"Frame: {frame_number}", style={'marginBottom': '4px'}),
                    html.Div(f"Pose: {'✓' if pose_detected else '✗'}", style={
                        'marginBottom': '8px', 
                        'color': '#10a37f' if pose_detected else '#ef4444'
                    }),
                    cluster_badge
                ], style={
                    'padding': '16px',
                    'textAlign': 'center',
                    'minHeight': '180px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': 'center'
                })
            ], className='frame-thumbnail', style={
                'borderColor': border_color,
                'borderWidth': border_width,
                'background': '#f7f7f8'
            })
        
        frame_elements.append(thumbnail)
    
    return frame_elements

# Callback to create cosine similarity graph
@callback(
    Output('cosine-similarity-graph', 'figure'),
    [Input('cosine-similarity-data', 'data'),
     Input('cluster-centroids', 'data')]
)
def create_cosine_similarity_graph(similarities, centroids):
    """Create cosine similarity graph.
    Section visibility is controlled by toggle_section_visibility callback."""
    if not similarities or not centroids:
        return dash.no_update
    
    # Extract frame numbers and similarity scores
    frame_numbers = []
    times = []
    similarity_data = {cluster_id: [] for cluster_id in centroids.keys()}
    
    for frame_idx, data in similarities.items():
        frame_numbers.append(data['frame_number'])
        times.append(data['time'])
        
        for cluster_id, score in data['similarities'].items():
            similarity_data[cluster_id].append(score)
    
    # Define minimal color palette for clusters (ChatGPT style)
    cluster_colors = [
        '#10a37f', '#8e8ea0', '#202123', '#565869', '#a5a5b1',
        '#353740', '#6b7280', '#9ca3af', '#d1d5db', '#e5e7eb'
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add similarity traces with minimal styling
    for i, (cluster_id, scores) in enumerate(similarity_data.items()):
        color = cluster_colors[int(cluster_id) % len(cluster_colors)]
        
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=scores,
            mode='lines',
            name=f'Cluster {cluster_id}',
            line=dict(color=color, width=2),
            opacity=0.7,
            hoverinfo='y+x',
            hovertemplate=f'<b>Cluster {cluster_id}</b><br>Frame: %{{x}}<br>Similarity: %{{y:.3f}}<extra></extra>'
        ))
    
    # Update layout with ChatGPT-esque minimal styling
    fig.update_layout(
        showlegend=True,
        xaxis=dict(
            title='Frame Number',
            titlefont=dict(size=13, color='#8e8ea0'),
            tickfont=dict(size=12, color='#8e8ea0'),
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            title='Cosine Similarity',
            titlefont=dict(size=13, color='#8e8ea0'),
            tickfont=dict(size=12, color='#8e8ea0'),
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=False
        ),
        margin=dict(l=50, r=30, t=50, b=40),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

# Callback to visualize wavelet segmentation results
@callback(
    [Output('wavelet-scalogram-graph', 'figure'),
     Output('wavelet-segmentation-graph', 'figure'),
     Output('wavelet-segmentation-info', 'children')],
    [Input('wavelet-segmentation-data', 'data'),
     Input('cosine-similarity-data', 'data'),
     Input('cluster-centroids', 'data')]
)
def visualize_wavelet_segmentation(wavelet_data, similarities, centroids):
    """Create visualization of frequency-based Morlet wavelet segmentation.
    Section visibility is controlled by toggle_section_visibility callback."""
    if not wavelet_data or not similarities or not centroids:
        return dash.no_update, dash.no_update, ""
    
    # Extract frame numbers and times from similarities
    frame_numbers = []
    times = []
    similarity_data = {cluster_id: [] for cluster_id in centroids.keys()}
    
    for frame_idx, data in similarities.items():
        frame_numbers.append(data['frame_number'])
        times.append(data['time'])
        
        for cluster_id, score in data['similarities'].items():
            similarity_data[cluster_id].append(score)
    
    # Define color palette
    cluster_colors = [
        '#10a37f', '#8e8ea0', '#202123', '#565869', '#a5a5b1',
        '#353740', '#6b7280', '#9ca3af', '#d1d5db', '#e5e7eb'
    ]
    
    # Segment colors for wavelet segments
    segment_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#E74C3C',
        '#3498DB', '#E67E22', '#9B59B6', '#1ABC9C', '#F39C12'
    ]
    
    # Get wavelet data
    scalogram = np.array(wavelet_data['scalogram'])
    frequencies = np.array(wavelet_data.get('frequencies', []))
    wavelet_frame_numbers = wavelet_data.get('frame_numbers', frame_numbers)
    wavelet_times = wavelet_data.get('times', times)
    freq_curve = wavelet_data.get('freq_curve', [])
    freq_change_rate = wavelet_data.get('freq_change_rate', [])
    segments = wavelet_data.get('segments', [])
    change_points = wavelet_data.get('change_points', [])
    
    # Create scalogram figure with subplots: scalogram + frequency curve + change rate
    from plotly.subplots import make_subplots
    
    scalogram_fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.55, 0.25, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            'Scalogram (Time-Frequency Power)',
            'Dominant Frequency Over Time',
            'Frequency Change Rate'
        ]
    )
    
    # ===== Row 1: Scalogram Heatmap =====
    scalogram_fig.add_trace(go.Heatmap(
        z=scalogram,
        x=wavelet_frame_numbers,
        y=frequencies,
        colorscale='Turbo',
        colorbar=dict(
            title='Power',
            titleside='right',
            titlefont=dict(size=11, color='#8e8ea0'),
            tickfont=dict(size=10, color='#8e8ea0'),
            len=0.5,
            y=0.8
        ),
        hovertemplate='Frame: %{x}<br>Frequency: %{y:.2f} Hz<br>Power: %{z:.3f}<extra></extra>'
    ), row=1, col=1)
    
    # Add dominant frequency line on scalogram (overlay)
    if freq_curve:
        scalogram_fig.add_trace(go.Scatter(
            x=wavelet_frame_numbers,
            y=freq_curve,
            mode='lines',
            name='Dominant Freq',
            line=dict(color='white', width=2, dash='dot'),
            opacity=0.8,
            hovertemplate='Frame: %{x}<br>Dom. Freq: %{y:.2f} Hz<extra></extra>'
        ), row=1, col=1)
    
    # ===== Row 2: Frequency Curve =====
    if freq_curve:
        scalogram_fig.add_trace(go.Scatter(
            x=wavelet_frame_numbers,
            y=freq_curve,
            mode='lines',
            name='Frequency',
            line=dict(color='#10a37f', width=2),
            fill='tozeroy',
            fillcolor='rgba(16, 163, 127, 0.15)',
            hovertemplate='Frame: %{x}<br>Frequency: %{y:.2f} Hz<extra></extra>'
        ), row=2, col=1)
        
        # Add segment frequency annotations
        for seg in segments:
            seg_id = seg.get('segment_id', 0)
            seg_color = segment_colors[seg_id % len(segment_colors)]
            dom_freq = seg.get('dominant_freq_hz', 0)
            mid_frame = (seg['start_frame'] + seg['end_frame']) // 2
            
            # Add horizontal line showing segment's dominant frequency
            scalogram_fig.add_shape(
                type="line",
                x0=seg['start_frame'], x1=seg['end_frame'],
                y0=dom_freq, y1=dom_freq,
                line=dict(color=seg_color, width=3, dash='solid'),
                row=2, col=1
            )
    
    # ===== Row 3: Frequency Change Rate =====
    if freq_change_rate:
        scalogram_fig.add_trace(go.Scatter(
            x=wavelet_frame_numbers,
            y=freq_change_rate,
            mode='lines',
            name='Change Rate',
            line=dict(color='#ef4444', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.15)',
            hovertemplate='Frame: %{x}<br>Change Rate: %{y:.2f} Hz/s<extra></extra>'
        ), row=3, col=1)
        
        # Add threshold line
        params = wavelet_data.get('parameters', {})
        threshold = params.get('threshold_hz_per_s', 2.0)
        scalogram_fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.6)",
            line_width=1,
            annotation_text=f"Threshold: {threshold} Hz/s",
            annotation_position="right",
            row=3, col=1
        )
    
    # Add segment boundaries to all rows
    for cp_idx in change_points[1:-1]:  # Skip first and last
        if cp_idx < len(wavelet_frame_numbers):
            frame_num = wavelet_frame_numbers[cp_idx]
            for row in [1, 2, 3]:
                scalogram_fig.add_vline(
                    x=frame_num,
                    line_dash="dash",
                    line_color="rgba(255, 255, 255, 0.9)" if row == 1 else "rgba(100, 100, 100, 0.6)",
                    line_width=2,
                    row=row, col=1
                )
    
    scalogram_fig.update_layout(
        height=580,
        showlegend=False,
        margin=dict(l=70, r=100, t=60, b=50),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12)
    )
    
    # Update y-axes
    scalogram_fig.update_yaxes(
        title_text='Frequency (Hz)',
        titlefont=dict(size=11, color='#8e8ea0'),
        tickfont=dict(size=10, color='#8e8ea0'),
        zeroline=False,
        showline=False,
        row=1, col=1
    )
    scalogram_fig.update_yaxes(
        title_text='Freq (Hz)',
        titlefont=dict(size=11, color='#8e8ea0'),
        tickfont=dict(size=10, color='#8e8ea0'),
        zeroline=False,
        showline=False,
        row=2, col=1
    )
    scalogram_fig.update_yaxes(
        title_text='Hz/s',
        titlefont=dict(size=11, color='#8e8ea0'),
        tickfont=dict(size=10, color='#8e8ea0'),
        zeroline=False,
        showline=False,
        row=3, col=1
    )
    
    # Update x-axis (shared, only show on bottom)
    scalogram_fig.update_xaxes(
        title_text='Frame Number',
        titlefont=dict(size=12, color='#8e8ea0'),
        tickfont=dict(size=10, color='#8e8ea0'),
        zeroline=False,
        showline=False,
        row=3, col=1
    )
    
    # ===== Create Segmentation Visualization =====
    num_clusters = len(centroids)
    seg_fig = make_subplots(
        rows=num_clusters, 
        cols=1,
        subplot_titles=[f'Cluster {cid}' for cid in sorted(centroids.keys())],
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Plot each cluster's similarity with segment boundaries
    for i, cluster_id in enumerate(sorted(centroids.keys())):
        row = i + 1
        color = cluster_colors[int(cluster_id) % len(cluster_colors)]
        
        # Add similarity trace
        seg_fig.add_trace(
            go.Scatter(
                x=frame_numbers,
                y=similarity_data[cluster_id],
                mode='lines',
                name=f'Cluster {cluster_id}',
                line=dict(color=color, width=2),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f'<b>Cluster {cluster_id}</b><br>Frame: %{{x}}<br>Similarity: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Add segment boundaries and shading
        for seg in segments:
            seg_id = seg.get('segment_id', 0)
            seg_color = segment_colors[seg_id % len(segment_colors)]
            
            # Add vertical line at segment start
            seg_fig.add_vline(
                x=seg['start_frame'],
                line_dash="solid",
                line_color=seg_color,
                line_width=2,
                opacity=0.6,
                row=row, col=1
            )
            
            # Add shaded region for segment (only on first row)
            if row == 1:
                seg_fig.add_vrect(
                    x0=seg['start_frame'],
                    x1=seg['end_frame'],
                    fillcolor=seg_color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )
        
        # Add segment labels with frequency info (only on first row)
        if row == 1:
            for seg in segments:
                seg_id = seg.get('segment_id', 0)
                mid_frame = (seg['start_frame'] + seg['end_frame']) // 2
                max_sim = max(max(similarity_data[cid]) for cid in centroids.keys())
                dom_freq = seg.get('dominant_freq_hz', 0)
                
                seg_fig.add_annotation(
                    x=mid_frame,
                    y=max_sim * 0.95,
                    text=f"<b>Seg {seg_id}</b><br>{dom_freq:.1f} Hz",
                    showarrow=False,
                    font=dict(size=9, color='#202123'),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='#e5e5e5',
                    borderwidth=1,
                    borderpad=3,
                    row=row, col=1
                )
    
    # Update segmentation figure layout
    seg_height = max(480, 180 * num_clusters + 100)
    seg_fig.update_layout(
        height=seg_height,
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12)
    )
    
    # Update axes
    seg_fig.update_xaxes(
        title_text='Frame Number',
        titlefont=dict(size=13, color='#8e8ea0'),
        tickfont=dict(size=12, color='#8e8ea0'),
        gridcolor='#f0f0f0',
        zeroline=False,
        showline=False,
        row=num_clusters, col=1
    )
    
    for i in range(1, num_clusters + 1):
        seg_fig.update_yaxes(
            title_text='Similarity',
            titlefont=dict(size=11, color='#8e8ea0'),
            tickfont=dict(size=10, color='#8e8ea0'),
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=False,
            row=i, col=1
        )
    
    # Create info text with segment details
    params = wavelet_data.get('parameters', {})
    freq_range = params.get('freq_range_hz', [0, 5])
    threshold = params.get('threshold_hz_per_s', 2.0)
    threshold_type = params.get('threshold_type', 'manual')
    fs = params.get('sampling_rate_hz', 30)
    
    # Build segment info
    segment_info = []
    for seg in segments:
        seg_id = seg.get('segment_id', 0)
        dom_freq = seg.get('dominant_freq_hz', 0)
        duration = seg.get('duration', 0)
        segment_info.append(f"Seg {seg_id}: {dom_freq:.2f} Hz ({duration:.1f}s)")
    
    # Format threshold display
    threshold_display = f"{threshold:.2f} Hz/s"
    if threshold_type == 'adaptive':
        threshold_display += " (adaptive)"
    
    info_text = html.Div([
        html.Div([
            html.Span(f"Method: Frequency-Based Morlet Wavelet | ", style={'marginRight': '8px', 'fontWeight': '600'}),
            html.Span(f"Segments: {wavelet_data['num_segments']} | ", style={'marginRight': '8px'}),
            html.Span(f"Freq Range: {freq_range[0]:.1f}-{freq_range[1]:.1f} Hz | ", style={'marginRight': '8px'}),
            html.Span(f"Threshold: {threshold_display} | ", style={'marginRight': '8px'}),
            html.Span(f"Sampling: {fs:.1f} Hz | ", style={'marginRight': '8px'}),
            html.Span(f"Total Frames: {wavelet_data['total_frames']}", style={'marginRight': '8px'}),
        ], style={'marginBottom': '8px'}),
        html.Div([
            html.Span("Segment Frequencies: ", style={'fontWeight': '600', 'marginRight': '8px'}),
            html.Span(" | ".join(segment_info), style={'fontSize': '12px', 'color': '#565869'}),
        ]) if segment_info else None
    ])
    
    return scalogram_fig, seg_fig, info_text

# Callback to create simplified frequency analysis graph (aligned with video)
@callback(
    [Output('simplified-freq-graph', 'figure'),
     Output('simplified-freq-info', 'children'),
     Output('simplified-freq-section', 'style')],
    [Input('wavelet-segmentation-data', 'data'),
     Input('simplified-view-toggle', 'value')]
)
def create_simplified_freq_graph(wavelet_data, simplified_view):
    """Create a simplified frequency analysis graph aligned with video scrollbar."""
    is_simplified = 'simplified' in (simplified_view or [])
    
    if not wavelet_data or not is_simplified:
        return dash.no_update, "", {'display': 'none'}
    
    from plotly.subplots import make_subplots
    
    # Get data from wavelet results
    frame_numbers = wavelet_data.get('frame_numbers', [])
    freq_curve = wavelet_data.get('freq_curve', [])
    freq_change_rate = wavelet_data.get('freq_change_rate', [])
    segments = wavelet_data.get('segments', [])
    change_points = wavelet_data.get('change_points', [])
    params = wavelet_data.get('parameters', {})
    threshold = params.get('threshold_hz_per_s', 2.0)
    
    if not frame_numbers or not freq_curve:
        return dash.no_update, "", {'display': 'none'}
    
    # Create figure with two rows - no titles
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    # Segment colors
    segment_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#E74C3C'
    ]
    
    # Add dominant frequency line
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=freq_curve,
        mode='lines',
        name='Dominant Frequency',
        line=dict(color='#10a37f', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(16, 163, 127, 0.15)',
        hovertemplate='Frame: %{x}<br>Freq: %{y:.2f} Hz<extra></extra>'
    ), row=1, col=1)
    
    # Add segment frequency lines
    for seg in segments:
        seg_id = seg.get('segment_id', 0)
        seg_color = segment_colors[seg_id % len(segment_colors)]
        dom_freq = seg.get('dominant_freq_hz', 0)
        start_frame = seg.get('start_frame', 0)
        end_frame = seg.get('end_frame', 0)
        
        # Add horizontal line for segment frequency
        fig.add_shape(
            type="line",
            x0=start_frame, x1=end_frame,
            y0=dom_freq, y1=dom_freq,
            line=dict(color=seg_color, width=2),
            row=1, col=1
        )
    
    # Add frequency change rate
    if freq_change_rate:
        fig.add_trace(go.Scatter(
            x=frame_numbers,
            y=freq_change_rate,
            mode='lines',
            name='Change Rate',
            line=dict(color='#ef4444', width=1),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.15)',
            hovertemplate='Frame: %{x}<br>Change: %{y:.2f} Hz/s<extra></extra>'
        ), row=2, col=1)
    
    # Add segment boundaries
    for cp_idx in change_points[1:-1]:  # Skip first and last
        if cp_idx < len(frame_numbers):
            frame_num = frame_numbers[cp_idx]
            for row in [1, 2]:
                fig.add_vline(
                    x=frame_num,
                    line_dash="dot",
                    line_color="rgba(80, 80, 80, 0.4)",
                    line_width=1,
                    row=row, col=1
                )
    
    # Update layout - minimal margins to align with video
    fig.update_layout(
        height=160,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(250,250,250,0.5)',
        paper_bgcolor='rgba(255,255,255,0)',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=9),
        hovermode='x unified'
    )
    
    # Update axes - no titles, minimal ticks
    fig.update_yaxes(
        showticklabels=False,
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False,
        showline=False,
        row=1, col=1
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False,
        showline=False,
        row=2, col=1
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        row=1, col=1
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        row=2, col=1
    )
    
    return fig, "", {'display': 'block', 'maxWidth': '1200px', 'margin': '-20px auto 16px auto', 'padding': '0', 'background': 'transparent'}

# Callback to control section visibility based on simplified view toggle
@callback(
    [Output('graph-section', 'style'),
     Output('peak-frames-section', 'style'),
     Output('cosine-similarity-section', 'style'),
     Output('wavelet-segmentation-section', 'style')],
    [Input('simplified-view-toggle', 'value'),
     Input('eventfulness-data', 'data'),
     Input('peak-frames', 'data'),
     Input('cosine-similarity-data', 'data'),
     Input('cluster-centroids', 'data'),
     Input('wavelet-segmentation-data', 'data')]
)
def toggle_section_visibility(simplified_view, eventfulness, peak_frames, similarities, centroids, wavelet_seg):
    """Toggle visibility of analysis sections based on simplified view toggle."""
    is_simplified = 'simplified' in (simplified_view or [])
    
    if is_simplified:
        # Hide all detailed sections in simplified view
        return (
            {'display': 'none'},  # graph-section
            {'display': 'none'},  # peak-frames-section
            {'display': 'none'},  # cosine-similarity-section
            {'display': 'none'}   # wavelet-segmentation-section
        )
    else:
        # Show sections based on data availability
        return (
            {'display': 'block'} if eventfulness else {'display': 'none'},
            {'display': 'block'} if peak_frames else {'display': 'none'},
            {'display': 'block'} if similarities and centroids else {'display': 'none'},
            {'display': 'block'} if wavelet_seg else {'display': 'none'}
        )

# Add route for video serving
@server.route('/video')
def serve_video():
    """Serve video files."""
    from flask import request, Response, send_file
    
    # Get the encoded path from the query string
    encoded_path = request.args.get('path', '')
    
    try:
        # Decode the path
        video_path = base64.b64decode(encoded_path).decode()
        
        # Check if the file exists
        if not os.path.exists(video_path):
            return Response("File not found", status=404)
        
        # Serve the file
        return send_file(video_path)
    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)

# Add route for frame serving
@server.route('/frame/<path:frame_path>')
def serve_frame(frame_path):
    """Serve frame images."""
    from flask import send_file, Response
    
    # For security, ensure the path is within the RESULTS_DIR
    full_path = os.path.join(RESULTS_DIR, frame_path)
    if not os.path.exists(full_path):
        return Response("Frame not found", status=404)
    
    return send_file(full_path)

# Callback to clear data folders (lossAccuracyReport and pose-photos)
@callback(
    Output('clear-data-status', 'children'),
    Input('clear-data-folders-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_data_folders(n_clicks):
    """Clear the lossAccuracyReport and pose-photos folders."""
    import shutil
    
    if not n_clicks:
        return ""
    
    folders_to_clear = [
        "/home/is1893/Mirror2/scripts/lossAccuracyReport",
        "/home/is1893/Mirror2/pose-photos",
        "/home/is1893/Mirror2/pose-results"
    ]
    
    results = []
    
    for folder_path in folders_to_clear:
        folder_name = os.path.basename(folder_path)
        try:
            if os.path.exists(folder_path):
                # Count items before clearing
                items = os.listdir(folder_path)
                item_count = len(items)
                
                # Remove all contents but keep the folder
                for item in items:
                    item_path = os.path.join(folder_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                results.append(f"✓ {folder_name}: cleared {item_count} items")
            else:
                results.append(f"⚠ {folder_name}: folder not found")
        except Exception as e:
            results.append(f"✗ {folder_name}: error - {str(e)}")
    
    return html.Div([
        html.Span("Data folders cleared: ", style={'color': '#10a37f', 'fontWeight': '500'}),
        html.Span(" | ".join(results))
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

