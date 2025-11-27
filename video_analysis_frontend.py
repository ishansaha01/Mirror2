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
from scipy.signal import find_peaks
from video_analysis_backend import VideoAnalysisBackend

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

# Get default video info and eventfulness data
default_video_path = DEFAULT_VIDEO
default_config_path, default_config = backend.find_matching_config(default_video_path)

default_video_info = None
default_eventfulness_data = None

if os.path.exists(default_video_path):
    video_info = backend.get_video_info(default_video_path)
    if video_info:
        default_video_info = video_info
        
        # Get default eventfulness data
        if default_config and "eventfulness" in default_config and len(default_config["eventfulness"]) > 0:
            default_eventfulness_data = {
                "data": default_config["eventfulness"][0],
                "full_vectors": default_config["eventfulness"],
                "fps": default_config.get("fps", video_info['fps']),
                "config_path": default_config_path
            }

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
    dcc.Store(id='fluss-segmentation-data', data=None),
    dcc.Store(id='motif-visualization-data', data=None),
    dcc.Store(id='pose-segmentation-data', data=None),
    
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
                html.Div(id='video-list', className='video-list-container'),
            ], className='sidebar'),
            
            # Main content area
            html.Div([
                # Analysis Control Section - Always visible at top when video loaded
                html.Div([
                    html.Div([
                        html.Button("Run Complete Analysis", id='complete-analysis-btn', n_clicks=0, 
                                  className='btn btn-primary', 
                                  style={'background': '#10a37f', 'color': '#ffffff', 'fontWeight': '600', 
                                        'padding': '12px 24px', 'fontSize': '15px', 'cursor': 'pointer'}),
                        html.Div([
                            dcc.Checklist(
                                id='delete-config-checkbox',
                                options=[{'label': ' Restart eventfulness data', 'value': 'delete'}],
                                value=[],
                                style={'fontSize': '12px', 'color': '#8e8ea0', 'opacity': '0.8'}
                            ),
                        ], style={'marginTop': '8px'}),
                        html.Div(id='analysis-status', style={'marginTop': '12px', 'fontSize': '13px', 'color': '#8e8ea0', 'minHeight': '20px'}),
                    ], style={'textAlign': 'center', 'maxWidth': '1200px', 'margin': '0 auto'}),
                ], className='analysis-control-section', id='analysis-control-section', style={'display': 'none'}),
                
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
                
                # Pose Segmentation Visualizations section (NEW)
                html.Div([
                    html.H3("Pose-Based Video Segmentation", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '8px'}),
                    html.Div("Segments created from peak-to-peak analysis with recursive refinement", 
                            style={'fontSize': '13px', 'color': '#8e8ea0', 'marginBottom': '12px'}),
                    html.Div(id='pose-segmentation-info', style={'fontSize': '13px', 'color': '#10a37f', 'marginBottom': '16px', 'fontWeight': '500'}),
                    html.Div(id='pose-segmentation-visualizations', style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}),
                ], className='section-card', id='pose-segmentation-section', style={'display': 'none'}),
                
                # Peak frames gallery section
                html.Div([
                    html.H3("Peak Frames", style={'fontSize': '16px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '12px'}),
                    html.Div(id='peak-frames-gallery', className='gallery'),
                ], className='section-card', id='peak-frames-section', style={'display': 'none'}),
                
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
     Output('fluss-segmentation-data', 'data', allow_duplicate=True),
     Output('motif-visualization-data', 'data', allow_duplicate=True),
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
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Find the index of the clicked button
    triggered_prop = ctx.triggered[0]['prop_id']
    if not triggered_prop or '.n_clicks' not in triggered_prop:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Find which button was clicked by checking which n_clicks changed
    clicked_index = None
    for i, clicks in enumerate(n_clicks):
        if clicks and clicks > 0:
            clicked_index = i
            break
    
    if clicked_index is None or clicked_index >= len(ids):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the video path from the clicked button
    video_path = ids[clicked_index]['path']
    
    # If clicking the same video, don't reload
    if video_path == current_video:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Validate video path exists
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get video info
    video_info = backend.get_video_info(video_path)
    if not video_info:
        logger.error(f"Failed to get video info for: {video_path}")
        return video_path, None, None, None, None, None, None, None, None, None, ""
    
    # Find matching config and eventfulness data
    config_path, config = backend.find_matching_config(video_path)
    eventfulness_data = None
    
    if config and "eventfulness" in config and len(config["eventfulness"]) > 0:
        eventfulness_data = {
            "data": config["eventfulness"][0],
            "full_vectors": config["eventfulness"],
            "fps": config.get("fps", video_info['fps']),
            "config_path": config_path
        }
    
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
     Output('pose-segmentation-section', 'style', allow_duplicate=True)],
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
    [Output('eventfulness-graph', 'figure', allow_duplicate=True),
     Output('graph-section', 'style', allow_duplicate=True)],
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
    """Update eventfulness graph with current video position."""
    if not current_video or not os.path.exists(current_video) or not eventfulness_data or not video_info:
        return dash.no_update, {'display': 'none'}
    
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
        # Initial graph creation - get peaks from peak_frames if available
        # Otherwise use scipy's find_peaks with backend-consistent parameters
        if peak_frames:
            peaks = sorted([int(k) for k in peak_frames.keys()])
            peak_values = [data[p] for p in peaks if p < len(data)]
        else:
            # Use backend-consistent parameters: height=0.3, distance=5
            peaks, _ = find_peaks(data, height=0.3, distance=5)
            peaks = peaks.tolist()
            peak_values = [data[p] for p in peaks]
        
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
    
    return fig, {'display': 'block'}

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
     Output('fluss-segmentation-data', 'data', allow_duplicate=True),
     Output('motif-visualization-data', 'data', allow_duplicate=True),
     Output('pose-segmentation-data', 'data', allow_duplicate=True),
     Output('analysis-status', 'children'),
     Output('graph-section', 'style', allow_duplicate=True),
     Output('peak-frames-section', 'style', allow_duplicate=True),
     Output('pose-segmentation-section', 'style', allow_duplicate=True)],
    [Input('complete-analysis-btn', 'n_clicks')],
    [State('current-video', 'data'),
     State('video-info', 'data'),
     State('delete-config-checkbox', 'value')],
    prevent_initial_call=True
)
def run_complete_analysis(n_clicks, video_path, video_info, delete_config):
    """Run the complete analysis workflow from start to finish."""
    if not n_clicks or not video_path or not video_info:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
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
        
        # Run the complete analysis workflow (FLUSS segmentation and pose segmentation enabled)
        pose_data, eventfulness_data, peak_frames, centroids, similarities, cluster_assignments, fluss_segmentation, pose_segmentation_results = backend.run_complete_analysis(
            video_path, num_workers=4)
        
        # Update status based on results - focus on pose segmentation
        if pose_data and eventfulness_data and peak_frames:
            status_lines = [
                html.Div(f"✓ Analysis complete! Processed {len(pose_data)} frames, found {len(peak_frames)} peaks.", style={'color': '#10a37f', 'marginBottom': '4px'}),
            ]
            
            # Add pose segmentation status if available
            if pose_segmentation_results:
                initial_count = len(pose_segmentation_results.get('initial_segments', []))
                viz_count = len(pose_segmentation_results.get('visualizations', {}))
                refined = pose_segmentation_results.get('refined_segments', {})
                
                status_lines.append(
                    html.Div(f"✓ Pose Segmentation: {initial_count} initial segments created from peaks", 
                            style={'color': '#10a37f', 'marginTop': '4px'})
                )
                
                # Show refinement results
                for strategy, results in refined.items():
                    final_count = len(results.get('final_segments', []))
                    status_lines.append(
                        html.Div(f"  • {strategy}: {initial_count} → {final_count} segments", 
                                style={'color': '#10a37f', 'marginLeft': '20px', 'fontSize': '0.9em'})
                    )
                
                status_lines.append(
                    html.Div(f"✓ Created {viz_count} visualization files", 
                            style={'color': '#10a37f', 'marginTop': '4px'})
                )
            
            status_msg = html.Div(status_lines)
        elif pose_data:
            status_msg = html.Div("⚠ Analysis partially complete. Pose estimation finished, but eventfulness data or peaks not found.", style={'color': '#fbd38d'})
        else:
            status_msg = html.Div("✗ Analysis failed. Check logs for details.", style={'color': '#ef4444'})
        
        # Check for motif visualization files
        motif_viz_data = None
        motif_viz_path = os.path.join(RESULTS_DIR, 'mstump_motifs.png')
        motif_summary_path = os.path.join(RESULTS_DIR, 'mstump_motifs_summary.png')
        
        if os.path.exists(motif_viz_path) and os.path.exists(motif_summary_path):
            motif_viz_data = {
                'motif_viz_path': motif_viz_path,
                'motif_summary_path': motif_summary_path,
                'timestamp': os.path.getmtime(motif_viz_path)
            }
        
        # Determine which sections to show - hide clustering/FLUSS, show pose segmentation
        graph_style = {'display': 'block'} if eventfulness_data else {'display': 'none'}
        peak_style = {'display': 'block'} if peak_frames else {'display': 'none'}
        pose_seg_style = {'display': 'block'} if pose_segmentation_results else {'display': 'none'}
        
        logger.info(f"Complete analysis finished. Results: pose_data={bool(pose_data)}, eventfulness={bool(eventfulness_data)}, peaks={len(peak_frames) if peak_frames else 0}, clusters={len(cluster_assignments) if cluster_assignments else 0}, fluss={bool(fluss_segmentation)}, motifs={bool(motif_viz_data)}")
        
        return eventfulness_data, peak_frames, cluster_assignments, centroids, similarities, pose_data, fluss_segmentation, motif_viz_data, pose_segmentation_results, status_msg, graph_style, peak_style, pose_seg_style
        
    except Exception as e:
        logger.error(f"Error in complete analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        status_msg = html.Div(f"✗ Error during analysis: {str(e)}", style={'color': '#ef4444'})
        return None, None, None, None, None, None, None, None, None, status_msg, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Callback to update pose segmentation visualizations
@callback(
    [Output('pose-segmentation-info', 'children'),
     Output('pose-segmentation-visualizations', 'children')],
    Input('pose-segmentation-data', 'data')
)
def update_pose_segmentation_display(pose_seg_data):
    """Display pose segmentation visualizations."""
    if not pose_seg_data:
        return "", html.Div("No pose segmentation data available.", 
                           style={'textAlign': 'center', 'color': '#8e8ea0', 'padding': '40px'})
    
    # Extract info
    initial_count = len(pose_seg_data.get('initial_segments', []))
    refined = pose_seg_data.get('refined_segments', {})
    visualizations = pose_seg_data.get('visualizations', {})
    
    # Create info summary
    info_lines = [f"Created {initial_count} initial segments from eventfulness peaks"]
    for strategy, results in refined.items():
        final_count = len(results.get('final_segments', []))
        info_lines.append(f" | {strategy}: {final_count} segments")
    
    info_text = " ".join(info_lines)
    
    # Create visualization elements
    viz_elements = []
    
    # Define order and titles for visualizations
    viz_order = [
        ('cosine_similarities', '📊 Cosine Similarity by Cluster Over Time'),
        ('timeline', 'Segmentation Timeline'),
        ('iterative_merge_similar', '🎬 Step-by-Step: Merge Similar Strategy'),
        ('iterative_hierarchical', '🎬 Step-by-Step: Hierarchical Strategy'),
        ('iterative_boundary_refinement', '🎬 Step-by-Step: Boundary Refinement Strategy'),
        ('similarity_matrix', 'Segment Similarity Matrix'),
        ('threshold_analysis', 'Adaptive Threshold Analysis'),
        ('statistics', 'Segment Statistics'),
        ('pose_changes', 'Pose Change Detection'),
        ('merge_history_merge_similar', 'Merge History Summary (Merge Similar)'),
        ('merge_history_hierarchical', 'Merge History Summary (Hierarchical)'),
        ('merge_history_boundary_refinement', 'Merge History Summary (Boundary Refinement)'),
        ('comparisons', 'Segment Comparisons'),
        ('segment_0', 'Segment 0 Creation Analysis'),
    ]
    
    for viz_key, viz_title in viz_order:
        if viz_key in visualizations:
            viz_path = visualizations[viz_key]
            if os.path.exists(viz_path):
                # Convert to base64 for display
                import base64
                with open(viz_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()
                
                viz_elements.append(
                    html.Div([
                        html.H4(viz_title, style={'fontSize': '14px', 'fontWeight': '600', 
                                                  'color': '#202123', 'marginBottom': '8px'}),
                        html.Img(
                            src=f'data:image/png;base64,{encoded}',
                            style={'width': '100%', 'borderRadius': '4px', 
                                  'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}
                        ),
                    ], style={'marginBottom': '20px'})
                )
    
    if not viz_elements:
        viz_elements = [html.Div("No visualizations found.", 
                                style={'textAlign': 'center', 'color': '#8e8ea0', 'padding': '20px'})]
    
    return info_text, viz_elements

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

# Callback to create cosine similarity graph (DISABLED - not shown in UI)
# Callback disabled - cosine similarity section removed from UI
# @callback(
#     [Output('cosine-similarity-graph', 'figure'),
#      Output('cosine-similarity-section', 'style')],
#     [Input('cosine-similarity-data', 'data'),
#      Input('cluster-centroids', 'data')]
# )
# def create_cosine_similarity_graph(similarities, centroids):
#     """Create cosine similarity graph."""
#     if not similarities or not centroids:
#         return dash.no_update, {'display': 'none'}
#     
#     # Extract frame numbers and similarity scores
#     frame_numbers = []
#     times = []
#     similarity_data = {cluster_id: [] for cluster_id in centroids.keys()}
#     
#     for frame_idx, data in similarities.items():
#         frame_numbers.append(data['frame_number'])
#         times.append(data['time'])
#         
#         for cluster_id, score in data['similarities'].items():
#             similarity_data[cluster_id].append(score)
#     
#     # Define minimal color palette for clusters (ChatGPT style)
#     cluster_colors = [
#         '#10a37f', '#8e8ea0', '#202123', '#565869', '#a5a5b1',
#         '#353740', '#6b7280', '#9ca3af', '#d1d5db', '#e5e7eb'
#     ]
#     
#     # Create figure
#     fig = go.Figure()
#     
#     # Add similarity traces with minimal styling
#     for i, (cluster_id, scores) in enumerate(similarity_data.items()):
#         color = cluster_colors[int(cluster_id) % len(cluster_colors)]
#         
#         fig.add_trace(go.Scatter(
#             x=frame_numbers,
#             y=scores,
#             mode='lines',
#             name=f'Cluster {cluster_id}',
#             line=dict(color=color, width=2),
#             opacity=0.7,
#             hoverinfo='y+x',
#             hovertemplate=f'<b>Cluster {cluster_id}</b><br>Frame: %{{x}}<br>Similarity: %{{y:.3f}}<extra></extra>'
#         ))
#     
#     # Update layout with ChatGPT-esque minimal styling
#     fig.update_layout(
#         showlegend=True,
#         xaxis=dict(
#             title='Frame Number',
#             titlefont=dict(size=13, color='#8e8ea0'),
#             tickfont=dict(size=12, color='#8e8ea0'),
#             gridcolor='#f0f0f0',
#             zeroline=False,
#             showline=False
#         ),
#         yaxis=dict(
#             title='Cosine Similarity',
#             titlefont=dict(size=13, color='#8e8ea0'),
#             tickfont=dict(size=12, color='#8e8ea0'),
#             gridcolor='#f0f0f0',
#             zeroline=False,
#             showline=False
#         ),
#         margin=dict(l=50, r=30, t=50, b=40),
#         hovermode='closest',
#         plot_bgcolor='#ffffff',
#         paper_bgcolor='#ffffff',
#         font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12),
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1,
#             font=dict(size=11),
#             bgcolor='rgba(255,255,255,0.8)'
#         )
#     )
#     
#     return fig, {'display': 'block'}

# Callback disabled - motif section removed from UI
# @callback(
#     [Output('motif-images-container', 'children'),
#      Output('motif-section', 'style')],
#     Input('motif-visualization-data', 'data')
# )
# def display_motif_visualizations(motif_data):
    """Display the motif visualization images."""
    if not motif_data:
        return html.Div(), {'display': 'none'}
    
    motif_viz_path = motif_data.get('motif_viz_path')
    motif_summary_path = motif_data.get('motif_summary_path')
    
    if not motif_viz_path or not motif_summary_path:
        return html.Div(), {'display': 'none'}
    
    # Check if files exist
    if not os.path.exists(motif_viz_path) or not os.path.exists(motif_summary_path):
        return html.Div("Motif visualization files not found.", 
                       style={'color': '#ef4444', 'textAlign': 'center', 'padding': '20px'}), {'display': 'block'}
    
    # Create relative paths for serving
    rel_motif_path = os.path.relpath(motif_viz_path, RESULTS_DIR)
    rel_summary_path = os.path.relpath(motif_summary_path, RESULTS_DIR)
    
    # Create image URLs
    motif_url = f"/frame/{rel_motif_path}"
    summary_url = f"/frame/{rel_summary_path}"
    
    # Create the image display elements
    content = html.Div([
        # Detailed motif comparison
        html.Div([
            html.H4("Detailed Motif Comparison", 
                   style={'fontSize': '14px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '8px'}),
            html.Div("Shows the top repeated patterns with their matching pairs across all dimensions", 
                    style={'fontSize': '12px', 'color': '#8e8ea0', 'marginBottom': '12px'}),
            html.Img(src=motif_url, 
                    style={'width': '100%', 'height': 'auto', 'border': '1px solid #e5e5e5', 
                           'borderRadius': '6px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'})
        ], style={'marginBottom': '24px'}),
        
        # Summary view
        html.Div([
            html.H4("Motif Locations in Time Series", 
                   style={'fontSize': '14px', 'fontWeight': '600', 'color': '#202123', 'marginBottom': '8px'}),
            html.Div("Shows where each motif occurs in the full time series (matching pairs have the same color)", 
                    style={'fontSize': '12px', 'color': '#8e8ea0', 'marginBottom': '12px'}),
            html.Img(src=summary_url, 
                    style={'width': '100%', 'height': 'auto', 'border': '1px solid #e5e5e5', 
                           'borderRadius': '6px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'})
        ])
    ])
    
    return content, {'display': 'block'}

# Callback disabled - FLUSS section removed from UI
# @callback(
#     [Output('fluss-segmentation-graph', 'figure'),
#      Output('fluss-segmentation-info', 'children'),
#      Output('fluss-segmentation-section', 'style')],
#     [Input('fluss-segmentation-data', 'data'),
#      Input('cosine-similarity-data', 'data'),
#      Input('cluster-centroids', 'data')]
# )
# def visualize_fluss_segmentation(fluss_data, similarities, centroids):
    """Create visualization of FLUSS segmentation results with GLOBAL segments across all clusters."""
    if not fluss_data or not similarities or not centroids:
        return dash.no_update, "", {'display': 'none'}
    
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
    
    # Create figure with subplots for each cluster + arc curve
    from plotly.subplots import make_subplots
    
    num_clusters = len(centroids)
    arc_curve = fluss_data.get('arc_curve', None)
    
    # Add extra row for arc curve if available
    total_rows = num_clusters + (1 if arc_curve else 0)
    
    # Create subplot titles
    subplot_titles = [f'Cluster {cid}' for cid in sorted(centroids.keys())]
    if arc_curve:
        subplot_titles.append('FLUSS Arc Curve (CAC)')
    
    fig = make_subplots(
        rows=total_rows, 
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08 if not arc_curve else 0.06,
        shared_xaxes=True
    )
    
    # Get GLOBAL change points and segments (not per-cluster)
    change_points = fluss_data.get('change_points', [])
    segments = fluss_data.get('segments', [])
    
    # Plot each cluster's similarity with GLOBAL segment boundaries
    for i, cluster_id in enumerate(sorted(centroids.keys())):
        row = i + 1
        color = cluster_colors[int(cluster_id) % len(cluster_colors)]
        
        # Add similarity trace
        fig.add_trace(
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
        
        # Add GLOBAL segment boundaries as vertical lines (same for all clusters)
        for cp_idx in change_points[1:-1]:  # Skip first and last
            # Map change point index to frame number
            if cp_idx < len(frame_numbers):
                frame_num = frame_numbers[cp_idx]
                fig.add_vline(
                    x=frame_num,
                    line_dash="dash",
                    line_color="rgba(255, 0, 0, 0.5)",
                    line_width=2,
                    row=row, col=1
                )
        
        # Add segment labels (only on first row to avoid clutter)
        if row == 1:
            for seg in segments:
                mid_frame = (seg['start_frame'] + seg['end_frame']) // 2
                # Find max similarity across all clusters for positioning
                max_sim = max(max(similarity_data[cid]) for cid in centroids.keys())
                fig.add_annotation(
                    x=mid_frame,
                    y=max_sim * 0.95,
                    text=f"<b>Segment {seg['segment_id']}</b>",
                    showarrow=False,
                    font=dict(size=11, color='#202123'),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='#e5e5e5',
                    borderwidth=1,
                    borderpad=4,
                    row=row, col=1
                )
    
    # Add FLUSS Arc Curve if available
    if arc_curve:
        arc_row = num_clusters + 1
        
        # The arc curve shows the likelihood of regime changes
        # Peaks in the arc curve correspond to change points
        fig.add_trace(
            go.Scatter(
                x=list(range(len(arc_curve))),
                y=arc_curve,
                mode='lines',
                name='Arc Curve',
                line=dict(color='#ef4444', width=2),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)',
                showlegend=False,
                hovertemplate='<b>Arc Curve</b><br>Index: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ),
            row=arc_row, col=1
        )
        
        # Mark detected change points on the arc curve
        for cp_idx in change_points[1:-1]:  # Skip first and last
            if cp_idx < len(arc_curve):
                fig.add_trace(
                    go.Scatter(
                        x=[cp_idx],
                        y=[arc_curve[cp_idx]],
                        mode='markers',
                        marker=dict(color='#ef4444', size=10, symbol='diamond', 
                                  line=dict(width=2, color='white')),
                        showlegend=False,
                        hovertemplate=f'<b>Change Point</b><br>Index: {cp_idx}<br>Arc Value: {arc_curve[cp_idx]:.3f}<extra></extra>'
                    ),
                    row=arc_row, col=1
                )
        
        # Update y-axis for arc curve
        fig.update_yaxes(
            title_text='Arc Curve Value',
            titlefont=dict(size=11, color='#8e8ea0'),
            tickfont=dict(size=10, color='#8e8ea0'),
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=False,
            row=arc_row, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=150 * total_rows + 100,
        showlegend=False,
        margin=dict(l=50, r=30, t=50, b=40),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size=12)
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='Index' if arc_curve else 'Frame Number',
        titlefont=dict(size=13, color='#8e8ea0'),
        tickfont=dict(size=12, color='#8e8ea0'),
        gridcolor='#f0f0f0',
        zeroline=False,
        showline=False,
        row=total_rows, col=1
    )
    
    for i in range(1, num_clusters + 1):
        fig.update_yaxes(
            title_text='Similarity',
            titlefont=dict(size=11, color='#8e8ea0'),
            tickfont=dict(size=10, color='#8e8ea0'),
            gridcolor='#f0f0f0',
            zeroline=False,
            showline=False,
            row=i, col=1
        )
    
    # Create info text
    total_segments = len(segments)
    params = fluss_data['parameters']
    info_text = html.Div([
        html.Span(f"Method: FLUSS (Matrix Profile) | ", style={'marginRight': '8px', 'fontWeight': '600'}),
        html.Span(f"Global Segments: {total_segments} | ", style={'marginRight': '8px'}),
        html.Span(f"Window: {params['window_size']} | ", style={'marginRight': '8px'}),
        html.Span(f"Regimes: {params.get('num_regimes', 'auto')} | ", style={'marginRight': '8px'}),
        html.Span(f"Min Length: {params['min_segment_length']} | ", style={'marginRight': '8px'}),
        html.Span(f"Dimensions: {params.get('vector_dimensions', 'N/A')}", style={'marginRight': '8px'}),
    ])
    
    return fig, info_text, {'display': 'block'}

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

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

