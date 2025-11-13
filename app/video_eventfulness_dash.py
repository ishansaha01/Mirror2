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
import datetime
from scipy.signal import find_peaks

# Set up logging
log_file = "/home/is1893/Mirror2/dataSets/test_data/results/sync_debug.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('video_sync')

# Define the base directory for browsing
DEFAULT_BASE_DIR = "/home/is1893/Mirror2/dataSets/test_data"
RESULTS_DIR = "/home/is1893/Mirror2/dataSets/test_data/results"
DEFAULT_VIDEO = "/home/is1893/Mirror2/dataSets/test_data/val/Hip_hop_dance_basic_teaching/Hip_hop_dance_basic_teaching.mp4"

# Function to map video frames to eventfulness datapoints
def map_frame_to_datapoint(frame_number, video_frame_count, eventfulness_length):
    """Maps a video frame number to the corresponding index in the eventfulness data array."""
    # Simple linear mapping
    if video_frame_count == eventfulness_length:
        # 1:1 mapping
        return min(frame_number, eventfulness_length - 1)
    
    # Calculate the ratio between video frames and eventfulness datapoints
    ratio = video_frame_count / eventfulness_length
    
    # Map the frame number to the corresponding eventfulness index
    eventfulness_index = min(int(frame_number / ratio), eventfulness_length - 1)
    
    return eventfulness_index

# Function to find matching config for a video
def find_matching_config(video_path):
    """Find the matching config.json file for a given video path."""
    config_files = glob.glob(os.path.join(RESULTS_DIR, "**/config.json"), recursive=True)
    
    video_path_normalized = os.path.normpath(video_path)
    video_filename = os.path.basename(video_path)
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            if "video_path" in config:
                config_video_path = config["video_path"]
                config_video_filename = os.path.basename(config_video_path)
                
                if (os.path.normpath(config_video_path) == video_path_normalized or 
                    config_video_filename == video_filename):
                    return config_file, config
        except Exception:
            continue
    
    return None, None

# Function to list directories and videos
def list_directories_and_videos(directory):
    """List directories and video files in the given directory."""
    items = []
    
    # Add parent directory if not at root
    if directory != "/":
        items.append({"name": "..", "type": "directory", "path": str(Path(directory).parent)})
    
    try:
        for item in sorted(os.listdir(directory)):
            item_path = os.path.join(directory, item)
            
            if os.path.isdir(item_path):
                items.append({"name": item, "type": "directory", "path": item_path})
            elif item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                items.append({
                    "name": item, 
                    "type": "video", 
                    "path": item_path
                })
    except Exception:
        pass
    
    return items

# Function to list only MP4 files in the val directory
def list_mp4_files(directory="/home/is1893/Mirror2/dataSets/test_data/val/"):
    """List only MP4 files in the given directory and its subdirectories."""
    items = []
    
    try:
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if file.lower().endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    # Get the relative path from the base directory for display
                    rel_path = os.path.relpath(root, directory)
                    display_name = file if rel_path == '.' else os.path.join(rel_path, file)
                    items.append({
                        "name": display_name, 
                        "type": "video", 
                        "path": file_path
                    })
    except Exception:
        pass
    
    return items

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define custom CSS for animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .pulse {
                animation: pulse 1s infinite;
            }
            
            .blink {
                animation: blink 1s infinite;
            }
            
            .volume-bar {
                display: inline-block;
                width: 8px;
                margin-right: 2px;
                border-radius: 2px;
                background-color: #ddd;
            }
            
            .volume-bar.active {
                background-color: #2196F3;
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

# Get default video info and eventfulness data
default_video_path = DEFAULT_VIDEO
default_config_path, default_config = find_matching_config(default_video_path)

# Get default video info
import cv2
default_video_info = None
default_eventfulness_data = None

if os.path.exists(default_video_path):
    cap = cv2.VideoCapture(default_video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        default_video_info = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
        
        # Get default eventfulness data
        if default_config and "eventfulness" in default_config and len(default_config["eventfulness"]) > 0:
            default_eventfulness_data = {
                "data": default_config["eventfulness"][0],
                "fps": default_config.get("fps", fps)
            }

# Define the app layout
app.layout = html.Div([
    html.H1("Video Eventfulness Analyzer"),
    
    # Store components to keep track of state
    dcc.Store(id='current-directory', data="/home/is1893/Mirror2/dataSets/test_data/val/"),
    dcc.Store(id='current-video', data=default_video_path),
    dcc.Store(id='video-info', data=default_video_info),
    dcc.Store(id='eventfulness-data', data=default_eventfulness_data),
    dcc.Store(id='peak-data', data=None),  # Store for detected peaks and dance steps
    
    # Main layout with file browser and content area
    html.Div([
        # File browser sidebar
        html.Div([
            html.H3("MP4 Files"),
            
            # Directory contents (no input field)
            html.Div(id='directory-contents'),
            
        ], style={'width': '25%', 'float': 'left', 'padding': '10px', 'borderRight': '1px solid #ccc'}),
        
        # Main content area
        html.Div([
            # Video info section
            html.Div(id='video-info-display'),
            
            # Video and graph section
            html.Div([
                # Video player
                html.Div([
                    html.H3("Video Player"),
                    html.Div(id='video-player-container'),
                ], id='video-section', style={'display': 'none'}),
                
                # Eventfulness graph
                html.Div([
                    html.H3("Eventfulness Data"),
                    dcc.Graph(
                        id='eventfulness-graph',
                        style={'height': '300px'},
                        config={'displayModeBar': False}
                    ),
                ], id='graph-section', style={'display': 'none'}),
                
                # Local maxima indicator
                html.Div([
                    html.H3("Local Maxima Detection"),
                    html.Div([
                        html.Div([
                            html.Div("Next Peak:", style={'fontWeight': 'bold', 'display': 'inline-block', 'marginRight': '10px'}),
                            html.Div(id='next-peak-time', style={'display': 'inline-block', 'fontSize': '1.2em'})
                        ]),
                        html.Div([
                            html.Div("Peak Value:", style={'fontWeight': 'bold', 'display': 'inline-block', 'marginRight': '10px'}),
                            html.Div(id='peak-value', style={'display': 'inline-block', 'marginRight': '10px'}),
                            html.Div(id='volume-indicator', style={'display': 'inline-block'})
                        ]),
                    ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'backgroundColor': '#f9f9f9'})
                ], id='dance-step-section', style={'display': 'none', 'marginTop': '20px'}),
            ]),
        ], style={'width': '75%', 'float': 'left', 'padding': '10px'}),
    ], style={'display': 'flex', 'flexFlow': 'row'}),
    
    # Interval for updating the graph marker
    dcc.Interval(
        id='graph-update-interval',
        interval=100,  # Update every 100ms
        n_intervals=0,
        disabled=True
    ),
])

# Callback to update directory contents
@callback(
    Output('directory-contents', 'children'),
    Input('current-directory', 'data')
)
def update_directory_contents(current_dir):
    # Always use the fixed directory
    fixed_directory = "/home/is1893/Mirror2/dataSets/test_data/val/"
    
    # Get only MP4 files from the fixed directory
    items = list_mp4_files(fixed_directory)
    
    # Create clickable items for videos
    content_elements = []
    for item in items:
        button = html.Button(
            f"🎬 {item['name']}",
            id={'type': 'video-button', 'path': item['path']},
            style={'display': 'block', 'width': '100%', 'textAlign': 'left', 'margin': '2px 0'}
        )
        content_elements.append(button)
    
    return content_elements

# Directory navigation callback removed as it's no longer needed

# Callback for video selection
@callback(
    [Output('current-video', 'data'),
     Output('video-info', 'data'),
     Output('eventfulness-data', 'data')],
    Input({'type': 'video-button', 'path': dash.ALL}, 'n_clicks'),
    State({'type': 'video-button', 'path': dash.ALL}, 'id'),
    prevent_initial_call=True
)
def select_video(n_clicks, ids):
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the index of the clicked button
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Find which button was clicked by matching the triggered ID
    for i, n in enumerate(n_clicks):
        if n is not None:  # This button has been clicked
            button_data = ids[i]
            video_path = button_data['path']
            break
    else:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_path, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    video_info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    # Find matching config and eventfulness data
    config_path, config = find_matching_config(video_path)
    eventfulness_data = None
    
    if config and "eventfulness" in config and len(config["eventfulness"]) > 0:
        eventfulness_data = {
            "data": config["eventfulness"][0],
            "fps": config.get("fps", fps)
        }
    
    return video_path, video_info, eventfulness_data

# Callback to update video info display
@callback(
    Output('video-info-display', 'children'),
    Input('current-video', 'data'),
    Input('video-info', 'data'),
    Input('eventfulness-data', 'data')
)
def update_video_info_display(video_path, video_info, eventfulness_data):
    if not video_path or not video_info:
        return html.Div("Select a video from the sidebar to begin.")
    
    video_filename = os.path.basename(video_path)
    
    info_elements = [
        html.H2(f"Selected Video: {video_filename}"),
        html.Div([
            html.Div([
                html.Strong("Resolution: "),
                html.Span(f"{video_info['width']}x{video_info['height']}")
            ], style={'margin': '5px 0'}),
            html.Div([
                html.Strong("Duration: "),
                html.Span(f"{video_info['duration']:.2f} seconds")
            ], style={'margin': '5px 0'}),
            html.Div([
                html.Strong("FPS: "),
                html.Span(f"{video_info['fps']:.2f}")
            ], style={'margin': '5px 0'}),
            html.Div([
                html.Strong("Total Frames: "),
                html.Span(f"{video_info['frame_count']}")
            ], style={'margin': '5px 0'}),
        ])
    ]
    
    if eventfulness_data:
        data_length = len(eventfulness_data['data'])
        mapping_ratio = video_info['frame_count'] / data_length
        
        info_elements.append(html.Div([
            html.Div([
                html.Strong("Eventfulness Data Points: "),
                html.Span(f"{data_length}")
            ], style={'margin': '5px 0'}),
            html.Div([
                html.Strong("Mapping Ratio: "),
                html.Span(f"{mapping_ratio:.4f} frames per data point")
            ], style={'margin': '5px 0'}),
        ]))
    else:
        info_elements.append(html.Div(
            "No eventfulness data available for this video.",
            style={'color': 'orange', 'fontWeight': 'bold', 'margin': '10px 0'}
        ))
    
    return html.Div(info_elements)

# Callback to update video player
@callback(
    Output('video-player-container', 'children'),
    Output('video-section', 'style'),
    Output('dance-step-section', 'style'),  # Also show the dance step section
    Input('current-video', 'data')
)
def update_video_player(video_path):
    if not video_path or not os.path.exists(video_path):
        return dash.no_update, {'display': 'none'}, {'display': 'none'}
    
    # Create a data URL for the video
    video_url = f"/video?path={base64.b64encode(video_path.encode()).decode()}"
    
    player = dash_player.DashPlayer(
        id='video-player',
        url=video_url,
        controls=True,
        width='100%',
        height='auto',
        # Use intervalCurrentTime instead of intervalDelay
        intervalCurrentTime=100,  # Update time every 100ms
        playing=False
    )
    
    return player, {'display': 'block'}, {'display': 'block'}

# Callback to update eventfulness graph
@callback(
    Output('eventfulness-graph', 'figure'),
    Output('graph-section', 'style'),
    Input('eventfulness-data', 'data'),
    Input('video-info', 'data'),
    Input('current-video', 'data')
)
def update_eventfulness_graph(eventfulness_data, video_info, current_video):
    # Skip if no video is selected or no data available
    if not current_video or not os.path.exists(current_video) or not eventfulness_data or not video_info:
        return dash.no_update, {'display': 'none'}
    if not eventfulness_data or not video_info:
        return dash.no_update, {'display': 'none'}
    
    data = eventfulness_data['data']
    fps = eventfulness_data.get('fps', video_info['fps'])
    
    # Calculate initial position
    initial_index = 0
    initial_value = data[0] if len(data) > 0 else 0
    
    # Detect local maxima (peaks) in the eventfulness data
    peaks = detect_local_maxima(data)
    peak_values = [data[p] for p in peaks]
    
    # We're only focusing on local maxima now
    
    # Find the next upcoming peak
    next_peak, distance_to_peak, next_peak_value = find_next_maximum(data, initial_index, peaks)
    
    # Log the initial graph setup and peak detection
    logger.info(f"Creating eventfulness graph with {len(data)} data points")
    logger.info(f"Initial marker position: index={initial_index}, value={initial_value:.3f}")
    logger.info(f"Detected {len(peaks)} peaks")
    if next_peak is not None:
        logger.info(f"Next peak at index {next_peak}, distance: {distance_to_peak}, value: {next_peak_value:.3f}")
    
    # Create the base graph
    fig = go.Figure()
    
    # Add eventfulness data line (trace index 0)
    x = list(range(len(data)))
    fig.add_trace(go.Scatter(
        x=x,
        y=data,
        mode='lines',
        name='Eventfulness',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.2)',
        hoverinfo='y+x',
        hoverlabel=dict(bgcolor='white')
    ))
    
    # Add markers for all peaks (trace index 1)
    fig.add_trace(go.Scatter(
        x=peaks,
        y=peak_values,
        mode='markers',
        name='Peaks',
        marker=dict(color='green', size=8, symbol='circle'),
        hoverinfo='text',
        hovertext=[f"Peak: {val:.3f}" for val in peak_values],
        visible=True
    ))
    
    # We're only showing local maxima now
    
    # Add a vertical line for current position (trace index 3)
    fig.add_trace(go.Scatter(
        x=[initial_index, initial_index],
        y=[min(data), max(data)],
        mode='lines',
        name='Current Position',
        line=dict(color='red', width=2),
        hoverinfo='none',
        visible=True
    ))
    
    # Add a point marker for the current value (trace index 4)
    fig.add_trace(go.Scatter(
        x=[initial_index],
        y=[initial_value],
        mode='markers+text',
        name='Current Value',
        marker=dict(color='red', size=10),
        text=[f"{initial_value:.3f}"],
        textposition="top right",
        hoverinfo='text',
        visible=True
    ))
    
    # Add a marker for the next upcoming peak (trace index 5)
    if next_peak is not None:
        fig.add_trace(go.Scatter(
            x=[next_peak],
            y=[next_peak_value],
            mode='markers+text',
            name='Next Peak',
            marker=dict(color='orange', size=12, symbol='diamond'),
            text=[f"Next: {next_peak_value:.3f}"],
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"Next peak: {next_peak_value:.3f}, Distance: {distance_to_peak}"],
            visible=True
        ))
    
    # Set up the layout
    fig.update_layout(
        title="Eventfulness Window (1s before, 3s after current position)",
        xaxis=dict(
            title="Data Point",
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            # Start with a reasonable window around the initial point
            range=[max(0, initial_index - 30), initial_index + 90],  # Approximate 1s before, 3s after
        ),
        yaxis=dict(
            title="Eventfulness",
            showgrid=True,
            zeroline=True,
            showticklabels=True,
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        uirevision='constant'  # Keep zoom level on updates
    )
    
    # Create a second x-axis for time
    max_time = len(data) / fps if fps > 0 else 0
    time_ticks = np.arange(0, max_time, 5)
    data_point_ticks = [int(t * fps) for t in time_ticks]
    
    # Add time ticks
    fig.update_layout(
        xaxis2=dict(
            title="Time (seconds)",
            overlaying='x',
            side='top',
            showgrid=False,
            tickvals=data_point_ticks,
            ticktext=[f"{int(t)}s" for t in time_ticks],
            range=[0, len(data)]
        )
    )
    
    return fig, {'display': 'block'}

# Callback to log video player events
@callback(
    Output('debug-info', 'children', allow_duplicate=True),
    [Input('video-player', 'playing'),
     Input('video-player', 'currentTime'),
     Input('video-player', 'seekTo')],
    State('debug-info', 'children'),
    prevent_initial_call=True
)
def log_video_events(playing, current_time, seek_to, current_debug):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_debug
    
    # Get the ID of the component that triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_prop = ctx.triggered[0]['prop_id'].split('.')[1]
    
    # Log the event
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    if trigger_id == 'video-player':
        if trigger_prop == 'playing':
            event_type = "PLAY" if playing else "PAUSE"
            # Handle None value for current_time
            time_str = f"{current_time:.3f}s" if current_time is not None else "N/A"
            log_msg = f"[{timestamp}] VIDEO EVENT: {event_type}, Time: {time_str}"
            logger.info(log_msg)
            
            # Update debug info
            debug_lines = current_debug.split('\n') if current_debug else []
            debug_lines = [log_msg] + debug_lines[:19]  # Add new line at the top, keep only last 20
            return '\n'.join(debug_lines)
        elif trigger_prop == 'seekTo' and seek_to is not None:
            log_msg = f"[{timestamp}] VIDEO EVENT: SEEK TO {seek_to:.3f}s"
            logger.info(log_msg)
            
            # Update debug info
            debug_lines = current_debug.split('\n') if current_debug else []
            debug_lines = [log_msg] + debug_lines[:19]  # Add new line at the top, keep only last 20
            return '\n'.join(debug_lines)
        elif trigger_prop == 'currentTime' and current_time is not None:
            # Only log time updates occasionally to avoid flooding
            try:
                # Safely check if we should log this time update
                if current_time % 1 < 0.1:  # Log roughly every second
                    log_msg = f"[{timestamp}] Time update: {current_time:.3f}s"
                    logger.info(log_msg)
                    
                    # Update debug info but less frequently
                    debug_lines = current_debug.split('\n') if current_debug else []
                    debug_lines = [log_msg] + debug_lines[:19]  # Add new line at the top, keep only last 20
                    return '\n'.join(debug_lines)
            except Exception as e:
                logger.error(f"Error in time update: {str(e)}")
    
    return current_debug

# Callback to update graph marker based on video time
@callback(
    Output('eventfulness-graph', 'figure', allow_duplicate=True),
    Output('graph-update-interval', 'disabled'),
    Output('debug-info', 'children'),
    Output('next-peak-time', 'children'),
    Output('peak-value', 'children'),
    Output('volume-indicator', 'children'),
    Input('graph-update-interval', 'n_intervals'),
    Input('video-player', 'currentTime'),
    State('video-info', 'data'),
    State('eventfulness-data', 'data'),
    State('debug-info', 'children'),
    State('eventfulness-graph', 'figure'),
    prevent_initial_call=True
)
def update_graph_marker(n_intervals, current_time, video_info, eventfulness_data, current_debug, current_figure):
    # Add more detailed check for current_time
    if current_time is None or not isinstance(current_time, (int, float)) or not video_info or not eventfulness_data or not current_figure:
        # Log the issue
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] Graph update skipped: current_time={current_time}, video_info={bool(video_info)}, eventfulness_data={bool(eventfulness_data)}"
        logger.info(log_msg)
        
        # Update debug info
        debug_lines = current_debug.split('\n') if current_debug else []
        debug_lines = [log_msg] + debug_lines[:19]
        return dash.no_update, True, '\n'.join(debug_lines), "N/A", "N/A", []
    
    # Calculate the current frame
    fps = video_info['fps']
    current_frame = int(current_time * fps)
    
    # Map to eventfulness data point
    data = eventfulness_data['data']
    data_index = map_frame_to_datapoint(current_frame, video_info['frame_count'], len(data))
    
    # Get the current value
    if 0 <= data_index < len(data):
        current_value = data[data_index]
    else:
        current_value = 0
    
    # Detect local maxima (peaks) in the eventfulness data
    peaks = detect_local_maxima(data)
    
    # Find the next upcoming peak
    next_peak, distance_to_peak, next_peak_value = find_next_maximum(data, data_index, peaks)
    
    # Calculate the window range (1 second before, 3 seconds after)
    # Convert seconds to data points
    data_fps = eventfulness_data.get('fps', fps)
    points_per_second = data_fps * (len(data) / video_info['frame_count'])
    
    # Calculate window boundaries in data points
    points_before = int(1 * points_per_second)  # 1 second before
    points_after = int(3 * points_per_second)   # 3 seconds after
    
    window_start = max(0, data_index - points_before)
    window_end = min(len(data) - 1, data_index + points_after)
    
    # Update the figure with the new marker position and window
    updated_figure = current_figure.copy()
    
    # Update x-axis range to show the window
    updated_figure['layout']['xaxis']['range'] = [window_start, window_end]
    
    # Update the markers for peaks (trace index 1)
    if len(updated_figure['data']) > 1:
        peak_values = [data[p] for p in peaks]
        updated_figure['data'][1]['x'] = peaks
        updated_figure['data'][1]['y'] = peak_values
        updated_figure['data'][1]['hovertext'] = [f"Peak: {val:.3f}" for val in peak_values]
    
    # Update the vertical line for current position (trace index 2)
    if len(updated_figure['data']) > 2:
        updated_figure['data'][2]['x'] = [data_index, data_index]
        updated_figure['data'][2]['y'] = [min(data), max(data)]
    
    # Update the marker point for current value (trace index 3)
    if len(updated_figure['data']) > 3:
        updated_figure['data'][3]['x'] = [data_index]
        updated_figure['data'][3]['y'] = [current_value]
        updated_figure['data'][3]['text'] = [f"{current_value:.3f}"]
    
    # Update the next peak marker (trace index 4)
    if len(updated_figure['data']) > 4 and next_peak is not None:
        # Calculate time to next peak in seconds
        time_to_peak = distance_to_peak / points_per_second
        
        # Update the marker for the next peak
        updated_figure['data'][4]['x'] = [next_peak]
        updated_figure['data'][4]['y'] = [next_peak_value]
        
        # Update the marker text to show time until next peak
        peak_text = f"Next: {time_to_peak:.1f}s"
        
        # Set marker style
        updated_figure['data'][4]['marker'] = dict(
            color='orange', 
            size=12, 
            symbol='diamond'
        )
            
        updated_figure['data'][4]['text'] = [peak_text]
        updated_figure['data'][4]['hovertext'] = [f"Next peak: {next_peak_value:.3f}, Time: {time_to_peak:.2f}s"]
        updated_figure['data'][4]['visible'] = True
    
    # Verify that the marker was updated correctly
    is_valid = verify_marker_position(updated_figure, data_index, current_value)
    
    # Log the synchronization data with marker position and window info
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"[{timestamp}] Time: {current_time:.3f}s, Frame: {current_frame}, Data idx: {data_index}, Value: {current_value:.3f}, Window: [{window_start}-{window_end}]"
    
    # Add information about the next peak
    if next_peak is not None:
        time_to_peak = distance_to_peak / points_per_second
        log_msg += f", Next peak: {time_to_peak:.2f}s (value: {next_peak_value:.3f})"
    
    logger.info(log_msg)
    
    # Update debug info (keep last 20 lines)
    debug_lines = current_debug.split('\n') if current_debug else []
    debug_lines = [log_msg] + debug_lines[:19]  # Add new line at the top, keep only last 20
    debug_info = '\n'.join(debug_lines)
    
    # Prepare peak indicator values
    next_peak_time_display = "N/A"
    peak_value_display = "N/A"
    
    if next_peak is not None:
        # Calculate time to next peak in seconds
        time_to_peak = distance_to_peak / points_per_second
        
        # Format the time display
        if time_to_peak < 1.0:
            next_peak_time_display = f"{time_to_peak:.2f} seconds"
        else:
            next_peak_time_display = f"{time_to_peak:.1f} seconds"
        
        # Format the peak value display
        peak_value_display = f"{next_peak_value:.3f}"
    
    # Create volume indicator bars
    volume_bars = []
    if next_peak is not None:
        # Create 10 volume bars
        num_bars = 10
        # Calculate how many bars should be active based on the peak value (0-1 scale)
        active_bars = min(num_bars, max(1, int(next_peak_value * num_bars)))
        
        for i in range(num_bars):
            bar_height = 10 + (i * 2)  # Increasing heights: 10px, 12px, 14px, etc.
            
            # Determine bar class
            if i < active_bars:
                bar_class = "volume-bar active"
            else:
                bar_class = "volume-bar"
            
            # Create the bar element
            bar = html.Div(
                "",
                className=bar_class,
                style={'height': f'{bar_height}px'}
            )
            volume_bars.append(bar)
    
    # Return the updated figure and indicators
    return updated_figure, False, debug_info, next_peak_time_display, peak_value_display, volume_bars

# Add a route for serving videos
@server.route('/video')
def serve_video():
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

# Add route for downloading the log file
@server.route('/download-log')
def download_log():
    from flask import send_file
    return send_file(log_file, as_attachment=True)

# Callback to clear the log file
@callback(
    Output('clear-log-button', 'n_clicks'),
    Input('clear-log-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_log(n_clicks):
    if n_clicks:
        # Clear the log file
        with open(log_file, 'w') as f:
            f.write(f"Log cleared at {datetime.datetime.now()}\n")
        logger.info("Log cleared by user")
    return 0

# Function to check dash_player version and properties
def check_dash_player():
    try:
        import dash_player
        version = getattr(dash_player, '__version__', 'unknown')
        
        # Try to get available props
        available_props = []
        try:
            # This is a common pattern for Dash components
            if hasattr(dash_player, '_js_dist'):
                for dist in dash_player._js_dist:
                    if 'namespace' in dist:
                        available_props.append(f"Namespace: {dist['namespace']}")
            
            # Try to get the DashPlayer class attributes
            player_attrs = dir(dash_player.DashPlayer)
            available_props.extend([attr for attr in player_attrs if not attr.startswith('_')])
        except Exception as e:
            available_props = [f"Error getting props: {str(e)}"]
        
        return {
            'version': version,
            'props': available_props
        }
    except Exception as e:
        return {
            'version': 'Error',
            'error': str(e)
        }

# Log dash_player info
dash_player_info = check_dash_player()
logger.info(f"dash_player version: {dash_player_info['version']}")
logger.info(f"dash_player props: {dash_player_info.get('props', [])}")

# Function to detect local maxima in eventfulness data
def detect_local_maxima(data):
    """
    Detects local maxima in eventfulness data.
    
    Args:
        data: List of eventfulness values
        
    Returns:
        List of indices where local maxima occur
    """
    
    # Find peaks using scipy's find_peaks
    peaks, properties = find_peaks(data, height = 0.3, distance = 5)
    
    # Convert to list of integers
    return [int(peak) for peak in peaks]

# Function to find the next upcoming local maximum
def find_next_maximum(data, current_index, peaks):
    """
    Finds the next upcoming local maximum after the current position.
    
    Args:
        data: List of eventfulness values
        current_index: Current position in the data
        peaks: List of peak indices
        
    Returns:
        Tuple of (next_peak_index, distance_to_peak, peak_value)
        If no upcoming peak is found, returns (None, None, None)
    """
    if not peaks:
        return None, None, None
    
    # Find the first peak that is after the current position
    upcoming_peaks = [p for p in peaks if p > current_index]
    
    if not upcoming_peaks:
        # No upcoming peaks, wrap around to the first peak
        next_peak = peaks[0]
        distance = (len(data) - current_index) + next_peak
    else:
        next_peak = upcoming_peaks[0]
        distance = next_peak - current_index
    
    # Get the peak value
    peak_value = data[next_peak]
    
    return next_peak, distance, peak_value

# Note: Simplified implementation to focus only on local maxima

# Function to verify marker updates
def verify_marker_position(figure, expected_position, expected_value):
    """Verify that the marker position in the figure matches the expected position"""
    if not figure or 'data' not in figure or len(figure['data']) < 3:
        logger.error(f"Figure verification failed: Invalid figure structure")
        return False
    
    # Check vertical line position (trace index 1)
    line_trace = figure['data'][1]
    if 'x' not in line_trace or len(line_trace['x']) != 2:
        logger.error(f"Figure verification failed: Invalid line trace structure")
        return False
    
    line_x = line_trace['x'][0]  # Both x values should be the same
    
    # Check marker position (trace index 2)
    marker_trace = figure['data'][2]
    if 'x' not in marker_trace or len(marker_trace['x']) != 1:
        logger.error(f"Figure verification failed: Invalid marker trace structure")
        return False
    
    marker_x = marker_trace['x'][0]
    marker_y = marker_trace['y'][0]
    
    # Check if positions match
    line_matches = abs(line_x - expected_position) < 0.001
    marker_matches = abs(marker_x - expected_position) < 0.001
    value_matches = abs(marker_y - expected_value) < 0.001
    
    if not (line_matches and marker_matches):
        logger.error(f"Marker position mismatch: Expected={expected_position}, Line={line_x}, Marker={marker_x}")
    
    if not value_matches:
        logger.error(f"Marker value mismatch: Expected={expected_value:.3f}, Actual={marker_y:.3f}")
    
    return line_matches and marker_matches and value_matches

# Add title and heading
app.title = "Video Eventfulness Analyzer - Firework Demo"

# Add debug info to page
app.layout.children.append(
    html.Div([
        html.H3("Debug Information"),
        html.Pre(id='debug-info', style={'border': '1px solid #ddd', 'padding': '10px', 'maxHeight': '200px', 'overflow': 'auto'}),
        html.Div([
            html.Button("Clear Log", id="clear-log-button", style={'margin': '10px'}),
            html.A("Download Log", id="download-log-link", href=f"/download-log", download="sync_debug.log", style={'margin': '10px'})
        ])
    ], style={'marginTop': '30px'})
)

# Run the app
if __name__ == '__main__':
    print(f"\nStarting Dash app with default video: {DEFAULT_VIDEO}")
    print("Access the app at: http://127.0.0.1:8050/")
    app.run_server(debug=True, port=8050)
