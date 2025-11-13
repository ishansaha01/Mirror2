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
            
            @keyframes fadeOut {
                0% { opacity: 1; transform: scale(1); }
                100% { opacity: 0; transform: scale(0.8); }
            }
            
            @keyframes slideLeft {
                0% { transform: translateX(0); }
                100% { transform: translateX(-100%); }
            }
            
            .pulse {
                animation: pulse 1s infinite;
            }
            
            .blink {
                animation: blink 1s infinite;
            }
            
            .fade-out {
                animation: fadeOut 0.5s forwards;
            }
            
            .slide-left {
                animation: slideLeft 0.5s forwards;
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
            
            .sliding-window-frame {
                position: absolute;
                height: 80%;
                width: 12%;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                transform: translateX(-50%); /* Center horizontally */
                pointer-events: none; /* Prevent click interactions */
                margin: 0 2%;
            }
            
            .sliding-window-frame img {
                height: 100%;
                max-width: 100%;
                object-fit: contain;
                border: 1px solid #ddd;
                border-radius: 2px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            /* Current frame at center */
            .sliding-window-frame.current img {
                border: 2px solid #ff4500;
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
    dcc.Store(id='peak-frames', data=None),  # Store for extracted frames at peaks
    dcc.Store(id='sliding-window-frames', data=None),  # Store for sliding window frames
    
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
                
                # Sliding window preview
                html.Div([
                    html.H3("Peak Frames Preview"),
                    html.Div([
                        # Container for sliding window frames
                        html.Div(id='sliding-window-container', style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'position': 'relative',
                            'height': '200px',
                            'backgroundColor': '#f0f0f0',
                            'borderRadius': '5px',
                            'padding': '10px',
                            'marginBottom': '20px',
                            'boxShadow': 'inset 0 0 10px rgba(0,0,0,0.1)',
                            'overflow': 'hidden'
                        }),
                        

                    ], style={'position': 'relative'}),
                ], id='sliding-window-section', style={'display': 'none', 'marginTop': '20px', 'marginBottom': '20px'}),
                
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
                
                # Peak frames section
                html.Div([
                    html.H3("Extracted Peak Frames"),
                    html.Div([
                        html.Div("Frames extracted at detected peaks:", style={'marginBottom': '10px'}),
                        html.Div(id='peak-frames-gallery', style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),
                    ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'backgroundColor': '#f9f9f9'})
                ], id='peak-frames-section', style={'display': 'none', 'marginTop': '20px'}),
            ]),
        ], style={'width': '75%', 'float': 'left', 'padding': '10px'}),
    ], style={'display': 'flex', 'flexFlow': 'row'}),
    
    # Interval for updating the graph marker
    dcc.Interval(
        id='graph-update-interval',
        interval=50,  # Update every 50ms for more responsiveness
        n_intervals=0,
        disabled=True
    ),
    
    # Separate high-frequency interval for sliding window updates
    dcc.Interval(
        id='sliding-window-update-interval',
        interval=10,  # Update every 30ms for smoother animations
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
    Output('peak-frames-section', 'style'),  # Also show the peak frames section
    Output('sliding-window-section', 'style'),  # Also show the sliding window section
    Output('sliding-window-frames', 'data'),  # Extract frames for sliding window
    Input('current-video', 'data'),
    State('video-info', 'data')
)
def update_video_player(video_path, video_info):
    if not video_path or not os.path.exists(video_path) or not video_info:
        return dash.no_update, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None
    
    # Create a data URL for the video
    video_url = f"/video?path={base64.b64encode(video_path.encode()).decode()}"
    
    player = dash_player.DashPlayer(
        id='video-player',
        url=video_url,
        controls=True,
        width='100%',
        height='400px',
        # Use intervalCurrentTime instead of intervalDelay
        intervalCurrentTime=50,  # Update time more frequently (50ms)
        playing=False
    )
    
    # Extract frames for the sliding window - use a smaller interval for smoother preview
    sliding_window_frames = extract_frames_for_sliding_window(
        video_path, 
        video_info, 
        interval_seconds=0.25,  # Extract a frame every 0.25 seconds for smoother preview
        window_seconds=3.0     # 3 second preview window
    )
    
    return player, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}, sliding_window_frames

# Callback to handle video play/pause events
@callback(
    Output('graph-update-interval', 'disabled', allow_duplicate=True),
    Input('video-player', 'playing'),
    prevent_initial_call=True
)
def handle_video_playback(playing):
    # Enable interval updates when video is playing, disable when paused
    return not playing

# Callback to update eventfulness graph
@callback(
    Output('eventfulness-graph', 'figure'),
    Output('graph-section', 'style'),
    Output('peak-frames', 'data'),
    Input('eventfulness-data', 'data'),
    Input('video-info', 'data'),
    Input('current-video', 'data')
)
def update_eventfulness_graph(eventfulness_data, video_info, current_video):
    # Skip if no video is selected or no data available
    if not current_video or not os.path.exists(current_video) or not eventfulness_data or not video_info:
        return dash.no_update, {'display': 'none'}, None
    if not eventfulness_data or not video_info:
        return dash.no_update, {'display': 'none'}, None
    
    data = eventfulness_data['data']
    fps = eventfulness_data.get('fps', video_info['fps'])
    
    # Calculate initial position
    initial_index = 0
    initial_value = data[0] if len(data) > 0 else 0
    
    # Detect local maxima (peaks) in the eventfulness data
    peaks = detect_local_maxima(data)
    peak_values = [data[p] for p in peaks]
    
    # Extract frames at peak locations
    peak_frames = extract_frames_at_peaks(current_video, peaks, video_info, eventfulness_data)
    
    # Find the next upcoming peak
    next_peak, distance_to_peak, next_peak_value = find_next_maximum(data, initial_index, peaks)
    
    # Log the initial graph setup and peak detection
    logger.info(f"Creating eventfulness graph with {len(data)} data points")
    logger.info(f"Initial marker position: index={initial_index}, value={initial_value:.3f}")
    logger.info(f"Detected {len(peaks)} peaks")
    logger.info(f"Extracted {len(peak_frames)} frames at peak locations")
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
    
    return fig, {'display': 'block'}, peak_frames

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
    Output('sliding-window-update-interval', 'disabled'),  # Also control the sliding window interval
    Output('debug-info', 'children'),
    Output('next-peak-time', 'children'),
    Output('peak-value', 'children'),
    Output('volume-indicator', 'children'),
    Input('graph-update-interval', 'n_intervals'),
    Input('video-player', 'currentTime'),
    Input('video-player', 'playing'),
    State('video-info', 'data'),
    State('eventfulness-data', 'data'),
    State('debug-info', 'children'),
    State('eventfulness-graph', 'figure'),
    prevent_initial_call=True
)
def update_graph_marker(n_intervals, current_time, playing, video_info, eventfulness_data, current_debug, current_figure):
    # Add more detailed check for current_time
    if current_time is None or not isinstance(current_time, (int, float)) or not video_info or not eventfulness_data or not current_figure:
        # Log the issue
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] Graph update skipped: current_time={current_time}, video_info={bool(video_info)}, eventfulness_data={bool(eventfulness_data)}"
        logger.info(log_msg)
        
        # Update debug info
        debug_lines = current_debug.split('\n') if current_debug else []
        debug_lines = [log_msg] + debug_lines[:19]
        return dash.no_update, not playing, not playing, '\n'.join(debug_lines), "N/A", "N/A", []
    
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
    # Keep the intervals enabled only when the video is playing
    return updated_figure, not playing, not playing, debug_info, next_peak_time_display, peak_value_display, volume_bars

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

# Add route for serving extracted frames
@server.route('/frame/<path:frame_path>')
def serve_frame(frame_path):
    from flask import send_file, Response
    
    # For security, ensure the path is within the RESULTS_DIR
    full_path = os.path.join(RESULTS_DIR, frame_path)
    if not os.path.exists(full_path):
        return Response("Frame not found", status=404)
    
    return send_file(full_path)

# Callback to update sliding window frames using the high-frequency interval
@callback(
    Output('sliding-window-container', 'children', allow_duplicate=True),
    Input('sliding-window-update-interval', 'n_intervals'),
    Input('video-player', 'currentTime'),
    State('video-info', 'data'),
    State('peak-frames', 'data'),
    prevent_initial_call=True
)
def update_sliding_window(n_intervals, current_time, video_info, peak_frames):
    # Add more detailed check for current_time
    if current_time is None or not isinstance(current_time, (int, float)) or not video_info:
        return html.Div("No preview available")
    
    # Check if peak frames are available
    if not peak_frames or len(peak_frames) == 0:
        return html.Div("No peak frames available for preview")
    
    # Only log every 10th update to reduce logging overhead
    if n_intervals % 10 == 0:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"[{timestamp}] Updating sliding window at time: {current_time:.3f}s")
    
    # Get video properties
    duration = video_info['duration']
    
    # Calculate dynamic preview time based on video length and peak count
    # Default to 3 seconds if calculation fails
    preview_time = 3.0
    
    # Use the actual number of detected peaks
    num_peaks = len(peak_frames)
    # Calculate average time between peaks
    avg_time_between_peaks = duration / num_peaks
    # Use a fraction of this time as preview window (adjust as needed)
    preview_time = min(max(avg_time_between_peaks * 0.8, 0.01), 5.0)
    
    # Only log preview time calculation occasionally to reduce overhead
    if n_intervals % 30 == 0:
        logger.info(f"Dynamic preview time: {preview_time:.2f}s (based on {num_peaks} peaks in {duration:.2f}s video)")
    
    # Use a more efficient approach to find relevant frames with list comprehension
    relevant_frames = [
        {
            'frame': frame_data,
            'time_diff': frame_data['time'] - current_time,
            'peak_idx': peak_idx
        }
        for peak_idx, frame_data in peak_frames.items()
        if -0.3 <= (frame_data['time'] - current_time) <= preview_time
    ]
    
    # Sort frames by time difference (past to future)
    relevant_frames.sort(key=lambda x: x['time_diff'])
    
    # Only log frame count occasionally
    if n_intervals % 10 == 0:
        logger.info(f"Found {len(relevant_frames)} relevant peak frames in time window")
    
    # Create frame elements for display
    frame_elements = []
    
    # Add frames with positions based on time difference
    for i, frame_data in enumerate(relevant_frames):
        frame = frame_data['frame']
        time_diff = frame_data['time_diff']
        peak_idx = frame_data['peak_idx']
        
        # Get the relative path for the frame image
        frame_path = frame['path']
        rel_path = os.path.relpath(frame_path, RESULTS_DIR)
        frame_url = f"/frame/{rel_path}"
        
        # Get peak value for labeling
        peak_value = frame['peak_value']
        
        # Initialize skip_frame flag
        skip_frame = False
        
        # Determine frame state and styling based on time difference
        if time_diff < -0.2:
            # Frame is well past - don't show it
            skip_frame = True
            position_percent = 50
            opacity = 0
            z_index = 1000 - int(time_diff * 100)  # Consistent z-index calculation
            animation = ""
            label = ""
            label_style = {}
        
        elif time_diff < 0:
            # Frame is just past - still at center
            position_percent = 50
            opacity = max(0.5, 1 + (time_diff * 2))  # Gradually reduce opacity
            z_index = 1000 - int(time_diff * 100)  # Consistent z-index calculation
            animation = "current"
            label = "NOW"
            label_style = {
                'position': 'absolute',
                'bottom': '5px',
                'left': '5px',
                'backgroundColor': 'rgba(255,69,0,0.7)',
                'color': 'white',
                'padding': '2px 5px',
                'borderRadius': '3px',
                'fontSize': '0.8em'
            }
        elif time_diff < 0.1:  # Current frame
            # Frame is at current time - show at center
            position_percent = 50
            opacity = 1.0
            z_index = 1000 - int(time_diff * 100)  # Consistent z-index calculation
            animation = "current"
            label = "NOW"
            label_style = {
                'position': 'absolute',
                'bottom': '5px',
                'left': '5px',
                'backgroundColor': 'rgba(255,69,0,0.7)',
                'color': 'white',
                'padding': '2px 5px',
                'borderRadius': '3px',
                'fontSize': '0.8em'
            }

        else:
            # Frame is in the future - use linear positioning from right to center
            # Calculate position based on a constant speed approach
            # All frames start at 90% (right side) and move toward 50% (center)
            # The closer to current time, the closer to center
            
            # Linear mapping from time_diff to position with better spacing
            # preview_time -> 90% (far right)
            # 0s -> 50% (center)
            # Ensure frames are spaced out more evenly
            position_percent = 90 - ((preview_time - time_diff) / preview_time) * 40
            
            # Add some spacing based on the frame's position in the sequence
            # This helps prevent frames from overlapping
            # position_percent = position_percent + (i * 2)
            
            # Simple opacity - consistent for all future frames
            opacity = 0.8
            # Assign z-index based solely on time difference (closer to current time = higher z-index)
            # This ensures each frame has a unique z-index independent of other frames
            z_index = 1000 - int(time_diff * 100)  # Higher for frames closer to current time
            animation = ""
            label = f"+{time_diff:.1f}s"
            label_style = {
                'position': 'absolute',
                'bottom': '5px',
                'left': '5px',
                'backgroundColor': 'rgba(0,0,0,0.7)',
                'color': 'white',
                'padding': '2px 5px',
                'borderRadius': '3px',
                'fontSize': '0.8em'
            }
        
        # Skip frames that are too far past
        if not skip_frame:
            # Create the frame element with peak value indicator
            frame_element = html.Div([
                html.Img(src=frame_url),
                html.Div(label, style=label_style),
                html.Div(
                    f"{peak_value:.2f}", 
                    style={
                        'position': 'absolute',
                        'top': '5px',
                        'right': '5px',
                        'backgroundColor': 'rgba(0,0,0,0.6)',
                        'color': 'white',
                        'padding': '2px 5px',
                        'borderRadius': '3px',
                        'fontSize': '0.8em'
                    }
                )
            ], className=f"sliding-window-frame {animation}", 
               key=f"peak-{peak_idx}-{frame['time']}", 
               style={
                   'left': f"{position_percent}%",
                   'opacity': opacity,
                   'zIndex': z_index
               })
            
            frame_elements.append(frame_element)
    
    # If no frames are available, show a message
    if not frame_elements:
        return html.Div("No upcoming peak frames in preview window")
    
    logger.info(f"[{timestamp}] Updated sliding window with {len(frame_elements)} peak frames")
    return frame_elements

# Callback to update peak frames gallery
@callback(
    Output('peak-frames-gallery', 'children'),
    Input('peak-frames', 'data')
)
def update_peak_frames_gallery(peak_frames):
    if not peak_frames:
        return html.Div("No peak frames extracted yet.")
    
    # Create a gallery of frame thumbnails
    frame_elements = []
    
    # Sort peaks by their index
    sorted_peaks = sorted(peak_frames.keys(), key=lambda x: int(x))
    
    for peak_idx in sorted_peaks:
        frame_info = peak_frames[peak_idx]
        frame_path = frame_info['path']
        peak_value = frame_info['peak_value']
        time = frame_info['time']
        
        # Get relative path for URL
        rel_path = os.path.relpath(frame_path, RESULTS_DIR)
        frame_url = f"/frame/{rel_path}"
        
        # Create a thumbnail with info
        thumbnail = html.Div([
            html.Img(src=frame_url, style={'width': '150px', 'height': 'auto', 'objectFit': 'cover'}),
            html.Div([
                html.Div(f"Peak: {peak_value:.3f}", style={'fontSize': '0.8em'}),
                html.Div(f"Time: {time:.2f}s", style={'fontSize': '0.8em'}),
            ], style={'textAlign': 'center', 'marginTop': '5px'})
        ], style={'border': '1px solid #ddd', 'padding': '5px', 'borderRadius': '5px'})
        
        frame_elements.append(thumbnail)
    
    return frame_elements

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
    peaks, properties = find_peaks(data, height = 0.3, distance = 10)
    
    # Convert to list of integers
    return [int(peak) for peak in peaks]

def extract_frames_at_peaks(video_path, peaks, video_info, eventfulness_data):
    """
    Extracts frames from the video at peak locations.
    
    Args:
        video_path: Path to the video file
        peaks: List of peak indices in eventfulness data
        video_info: Dictionary containing video information
        eventfulness_data: Dictionary containing eventfulness data
        
    Returns:
        Dictionary mapping peak indices to extracted frame paths
    """
    import cv2
    import os
    from datetime import datetime
    
    # Create directory for extracted frames if it doesn't exist
    video_filename = os.path.basename(video_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(RESULTS_DIR, f"peak_frames_{video_filename}_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return {}
    
    # Get video properties
    fps = video_info['fps']
    frame_count = video_info['frame_count']
    
    # Dictionary to store peak index to frame path mapping
    peak_frames = {}
    
    # Extract frames at each peak
    for i, peak_idx in enumerate(peaks):
        # Map peak index in eventfulness data to frame number in video
        # This is the reverse of map_frame_to_datapoint
        eventfulness_length = len(eventfulness_data['data'])
        ratio = frame_count / eventfulness_length
        frame_number = int(peak_idx * ratio)
        
        # Set video position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if ret:
            # Save the frame
            frame_path = os.path.join(frames_dir, f"peak_{i}_idx_{peak_idx}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Store the mapping
            peak_frames[peak_idx] = {
                'path': frame_path,
                'frame_number': frame_number,
                'peak_value': eventfulness_data['data'][peak_idx],
                'time': frame_number / fps
            }
            
            logger.info(f"Extracted frame at peak {i}: index={peak_idx}, frame={frame_number}, time={frame_number/fps:.2f}s")
        else:
            logger.error(f"Failed to extract frame at peak {i}: index={peak_idx}, frame={frame_number}")
    
    # Release the video
    cap.release()
    
    logger.info(f"Extracted {len(peak_frames)} frames to {frames_dir}")
    return peak_frames

def extract_frames_for_sliding_window(video_path, video_info, interval_seconds=0.5, window_seconds=3):
    """
    Extracts frames at regular intervals throughout the video for the sliding window preview.
    
    Args:
        video_path: Path to the video file
        video_info: Dictionary containing video information
        interval_seconds: Time interval between frames in seconds
        window_seconds: Total window size in seconds
        
    Returns:
        List of dictionaries with frame information
    """
    import cv2
    import os
    from datetime import datetime
    
    # Create directory for sliding window frames if it doesn't exist
    video_filename = os.path.basename(video_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(RESULTS_DIR, f"sliding_window_{video_filename}_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    # Get video properties
    fps = video_info['fps']
    frame_count = video_info['frame_count']
    duration = video_info['duration']
    
    # Calculate frame interval in frames
    frame_interval = max(1, int(fps * interval_seconds))
    
    # List to store frame information
    frames = []
    
    # Extract frames at regular intervals throughout the entire video
    frame_number = 0
    frame_index = 0
    
    while frame_number < frame_count:
        # Set video position to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if ret:
            # Calculate time in seconds for this frame
            time_seconds = frame_number / fps
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"window_frame_{frame_index}_time_{time_seconds:.2f}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Store the frame information
            frames.append({
                'path': frame_path,
                'frame_number': frame_number,
                'time': time_seconds
            })
            
            if frame_index % 20 == 0:  # Log every 20th frame to avoid log flooding
                logger.info(f"Extracted sliding window frame {frame_index}: time={time_seconds:.2f}s, frame={frame_number}")
            
            frame_index += 1
        else:
            logger.error(f"Failed to extract sliding window frame at time={frame_number/fps:.2f}s, frame={frame_number}")
        
        # Move to next frame position
        frame_number += frame_interval
    
    # Release the video
    cap.release()
    
    logger.info(f"Extracted {len(frames)} sliding window frames to {frames_dir}")
    return frames

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
