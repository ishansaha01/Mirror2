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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import mediapipe as mp

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
                "data": default_config["eventfulness"][0],  # First dimension for visualization
                "full_vectors": default_config["eventfulness"],  # Full eventfulness vectors (all dimensions)
                "fps": default_config.get("fps", fps),
                "config_path": default_config_path
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
    dcc.Store(id='cluster-assignments', data=None),  # Store for cluster assignments
    dcc.Store(id='pca-results', data=None),  # Store for PCA analysis results
    
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
                        html.Div([
                            html.Div("Frames extracted at detected peaks:", style={'marginBottom': '10px', 'display': 'inline-block', 'marginRight': '20px'}),
                            html.Button("Cluster Vectors", id='cluster-button', n_clicks=0, style={
                                'marginBottom': '10px',
                                'padding': '8px 16px',
                                'backgroundColor': '#2196F3',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '4px',
                                'cursor': 'pointer'
                            }),
                            html.Div([
                                dcc.RadioItems(
                                    id='frame-view-toggle',
                                    options=[
                                        {'label': 'Original Frames', 'value': 'original'},
                                        {'label': 'Pose Annotated Frames', 'value': 'pose'}
                                    ],
                                    value='original',
                                    labelStyle={'display': 'inline-block', 'marginRight': '15px'},
                                    style={'marginLeft': '20px', 'display': 'inline-block'}
                                ),
                            ]),
                        ], style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label("Vector Type:", style={'marginRight': '10px'}),
                            dcc.Dropdown(
                                id='vector-type',
                                options=[
                                    {'label': 'Eventfulness Vector', 'value': 'eventfulness'},
                                    {'label': 'Pose Vector', 'value': 'pose'},
                                    {'label': 'Combined Vectors', 'value': 'combined'}
                                ],
                                value='eventfulness',
                                style={'width': '200px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                            html.Label("Max Number of Clusters:", style={'marginRight': '10px'}),
                            dcc.Input(
                                id='max-clusters',
                                type='number',
                                value=40,
                                min=2,
                                max=40,
                                style={'width': '80px', 'display': 'inline-block', 'marginRight': '20px'}
                            ),
                        ], id='cluster-controls', style={'marginBottom': '10px', 'display': 'block'}),
                        html.Div(id='cluster-info', style={'marginBottom': '10px', 'fontSize': '0.9em', 'color': '#666'}),
                        html.Div([
                            html.H4("K-Means Clustering Results"),
                            dcc.Graph(
                                id='cluster-visualization',
                                style={'height': '400px'},
                                config={'displayModeBar': True}
                            ),
                            html.H4("Silhouette Score Analysis", style={'marginTop': '20px'}),
                            dcc.Graph(
                                id='silhouette-plot',
                                style={'height': '300px'},
                                config={'displayModeBar': True}
                            ),
                        ], id='cluster-viz-section', style={'display': 'none', 'marginBottom': '20px'}),
                        
                        # PCA Analysis Section
                        html.Div([
                            html.H4("PCA Analysis"),
                            html.Div([
                                html.Label("PCA Components:", style={'marginRight': '10px'}),
                                html.Div([
                                    dcc.Slider(
                                        id='pca-components-slider',
                                        min=2,
                                        max=10,
                                        step=1,
                                        value=3,
                                        marks={i: str(i) for i in range(2, 11)},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                ], style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}),
                                html.Button("Run PCA Analysis", id='pca-button', n_clicks=0, style={
                                    'marginLeft': '20px',
                                    'padding': '8px 16px',
                                    'backgroundColor': '#4CAF50',
                                    'color': 'white',
                                    'border': 'none',
                                    'borderRadius': '4px',
                                    'cursor': 'pointer'
                                }),
                            ], style={'marginBottom': '15px'}),
                            html.Div(id='pca-info', style={'marginBottom': '10px', 'fontSize': '0.9em', 'color': '#666'}),
                            dcc.Graph(
                                id='pca-visualization',
                                style={'height': '400px'},
                                config={'displayModeBar': True}
                            ),
                            dcc.Graph(
                                id='pca-variance-plot',
                                style={'height': '300px'},
                                config={'displayModeBar': True}
                            ),
                        ], id='pca-section', style={'display': 'none', 'marginBottom': '20px', 'marginTop': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'backgroundColor': '#f9f9f9'}),
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
            "data": config["eventfulness"][0],  # First dimension for visualization
            "full_vectors": config["eventfulness"],  # Full eventfulness vectors (all dimensions)
            "fps": config.get("fps", fps),
            "config_path": config_path
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
    preview_time = min(max(avg_time_between_peaks * 2, 0.01), 5.0)
    
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

# Callback to handle clustering button and show controls
@callback(
    [Output('cluster-controls', 'style'),
     Output('cluster-assignments', 'data'),
     Output('cluster-info', 'children'),
     Output('cluster-viz-section', 'style')],
    [Input('cluster-button', 'n_clicks'),
     Input('vector-type', 'value')],
    [State('peak-frames', 'data'),
     State('max-clusters', 'value'),
     State('cluster-assignments', 'data')],
    prevent_initial_call=True
)
def handle_clustering(n_clicks, vector_type, peak_frames, max_clusters, existing_assignments):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {'display': 'none'}, existing_assignments, "", {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Show controls when button is clicked
    if trigger_id == 'cluster-button' and n_clicks > 0:
        if not peak_frames:
            return {'display': 'block'}, existing_assignments, html.Div("No peak frames available for clustering.", style={'color': 'red'}), {'display': 'none'}
        
        # Perform clustering with K-means and silhouette score analysis
        max_k = max_clusters if max_clusters else 40
        
        cluster_assignments, cluster_info = cluster_eventfulness_vectors(
            peak_frames, 
            max_clusters=max_k,
            vector_type=vector_type
        )
        
        if cluster_assignments is None:
            return {'display': 'block'}, existing_assignments, html.Div("Clustering failed. Check logs for details.", style={'color': 'red'}), {'display': 'none'}
        
        # Store silhouette score data in the cluster_assignments store for visualization
        if 'k_values' in cluster_info and 'silhouette_scores' in cluster_info:
            cluster_assignments['k_values_data'] = json.dumps(cluster_info['k_values'])
            cluster_assignments['silhouette_scores_data'] = json.dumps(cluster_info['silhouette_scores'])
            cluster_assignments['penalized_scores_data'] = json.dumps(cluster_info['penalized_scores'])
            cluster_assignments['best_k_value'] = str(cluster_info['n_clusters'])
        
        # Create info display with silhouette score information
        info_text = (
            f"Vector Type: {cluster_info['vector_type'].capitalize()}, "
            f"Optimal Clusters: {cluster_info['n_clusters']}, "
            f"Silhouette Score: {cluster_info['silhouette_score']:.4f}, "
            f"Samples: {cluster_info['n_samples']}"
        )
        
        # Create a more detailed info display with silhouette scores
        info_div = html.Div([
            html.Div(info_text, style={'color': '#2196F3', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.Div("Silhouette scores by cluster count:", style={'fontSize': '0.9em', 'marginTop': '5px'}),
            html.Div([
                html.Span(f"K={k}: {score:.4f}", 
                    style={
                        'marginRight': '10px', 
                        'fontWeight': 'bold' if k == cluster_info['n_clusters'] else 'normal',
                        'color': '#2196F3' if k == cluster_info['n_clusters'] else 'inherit'
                    }
                ) 
                for k, score in zip(cluster_info['k_values'], cluster_info['silhouette_scores'])
            ], style={'fontSize': '0.9em'})
        ])
        
        return {'display': 'block'}, cluster_assignments, info_div, {'display': 'block'}
    
    # Show controls when vector type changes (but don't re-cluster)
    if trigger_id == 'vector-type':
        viz_style = {'display': 'block'} if existing_assignments else {'display': 'none'}
        return {'display': 'block'}, existing_assignments, "", viz_style
    
    return {'display': 'none'}, existing_assignments, "", {'display': 'none'}

# Callback to update peak frames gallery with cluster colors
@callback(
    Output('peak-frames-gallery', 'children'),
    [Input('peak-frames', 'data'),
     Input('cluster-assignments', 'data'),
     Input('frame-view-toggle', 'value')]
)
def update_peak_frames_gallery(peak_frames, cluster_assignments, frame_view):
    if not peak_frames:
        return html.Div("No peak frames extracted yet.")
    
    # Define color palette for clusters
    cluster_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#E74C3C'
    ]
    
    # Create a gallery of frame thumbnails
    frame_elements = []
    
    # Sort peaks by cluster first, then by time within each cluster
    # If no clusters, sort by peak index
    if cluster_assignments:
        # Group by cluster
        cluster_groups = {}
        for peak_idx in peak_frames.keys():
            cluster_id = cluster_assignments.get(str(peak_idx), -999)  # -999 for unclustered
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(peak_idx)
        
        # Sort clusters (noise points last)
        sorted_cluster_ids = sorted([c for c in cluster_groups.keys() if c != -999])
        if -999 in cluster_groups:
            sorted_cluster_ids.append(-999)
        if -1 in cluster_groups:  # DBSCAN noise
            sorted_cluster_ids.remove(-1)
            sorted_cluster_ids.append(-1)
        
        # Sort peaks: by cluster, then by time within cluster
        sorted_peaks = []
        for cluster_id in sorted_cluster_ids:
            cluster_peaks = cluster_groups[cluster_id]
            # Sort by time within cluster
            cluster_peaks_sorted = sorted(cluster_peaks, key=lambda x: peak_frames[x]['time'])
            sorted_peaks.extend(cluster_peaks_sorted)
    else:
        # No clustering, sort by peak index
        sorted_peaks = sorted(peak_frames.keys(), key=lambda x: int(x))
    
    for peak_idx in sorted_peaks:
        frame_info = peak_frames[peak_idx]
        
        # Choose which frame path to use based on toggle
        if frame_view == 'pose' and 'annotated_path' in frame_info:
            frame_path = frame_info['annotated_path']
        else:
            frame_path = frame_info['path']
            
        peak_value = frame_info['peak_value']
        time = frame_info['time']
        eventfulness_vector = frame_info.get('eventfulness_vector', None)
        pose_vector = frame_info.get('pose_vector', None)
        pose_detected = frame_info.get('pose_detected', False)
        
        # Get cluster assignment if available
        cluster_id = None
        if cluster_assignments and str(peak_idx) in cluster_assignments:
            cluster_id = cluster_assignments[str(peak_idx)]
        
        # Calculate magnitude of eventfulness vector if available
        vector_magnitude = None
        if eventfulness_vector:
            # Magnitude = ||v|| = sqrt(sum(v[i]^2))
            vector_magnitude = np.linalg.norm(eventfulness_vector)
        
        # Calculate magnitude of pose vector if available
        pose_magnitude = None
        if pose_vector:
            # Magnitude = ||v|| = sqrt(sum(v[i]^2))
            pose_magnitude = np.linalg.norm(pose_vector)
        
        # Get relative path for URL
        rel_path = os.path.relpath(frame_path, RESULTS_DIR)
        frame_url = f"/frame/{rel_path}"
        
        # Determine border color based on cluster
        border_color = '#ddd'
        border_width = '1px'
        if cluster_id is not None:
            if cluster_id == -1:  # Noise point (DBSCAN)
                border_color = '#999'
                border_width = '2px'
            else:
                border_color = cluster_colors[cluster_id % len(cluster_colors)]
                border_width = '3px'
        
        # Create info div with peak value, time, magnitude, and cluster
        info_divs = [
            html.Div(f"Peak: {peak_value:.3f}", style={'fontSize': '0.8em'}),
            html.Div(f"Time: {time:.2f}s", style={'fontSize': '0.8em'}),
        ]
        
        # Add magnitude if available
        if vector_magnitude is not None:
            info_divs.append(
                html.Div(f"Eventfulness Mag: {vector_magnitude:.3f}", style={'fontSize': '0.8em', 'fontWeight': 'bold', 'color': '#2196F3'})
            )
        
        # Add pose information if available
        if pose_detected:
            info_divs.append(
                html.Div(f"Pose Detected ✓", style={'fontSize': '0.8em', 'fontWeight': 'bold', 'color': 'green'})
            )
            if pose_magnitude is not None:
                info_divs.append(
                    html.Div(f"Pose Mag: {pose_magnitude:.3f}", style={'fontSize': '0.8em', 'color': 'green'})
                )
        else:
            info_divs.append(
                html.Div("No Pose Detected", style={'fontSize': '0.8em', 'color': 'red'})
            )
        
        # Add cluster assignment if available
        if cluster_id is not None:
            if cluster_id == -1:
                cluster_text = "Noise"
                cluster_color = '#999'
            else:
                cluster_text = f"Cluster {cluster_id}"
                cluster_color = border_color
            info_divs.append(
                html.Div(cluster_text, style={
                    'fontSize': '0.9em', 
                    'fontWeight': 'bold', 
                    'color': cluster_color,
                    'marginTop': '3px',
                    'padding': '2px 5px',
                    'backgroundColor': 'rgba(255,255,255,0.8)',
                    'borderRadius': '3px'
                })
            )
        
        # Create a thumbnail with info
        thumbnail = html.Div([
            html.Img(src=frame_url, style={'width': '150px', 'height': 'auto', 'objectFit': 'cover'}),
            html.Div(info_divs, style={'textAlign': 'center', 'marginTop': '5px'})
        ], style={
            'border': f'{border_width} solid {border_color}', 
            'padding': '5px', 
            'borderRadius': '5px',
            'backgroundColor': 'white' if cluster_id is None else 'rgba(255,255,255,0.95)'
        })
        
        frame_elements.append(thumbnail)
    
    return frame_elements

# Callback to create cluster visualization and silhouette plot
@callback(
    [Output('cluster-visualization', 'figure'),
     Output('silhouette-plot', 'figure')],
    [Input('peak-frames', 'data'),
     Input('cluster-assignments', 'data'),
     Input('vector-type', 'value')]
)
def update_cluster_visualization(peak_frames, cluster_assignments, vector_type):
    if not peak_frames or not cluster_assignments:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No clustering data available. Click 'Cluster Vectors' to perform clustering.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_fig.update_layout(
            title="K-Means Clustering Visualization",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        empty_silhouette_fig = go.Figure()
        empty_silhouette_fig.add_annotation(
            text="Run clustering to see silhouette score analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_silhouette_fig.update_layout(
            title="Silhouette Score Analysis",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return empty_fig, empty_silhouette_fig
    
    # Extract vectors and cluster assignments based on vector_type
    vectors = []
    peak_indices = []
    cluster_labels = []
    
    for peak_idx, frame_info in peak_frames.items():
        feature_vector = None
        
        if vector_type == 'eventfulness':
            # Use eventfulness vector
            feature_vector = frame_info.get('eventfulness_vector', None)
            
        elif vector_type == 'pose':
            # Use pose vector
            pose_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            if pose_vector is not None and pose_detected:
                feature_vector = pose_vector
                
        elif vector_type == 'combined':
            # Use both vectors
            eventfulness_vector = frame_info.get('eventfulness_vector', None)
            pose_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            
            if eventfulness_vector is not None and pose_vector is not None and pose_detected:
                # Normalize each vector separately before combining
                if len(eventfulness_vector) > 0:
                    e_norm = np.linalg.norm(eventfulness_vector)
                    if e_norm > 0:
                        eventfulness_vector = [x / e_norm for x in eventfulness_vector]
                
                if len(pose_vector) > 0:
                    p_norm = np.linalg.norm(pose_vector)
                    if p_norm > 0:
                        pose_vector = [x / p_norm for x in pose_vector]
                
                # Combine the vectors
                feature_vector = eventfulness_vector + pose_vector
        
        # Add vector if available and has cluster assignment
        if feature_vector is not None:
            cluster_id = cluster_assignments.get(str(peak_idx), None)
            if cluster_id is not None:
                vectors.append(feature_vector)
                peak_indices.append(peak_idx)
                cluster_labels.append(cluster_id)
    
    if len(vectors) < 2:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Not enough {vector_type} vectors for visualization (need at least 2).",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_fig.update_layout(
            title=f"K-Means Clustering - {vector_type.capitalize()} Vectors",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        empty_silhouette_fig = go.Figure()
        empty_silhouette_fig.add_annotation(
            text=f"Not enough {vector_type} vectors for silhouette analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_silhouette_fig.update_layout(
            title="Silhouette Score Analysis",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return empty_fig, empty_silhouette_fig
    
    # Convert to numpy array
    X = np.array(vectors)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use t-SNE for dimensionality reduction (2D visualization)
    try:
        # Use PCA first if we have too many dimensions (t-SNE can be slow)
        if X_scaled.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
        X_2d = tsne.fit_transform(X_reduced)
    except Exception as e:
        logger.error(f"Error in t-SNE: {str(e)}")
        # Fallback to PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=.99)
        X_2d = pca.fit_transform(X_scaled)
    
    # Define color palette for clusters
    cluster_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#E74C3C'
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Group points by cluster
    unique_clusters = sorted(set(cluster_labels))
    
    for cluster_id in unique_clusters:
        # Get indices for this cluster
        cluster_mask = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        
        if cluster_id == -1:
            # Noise points
            cluster_name = "Noise"
            color = '#999999'
            symbol = 'x'
            size = 8
        else:
            cluster_name = f"Cluster {cluster_id}"
            color = cluster_colors[cluster_id % len(cluster_colors)]
            symbol = 'circle'
            size = 10
        
        # Extract coordinates for this cluster
        x_coords = [X_2d[i, 0] for i in cluster_mask]
        y_coords = [X_2d[i, 1] for i in cluster_mask]
        
        # Create hover text with peak info
        hover_texts = []
        for i in cluster_mask:
            peak_idx = peak_indices[i]
            frame_info = peak_frames[peak_idx]
            
            # Add vector-specific information to hover text
            hover_info = [
                f"Peak Index: {peak_idx}",
                f"Time: {frame_info['time']:.2f}s",
                f"Peak Value: {frame_info['peak_value']:.3f}",
                f"Cluster: {cluster_name}"
            ]
            
            # Add vector magnitude information based on vector type
            if vector_type == 'eventfulness' or vector_type == 'combined':
                if 'eventfulness_vector' in frame_info and frame_info['eventfulness_vector'] is not None:
                    e_mag = np.linalg.norm(frame_info['eventfulness_vector'])
                    hover_info.append(f"Eventfulness Mag: {e_mag:.3f}")
            
            if vector_type == 'pose' or vector_type == 'combined':
                if 'pose_vector' in frame_info and frame_info['pose_vector'] is not None:
                    p_mag = np.linalg.norm(frame_info['pose_vector'])
                    hover_info.append(f"Pose Mag: {p_mag:.3f}")
                    hover_info.append(f"Pose Detected: {frame_info.get('pose_detected', False)}")
            
            hover_texts.append("<br>".join(hover_info))
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name=cluster_name,
            marker=dict(
                color=color,
                size=size,
                symbol=symbol,
                line=dict(width=1, color='white') if cluster_id != -1 else dict(width=0)
            ),
            text=hover_texts,
            hoverinfo='text',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=f"K-Means Clustering - {vector_type.capitalize()} Vectors (t-SNE)",
        xaxis=dict(title="t-SNE Dimension 1", showgrid=True),
        yaxis=dict(title="t-SNE Dimension 2", showgrid=True),
        hovermode='closest',
        plot_bgcolor='white',
        width=800,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create silhouette score plot if we have the data in the cluster_assignments store
    silhouette_fig = go.Figure()
    
    # Try to extract silhouette scores from the cluster_assignments store
    # This requires that we've used our new clustering approach
    k_values = []
    silhouette_scores = []
    penalized_scores = []
    best_k = None
    
    # Check if we have any silhouette score data stored in the cluster_assignments
    if 'k_values_data' in cluster_assignments:
        k_values = json.loads(cluster_assignments['k_values_data'])
    if 'silhouette_scores_data' in cluster_assignments:
        silhouette_scores = json.loads(cluster_assignments['silhouette_scores_data'])
    if 'penalized_scores_data' in cluster_assignments:
        penalized_scores = json.loads(cluster_assignments['penalized_scores_data'])
    if 'best_k_value' in cluster_assignments:
        best_k = int(cluster_assignments['best_k_value'])
    
    if k_values and silhouette_scores:
        # Plot original silhouette scores
        silhouette_fig.add_trace(go.Scatter(
            x=k_values,
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Scores',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Plot penalized scores if available
        if penalized_scores:
            silhouette_fig.add_trace(go.Scatter(
                x=k_values,
                y=penalized_scores,
                mode='lines+markers',
                name='Penalized Scores',
                line=dict(color='green', width=2, dash='dot'),
                marker=dict(size=8)
            ))
        
        # Highlight the chosen k value
        if best_k is not None and best_k in k_values:
            idx = k_values.index(best_k)
            silhouette_fig.add_trace(go.Scatter(
                x=[best_k],
                y=[silhouette_scores[idx]],
                mode='markers',
                name=f'Selected k={best_k}',
                marker=dict(color='red', size=12, symbol='star')
            ))
        
        silhouette_fig.update_layout(
            title="Silhouette Score Analysis for K Selection",
            xaxis=dict(title="Number of Clusters (k)", tickmode='linear', dtick=1),
            yaxis=dict(title="Silhouette Score"),
            hovermode='closest',
            plot_bgcolor='white',
            width=800,
            height=300,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    else:
        # If we don't have silhouette data, create a placeholder
        silhouette_fig.add_annotation(
            text="Silhouette score data not available. Run clustering again to see analysis.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        silhouette_fig.update_layout(
            title="Silhouette Score Analysis",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    
    return fig, silhouette_fig

# Callback to handle PCA analysis button
@callback(
    [Output('pca-results', 'data'),
     Output('pca-info', 'children'),
     Output('pca-section', 'style'),
     Output('pca-visualization', 'figure'),
     Output('pca-variance-plot', 'figure')],
    [Input('pca-button', 'n_clicks'),
     Input('vector-type', 'value')],
    [State('peak-frames', 'data'),
     State('pca-components-slider', 'value')],
    prevent_initial_call=True
)
def handle_pca_analysis(n_clicks, vector_type, peak_frames, n_components):
    if not n_clicks or not peak_frames:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No PCA analysis performed yet. Click 'Run PCA Analysis' to begin.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_fig.update_layout(
            title="PCA Visualization",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        empty_variance_fig = go.Figure()
        empty_variance_fig.update_layout(
            title="Explained Variance",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return None, "", {'display': 'none'}, empty_fig, empty_variance_fig
    
    # Extract vectors based on vector_type
    vectors = []
    peak_indices = []
    
    for peak_idx, frame_info in peak_frames.items():
        feature_vector = None
        
        if vector_type == 'eventfulness':
            # Use eventfulness vector
            feature_vector = frame_info.get('eventfulness_vector', None)
            
        elif vector_type == 'pose':
            # Use pose vector
            pose_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            if pose_vector is not None and pose_detected:
                feature_vector = pose_vector
                
        elif vector_type == 'combined':
            # Use both vectors
            eventfulness_vector = frame_info.get('eventfulness_vector', None)
            pose_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            
            if eventfulness_vector is not None and pose_vector is not None and pose_detected:
                # Normalize each vector separately before combining
                if len(eventfulness_vector) > 0:
                    e_norm = np.linalg.norm(eventfulness_vector)
                    if e_norm > 0:
                        eventfulness_vector = [x / e_norm for x in eventfulness_vector]
                
                if len(pose_vector) > 0:
                    p_norm = np.linalg.norm(pose_vector)
                    if p_norm > 0:
                        pose_vector = [x / p_norm for x in pose_vector]
                
                # Combine the vectors
                feature_vector = eventfulness_vector + pose_vector
        
        # Add vector if available
        if feature_vector is not None:
            vectors.append(feature_vector)
            peak_indices.append(peak_idx)
    
    if len(vectors) < 2:
        # Not enough vectors for PCA
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Not enough {vector_type} vectors for PCA analysis (need at least 2).",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        empty_fig.update_layout(
            title=f"PCA Analysis - {vector_type.capitalize()} Vectors",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        empty_variance_fig = go.Figure()
        empty_variance_fig.update_layout(
            title="Explained Variance",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return None, html.Div(f"Not enough {vector_type} vectors for PCA analysis.", style={'color': 'red'}), {'display': 'block'}, empty_fig, empty_variance_fig
    
    # Perform PCA analysis
    pca_results = perform_pca_analysis(vectors, n_components=n_components)
    
    if pca_results is None:
        # PCA analysis failed
        return None, html.Div("PCA analysis failed. Check logs for details.", style={'color': 'red'}), {'display': 'block'}, empty_fig, empty_variance_fig
    
    # Create info display
    info_text = f"Vector Type: {vector_type.capitalize()}, Components: {pca_results['n_components']}, Total Explained Variance: {pca_results['total_explained_variance']:.4f}"
    
    # Create PCA visualization figure
    pca_fig = go.Figure()
    
    # Get the first two PCA components
    X_pca = pca_results['pca_vectors']
    
    # Create scatter plot of PCA components
    for i, peak_idx in enumerate(peak_indices):
        frame_info = peak_frames[peak_idx]
        time = frame_info['time']
        peak_value = frame_info['peak_value']
        
        # Create hover text
        hover_text = f"Peak Index: {peak_idx}<br>Time: {time:.2f}s<br>Peak Value: {peak_value:.3f}"
        
        # Add point to scatter plot
        pca_fig.add_trace(go.Scatter(
            x=[X_pca[i, 0]],
            y=[X_pca[i, 1]],
            mode='markers+text',
            name=f'Peak {peak_idx}',
            text=[f"{i}"],
            textposition="top center",
            marker=dict(
                size=10,
                color=peak_value,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Peak Value")
            ),
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
    
    # Update layout
    pca_fig.update_layout(
        title=f"PCA of {vector_type.capitalize()} Vectors (First Two Components)",
        xaxis=dict(title=f"PC1 ({pca_results['explained_variance_ratio'][0]:.2%} variance)"),
        yaxis=dict(title=f"PC2 ({pca_results['explained_variance_ratio'][1]:.2%} variance)"),
        hovermode='closest',
        plot_bgcolor='white',
        width=800,
        height=500
    )
    
    # Create explained variance plot
    variance_fig = go.Figure()
    
    # Add cumulative explained variance
    cumulative_variance = np.cumsum(pca_results['explained_variance_ratio'])
    variance_fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance) + 1)),
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add individual explained variance
    variance_fig.add_trace(go.Bar(
        x=list(range(1, len(pca_results['explained_variance_ratio']) + 1)),
        y=pca_results['explained_variance_ratio'],
        name='Individual Explained Variance',
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    # Update layout
    variance_fig.update_layout(
        title='Explained Variance by PCA Components',
        xaxis=dict(title='Number of Components', tickmode='linear', dtick=1),
        yaxis=dict(title='Explained Variance Ratio', range=[0, 1]),
        hovermode='x',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white'
    )
    
    # Return results
    return pca_results, html.Div(info_text, style={'color': '#4CAF50', 'fontWeight': 'bold'}), {'display': 'block'}, pca_fig, variance_fig

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

# Function to perform PCA analysis on vectors
def perform_pca_analysis(vectors, n_components=0.95):
    """
    Performs PCA analysis on a set of vectors.
    
    Args:
        vectors: List of vectors to analyze
        n_components: Number of components to keep (if float, represents variance to preserve)
        
    Returns:
        Dictionary containing PCA results
    """
    if len(vectors) < 2:
        logger.warning("Not enough vectors for PCA analysis (need at least 2)")
        return None
    
    # Convert to numpy array
    X = np.array(vectors)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create results dictionary
    pca_results = {
        'pca_vectors': X_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'n_components': pca.n_components_,
        'total_explained_variance': sum(pca.explained_variance_ratio_),
        'scaler': scaler,
        'pca': pca
    }
    
    logger.info(f"PCA analysis completed: {len(vectors)} vectors reduced to {pca.n_components_} components")
    logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca_results

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

# Function to cluster eventfulness vectors using K-means with silhouette score analysis
def cluster_eventfulness_vectors(peak_frames, max_clusters=6, vector_type='eventfulness'):
    """
    Clusters vectors from peak frames using K-means with silhouette score analysis
    to determine the optimal number of clusters.
    
    Args:
        peak_frames: Dictionary mapping peak indices to frame info (including eventfulness_vector and pose_vector)
        max_clusters: Maximum number of clusters to consider
        vector_type: Type of vector to use ('eventfulness', 'pose', or 'combined')
        
    Returns:
        Dictionary mapping peak indices to cluster assignments, and cluster info
    """
    if not peak_frames:
        return None, None
    
    # Extract vectors and corresponding peak indices
    vectors = []
    peak_indices = []
    
    for peak_idx, frame_info in peak_frames.items():
        if vector_type == 'eventfulness':
            # Use eventfulness vector only
            feature_vector = frame_info.get('eventfulness_vector', None)
            if feature_vector is not None:
                vectors.append(feature_vector)
                peak_indices.append(peak_idx)
        
        elif vector_type == 'pose':
            # Use pose vector only
            feature_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            if feature_vector is not None and pose_detected:
                vectors.append(feature_vector)
                peak_indices.append(peak_idx)
        
        elif vector_type == 'combined':
            # Use both eventfulness and pose vectors
            eventfulness_vector = frame_info.get('eventfulness_vector', None)
            pose_vector = frame_info.get('pose_vector', None)
            pose_detected = frame_info.get('pose_detected', False)
            
            if eventfulness_vector is not None and pose_vector is not None and pose_detected:
                # Normalize each vector separately before combining
                if len(eventfulness_vector) > 0:
                    e_norm = np.linalg.norm(eventfulness_vector)
                    if e_norm > 0:
                        eventfulness_vector = [x / e_norm for x in eventfulness_vector]
                
                if len(pose_vector) > 0:
                    p_norm = np.linalg.norm(pose_vector)
                    if p_norm > 0:
                        pose_vector = [x / p_norm for x in pose_vector]
                
                # Combine the vectors
                combined_vector = eventfulness_vector + pose_vector
                vectors.append(combined_vector)
                peak_indices.append(peak_idx)
    
    if len(vectors) < 2:
        logger.warning(f"Not enough {vector_type} vectors for clustering (need at least 2)")
        return None, None
    
    # Convert to numpy array
    X = np.array(vectors)
    
    # Standardize the vectors (similar to PCA preprocessing)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Standardized {len(vectors)} {vector_type} vectors before clustering")
    
    # Try different k values and calculate silhouette scores
    # Limit max_clusters to the number of vectors
    max_k = min(max_clusters, len(vectors) - 1)
    k_values = range(2, max_k + 1) if max_k >= 2 else [2]
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score if we have at least 2 clusters
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            logger.info(f"K={k}, Silhouette Score: {score:.4f}")
        else:
            silhouette_scores.append(0)
            logger.info(f"K={k}, Only one cluster found")
    
    # Apply a penalty to favor smaller k values
    penalty_factor = 0.00  # Adjust this value to control how much to penalize larger cluster counts
    penalized_scores = [score - (k * penalty_factor) for score, k in zip(silhouette_scores, k_values)]
    
    # Get best k from penalized scores
    if not silhouette_scores:
        best_k = 2  # Default if we couldn't calculate scores
    else:
        best_k = list(k_values)[np.argmax(penalized_scores)]
    
    logger.info(f"Selected optimal number of clusters: {best_k}")
    
    # Perform clustering with the optimal k
    cluster_assignments = {}
    cluster_info = {}
    
    try:
        # Apply K-means with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score for the final clustering
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = 0
        
        cluster_info = {
            'algorithm': 'K-Means',
            'vector_type': vector_type,
            'n_clusters': best_k,
            'n_samples': len(vectors),
            'silhouette_score': float(sil_score),
            'inertia': float(kmeans.inertia_),
            'k_values': list(k_values),
            'silhouette_scores': silhouette_scores,
            'penalized_scores': penalized_scores
        }
        
        # Map peak indices to cluster assignments
        for i, peak_idx in enumerate(peak_indices):
            cluster_assignments[str(peak_idx)] = int(labels[i])
        
        logger.info(f"Clustering completed: {cluster_info}")
        return cluster_assignments, cluster_info
        
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        return None, None

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

def extract_frames_at_peaks(video_path, peaks, video_info, eventfulness_data):
    """
    Extracts frames from the video at peak locations, extracts full eventfulness vectors,
    and performs pose estimation on each frame.
    
    Args:
        video_path: Path to the video file
        peaks: List of peak indices in eventfulness data
        video_info: Dictionary containing video information
        eventfulness_data: Dictionary containing eventfulness data
        
    Returns:
        Dictionary mapping peak indices to extracted frame paths, eventfulness vectors, and pose vectors
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
    
    # Get full eventfulness vectors if available
    full_vectors = eventfulness_data.get('full_vectors', None)
    config_path = eventfulness_data.get('config_path', None)
    
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
            # Perform pose estimation on the frame
            annotated_frame, pose_vector, pose_success = perform_pose_estimation(frame)
            
            # Save the original frame
            frame_path = os.path.join(frames_dir, f"peak_{i}_idx_{peak_idx}_frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Save the annotated frame with pose landmarks
            annotated_frame_path = os.path.join(frames_dir, f"pose_annotated_peak_{i}_idx_{peak_idx}_frame_{frame_number}.jpg")
            cv2.imwrite(annotated_frame_path, annotated_frame)
            
            # Extract full eventfulness vector for this peak
            eventfulness_vector = None
            if full_vectors and peak_idx < len(full_vectors[0]):
                # Extract the full vector: [full_vectors[dim][peak_idx] for dim in range(len(full_vectors))]
                eventfulness_vector = [full_vectors[dim][peak_idx] for dim in range(len(full_vectors))]
                logger.info(f"Extracted full eventfulness vector for peak {i}: index={peak_idx}, vector_length={len(eventfulness_vector)}")
            else:
                logger.warning(f"Full eventfulness vectors not available for peak {i}: index={peak_idx}")
            
            # Store the mapping with pose vector
            peak_frames[peak_idx] = {
                'path': frame_path,
                'annotated_path': annotated_frame_path,
                'frame_number': frame_number,
                'peak_value': eventfulness_data['data'][peak_idx],
                'eventfulness_vector': eventfulness_vector,  # Full vector
                'pose_vector': pose_vector,  # Pose landmarks vector
                'pose_detected': pose_success,  # Whether pose detection was successful
                'time': frame_number / fps
            }
            
            logger.info(f"Extracted frame at peak {i}: index={peak_idx}, frame={frame_number}, time={frame_number/fps:.2f}s, pose_detected={pose_success}")
        else:
            logger.error(f"Failed to extract frame at peak {i}: index={peak_idx}, frame={frame_number}")
    
    # Release the video
    cap.release()
    
    # Save peak frames data with eventfulness vectors and pose vectors to config.json if config_path is available
    if config_path and peak_frames:
        try:
            # Load existing config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create peak_frames_data structure
            peak_frames_data = {}
            for peak_idx, frame_info in peak_frames.items():
                peak_frames_data[str(peak_idx)] = {
                    'frame_number': frame_info['frame_number'],
                    'time': frame_info['time'],
                    'peak_value': frame_info['peak_value'],
                    'eventfulness_vector': frame_info['eventfulness_vector'],
                    'pose_vector': frame_info['pose_vector'],
                    'pose_detected': frame_info['pose_detected'],
                    'frame_path': frame_info['path'],
                    'annotated_path': frame_info['annotated_path']
                }
            
            # Add peak_frames_data to config
            config['peak_frames'] = peak_frames_data
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved peak frames data with eventfulness and pose vectors to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save peak frames data to config.json: {str(e)}")
    
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

# Function to perform pose estimation on an image
def perform_pose_estimation(image):
    """
    Performs pose estimation on an image using MediaPipe.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Tuple containing:
        - annotated_image: Image with pose landmarks drawn
        - pose_vector: Flattened vector of pose landmarks (x, y, z, visibility for each landmark)
        - success: Boolean indicating if pose estimation was successful
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        results = pose.process(image_rgb)
        
        # Create pose vector
        pose_vector = []
        success = False
        
        if results.pose_landmarks:
            success = True
            # Extract landmarks into a flat vector
            for landmark in results.pose_landmarks.landmark:
                pose_vector.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            
            # Create annotated image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        else:
            # If no landmarks detected, return the original image and an empty vector
            annotated_image = image
            pose_vector = [0] * (33 * 4)  # 33 landmarks with x, y, z, visibility
        
        return annotated_image, pose_vector, success

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
