import cv2
import numpy as np
import os
import json
import logging
import datetime
import concurrent.futures
import glob
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp

# Set up logging
log_file = "/home/is1893/Mirror2/video_analysis.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('video_analysis')

# DTW imports - try dtaidistance first, fallback to scipy
try:
    from dtaidistance import dtw
    DTW_LIBRARY = 'dtaidistance'
    logger.info("Using dtaidistance library for DTW computations")
except ImportError:
    try:
        from scipy.spatial.distance import euclidean
        from scipy.ndimage import uniform_filter1d
        DTW_LIBRARY = 'scipy'
        logger.info("dtaidistance not available, using scipy for DTW computations")
    except ImportError:
        DTW_LIBRARY = None
        logger.warning("No DTW library available. DTW segmentation will not be available.")

# Define constants
RESULTS_DIR = "/home/is1893/Mirror2/dataSets/test_data/results"

class VideoAnalysisBackend:
    """Backend class for video analysis with concurrent processing capabilities."""
    
    # Peak detection parameters - shared between backend and frontend
    PEAK_DETECTION_DISTANCE = 1  # Minimum distance between peaks
    PEAK_DETECTION_PROMINENCE = 0.2  # Minimum prominence for peaks (how much they stand out)
    
    def __init__(self):
        """Initialize the VideoAnalysisBackend."""
        self.mp_pose = mp.solutions.pose
        # Define which landmarks to keep (reduced face, hand, and foot vectors)
        # 0: nose, 11-16: shoulders/elbows/wrists, 23-28: hips/knees/ankles
        self.LANDMARKS_TO_KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    @staticmethod
    def detect_peaks(data, distance=None, prominence=None):
        """
        Detect peaks in eventfulness data using consistent parameters.
        This method is used by both backend and frontend to ensure consistency.
        
        Args:
            data: List or array of eventfulness values
            distance: Minimum distance between peaks (default: class constant)
            prominence: Minimum prominence for peaks - how much they stand out (default: class constant)
            
        Returns:
            Tuple of (peak_indices, peak_values, detection_params)
        """
        from scipy.signal import find_peaks
        
        # Use class defaults if not specified
        if distance is None:
            distance = VideoAnalysisBackend.PEAK_DETECTION_DISTANCE
        if prominence is None:
            prominence = VideoAnalysisBackend.PEAK_DETECTION_PROMINENCE
        
        # Find peaks using scipy's find_peaks with prominence
        peaks, properties = find_peaks(data, prominence=prominence, distance=distance)
        
        # Convert to list of integers and get peak values
        peaks = [int(peak) for peak in peaks]
        peak_values = [data[p] for p in peaks]
        
        # Store detection parameters for reproducibility
        detection_params = {
            'distance': distance,
            'prominence': prominence,
            'total_peaks_detected': len(peaks),
            'peaks_kept': len(peaks)
        }
        
        return peaks, peak_values, detection_params
    
    def get_video_info(self, video_path):
        """Extract basic information from a video file."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
            
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
            "duration": duration,
            "video_path": video_path
        }
        
        return video_info
    
    def save_uploaded_video(self, file_content, original_filename, category_name=None):
        """
        Saves an uploaded video file to the appropriate directory structure.
        
        Args:
            file_content: Binary content of the uploaded video file
            original_filename: Original filename of the uploaded video
            category_name: Optional category name for organizing videos (e.g., 'JumpJack', 'Pushup')
                          If None, uses the filename without extension as category
            
        Returns:
            Tuple of (success: bool, video_path: str, message: str)
        """
        try:
            # Validate file extension
            if not original_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return False, None, "Invalid file format. Please upload a video file (.mp4, .avi, .mov, .mkv)"
            
            # Clean the filename
            safe_filename = os.path.basename(original_filename)
            filename_without_ext = os.path.splitext(safe_filename)[0]
            
            # Determine category name
            if category_name is None:
                category_name = filename_without_ext
            
            # Clean category name (remove special characters)
            category_name = "".join(c for c in category_name if c.isalnum() or c in (' ', '-', '_')).strip()
            if not category_name:
                category_name = "Uploaded"
            
            # Create directory structure: /val/{category_name}/
            val_dir = "/home/is1893/Mirror2/dataSets/test_data/val"
            category_dir = os.path.join(val_dir, category_name)
            os.makedirs(category_dir, exist_ok=True)
            
            # Determine final filename (use category name + .mp4 for consistency)
            final_filename = f"{category_name}.mp4"
            video_path = os.path.join(category_dir, final_filename)
            
            # Check if file already exists
            if os.path.exists(video_path):
                # Add timestamp to make unique
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                final_filename = f"{category_name}_{timestamp}.mp4"
                video_path = os.path.join(category_dir, final_filename)
            
            # Save the file
            with open(video_path, 'wb') as f:
                f.write(file_content)
            
            # Verify the video can be opened
            video_info = self.get_video_info(video_path)
            if not video_info:
                # If video is invalid, delete it
                os.remove(video_path)
                return False, None, "Uploaded file is not a valid video or cannot be processed"
            
            logger.info(f"Successfully saved uploaded video: {video_path}")
            logger.info(f"Video info: {video_info['duration']:.1f}s, {video_info['width']}x{video_info['height']}, {video_info['fps']:.1f} fps")
            
            return True, video_path, f"Video uploaded successfully to {category_name}/{final_filename}"
            
        except Exception as e:
            logger.error(f"Error saving uploaded video: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None, f"Error saving video: {str(e)}"
    
    def find_matching_config(self, video_path):
        """Find the matching config.json file for a given video path."""
        config_files = glob.glob(os.path.join(
            RESULTS_DIR, "**/config.json"), recursive=True)
            
        # Use realpath to resolve symlinks for proper path comparison
        video_path_normalized = os.path.realpath(video_path)
        video_filename = os.path.basename(video_path)
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if "video_path" in config:
                    config_video_path = config["video_path"]
                    config_video_filename = os.path.basename(config_video_path)
                    
                    # Use realpath to resolve symlinks for proper path comparison
                    if (os.path.realpath(config_video_path) == video_path_normalized or
                            config_video_filename == video_filename):
                        return config_file, config
            except Exception as e:
                logger.error(f"Error reading config file {config_file}: {str(e)}")
                continue
                
        return None, None
    
    def perform_pose_estimation(self, image, draw=False, normalize_centroid=True):
        """
        Performs pose estimation on an image using MediaPipe with reduced face, hand, and foot vectors.
        
        Args:
            image: OpenCV image (BGR format)
            draw: Boolean indicating whether to draw landmarks on the image
            normalize_centroid: Boolean indicating whether to normalize landmarks by centroid
            
        Returns:
            Tuple containing:
            - annotated_image: Image with pose landmarks drawn (or original image if draw=False)
            - pose_vector: Flattened vector of pose landmarks
            - success: Boolean indicating if pose estimation was successful
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        with self.mp_pose.Pose(
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
                
                # If centroid normalization is requested, calculate the centroid
                if normalize_centroid:
                    # Calculate centroid of all landmarks we're keeping
                    centroid_x = 0
                    centroid_y = 0
                    centroid_z = 0
                    valid_landmarks = 0
                    
                    for idx in self.LANDMARKS_TO_KEEP:
                        landmark = results.pose_landmarks.landmark[idx]
                        # Only use landmarks with good visibility for centroid calculation
                        if landmark.visibility > 0.5:
                            centroid_x += landmark.x
                            centroid_y += landmark.y
                            centroid_z += landmark.z
                            valid_landmarks += 1
                            
                    # Avoid division by zero
                    if valid_landmarks > 0:
                        centroid_x /= valid_landmarks
                        centroid_y /= valid_landmarks
                        centroid_z /= valid_landmarks
                        
                        # Extract only the selected landmarks into a flat vector, normalized by centroid
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if idx in self.LANDMARKS_TO_KEEP:
                                # Normalize by subtracting centroid
                                pose_vector.extend([
                                    landmark.x - centroid_x,
                                    landmark.y - centroid_y,
                                    landmark.z - centroid_z,
                                    landmark.visibility
                                ])
                    else:
                        # If no valid landmarks for centroid, just use the original landmarks
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if idx in self.LANDMARKS_TO_KEEP:
                                pose_vector.extend(
                                    [landmark.x, landmark.y, landmark.z, landmark.visibility])
                else:
                    # No normalization, just extract the landmarks
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx in self.LANDMARKS_TO_KEEP:
                            pose_vector.extend(
                                [landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Create annotated image
                if draw:
                    annotated_image = image.copy()
                    
                    # Draw only the landmarks we want to keep
                    for idx in self.LANDMARKS_TO_KEEP:
                        landmark = results.pose_landmarks.landmark[idx]
                        # Convert normalized coordinates to pixel coordinates
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        # Draw a circle at the landmark position
                        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw connections between landmarks we want to keep
                    # Define the connections we want to draw
                    connections = [
                        (0, 11), (0, 12),  # nose to shoulders
                        (11, 12),  # shoulder to shoulder
                        (11, 13), (13, 15),  # left arm
                        (12, 14), (14, 16),  # right arm
                        (11, 23), (12, 24),  # shoulders to hips
                        (23, 24),  # hip to hip
                        (23, 25), (25, 27),  # left leg
                        (24, 26), (26, 28)   # right leg
                    ]
                    
                    # Draw each connection
                    for connection in connections:
                        idx1, idx2 = connection
                        # Only draw if both landmarks are in our keep list
                        if idx1 in self.LANDMARKS_TO_KEEP and idx2 in self.LANDMARKS_TO_KEEP:
                            landmark1 = results.pose_landmarks.landmark[idx1]
                            landmark2 = results.pose_landmarks.landmark[idx2]
                            
                            # Convert normalized coordinates to pixel coordinates
                            x1 = int(landmark1.x * image.shape[1])
                            y1 = int(landmark1.y * image.shape[0])
                            x2 = int(landmark2.x * image.shape[1])
                            y2 = int(landmark2.y * image.shape[0])
                            
                            # Draw a line between the landmarks
                            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # If drawing is disabled, return the original image
                    annotated_image = image
            else:
                # If no landmarks detected, return the original image and an empty vector
                annotated_image = image
                # Only for the landmarks we're keeping (13 landmarks with x, y, z, visibility)
                pose_vector = [0] * (len(self.LANDMARKS_TO_KEEP) * 4)
                
            return annotated_image, pose_vector, success
    
    def process_video_chunk(self, video_path, start_frame, end_frame, sample_rate=1):
        """
        Process a chunk of video frames for parallel processing.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            sample_rate: Sample every Nth frame
            
        Returns:
            Dictionary of pose data for processed frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return {}
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Set the starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Dictionary to store pose data for each frame
        pose_data = {}
        
        # Process frames in the chunk
        frame_number = start_frame
        while frame_number < end_frame:
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process only every Nth frame
            if (frame_number - start_frame) % sample_rate == 0:
                # Calculate time in seconds
                time = frame_number / fps if fps > 0 else 0
                
                # Perform pose estimation WITH normalization (for cosine similarity)
                _, pose_vector_normalized, pose_detected = self.perform_pose_estimation(
                    frame, draw=False, normalize_centroid=True)
                
                # Perform pose estimation WITHOUT normalization (for clustering)
                _, pose_vector_raw, _ = self.perform_pose_estimation(
                    frame, draw=False, normalize_centroid=False)
                
                # Store the pose data
                if pose_detected:
                    pose_data[str(frame_number)] = {
                        'frame_number': frame_number,
                        'time': time,
                        'pose_vector': pose_vector_normalized,  # For cosine similarity
                        'pose_vector_raw': pose_vector_raw,     # For clustering
                    }
                    
            # Increment frame counter
            frame_number += 1
            
        # Release the video
        cap.release()
        
        return pose_data
    
    def perform_full_video_pose_estimation(self, video_path, video_info, sample_rate=1, num_workers=4):
        """
        Performs pose estimation on the entire video using multiple workers for parallel processing.
        
        Args:
            video_path: Path to the video file
            video_info: Dictionary containing video information
            sample_rate: Sample every Nth frame (default: 1)
            num_workers: Number of parallel workers (default: 4)
            
        Returns:
            Dictionary containing frame-by-frame pose estimation results
        """
        if not video_path or not video_info:
            return {}
            
        # Get video properties
        frame_count = video_info['frame_count']
        
        # Calculate chunk size for each worker
        chunk_size = frame_count // num_workers
        if chunk_size < 10:  # If video is very short, don't parallelize
            num_workers = 1
            chunk_size = frame_count
            
        # Create chunks
        chunks = []
        for i in range(num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < num_workers - 1 else frame_count
            chunks.append((start, end))
            
        logger.info(f"Processing video with {num_workers} workers, chunk size: {chunk_size} frames")
        
        # Process chunks in parallel
        pose_data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks
            future_to_chunk = {
                executor.submit(self.process_video_chunk, video_path, start, end, sample_rate): (start, end)
                for start, end in chunks
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_chunk):
                start, end = future_to_chunk[future]
                try:
                    chunk_data = future.result()
                    pose_data.update(chunk_data)
                    logger.info(f"Completed chunk {start}-{end}, processed {len(chunk_data)} frames")
                except Exception as e:
                    logger.error(f"Error processing chunk {start}-{end}: {str(e)}")
                    
        logger.info(f"Completed pose estimation on {len(pose_data)} frames")
        return pose_data
    
    def cluster_eventfulness_vectors(self, peak_frames, max_clusters=6, vector_type='pose'):
        """
        Clusters vectors from peak frames using K-means with silhouette score analysis.
        
        Args:
            peak_frames: Dictionary mapping peak indices to frame info
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
                # Use RAW (non-normalized) pose vector for clustering
                feature_vector = frame_info.get('pose_vector_raw', None)
                # Fallback to normalized if raw not available (backward compatibility)
                if feature_vector is None:
                    feature_vector = frame_info.get('pose_vector', None)
                pose_detected = frame_info.get('pose_detected', False)
                if feature_vector is not None and pose_detected:
                    vectors.append(feature_vector)
                    peak_indices.append(peak_idx)
                    
            elif vector_type == 'combined':
                # Use both eventfulness and RAW (non-normalized) pose vectors for clustering
                eventfulness_vector = frame_info.get('eventfulness_vector', None)
                pose_vector = frame_info.get('pose_vector_raw', None)
                # Fallback to normalized if raw not available (backward compatibility)
                if pose_vector is None:
                    pose_vector = frame_info.get('pose_vector', None)
                pose_detected = frame_info.get('pose_detected', False)
                
                if eventfulness_vector is not None and pose_vector is not None and pose_detected:
                    # Normalize each vector separately before combining
                    if len(eventfulness_vector) > 0:
                        e_norm = np.linalg.norm(eventfulness_vector)
                        if e_norm > 0:
                            eventfulness_vector = [
                                x / e_norm for x in eventfulness_vector]
                            
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
        
        # Standardize the vectors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Standardized {len(vectors)} {vector_type} vectors before clustering")
        
        # Try different k values and calculate silhouette scores
        # Limit max_clusters to the number of vectors
        max_k = min(max_clusters, len(vectors) - 1)
        k_values = range(2, max_k + 1) if max_k >= 2 else [2]
        silhouette_scores = []
        
        # Use ThreadPoolExecutor for parallel silhouette score calculation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define a function to compute silhouette score for a specific k
            def compute_silhouette(k):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                # Calculate silhouette score if we have at least 2 clusters
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    logger.info(f"K={k}, Silhouette Score: {score:.4f}")
                    return score
                else:
                    logger.info(f"K={k}, Only one cluster found")
                    return 0
                    
            # Submit tasks for each k value
            future_to_k = {executor.submit(compute_silhouette, k): k for k in k_values}
            
            # Collect results
            results = {}
            for future in concurrent.futures.as_completed(future_to_k):
                k = future_to_k[future]
                try:
                    score = future.result()
                    results[k] = score
                except Exception as e:
                    logger.error(f"Error calculating silhouette score for k={k}: {str(e)}")
                    results[k] = 0
                    
            # Sort results by k value
            silhouette_scores = [results[k] for k in k_values]
        
        # Apply a penalty to favor smaller k values
        penalty_factor = 0.00
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
    
    def evaluate_cluster_quality(self, peak_frames, cluster_assignments, vector_type='pose'):
        """
        Evaluates the quality of clusters and identifies outliers/noisy data points.
        
        Uses multiple metrics:
        1. Silhouette coefficient per sample (measures how well each point fits its cluster)
        2. Distance to cluster centroid (identifies points far from cluster center)
        3. Cluster cohesion (intra-cluster distance)
        4. Cluster separation (inter-cluster distance)
        
        Args:
            peak_frames: Dictionary of peak frames with vectors
            cluster_assignments: Dictionary mapping peak indices to cluster IDs
            vector_type: Type of vector used for clustering ('pose', 'eventfulness', 'combined')
            
        Returns:
            Dictionary with cluster quality metrics and outlier identification
        """
        from sklearn.metrics import silhouette_samples
        from scipy.spatial.distance import cdist
        
        if not peak_frames or not cluster_assignments:
            return None
            
        # Extract vectors and labels in the same order
        vectors = []
        peak_indices = []
        labels = []
        
        for peak_idx, frame_info in peak_frames.items():
            if str(peak_idx) not in cluster_assignments:
                continue
                
            # Get the appropriate vector based on vector_type
            if vector_type == 'pose':
                feature_vector = frame_info.get('pose_vector_raw', None)
                if feature_vector is None:
                    feature_vector = frame_info.get('pose_vector', None)
            elif vector_type == 'eventfulness':
                feature_vector = frame_info.get('eventfulness_vector', None)
            elif vector_type == 'combined':
                ev = frame_info.get('eventfulness_vector', None)
                pv = frame_info.get('pose_vector_raw', None) or frame_info.get('pose_vector', None)
                if ev and pv:
                    feature_vector = ev + pv
                else:
                    feature_vector = None
            else:
                feature_vector = None
                
            if feature_vector is not None:
                vectors.append(feature_vector)
                peak_indices.append(peak_idx)
                labels.append(cluster_assignments[str(peak_idx)])
        
        if len(vectors) < 2:
            return None
            
        # Convert to numpy arrays
        X = np.array(vectors)
        labels_array = np.array(labels)
        
        # Standardize (same as during clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Calculate silhouette coefficient for each sample
        silhouette_vals = silhouette_samples(X_scaled, labels_array)
        
        # 2. Calculate cluster centroids
        unique_labels = np.unique(labels_array)
        centroids = {}
        for label in unique_labels:
            cluster_points = X_scaled[labels_array == label]
            centroids[label] = np.mean(cluster_points, axis=0)
        
        # 3. Calculate distance of each point to its cluster centroid
        distances_to_centroid = []
        for i, label in enumerate(labels_array):
            centroid = centroids[label]
            distance = np.linalg.norm(X_scaled[i] - centroid)
            distances_to_centroid.append(distance)
        distances_to_centroid = np.array(distances_to_centroid)
        
        # 4. Calculate cluster-level metrics
        cluster_metrics = {}
        for label in unique_labels:
            cluster_mask = labels_array == label
            cluster_points = X_scaled[cluster_mask]
            cluster_silhouettes = silhouette_vals[cluster_mask]
            cluster_distances = distances_to_centroid[cluster_mask]
            
            # Intra-cluster cohesion (average distance to centroid)
            cohesion = np.mean(cluster_distances)
            
            # Cluster size
            size = np.sum(cluster_mask)
            
            # Average silhouette for this cluster
            avg_silhouette = np.mean(cluster_silhouettes)
            
            cluster_metrics[int(label)] = {
                'size': int(size),
                'cohesion': float(cohesion),
                'avg_silhouette': float(avg_silhouette),
                'avg_distance_to_centroid': float(np.mean(cluster_distances)),
                'std_distance_to_centroid': float(np.std(cluster_distances)),
                'min_silhouette': float(np.min(cluster_silhouettes)),
                'max_silhouette': float(np.max(cluster_silhouettes))
            }
        
        # 5. Identify outliers using multiple criteria
        outlier_info = {}
        for i, peak_idx in enumerate(peak_indices):
            label = labels_array[i]
            silhouette_val = silhouette_vals[i]
            distance = distances_to_centroid[i]
            
            # Calculate z-score for distance within cluster
            cluster_distances = distances_to_centroid[labels_array == label]
            mean_dist = np.mean(cluster_distances)
            std_dist = np.std(cluster_distances)
            z_score = (distance - mean_dist) / std_dist if std_dist > 0 else 0
            
            # Determine if point is an outlier based on multiple criteria
            is_outlier = False
            outlier_reasons = []
            
            # Criterion 1: Negative silhouette (point is closer to another cluster)
            if silhouette_val < 0:
                is_outlier = True
                outlier_reasons.append('negative_silhouette')
            
            # Criterion 2: Very low silhouette (poorly clustered)
            elif silhouette_val < 0.1:
                is_outlier = True
                outlier_reasons.append('low_silhouette')
            
            # Criterion 3: High z-score (far from cluster centroid)
            if z_score > 2.5:
                is_outlier = True
                outlier_reasons.append('high_distance')
            
            outlier_info[peak_idx] = {
                'cluster_id': int(label),
                'silhouette': float(silhouette_val),
                'distance_to_centroid': float(distance),
                'z_score': float(z_score),
                'is_outlier': is_outlier,
                'outlier_reasons': outlier_reasons,
                'quality_score': float(silhouette_val)  # Overall quality (higher is better)
            }
        
        # 6. Calculate overall clustering quality
        overall_silhouette = np.mean(silhouette_vals)
        num_outliers = sum(1 for info in outlier_info.values() if info['is_outlier'])
        outlier_percentage = (num_outliers / len(outlier_info)) * 100 if len(outlier_info) > 0 else 0
        
        quality_report = {
            'overall_silhouette': float(overall_silhouette),
            'num_samples': len(vectors),
            'num_clusters': len(unique_labels),
            'num_outliers': num_outliers,
            'outlier_percentage': float(outlier_percentage),
            'cluster_metrics': cluster_metrics,
            'outlier_info': outlier_info
        }
        
        logger.info(f"Cluster Quality: Silhouette={overall_silhouette:.3f}, Outliers={num_outliers}/{len(vectors)} ({outlier_percentage:.1f}%)")
        
        return quality_report
    
    def filter_outliers_from_clusters(self, peak_frames, cluster_assignments, quality_report):
        """
        Filters out outlier data points from cluster assignments based on quality metrics.
        
        Args:
            peak_frames: Dictionary of peak frames
            cluster_assignments: Original cluster assignments
            quality_report: Quality report from evaluate_cluster_quality()
            
        Returns:
            Filtered cluster assignments (outliers removed), outlier assignments
        """
        if not quality_report or 'outlier_info' not in quality_report:
            return cluster_assignments, {}
            
        filtered_assignments = {}
        outlier_assignments = {}
        
        for peak_idx_str, cluster_id in cluster_assignments.items():
            peak_idx = int(peak_idx_str) if isinstance(peak_idx_str, str) else peak_idx_str
            
            if peak_idx in quality_report['outlier_info']:
                outlier_data = quality_report['outlier_info'][peak_idx]
                
                if outlier_data['is_outlier']:
                    # Mark as outlier
                    outlier_assignments[peak_idx_str] = {
                        'original_cluster': cluster_id,
                        'silhouette': outlier_data['silhouette'],
                        'reasons': outlier_data['outlier_reasons']
                    }
                else:
                    # Keep in filtered assignments
                    filtered_assignments[peak_idx_str] = cluster_id
            else:
                # Keep if not evaluated
                filtered_assignments[peak_idx_str] = cluster_id
        
        logger.info(f"Filtered clusters: Kept {len(filtered_assignments)}, Removed {len(outlier_assignments)} outliers")
        
        return filtered_assignments, outlier_assignments
    
    def calculate_cluster_centroids(self, peak_frames, cluster_assignments):
        """
        Calculates the centroid pose vector for each cluster.
        Uses NORMALIZED pose vectors to match what will be compared in cosine similarity.
        
        Note: Clustering uses raw (non-normalized) vectors, but centroids are calculated
        from normalized vectors for proper cosine similarity comparison.
        
        Args:
            peak_frames: Dictionary of peak frames with pose vectors
            cluster_assignments: Dictionary mapping peak indices to cluster IDs
            
        Returns:
            Dictionary mapping cluster IDs to centroid pose vectors (normalized)
        """
        if not peak_frames or not cluster_assignments:
            return {}
            
        # Group NORMALIZED pose vectors by cluster (for cosine similarity)
        clusters = {}
        for peak_idx, frame_info in peak_frames.items():
            if str(peak_idx) in cluster_assignments and frame_info.get('pose_detected', False):
                cluster_id = cluster_assignments[str(peak_idx)]
                # Use normalized pose vector for centroid (matches cosine similarity space)
                pose_vector = frame_info.get('pose_vector', None)
                
                if pose_vector is not None:
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(pose_vector)
                    
        # Calculate centroid for each cluster
        centroids = {}
        for cluster_id, vectors in clusters.items():
            if vectors:
                # Convert to numpy array for easier calculation
                vectors_array = np.array(vectors)
                centroid = np.mean(vectors_array, axis=0)
                centroids[cluster_id] = centroid.tolist()
                
        return centroids
    
    def compute_cosine_similarities(self, pose_data, cluster_centroids, batch_size=1000):
        """
        Computes cosine similarity between each frame's pose vector and each cluster centroid.
        Uses batched processing for efficiency.
        
        Args:
            pose_data: Dictionary of frame-by-frame pose data
            cluster_centroids: Dictionary mapping cluster IDs to centroid pose vectors
            batch_size: Number of frames to process in each batch
            
        Returns:
            Dictionary mapping frame numbers to similarity scores for each cluster
        """
        if not pose_data or cluster_centroids is None or len(cluster_centroids) == 0:
            return {}
            
        # Dictionary to store similarity scores
        similarities = {}
        
        # Convert centroids to numpy array
        centroid_ids = sorted(cluster_centroids.keys())
        try:
            centroid_vectors = np.array([cluster_centroids[cid] for cid in centroid_ids])
            logger.info(f"Centroid vectors shape: {centroid_vectors.shape}, dtype: {centroid_vectors.dtype}")
        except Exception as e:
            logger.error(f"Error converting centroids to numpy array: {str(e)}")
            return {}
        
        # Get all frame indices and sort them
        frame_indices = sorted(pose_data.keys(), key=lambda x: pose_data[x]['frame_number'])
        
        # Process frames in batches
        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i+batch_size]
            batch_vectors = []
            batch_data = []
            
            # Collect vectors and data for this batch
            for frame_idx in batch_indices:
                frame_data = pose_data[frame_idx]
                pose_vector = frame_data.get('pose_vector', None)
                
                if pose_vector is not None:
                    # Ensure pose_vector is a list, not a numpy array
                    if isinstance(pose_vector, np.ndarray):
                        pose_vector = pose_vector.tolist()
                    batch_vectors.append(pose_vector)
                    batch_data.append((frame_idx, frame_data))
            
            if not batch_vectors:
                continue
                
            # Convert to numpy array
            try:
                batch_vectors_array = np.array(batch_vectors)
                logger.info(f"Batch vectors shape: {batch_vectors_array.shape}, Centroid vectors shape: {centroid_vectors.shape}")
                
                # Compute cosine similarity for the entire batch at once
                sim_scores_batch = cosine_similarity(batch_vectors_array, centroid_vectors)
            except Exception as e:
                logger.error(f"Error computing cosine similarity for batch: {str(e)}")
                logger.error(f"Batch vectors type: {type(batch_vectors[0]) if batch_vectors else 'empty'}")
                logger.error(f"Centroid vectors type: {type(centroid_vectors)}")
                raise
            
            # Store results
            for j, (frame_idx, frame_data) in enumerate(batch_data):
                similarities[frame_idx] = {
                    'frame_number': frame_data['frame_number'],
                    'time': frame_data['time'],
                    'similarities': {str(centroid_ids[k]): float(score) 
                                    for k, score in enumerate(sim_scores_batch[j])}
                }
                
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(frame_indices) + batch_size - 1)//batch_size}, "
                       f"{len(batch_indices)} frames")
                
        return similarities
    
    def smooth_cosine_similarities(self, similarities, window_size=5):
        """
        Applies a running average (moving average) to smooth cosine similarity values.
        
        Args:
            similarities: Dictionary mapping frame numbers to similarity scores
            window_size: Size of the moving average window (default: 5)
            
        Returns:
            Dictionary with smoothed similarity scores
        """
        if not similarities or window_size < 1:
            return similarities
            
        # Sort frames by frame number
        sorted_frames = sorted(similarities.keys(), key=lambda x: similarities[x]['frame_number'])
        
        if len(sorted_frames) < window_size:
            logger.warning(f"Not enough frames ({len(sorted_frames)}) for smoothing window size {window_size}")
            return similarities
            
        # Get cluster IDs from first frame
        first_frame = sorted_frames[0]
        cluster_ids = list(similarities[first_frame]['similarities'].keys())
        
        # Create smoothed similarities dictionary
        smoothed_similarities = {}
        
        # For each cluster, smooth its similarity values across frames
        for cluster_id in cluster_ids:
            # Extract similarity values for this cluster across all frames
            similarity_values = []
            for frame_idx in sorted_frames:
                sim_value = similarities[frame_idx]['similarities'].get(cluster_id, 0.0)
                similarity_values.append(sim_value)
            
            # Apply moving average
            smoothed_values = []
            for i in range(len(similarity_values)):
                # Calculate window bounds
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(similarity_values), i + window_size // 2 + 1)
                
                # Compute average over window
                window_values = similarity_values[start_idx:end_idx]
                smoothed_value = np.mean(window_values)
                smoothed_values.append(smoothed_value)
            
            # Store smoothed values back
            for i, frame_idx in enumerate(sorted_frames):
                if frame_idx not in smoothed_similarities:
                    smoothed_similarities[frame_idx] = {
                        'frame_number': similarities[frame_idx]['frame_number'],
                        'time': similarities[frame_idx]['time'],
                        'similarities': {},
                        'similarities_raw': similarities[frame_idx]['similarities'].copy()  # Keep original
                    }
                smoothed_similarities[frame_idx]['similarities'][cluster_id] = float(smoothed_values[i])
        
        logger.info(f"Applied moving average smoothing with window size {window_size} to {len(sorted_frames)} frames")
        return smoothed_similarities
    
    def compute_dtw_distance(self, series1, series2, window=None):
        """
        Computes Dynamic Time Warping (DTW) distance between two time series.
        
        Args:
            series1: First time series (numpy array or list)
            series2: Second time series (numpy array or list)
            window: Sakoe-Chiba window constraint (optional)
            
        Returns:
            DTW distance (normalized by series length)
        """
        if DTW_LIBRARY is None:
            logger.error("No DTW library available. Cannot compute DTW distance.")
            return float('inf')
        
        series1 = np.array(series1)
        series2 = np.array(series2)
        
        if DTW_LIBRARY == 'dtaidistance':
            # Use dtaidistance library for efficient DTW
            try:
                if window is not None:
                    distance = dtw.distance(series1, series2, window=window)
                else:
                    distance = dtw.distance(series1, series2)
                # Normalize by average length
                normalized_distance = distance / ((len(series1) + len(series2)) / 2)
                return normalized_distance
            except Exception as e:
                logger.error(f"Error computing DTW with dtaidistance: {str(e)}")
                return float('inf')
        
        elif DTW_LIBRARY == 'scipy':
            # Fallback: implement basic DTW using dynamic programming
            n, m = len(series1), len(series2)
            
            # Initialize DTW matrix
            dtw_matrix = np.full((n + 1, m + 1), float('inf'))
            dtw_matrix[0, 0] = 0
            
            # Apply window constraint if specified
            if window is None:
                window = max(n, m)
            
            # Fill DTW matrix
            for i in range(1, n + 1):
                for j in range(max(1, i - window), min(m + 1, i + window + 1)):
                    cost = abs(series1[i-1] - series2[j-1])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
            
            # Normalize by path length (average of both series lengths)
            normalized_distance = dtw_matrix[n, m] / ((n + m) / 2)
            return normalized_distance
        
        return float('inf')
    
    def detect_dtw_change_points(self, time_series, window_size=10, threshold=1.5, min_segment_length=5):
        """
        Detects change points in a univariate time series using DTW distance between sliding windows.
        
        Args:
            time_series: 1D numpy array or list of time series values
            window_size: Size of the sliding window for comparison
            threshold: Threshold multiplier for detecting change points (relative to median DTW)
            min_segment_length: Minimum number of frames between change points
            
        Returns:
            List of change point indices
        """
        if DTW_LIBRARY is None:
            logger.warning("No DTW library available. Returning simple segmentation.")
            return [0, len(time_series)]
        
        time_series = np.array(time_series)
        n = len(time_series)
        
        # Need at least 2*window_size points for meaningful comparison
        if n < 2 * window_size:
            logger.warning(f"Time series too short ({n} points) for window size {window_size}")
            return [0, n]
        
        # Compute DTW distances between consecutive windows
        dtw_distances = []
        positions = []
        
        # Slide window and compute DTW between adjacent windows
        step_size = max(1, window_size // 2)  # 50% overlap
        
        for i in range(0, n - 2 * window_size + 1, step_size):
            window1 = time_series[i:i + window_size]
            window2 = time_series[i + window_size:i + 2 * window_size]
            
            # Compute DTW distance between windows
            distance = self.compute_dtw_distance(window1, window2, window=window_size)
            dtw_distances.append(distance)
            positions.append(i + window_size)  # Change point at window boundary
        
        if not dtw_distances:
            return [0, n]
        
        dtw_distances = np.array(dtw_distances)
        
        # Detect peaks in DTW distances (potential change points)
        # Use threshold relative to median + threshold * std
        median_dtw = np.median(dtw_distances)
        std_dtw = np.std(dtw_distances)
        dtw_threshold = median_dtw + threshold * std_dtw
        
        logger.info(f"DTW distances - median: {median_dtw:.4f}, std: {std_dtw:.4f}, threshold: {dtw_threshold:.4f}")
        
        # Find change points where DTW distance exceeds threshold
        change_points = [0]  # Always start with 0
        
        for i, (pos, dist) in enumerate(zip(positions, dtw_distances)):
            if dist > dtw_threshold:
                # Check if far enough from last change point
                if pos - change_points[-1] >= min_segment_length:
                    change_points.append(pos)
        
        # Always end with the last index
        if change_points[-1] != n:
            change_points.append(n)
        
        logger.info(f"Detected {len(change_points) - 1} segments with {len(change_points) - 2} change points")
        
        return change_points
    
    def detect_dtw_change_points_vector(self, vector_time_series, window_size=10, threshold=1.5, min_segment_length=5):
        """
        Detects change points in a MULTIVARIATE time series using DTW distance between sliding windows.
        
        This treats each time point as a vector (e.g., similarity to all clusters) and compares
        windows of vectors using multivariate DTW distance.
        
        Args:
            vector_time_series: 2D numpy array (n_timepoints × n_dimensions)
            window_size: Size of the sliding window for comparison
            threshold: Threshold multiplier for detecting change points (relative to median DTW)
            min_segment_length: Minimum number of frames between change points
            
        Returns:
            List of change point indices
        """
        if DTW_LIBRARY is None:
            logger.warning("No DTW library available. Returning simple segmentation.")
            return [0, len(vector_time_series)]
        
        vector_time_series = np.array(vector_time_series)
        n = len(vector_time_series)
        
        # Need at least 2*window_size points for meaningful comparison
        if n < 2 * window_size:
            logger.warning(f"Time series too short ({n} points) for window size {window_size}")
            return [0, n]
        
        # Compute DTW distances between consecutive windows of vectors
        dtw_distances = []
        positions = []
        
        # Slide window and compute DTW between adjacent windows
        step_size = max(1, window_size // 2)  # 50% overlap
        
        for i in range(0, n - 2 * window_size + 1, step_size):
            window1 = vector_time_series[i:i + window_size]  # Shape: (window_size, n_dims)
            window2 = vector_time_series[i + window_size:i + 2 * window_size]
            
            # Compute multivariate DTW distance
            distance = self.compute_multivariate_dtw_distance(window1, window2, window=window_size)
            dtw_distances.append(distance)
            positions.append(i + window_size)  # Change point at window boundary
        
        if not dtw_distances:
            return [0, n]
        
        dtw_distances = np.array(dtw_distances)
        
        # Detect peaks in DTW distances (potential change points)
        median_dtw = np.median(dtw_distances)
        std_dtw = np.std(dtw_distances)
        dtw_threshold = median_dtw + threshold * std_dtw
        
        logger.info(f"Vector DTW distances - median: {median_dtw:.4f}, std: {std_dtw:.4f}, threshold: {dtw_threshold:.4f}")
        
        # Find change points where DTW distance exceeds threshold
        change_points = [0]  # Always start with 0
        
        for i, (pos, dist) in enumerate(zip(positions, dtw_distances)):
            if dist > dtw_threshold:
                # Check if far enough from last change point
                if pos - change_points[-1] >= min_segment_length:
                    change_points.append(pos)
        
        # Always end with the last index
        if change_points[-1] != n:
            change_points.append(n)
        
        logger.info(f"Detected {len(change_points) - 1} segments with {len(change_points) - 2} change points")
        
        return change_points
    
    def compute_multivariate_dtw_distance(self, series1, series2, window=None):
        """
        Computes DTW distance between two MULTIVARIATE time series.
        
        Args:
            series1: 2D array (n_timepoints1 × n_dimensions)
            series2: 2D array (n_timepoints2 × n_dimensions)
            window: Sakoe-Chiba window constraint (optional)
            
        Returns:
            DTW distance (normalized)
        """
        if DTW_LIBRARY is None:
            return float('inf')
        
        series1 = np.array(series1)
        series2 = np.array(series2)
        
        # For multivariate DTW, we compute DTW on each dimension and aggregate
        # This is a common approach: sum or average DTW distances across dimensions
        
        if len(series1.shape) == 1:
            # Univariate case
            return self.compute_dtw_distance(series1, series2, window)
        
        n_dims = series1.shape[1]
        total_distance = 0.0
        
        for dim in range(n_dims):
            dim_distance = self.compute_dtw_distance(series1[:, dim], series2[:, dim], window)
            total_distance += dim_distance
        
        # Average across dimensions for normalized comparison
        avg_distance = total_distance / n_dims
        
        return avg_distance
    
    def merge_nearby_change_points(self, change_points, min_distance):
        """
        Merges change points that are too close together.
        
        Args:
            change_points: List of change point indices
            min_distance: Minimum distance between change points
            
        Returns:
            Filtered list of change points
        """
        if len(change_points) <= 2:
            return change_points
        
        merged = [change_points[0]]
        
        for i in range(1, len(change_points) - 1):
            if change_points[i] - merged[-1] >= min_distance:
                merged.append(change_points[i])
        
        # Always keep the last point
        merged.append(change_points[-1])
        
        return merged
    
    def create_segments_from_boundaries(self, change_points, frame_numbers, times):
        """
        Creates segment objects from change point boundaries.
        
        Args:
            change_points: List of change point indices (in time series space)
            frame_numbers: List of actual frame numbers corresponding to time series
            times: List of timestamps corresponding to frames
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            
            # Map to actual frame numbers
            start_frame = frame_numbers[start_idx]
            end_frame = frame_numbers[min(end_idx, len(frame_numbers) - 1)]
            
            # Get timestamps
            start_time = times[start_idx]
            end_time = times[min(end_idx, len(times) - 1)]
            
            segment = {
                'segment_id': i,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(end_time - start_time),
                'num_frames': int(end_idx - start_idx)
            }
            
            segments.append(segment)
        
        return segments
    
    def segment_cosine_similarity_with_dtw(self, similarities, window_size=10, threshold=1.5, min_segment_length=5):
        """
        Segments cosine similarity time series using DTW-based change point detection.
        
        This method treats the cosine similarities as a MULTIVARIATE time series (vector at each time point)
        and finds global change points where the overall similarity pattern changes, rather than
        segmenting each cluster independently.
        
        Args:
            similarities: Dictionary mapping frame indices to similarity scores
                         (output from compute_cosine_similarities or smooth_cosine_similarities)
            window_size: Size of sliding window for DTW comparison (default: 10)
            threshold: Threshold multiplier for change detection (default: 1.5)
            min_segment_length: Minimum frames per segment (default: 5)
            
        Returns:
            Dictionary containing segmentation results:
            {
                'method': 'dtw_change_detection_vector',
                'segments': [segment_dicts],  # Global segments
                'change_points': [indices],    # Global change points
                'parameters': {parameter_dict}
            }
        """
        if not similarities:
            logger.warning("No similarity data provided for DTW segmentation")
            return None
        
        if DTW_LIBRARY is None:
            logger.error("No DTW library available. Cannot perform DTW segmentation.")
            return None
        
        logger.info(f"Starting vector-based DTW segmentation with window_size={window_size}, "
                   f"threshold={threshold}, min_segment_length={min_segment_length}")
        
        # Extract and sort frames by frame number
        sorted_frames = sorted(similarities.items(), 
                              key=lambda x: similarities[x[0]]['frame_number'])
        
        if not sorted_frames:
            return None
        
        # Get cluster IDs from first frame
        first_frame_key = sorted_frames[0][0]
        cluster_ids = sorted(list(similarities[first_frame_key]['similarities'].keys()))
        
        # Extract frame numbers and times
        frame_numbers = [similarities[frame_key]['frame_number'] for frame_key, _ in sorted_frames]
        times = [similarities[frame_key]['time'] for frame_key, _ in sorted_frames]
        
        # Build multivariate time series: each row is a time point, each column is a cluster
        vector_time_series = []
        for frame_key, _ in sorted_frames:
            vector = [similarities[frame_key]['similarities'].get(cid, 0.0) for cid in cluster_ids]
            vector_time_series.append(vector)
        
        vector_time_series = np.array(vector_time_series)
        logger.info(f"Created vector time series: shape {vector_time_series.shape} "
                   f"({vector_time_series.shape[0]} frames × {vector_time_series.shape[1]} clusters)")
        
        # Detect change points using DTW on the vector time series
        change_points = self.detect_dtw_change_points_vector(
            vector_time_series,
            window_size=window_size,
            threshold=threshold,
            min_segment_length=min_segment_length
        )
        
        # Merge nearby change points
        change_points = self.merge_nearby_change_points(change_points, min_segment_length)
        
        # Create segment objects
        segments = self.create_segments_from_boundaries(
            change_points,
            frame_numbers,
            times
        )
        
        logger.info(f"Created {len(segments)} global segments from {len(change_points) - 2} change points")
        
        # Compile results
        results = {
            'method': 'dtw_change_detection_vector',
            'segments': segments,  # Global segments (not per-cluster)
            'change_points': change_points,  # Global change points
            'cluster_ids': cluster_ids,
            'parameters': {
                'window_size': window_size,
                'threshold': threshold,
                'min_segment_length': min_segment_length,
                'dtw_library': DTW_LIBRARY,
                'vector_dimensions': len(cluster_ids)
            },
            'num_clusters': len(cluster_ids),
            'total_frames': len(frame_numbers)
        }
        
        logger.info(f"Vector-based DTW segmentation completed: {len(segments)} segments")
        
        return results
    
    def segment_by_peaks_with_merging(self, similarities, peak_frames, similarity_threshold=0.85, max_passes=10):
        """
        Segments the time series using eventfulness peaks as initial boundaries,
        then iteratively merges similar neighboring segments.
        
        Args:
            similarities: Dictionary mapping frame indices to similarity scores
            peak_frames: Dictionary of peak frames from eventfulness detection
            similarity_threshold: Threshold for merging segments (0-1, higher = more similar required)
            max_passes: Maximum number of merging passes
            
        Returns:
            Dictionary containing segmentation results with merge history
        """
        if not similarities or not peak_frames:
            logger.warning("No similarity data or peak frames provided for peak-based segmentation")
            return None
        
        logger.info(f"Starting peak-based segmentation with {len(peak_frames)} peaks")
        
        # Extract and sort frames by frame number
        sorted_frames = sorted(similarities.items(), 
                              key=lambda x: similarities[x[0]]['frame_number'])
        
        if not sorted_frames:
            return None
        
        # Get cluster IDs from first frame
        first_frame_key = sorted_frames[0][0]
        cluster_ids = sorted(list(similarities[first_frame_key]['similarities'].keys()))
        
        # Extract frame numbers and times
        frame_numbers = [similarities[frame_key]['frame_number'] for frame_key, _ in sorted_frames]
        times = [similarities[frame_key]['time'] for frame_key, _ in sorted_frames]
        
        # Build multivariate time series
        vector_time_series = []
        for frame_key, _ in sorted_frames:
            vector = [similarities[frame_key]['similarities'].get(cid, 0.0) for cid in cluster_ids]
            vector_time_series.append(vector)
        
        vector_time_series = np.array(vector_time_series)
        
        # Create initial segments from peaks
        # Sort peaks by frame number
        peak_frame_numbers = sorted([peak_frames[pk]['frame_number'] for pk in peak_frames.keys()])
        
        # Map peak frame numbers to indices in the time series
        peak_indices = []
        for peak_frame in peak_frame_numbers:
            # Find closest index in frame_numbers
            closest_idx = min(range(len(frame_numbers)), 
                            key=lambda i: abs(frame_numbers[i] - peak_frame))
            peak_indices.append(closest_idx)
        
        # Remove duplicates and sort
        peak_indices = sorted(list(set(peak_indices)))
        
        # Create initial boundaries: start, peaks, end
        initial_boundaries = [0] + peak_indices + [len(frame_numbers)]
        initial_boundaries = sorted(list(set(initial_boundaries)))
        
        logger.info(f"Created {len(initial_boundaries) - 1} initial segments from {len(peak_indices)} peaks")
        
        # Create initial segments
        segments = self.create_segments_from_boundaries(
            initial_boundaries,
            frame_numbers,
            times
        )
        
        # Add similarity vectors to each segment
        for seg in segments:
            start_idx = initial_boundaries[seg['segment_id']]
            end_idx = initial_boundaries[seg['segment_id'] + 1]
            seg['start_idx'] = start_idx
            seg['end_idx'] = end_idx
            
            # Calculate mean similarity vector for this segment
            seg_vectors = vector_time_series[start_idx:end_idx]
            seg['mean_vector'] = np.mean(seg_vectors, axis=0)
            seg['std_vector'] = np.std(seg_vectors, axis=0)
        
        # Iterative merging process
        merge_history = []
        current_segments = segments.copy()
        
        for pass_num in range(max_passes):
            if len(current_segments) <= 1:
                logger.info(f"Only 1 segment remaining, stopping merging at pass {pass_num}")
                break
            
            # Calculate similarity between all neighboring segments
            merge_candidates = []
            
            for i in range(len(current_segments) - 1):
                seg1 = current_segments[i]
                seg2 = current_segments[i + 1]
                
                # Calculate cosine similarity between mean vectors
                vec1 = seg1['mean_vector']
                vec2 = seg2['mean_vector']
                
                # Normalize vectors
                vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
                vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
                
                # Cosine similarity
                similarity = np.dot(vec1_norm, vec2_norm)
                
                merge_candidates.append({
                    'index': i,
                    'seg1_id': seg1['segment_id'],
                    'seg2_id': seg2['segment_id'],
                    'similarity': float(similarity)
                })
            
            if not merge_candidates:
                logger.info(f"No merge candidates found at pass {pass_num}")
                break
            
            # Find the most similar pair
            best_candidate = max(merge_candidates, key=lambda x: x['similarity'])
            
            # Check if similarity exceeds threshold
            if best_candidate['similarity'] < similarity_threshold:
                logger.info(f"Pass {pass_num}: Best similarity {best_candidate['similarity']:.3f} "
                          f"below threshold {similarity_threshold}, stopping merging")
                break
            
            # Merge the segments
            idx = best_candidate['index']
            seg1 = current_segments[idx]
            seg2 = current_segments[idx + 1]
            
            # Create merged segment
            merged_segment = {
                'segment_id': seg1['segment_id'],  # Keep first segment's ID
                'start_frame': seg1['start_frame'],
                'end_frame': seg2['end_frame'],
                'start_time': seg1['start_time'],
                'end_time': seg2['end_time'],
                'duration': seg2['end_time'] - seg1['start_time'],
                'num_frames': seg1['num_frames'] + seg2['num_frames'],
                'start_idx': seg1['start_idx'],
                'end_idx': seg2['end_idx'],
                'merged_from': [seg1['segment_id'], seg2['segment_id']]
            }
            
            # Recalculate mean vector for merged segment
            merged_vectors = vector_time_series[merged_segment['start_idx']:merged_segment['end_idx']]
            merged_segment['mean_vector'] = np.mean(merged_vectors, axis=0)
            merged_segment['std_vector'] = np.std(merged_vectors, axis=0)
            
            # Record merge
            merge_history.append({
                'pass': pass_num,
                'merged_segments': [seg1['segment_id'], seg2['segment_id']],
                'similarity': best_candidate['similarity'],
                'new_segment_id': merged_segment['segment_id']
            })
            
            # Update segments list
            new_segments = current_segments[:idx] + [merged_segment] + current_segments[idx + 2:]
            current_segments = new_segments
            
            logger.info(f"Pass {pass_num}: Merged segments {seg1['segment_id']} and {seg2['segment_id']} "
                       f"(similarity: {best_candidate['similarity']:.3f}), "
                       f"{len(current_segments)} segments remaining")
        
        # Renumber final segments
        for i, seg in enumerate(current_segments):
            seg['final_segment_id'] = i
        
        logger.info(f"Peak-based segmentation completed: {len(segments)} initial segments -> "
                   f"{len(current_segments)} final segments after {len(merge_history)} merges")
        
        # Compile results
        results = {
            'method': 'peak_based_with_merging',
            'initial_segments': segments,
            'final_segments': current_segments,
            'merge_history': merge_history,
            'cluster_ids': cluster_ids,
            'parameters': {
                'similarity_threshold': similarity_threshold,
                'max_passes': max_passes,
                'num_peaks': len(peak_indices)
            },
            'num_clusters': len(cluster_ids),
            'total_frames': len(frame_numbers),
            'initial_segment_count': len(segments),
            'final_segment_count': len(current_segments)
        }
        
        return results
    
    def handle_full_video_analysis(self, video_path, peak_frames=None, cluster_assignments=None, num_workers=4, existing_pose_data=None, 
                                   perform_dtw_segmentation=False, dtw_window_size=10, dtw_threshold=1.5, dtw_min_segment_length=5,
                                   perform_peak_segmentation=False, peak_similarity_threshold=0.85, peak_max_passes=10):
        """
        Performs full video analysis with pose estimation, clustering, and similarity calculation.
        Uses parallel processing for improved performance.
        
        Args:
            video_path: Path to the video file
            peak_frames: Dictionary of peak frames (optional)
            cluster_assignments: Dictionary of cluster assignments (optional)
            num_workers: Number of parallel workers
            existing_pose_data: Existing pose data to avoid reprocessing the entire video (optional)
            perform_dtw_segmentation: Whether to perform DTW-based segmentation on similarity time series (default: False)
            dtw_window_size: Window size for DTW segmentation (default: 10)
            dtw_threshold: Threshold for DTW change detection (default: 1.5)
            dtw_min_segment_length: Minimum segment length for DTW (default: 5)
            perform_peak_segmentation: Whether to perform peak-based segmentation with merging (default: False)
            peak_similarity_threshold: Similarity threshold for merging segments (default: 0.85)
            peak_max_passes: Maximum number of merging passes (default: 10)
            
        Returns:
            Tuple containing:
            - pose_data: Dictionary of pose data for each frame
            - centroids: Dictionary of cluster centroids
            - similarities: Dictionary of cosine similarities
            - cluster_assignments: Dictionary of cluster assignments
            - dtw_segmentation: Dictionary of DTW segmentation results (None if not performed)
            - peak_segmentation: Dictionary of peak-based segmentation results (None if not performed)
        """
        # Log the start of processing
        if peak_frames:
            logger.info(f"Starting video analysis for {video_path} with {len(peak_frames)} peak frames")
        else:
            logger.info(f"Starting full video pose estimation for {video_path}")
        
        try:
            # Get video info
            video_info = self.get_video_info(video_path)
            if not video_info:
                logger.error(f"Failed to get video info for {video_path}")
                return None, None, None, None
                
            # Process every frame for cosine similarity calculation
            sample_rate = 1
            
            # Step 1: Perform pose estimation
            if existing_pose_data:
                logger.info("Step 1/5: Using existing pose data, skipping pose estimation...")
                pose_data = existing_pose_data
                logger.info(f"Using existing pose data with {len(pose_data)} frames")
            elif peak_frames and not existing_pose_data:
                logger.info("Step 1/5: Performing pose estimation only on peak frames...")
                # Extract frame numbers from peak frames
                peak_frame_numbers = [info['frame_number'] for info in peak_frames.values() if 'frame_number' in info]
                if not peak_frame_numbers:
                    logger.warning("No valid frame numbers found in peak_frames, falling back to full video processing")
                    pose_data = self.perform_full_video_pose_estimation(
                        video_path, video_info, sample_rate, num_workers=num_workers)
                else:
                    logger.info(f"Processing {len(peak_frame_numbers)} peak frames")
                    pose_data = {}
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Could not open video: {video_path}")
                        return None, None, None, None
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    for frame_number in peak_frame_numbers:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        if ret:
                            time = frame_number / fps if fps > 0 else 0
                            _, pose_vector, pose_detected = self.perform_pose_estimation(
                                frame, draw=False, normalize_centroid=True)
                            if pose_detected:
                                pose_data[str(frame_number)] = {
                                    'frame_number': frame_number,
                                    'time': time,
                                    'pose_vector': pose_vector,
                                }
                    cap.release()
                    logger.info(f"Completed pose estimation on {len(pose_data)} peak frames")
            else:
                logger.info("Step 1/5: Starting pose estimation on the entire video...")
                pose_data = self.perform_full_video_pose_estimation(
                    video_path, video_info, sample_rate, num_workers=num_workers)
                logger.info(f"Completed pose estimation on {len(pose_data)} frames")
            
            # Calculate cluster centroids if clusters exist
            centroids = None
            similarities = None
            
            # Step 2: If we don't have cluster assignments but we have peak frames, perform clustering
            if peak_frames and not cluster_assignments:
                logger.info(f"Step 2/5: No existing clusters found. Performing clustering on {len(peak_frames)} peak frames...")
                # Use default max_clusters=40 and vector_type='pose'
                cluster_assignments, cluster_info = self.cluster_eventfulness_vectors(
                    peak_frames, max_clusters=40, vector_type='pose')
                    
                if cluster_assignments:
                    num_clusters = len(set(cluster_assignments.values()))
                    logger.info(f"Created {num_clusters} clusters from {len(peak_frames)} peak frames")
                    
                    # Display notification about clustering
                    if cluster_info and 'silhouette_score' in cluster_info:
                        logger.info(f"Silhouette Score: {cluster_info['silhouette_score']:.4f}")
                        logger.info(f"Cluster distribution: {dict([(c, list(cluster_assignments.values()).count(c)) for c in set(cluster_assignments.values())])}")
                    
                    # Evaluate cluster quality and filter outliers
                    logger.info("Step 2b/5: Evaluating cluster quality and filtering outliers...")
                    quality_report = self.evaluate_cluster_quality(peak_frames, cluster_assignments, vector_type='pose')
                    
                    if quality_report:
                        # Filter out outliers
                        filtered_assignments, outlier_assignments = self.filter_outliers_from_clusters(
                            peak_frames, cluster_assignments, quality_report)
                        
                        # Update cluster_assignments to filtered version
                        cluster_assignments = filtered_assignments
                        
                        # Log quality metrics
                        logger.info(f"Cluster Quality Metrics:")
                        logger.info(f"  Overall Silhouette: {quality_report['overall_silhouette']:.3f}")
                        logger.info(f"  Outliers Removed: {quality_report['num_outliers']} ({quality_report['outlier_percentage']:.1f}%)")
                        
                        # Log per-cluster metrics
                        for cluster_id, metrics in quality_report['cluster_metrics'].items():
                            logger.info(f"  Cluster {cluster_id}: size={metrics['size']}, "
                                      f"silhouette={metrics['avg_silhouette']:.3f}, "
                                      f"cohesion={metrics['cohesion']:.3f}")
                else:
                    logger.warning("Clustering failed - no cluster assignments were created")
            
            # Step 3: Calculate cluster centroids
            if peak_frames and cluster_assignments:
                logger.info("Step 3/5: Calculating cluster centroids...")
                centroids = self.calculate_cluster_centroids(peak_frames, cluster_assignments)
                if centroids:
                    logger.info(f"Calculated centroids for {len(centroids)} clusters")
                    # Log the number of frames in each cluster
                    cluster_counts = {}
                    for peak_idx, cluster_id in cluster_assignments.items():
                        if cluster_id not in cluster_counts:
                            cluster_counts[cluster_id] = 0
                        cluster_counts[cluster_id] += 1
                    logger.info(f"Frames per cluster: {cluster_counts}")
                else:
                    logger.warning("Failed to calculate cluster centroids")
                
                # Step 4: Compute cosine similarities with batched processing
                if pose_data and centroids:
                    logger.info(f"Step 4/5: Computing cosine similarities for {len(pose_data)} frames against {len(centroids)} clusters...")
                    similarities = self.compute_cosine_similarities(pose_data, centroids, batch_size=1000)
                    logger.info(f"Computed cosine similarities for {len(similarities)} frames")
                    
                    # Step 5: Smooth cosine similarities with running average
                    logger.info("Step 5/5: Smoothing cosine similarities with running average...")
                    similarities = self.smooth_cosine_similarities(similarities, window_size=5)
                    logger.info(f"Smoothed cosine similarities for {len(similarities)} frames")
                    
                    # Optional Step 6: Perform DTW-based segmentation
                    dtw_segmentation = None
                    if perform_dtw_segmentation and similarities:
                        logger.info("Step 6a/7: Performing DTW-based segmentation on similarity time series...")
                        dtw_segmentation = self.segment_cosine_similarity_with_dtw(
                            similarities,
                            window_size=dtw_window_size,
                            threshold=dtw_threshold,
                            min_segment_length=dtw_min_segment_length
                        )
                        if dtw_segmentation:
                            logger.info(f"DTW segmentation completed: {dtw_segmentation['num_clusters']} clusters, "
                                      f"{dtw_segmentation['total_frames']} frames")
                        else:
                            logger.warning("DTW segmentation failed")
                    else:
                        dtw_segmentation = None
                    
                    # Optional Step 7: Perform peak-based segmentation with merging
                    peak_segmentation = None
                    if perform_peak_segmentation and similarities and peak_frames:
                        logger.info("Step 6b/7: Performing peak-based segmentation with iterative merging...")
                        peak_segmentation = self.segment_by_peaks_with_merging(
                            similarities,
                            peak_frames,
                            similarity_threshold=peak_similarity_threshold,
                            max_passes=peak_max_passes
                        )
                        if peak_segmentation:
                            logger.info(f"Peak-based segmentation completed: {peak_segmentation['initial_segment_count']} "
                                      f"initial segments -> {peak_segmentation['final_segment_count']} final segments "
                                      f"after {len(peak_segmentation['merge_history'])} merges")
                        else:
                            logger.warning("Peak-based segmentation failed")
                    else:
                        peak_segmentation = None
                else:
                    logger.warning(f"Skipping similarity calculation - pose_data: {bool(pose_data)}, centroids: {bool(centroids)}")
                    dtw_segmentation = None
                    peak_segmentation = None
            else:
                dtw_segmentation = None
                peak_segmentation = None
            
            return pose_data, centroids, similarities, cluster_assignments, dtw_segmentation, peak_segmentation
            
        except Exception as e:
            logger.error(f"Error in full video analysis: {str(e)}")
            return None, None, None, cluster_assignments, None, None
    
    def run_complete_analysis(self, video_path, num_workers=4):
        """
        Runs the complete analysis workflow with parallel processing:
        1. Submit eventfulness prediction job (runs in background via SLURM)
        2. Pose estimation on entire video (runs in parallel with eventfulness)
        3. Wait for eventfulness to complete, then load data and extract peak frames
        4. Run clustering and similarity calculation
        
        Args:
            video_path: Path to the video file
            num_workers: Number of parallel workers for pose estimation
            
        Returns:
            Tuple containing:
            - pose_data: Dictionary of pose data for each frame
            - eventfulness_data: Dictionary with eventfulness data and config path
            - peak_frames: Dictionary of peak frames
            - centroids: Dictionary of cluster centroids
            - similarities: Dictionary of cosine similarities
            - cluster_assignments: Dictionary of cluster assignments
        """
        import subprocess
        import time
        import glob
        import json
        
        logger.info(f"Starting complete analysis workflow for: {video_path}")
        
        try:
            # Get video info first
            video_info = self.get_video_info(video_path)
            if not video_info:
                logger.error(f"Failed to get video info for {video_path}")
                return None, None, None, None, None, None
            
            logger.info(f"Video info: {video_info}")
            
            # Step 1: Submit eventfulness prediction job (runs in background)
            logger.info("Step 1/5: Submitting eventfulness prediction job...")
            
            # Launch the slurm job
            slurm_script = "/home/is1893/Mirror2/scripts/adroit_predict.slurm"
            
            # Check if running in a cluster environment
            in_cluster = os.path.exists("/scratch/network")
            eventfulness_job_submitted = False
            eventfulness_start_time = None
            
            if in_cluster:
                # If in cluster, submit as a job (non-blocking)
                try:
                    subprocess.run(["sbatch", slurm_script], check=True)
                    logger.info("Submitted eventfulness prediction job to SLURM (running in background)")
                    eventfulness_job_submitted = True
                    eventfulness_start_time = time.time()
                except Exception as e:
                    logger.error(f"Error submitting SLURM job: {str(e)}")
            else:
                # If not in cluster, we'll run it after pose estimation
                logger.info("Not running in cluster environment, will execute prediction script after pose estimation")
            
            # Step 2: Start pose estimation (while eventfulness prediction runs in parallel)
            logger.info("Step 2/5: Starting pose estimation on entire video (while eventfulness prediction runs)...")
            pose_data = self.perform_full_video_pose_estimation(
                video_path, video_info, sample_rate=1, num_workers=num_workers)
            
            if not pose_data:
                logger.error("No pose data was generated")
                return None, None, None, None, None, None
            
            logger.info(f"Processed {len(pose_data)} frames with pose estimation")
            
            # Step 2b: Wait for eventfulness prediction to complete (if it was submitted)
            if eventfulness_job_submitted:
                logger.info("Checking if eventfulness prediction has completed...")
                results_found = False
                max_wait_time = 600  # 10 minutes max wait time
                elapsed_time = time.time() - eventfulness_start_time
                remaining_time = max_wait_time - elapsed_time
                
                if remaining_time > 0:
                    check_start = time.time()
                    while time.time() - check_start < remaining_time:
                        # Check for eventfulness results in config.json
                        # The SLURM job updates the config.json file with eventfulness data
                        config_path, config = self.find_matching_config(video_path)
                        if config and "eventfulness" in config and len(config["eventfulness"]) > 0:
                            results_found = True
                            total_elapsed = time.time() - eventfulness_start_time
                            logger.info(f"Found eventfulness results in config: {config_path} (took {total_elapsed:.1f}s)")
                            break
                        
                        logger.info("Still waiting for eventfulness results...")
                        time.sleep(1)  # Check every 30 seconds
                else:
                    logger.info("Pose estimation took longer than expected, checking for results now...")
                    config_path, config = self.find_matching_config(video_path)
                    if config and "eventfulness" in config and len(config["eventfulness"]) > 0:
                        results_found = True
                        logger.info(f"Found eventfulness results in config: {config_path}")
                
                if not results_found:
                    logger.warning("Eventfulness prediction did not complete within the timeout period")
                    logger.info("Proceeding without eventfulness data")
            elif not in_cluster:
                # Run prediction directly if not in cluster
                logger.info("Running eventfulness prediction directly (not in cluster)...")
                try:
                    subprocess.run([
                        "python", "/home/is1893/Mirror2/scripts/predict.py",
                        "--data_dir", "/home/is1893/Mirror2/dataSets/test_data",
                        "--ngpu", "1", "--nepoch", "1", "--nworker", "4",
                        "--label_type", "none",
                        "--num_accS_dir", "4", "--num_velS_dir", "4", "--num_blurrs", "4",
                        "--prediction_window_step", "24",
                        "--load_model", "--load_model_dir", "/home/is1893/Mirror2/checkpoints", "--load_epoch", "61"
                    ], check=True)
                    logger.info("Eventfulness prediction completed")
                except Exception as e:
                    logger.error(f"Error running eventfulness prediction: {str(e)}")
            
            # Step 3: Load eventfulness data and extract peak frames
            logger.info("Step 3/5: Loading eventfulness data and extracting peak frames...")
            
            # Find matching config.json file for eventfulness data (matching dash app approach)
            peak_frames = None
            eventfulness_data_dict = None
            
            logger.info(f"Searching for eventfulness data in config.json for video: {video_path}")
            config_path, config = self.find_matching_config(video_path)
            
            if config and "eventfulness" in config and len(config["eventfulness"]) > 0:
                logger.info(f"Found eventfulness data in config: {config_path}")
                
                # Prepare eventfulness data dict (matching dash app structure)
                eventfulness_data_dict = {
                    "data": config["eventfulness"][0],  # First dimension for visualization
                    "full_vectors": config["eventfulness"],  # Full eventfulness vectors (all dimensions)
                    "fps": config.get("fps", video_info['fps']),
                    "config_path": config_path
                }
                
                # Get the eventfulness data for peak detection
                data = eventfulness_data_dict['data']
                logger.info(f"Eventfulness data loaded: {len(data)} data points")
                
                # Detect peaks using the centralized method
                peaks, peak_values, detection_params = self.detect_peaks(data)
                
                # Store peak detection parameters in eventfulness data for frontend consistency
                eventfulness_data_dict['peak_indices'] = peaks
                eventfulness_data_dict['peak_values'] = peak_values
                eventfulness_data_dict['peak_detection_params'] = detection_params
                
                logger.info(f"Detected {detection_params['total_peaks_detected']} peaks "
                          f"(prominence >= {detection_params['prominence']}, distance >= {detection_params['distance']})")
                if len(peaks) > 0:
                    logger.info(f"Peak indices: {peaks[:10]}{'...' if len(peaks) > 10 else ''}")
                    logger.info(f"Peak values range: [{min(peak_values):.3f}, {max(peak_values):.3f}]")
                
                # Create peak frames dictionary and extract frame images
                peak_frames = {}
                eventfulness_length = len(data)
                full_vectors = eventfulness_data_dict['full_vectors']
                
                # Create directory for extracted frames
                video_filename = os.path.basename(video_path).replace('.mp4', '')
                timestamp = int(os.path.getmtime(video_path)) if os.path.exists(video_path) else int(datetime.datetime.now().timestamp())
                frame_dir = os.path.join(RESULTS_DIR, f"peak_frames_{video_filename}_{timestamp}")
                os.makedirs(frame_dir, exist_ok=True)
                logger.info(f"Saving peak frames to: {frame_dir}")
                
                # Open video to extract frames
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video to extract peak frames: {video_path}")
                
                for i, peak_idx in enumerate(peaks):
                    # Map peak index to frame number
                    ratio = video_info['frame_count'] / eventfulness_length if eventfulness_length > 0 else 1
                    frame_number = int(peak_idx * ratio)
                    frame_number = min(frame_number, video_info['frame_count'] - 1)
                    
                    # Find the closest frame in pose_data
                    closest_frame = min(pose_data.keys(), key=lambda x: abs(int(x) - frame_number))
                    frame_data = pose_data[closest_frame]
                    
                    # Get eventfulness value and full vector for this peak
                    eventfulness_value = data[peak_idx]
                    
                    # Extract full eventfulness vector for this peak
                    # full_vectors is a list of dimensions, each dimension is a list of values
                    eventfulness_vector = None
                    if full_vectors and peak_idx < len(full_vectors[0]):
                        eventfulness_vector = [full_vectors[dim][peak_idx] for dim in range(len(full_vectors))]
                        if i < 5:  # Log first 5 peaks in detail
                            logger.info(f"Peak {i}: index={peak_idx}, frame={frame_number}, time={frame_data['time']:.2f}s, value={eventfulness_value:.3f}, vector_length={len(eventfulness_vector)}")
                    else:
                        logger.warning(f"Peak {i}: Could not extract full eventfulness vector for peak_idx={peak_idx}")
                    
                    # Extract and save the actual frame image
                    frame_path = None
                    annotated_path = None
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        if ret:
                            # Save original frame
                            frame_path = os.path.join(frame_dir, f"frame_{frame_number:06d}.jpg")
                            cv2.imwrite(frame_path, frame)
                            
                            # Create annotated frame with pose landmarks
                            annotated_frame, _, _ = self.perform_pose_estimation(frame, draw=True, normalize_centroid=True)
                            annotated_path = os.path.join(frame_dir, f"frame_{frame_number:06d}_annotated.jpg")
                            cv2.imwrite(annotated_path, annotated_frame)
                    
                    peak_frames[peak_idx] = {
                        'frame_number': frame_data['frame_number'],
                        'time': frame_data['time'],
                        'pose_vector': frame_data['pose_vector'],  # Normalized for cosine similarity
                        'pose_vector_raw': frame_data.get('pose_vector_raw', frame_data['pose_vector']),  # Raw for clustering
                        'pose_detected': True,
                        'peak_value': eventfulness_value,
                        'eventfulness_value': eventfulness_value,
                        'eventfulness_vector': eventfulness_vector,
                        'path': frame_path,
                        'annotated_path': annotated_path
                    }
                
                # Release video capture
                if cap.isOpened():
                    cap.release()
                
                logger.info(f"Created {len(peak_frames)} peak frames from detected peaks and saved images to {frame_dir}")
            else:
                logger.warning(f"No eventfulness data found in config.json for video: {video_path}")
            
            # Step 4: Run clustering on the peak frames if available
            centroids = None
            similarities = None
            cluster_assignments = None
            dtw_segmentation = None
            
            if peak_frames:
                logger.info("Step 4/5: Performing clustering on peak frames...")
                # Pass the existing pose_data to avoid reprocessing the entire video
                # Enable both DTW and peak-based segmentation
                _, centroids, similarities, cluster_assignments, dtw_segmentation, peak_segmentation = self.handle_full_video_analysis(
                    video_path, peak_frames=peak_frames, num_workers=num_workers, existing_pose_data=pose_data,
                    perform_dtw_segmentation=True,  # Enable DTW segmentation
                    dtw_window_size=10,
                    dtw_threshold=1.5,
                    dtw_min_segment_length=5,
                    perform_peak_segmentation=True,  # Enable peak-based segmentation
                    peak_similarity_threshold=0.95,
                    peak_max_passes=10)
                
                if centroids:
                    logger.info(f"Created {len(centroids)} clusters")
                else:
                    logger.warning("No centroids were generated")
                    
                if similarities:
                    logger.info(f"Computed similarities for {len(similarities)} frames")
                else:
                    logger.warning("No similarities were computed")
                    
                if cluster_assignments:
                    logger.info(f"Assigned {len(cluster_assignments)} peaks to clusters")
                else:
                    logger.warning("No cluster assignments were made")
                    
                if dtw_segmentation:
                    num_segments = len(dtw_segmentation.get('segments', []))
                    logger.info(f"DTW segmentation completed: {dtw_segmentation['num_clusters']} clusters, "
                              f"{num_segments} global segments")
                else:
                    logger.warning("No DTW segmentation results")
                
                if peak_segmentation:
                    logger.info(f"Peak-based segmentation completed: {peak_segmentation['initial_segment_count']} "
                              f"initial segments -> {peak_segmentation['final_segment_count']} final segments")
                else:
                    logger.warning("No peak-based segmentation results")
            else:
                logger.info("Skipping clustering step - no peak frames available")
            
            logger.info("Complete analysis workflow finished")
            return pose_data, eventfulness_data_dict, peak_frames, centroids, similarities, cluster_assignments, dtw_segmentation, peak_segmentation
            
        except Exception as e:
            logger.error(f"Error in complete analysis workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None, None, None, None

# Example usage (commented out - use run_complete_analysis method instead)
# if __name__ == "__main__":
#     try:
#         import subprocess
#         import time
#         import glob
#         import json
#         
#         backend = VideoAnalysisBackend()
#         video_path = "/home/is1893/Mirror2/dataSets/test_data/val/JumpJack/JumpJack.mp4"
#         
#         print(f"Starting analysis of video: {video_path}")
#         print(f"File exists: {os.path.exists(video_path)}")
#         
#         # Get video info first
#         video_info = backend.get_video_info(video_path)
#         print(f"Video info: {video_info}")
#         
#         # Step 1: Start pose estimation
#         print("Step 1: Starting pose estimation...")
#         pose_data, _, _, _ = backend.handle_full_video_analysis(
#             video_path, num_workers=4)
#             
#         if pose_data:
#             print(f"Processed {len(pose_data)} frames with pose estimation")
#             
#             # Step 2: Run eventfulness prediction using the slurm script
#             print("Step 2: Running eventfulness prediction with adroit_predict.slurm...")
#             
#             # Launch the slurm job
#             slurm_script = "/home/is1893/Mirror2/scripts/adroit_predict.slurm"
#             
#             # Check if running in a cluster environment
#             in_cluster = os.path.exists("/scratch/network")
#             
#             if in_cluster:
#                 # If in cluster, submit as a job
#                 subprocess.run(["sbatch", slurm_script], check=True)
#                 print("Submitted eventfulness prediction job to SLURM")
#                 
#                 # Wait for job to complete by checking for results
#                 print("Waiting for eventfulness prediction to complete...")
#                 results_found = False
#                 max_wait_time = 600  # 10 minutes max wait time
#                 start_time = time.time()
#                 
#                 while time.time() - start_time < max_wait_time:
#                     # Check for eventfulness results
#                     result_files = glob.glob("/scratch/network/is1893/mirror2_data/dataSets/test_data/results/*/eventfulness.json")
#                     if result_files:
#                         results_found = True
#                         print(f"Found eventfulness results: {result_files}")
#                         break
#                     
#                     print("Still waiting for eventfulness results...")
#                     time.sleep(30)  # Check every 30 seconds
#                 
#                 if not results_found:
#                     print("Warning: Eventfulness prediction did not complete within the timeout period")
#                     print("Proceeding without eventfulness data")
#             else:
#                 # If not in cluster, run the command directly
#                 print("Not running in cluster environment, executing script directly")
#                 subprocess.run([
#                     "python", "/home/is1893/Mirror2/scripts/predict.py",
#                     "--data_dir", "/home/is1893/Mirror2/dataSets/test_data",
#                     "--ngpu", "1", "--nepoch", "1", "--nworker", "4",
#                     "--label_type", "none",
#                     "--num_accS_dir", "4", "--num_velS_dir", "4", "--num_blurrs", "4",
#                     "--prediction_window_step", "24",
#                     "--load_model", "--load_model_dir", "/home/is1893/Mirror2/checkpoints", "--load_epoch", "61"
#                 ], check=True)
#             
#             # Step 3: Load eventfulness data and extract peak frames
#             print("Step 3: Loading eventfulness data and extracting peak frames...")
#             
#             # Find the eventfulness data
#             eventfulness_files = glob.glob("/home/is1893/Mirror2/dataSets/test_data/results/*/eventfulness.json")
#             if not eventfulness_files and in_cluster:
#                 eventfulness_files = glob.glob("/scratch/network/is1893/mirror2_data/dataSets/test_data/results/*/eventfulness.json")
#             
#             peak_frames = None
#             
#             if eventfulness_files:
#                 latest_file = max(eventfulness_files, key=os.path.getmtime)
#                 print(f"Using eventfulness data from: {latest_file}")
#                 
#                 # Load the eventfulness data
#                 with open(latest_file, 'r') as f:
#                     eventfulness_data = json.load(f)
#                 
#                 # Extract peaks from eventfulness data
#                 if 'peaks' in eventfulness_data:
#                     peaks = eventfulness_data['peaks']
#                     print(f"Found {len(peaks)} peaks in eventfulness data")
#                     
#                     # Create peak frames dictionary
#                     peak_frames = {}
#                     for i, peak_idx in enumerate(peaks):
#                         # Map peak index to frame number
#                         eventfulness_length = len(eventfulness_data.get('data', []))
#                         ratio = video_info['frame_count'] / eventfulness_length if eventfulness_length > 0 else 1
#                         frame_number = int(peak_idx * ratio)
#                         
#                         # Find the closest frame in pose_data
#                         closest_frame = min(pose_data.keys(), key=lambda x: abs(int(x) - frame_number))
#                         frame_data = pose_data[closest_frame]
#                         
#                         peak_frames[peak_idx] = {
#                             'frame_number': frame_data['frame_number'],
#                             'time': frame_data['time'],
#                             'pose_vector': frame_data['pose_vector'],
#                             'pose_detected': True,
#                             'eventfulness_value': eventfulness_data['data'][peak_idx] if 'data' in eventfulness_data and peak_idx < len(eventfulness_data['data']) else 0
#                         }
#                     
#                     print(f"Created {len(peak_frames)} peak frames from eventfulness data")
#                 else:
#                     print("No peaks found in eventfulness data")
#             else:
#                 print("No eventfulness data found")
#             
#             # Step 4: Run clustering on the peak frames if available
#             if peak_frames:
#                 print("Step 4: Performing clustering on peak frames...")
#                 # Pass the existing pose_data to avoid reprocessing the entire video
#                 _, centroids, similarities, cluster_assignments = backend.handle_full_video_analysis(
#                     video_path, peak_frames=peak_frames, num_workers=4, existing_pose_data=pose_data)
#                     
#                 if centroids:
#                     print(f"Created {len(centroids)} clusters")
#                 else:
#                     print("No centroids were generated")
#                     
#                 if similarities:
#                     print(f"Computed similarities for {len(similarities)} frames")
#                 else:
#                     print("No similarities were computed")
#                     
#                 if cluster_assignments:
#                     print(f"Assigned {len(cluster_assignments)} peaks to clusters")
#                     print(f"Cluster assignments: {cluster_assignments}")
#                 else:
#                     print("No cluster assignments were made")
#             else:
#                 print("Skipping clustering step - no peak frames available")
#         else:
#             print("No pose data was generated")
#     except Exception as e:
#         print(f"Error running backend: {str(e)}")
#         import traceback
#         traceback.print_exc()