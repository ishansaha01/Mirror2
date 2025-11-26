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

# Define constants
RESULTS_DIR = "/home/is1893/Mirror2/dataSets/test_data/results"

class VideoAnalysisBackend:
    """Backend class for video analysis with concurrent processing capabilities."""
    
    def __init__(self):
        """Initialize the VideoAnalysisBackend."""
        self.mp_pose = mp.solutions.pose
        # Define which landmarks to keep (reduced face, hand, and foot vectors)
        # 0: nose, 11-16: shoulders/elbows/wrists, 23-28: hips/knees/ankles
        self.LANDMARKS_TO_KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
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
    
    def find_matching_config(self, video_path):
        """Find the matching config.json file for a given video path."""
        config_files = glob.glob(os.path.join(
            RESULTS_DIR, "**/config.json"), recursive=True)
            
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
            except Exception as e:
                logger.error(f"Error reading config file {config_file}: {str(e)}")
                continue
                
        return None, None
    
    def perform_pose_estimation(self, image, draw=False, normalize_centroid=False):
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
                
                # Perform pose estimation without normalization
                _, pose_vector, pose_detected = self.perform_pose_estimation(
                    frame, draw=False, normalize_centroid=False)
                
                # Store the pose data
                if pose_detected:
                    pose_data[str(frame_number)] = {
                        'frame_number': frame_number,
                        'time': time,
                        'pose_vector': pose_vector,
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
    
    def calculate_cluster_centroids(self, peak_frames, cluster_assignments):
        """
        Calculates the centroid pose vector for each cluster.
        
        Args:
            peak_frames: Dictionary of peak frames with pose vectors
            cluster_assignments: Dictionary mapping peak indices to cluster IDs
            
        Returns:
            Dictionary mapping cluster IDs to centroid pose vectors
        """
        if not peak_frames or not cluster_assignments:
            return {}
            
        # Group pose vectors by cluster
        clusters = {}
        for peak_idx, frame_info in peak_frames.items():
            if str(peak_idx) in cluster_assignments and frame_info.get('pose_detected', False):
                cluster_id = cluster_assignments[str(peak_idx)]
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
        if not pose_data or not cluster_centroids:
            return {}
            
        # Dictionary to store similarity scores
        similarities = {}
        
        # Convert centroids to numpy array
        centroid_ids = sorted(cluster_centroids.keys())
        centroid_vectors = np.array([cluster_centroids[cid] for cid in centroid_ids])
        
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
                    batch_vectors.append(pose_vector)
                    batch_data.append((frame_idx, frame_data))
            
            if not batch_vectors:
                continue
                
            # Convert to numpy array
            batch_vectors_array = np.array(batch_vectors)
            
            # Compute cosine similarity for the entire batch at once
            sim_scores_batch = cosine_similarity(batch_vectors_array, centroid_vectors)
            
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
    
    def handle_full_video_analysis(self, video_path, peak_frames=None, cluster_assignments=None, num_workers=4, existing_pose_data=None):
        """
        Performs full video analysis with pose estimation, clustering, and similarity calculation.
        Uses parallel processing for improved performance.
        
        Args:
            video_path: Path to the video file
            peak_frames: Dictionary of peak frames (optional)
            cluster_assignments: Dictionary of cluster assignments (optional)
            num_workers: Number of parallel workers
            existing_pose_data: Existing pose data to avoid reprocessing the entire video (optional)
            
        Returns:
            Tuple containing:
            - pose_data: Dictionary of pose data for each frame
            - centroids: Dictionary of cluster centroids
            - similarities: Dictionary of cosine similarities
            - cluster_assignments: Dictionary of cluster assignments
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
                logger.info("Step 1/4: Using existing pose data, skipping pose estimation...")
                pose_data = existing_pose_data
                logger.info(f"Using existing pose data with {len(pose_data)} frames")
            elif peak_frames and not existing_pose_data:
                logger.info("Step 1/4: Performing pose estimation only on peak frames...")
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
                                frame, draw=False, normalize_centroid=False)
                            if pose_detected:
                                pose_data[str(frame_number)] = {
                                    'frame_number': frame_number,
                                    'time': time,
                                    'pose_vector': pose_vector,
                                }
                    cap.release()
                    logger.info(f"Completed pose estimation on {len(pose_data)} peak frames")
            else:
                logger.info("Step 1/4: Starting pose estimation on the entire video...")
                pose_data = self.perform_full_video_pose_estimation(
                    video_path, video_info, sample_rate, num_workers=num_workers)
                logger.info(f"Completed pose estimation on {len(pose_data)} frames")
            
            # Calculate cluster centroids if clusters exist
            centroids = None
            similarities = None
            
            # Step 2: If we don't have cluster assignments but we have peak frames, perform clustering
            if peak_frames and not cluster_assignments:
                logger.info(f"Step 2/4: No existing clusters found. Performing clustering on {len(peak_frames)} peak frames...")
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
                else:
                    logger.warning("Clustering failed - no cluster assignments were created")
            
            # Step 3: Calculate cluster centroids
            if peak_frames and cluster_assignments:
                logger.info("Step 3/4: Calculating cluster centroids...")
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
                    logger.info(f"Step 4/4: Computing cosine similarities for {len(pose_data)} frames against {len(centroids)} clusters...")
                    similarities = self.compute_cosine_similarities(pose_data, centroids, batch_size=1000)
                    logger.info(f"Computed cosine similarities for {len(similarities)} frames")
                else:
                    logger.warning(f"Skipping similarity calculation - pose_data: {bool(pose_data)}, centroids: {bool(centroids)}")
            
            return pose_data, centroids, similarities, cluster_assignments
            
        except Exception as e:
            logger.error(f"Error in full video analysis: {str(e)}")
            return None, None, None, cluster_assignments
    
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
                        time.sleep(30)  # Check every 30 seconds
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
                
                # Detect local maxima (peaks) using scipy's find_peaks
                # This matches the dash app's detect_local_maxima() function
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(data, height=0.3, distance=5)
                peaks = [int(peak) for peak in peaks]
                peak_values = [data[p] for p in peaks]
                
                logger.info(f"Detected {len(peaks)} peaks using find_peaks (height=0.3, distance=5)")
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
                            annotated_frame, _, _ = self.perform_pose_estimation(frame, draw=True, normalize_centroid=False)
                            annotated_path = os.path.join(frame_dir, f"frame_{frame_number:06d}_annotated.jpg")
                            cv2.imwrite(annotated_path, annotated_frame)
                    
                    peak_frames[peak_idx] = {
                        'frame_number': frame_data['frame_number'],
                        'time': frame_data['time'],
                        'pose_vector': frame_data['pose_vector'],
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
            
            if peak_frames:
                logger.info("Step 4/5: Performing clustering on peak frames...")
                # Pass the existing pose_data to avoid reprocessing the entire video
                _, centroids, similarities, cluster_assignments = self.handle_full_video_analysis(
                    video_path, peak_frames=peak_frames, num_workers=num_workers, existing_pose_data=pose_data)
                
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
            else:
                logger.info("Skipping clustering step - no peak frames available")
            
            logger.info("Complete analysis workflow finished")
            return pose_data, eventfulness_data_dict, peak_frames, centroids, similarities, cluster_assignments
            
        except Exception as e:
            logger.error(f"Error in complete analysis workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None, None

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