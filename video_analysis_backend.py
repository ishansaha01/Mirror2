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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up logging
log_file = "/home/is1893/Mirror2/video_analysis.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('video_analysis')

# Segmentation library imports - using STUMPY for FLUSS segmentation
try:
    import stumpy
    SEGMENTATION_LIBRARY = 'stumpy'
    logger.info("Using STUMPY library for FLUSS-based segmentation")
except ImportError:
    SEGMENTATION_LIBRARY = None
    logger.warning("STUMPY not available. Time series segmentation will not be available.")

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
    
    def plot_cosine_similarities_by_cluster(self, similarities, cluster_assignments=None,
                                           peak_frames=None, output_dir=None, video_name='video', 
                                           show_raw=False):
        """
        Plots cosine similarity time series for each cluster.
        
        Args:
            similarities: Dictionary mapping frame numbers to similarity scores
            cluster_assignments: Optional dictionary {peak_idx: cluster_id} for peak frames
            peak_frames: Optional dictionary {peak_idx: {frame_number, time, ...}} for peak frame info
            output_dir: Directory to save the plot
            video_name: Name of the video for the plot title
            show_raw: Whether to show raw similarities alongside smoothed (if available)
            
        Returns:
            Path to saved plot file
        """
        if not similarities:
            logger.warning("No similarities to plot")
            return None
        
        # Sort frames by frame number
        sorted_frames = sorted(similarities.keys(), key=lambda x: similarities[x]['frame_number'])
        
        # Get cluster IDs
        first_frame = sorted_frames[0]
        cluster_ids = sorted(similarities[first_frame]['similarities'].keys())
        num_clusters = len(cluster_ids)
        
        # Extract frame numbers and times
        frame_numbers = [similarities[f]['frame_number'] for f in sorted_frames]
        times = [similarities[f]['time'] for f in sorted_frames]
        
        # Create figure with subplots for each cluster
        fig, axes = plt.subplots(num_clusters, 1, figsize=(14, 3 * num_clusters), sharex=True)
        if num_clusters == 1:
            axes = [axes]
        
        # Color palette for clusters
        colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
        
        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes[idx]
            
            # Extract similarity values for this cluster
            sim_values = [similarities[f]['similarities'][cluster_id] for f in sorted_frames]
            
            # Plot smoothed similarities
            ax.plot(times, sim_values, color=colors[idx], linewidth=2, 
                   label=f'Cluster {cluster_id} (smoothed)' if 'similarities_raw' in similarities[sorted_frames[0]] else f'Cluster {cluster_id}')
            
            # Plot raw similarities if available and requested
            if show_raw and 'similarities_raw' in similarities[sorted_frames[0]]:
                raw_values = [similarities[f]['similarities_raw'][cluster_id] for f in sorted_frames]
                ax.plot(times, raw_values, color=colors[idx], alpha=0.3, linewidth=1, 
                       linestyle='--', label=f'Cluster {cluster_id} (raw)')
            
            # Highlight peak frames if cluster assignments provided
            if cluster_assignments and peak_frames:
                peak_times = []
                # cluster_assignments is {peak_idx: cluster_id} where both are ints/strings
                for peak_idx, assigned_cluster_id in cluster_assignments.items():
                    # Convert to string for comparison
                    if str(assigned_cluster_id) == str(cluster_id):
                        # Get frame info from peak_frames
                        if peak_idx in peak_frames:
                            peak_frame_info = peak_frames[peak_idx]
                            if 'time' in peak_frame_info:
                                peak_times.append(peak_frame_info['time'])
                
                if peak_times:
                    # Add vertical lines at peak frames
                    for pt in peak_times:
                        ax.axvline(x=pt, color=colors[idx], alpha=0.3, linestyle=':', linewidth=1)
                    
                    # Add scatter points at peaks
                    peak_sims = []
                    for pt in peak_times:
                        # Find similarity value at this time
                        for i, t in enumerate(times):
                            if abs(t - pt) < 0.01:  # Close enough
                                peak_sims.append(sim_values[i])
                                break
                    
                    if peak_sims:
                        ax.scatter(peak_times, peak_sims, color=colors[idx], s=100, 
                                 zorder=5, edgecolors='black', linewidths=1.5,
                                 label=f'{len(peak_times)} peaks')
            
            # Formatting
            ax.set_ylabel('Cosine Similarity', fontsize=10, fontweight='bold')
            ax.set_title(f'Cluster {cluster_id} Similarity Over Time', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='upper right', fontsize=9)
            
            # Add horizontal line at 0.5 for reference
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Set x-label on bottom plot
        axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Cosine Similarity Time Series by Cluster - {video_name}', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{video_name}_cosine_similarities_by_cluster.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved cosine similarity plot to: {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def detect_fluss_change_points(self, time_series, window_size=10, num_regimes=None, exclusion_zone=None):
        """
        Detects change points in a univariate time series using STUMPY's FLUSS algorithm.
        
        FLUSS (Fast Low-cost Unipotent Semantic Segmentation) uses matrix profiles to find
        regime changes in time series data.
        
        Args:
            time_series: 1D numpy array or list of time series values
            window_size: Subsequence window size for matrix profile computation
            num_regimes: Number of regimes to segment into (if None, auto-detect)
            exclusion_zone: Exclusion zone for matrix profile (if None, uses m//4)
            
        Returns:
            List of change point indices
        """
        if SEGMENTATION_LIBRARY is None:
            logger.warning("STUMPY library not available. Returning simple segmentation.")
            return [0, len(time_series)]
        
        time_series = np.array(time_series, dtype=np.float64)
        n = len(time_series)
        
        # Need sufficient data points for FLUSS
        if n < 2 * window_size:
            logger.warning(f"Time series too short ({n} points) for window size {window_size}")
            return [0, n]
        
        try:
            # Compute matrix profile
            logger.info(f"Computing matrix profile for time series of length {n} with window size {window_size}")
            mp = stumpy.stump(time_series, m=window_size)
            
            # Compute arc curve (corrected arc curve) for FLUSS
            logger.info("Computing FLUSS arc curve for regime change detection")
            cac, regime_locations = stumpy.fluss(mp[:, 1], L=window_size, n_regimes=num_regimes, excl_factor=1)
            
            # regime_locations contains the indices where regimes change
            change_points = [0]
            
            if regime_locations is not None and len(regime_locations) > 0:
                # Add detected regime change points
                for loc in sorted(regime_locations):
                    if 0 < loc < n:
                        change_points.append(int(loc))
            
            # Always end with the last index
            if change_points[-1] != n:
                change_points.append(n)
            
            logger.info(f"FLUSS detected {len(change_points) - 1} segments with {len(change_points) - 2} change points")
            
            return change_points
            
        except Exception as e:
            logger.error(f"Error in FLUSS segmentation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [0, n]
    
    def visualize_motifs(self, mp_distances, mp_indices, T, window_size, vector_time_series, num_motifs=3, output_dir=None):
        """
        Visualizes the top motifs identified by mSTUMP.
        
        Args:
            mp_distances: Matrix profile distances from mSTUMP (n_dims × n_timepoints)
                Each row is the k-dimensional matrix profile (k=1, 2, ..., n_dims)
            mp_indices: Matrix profile indices from mSTUMP (n_dims × n_timepoints)
                Each row contains the indices of nearest neighbors for each k-dimensional profile
            T: Transposed time series (n_dimensions × n_timepoints)
            window_size: Window size used for mSTUMP
            vector_time_series: Original time series (n_timepoints × n_dimensions)
            num_motifs: Number of top motifs to visualize
            output_dir: Directory to save visualizations (defaults to RESULTS_DIR)
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Visualizing top {num_motifs} motifs from mSTUMP matrix profile")
        
        # Use the full multidimensional matrix profile (last row = all dimensions considered)
        full_mp_distances = mp_distances[-1]
        full_mp_indices = mp_indices[-1].astype(int)
        
        # Find top motifs (lowest distances = most similar pairs)
        # Exclude trivial matches (self-matches and overlapping subsequences)
        valid_motifs = []
        
        for i in range(len(full_mp_distances)):
            if full_mp_distances[i] < np.inf:
                match_idx = full_mp_indices[i]
                # Ensure non-trivial match (not overlapping)
                if abs(i - match_idx) >= window_size:
                    valid_motifs.append((i, match_idx, full_mp_distances[i]))
        
        # Sort by distance (lowest = best motifs)
        valid_motifs.sort(key=lambda x: x[2])
        
        # Take top motifs
        top_motifs = valid_motifs[:num_motifs]
        
        if len(top_motifs) == 0:
            logger.warning("No valid motifs found")
            return
        
        logger.info(f"Found {len(top_motifs)} motifs to visualize")
        
        # Create visualization
        n_dims = T.shape[0]
        fig, axes = plt.subplots(num_motifs, n_dims + 1, figsize=(20, 4 * num_motifs))
        
        if num_motifs == 1:
            axes = axes.reshape(1, -1)
        
        for motif_idx, (idx1, idx2, distance) in enumerate(top_motifs):
            # Extract the two matching subsequences
            subseq1 = vector_time_series[idx1:idx1 + window_size]
            subseq2 = vector_time_series[idx2:idx2 + window_size]
            
            # Plot each dimension
            for dim in range(n_dims):
                ax = axes[motif_idx, dim]
                ax.plot(subseq1[:, dim], label=f'Motif at t={idx1}', linewidth=2, alpha=0.7)
                ax.plot(subseq2[:, dim], label=f'Match at t={idx2}', linewidth=2, alpha=0.7)
                ax.set_title(f'Motif {motif_idx + 1} - Dimension {dim + 1}\nDistance: {distance:.4f}')
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Plot matrix profile with motif locations highlighted
            ax = axes[motif_idx, n_dims]
            ax.plot(full_mp_distances, linewidth=1, color='gray', alpha=0.5)
            ax.scatter([idx1, idx2], [full_mp_distances[idx1], full_mp_distances[idx2]], 
                      color='red', s=100, zorder=5, label='Motif pair')
            ax.set_title(f'Matrix Profile\nMotif pair at t={idx1} and t={idx2}')
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Matrix Profile Distance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'mstump_motifs.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved motif visualization to {output_path}")
        
        # Also create a summary plot showing all motif locations on the time series
        fig2, axes2 = plt.subplots(n_dims, 1, figsize=(16, 3 * n_dims))
        
        if n_dims == 1:
            axes2 = [axes2]
        
        for dim in range(n_dims):
            ax = axes2[dim]
            # Plot full time series for this dimension
            ax.plot(vector_time_series[:, dim], linewidth=1, color='black', alpha=0.5, label='Time series')
            
            # Highlight motif locations
            colors = plt.cm.Set1(np.linspace(0, 1, num_motifs))
            for motif_idx, (idx1, idx2, distance) in enumerate(top_motifs):
                # Highlight motif pair regions
                ax.axvspan(idx1, idx1 + window_size, alpha=0.3, color=colors[motif_idx], 
                          label=f'Motif {motif_idx + 1} (d={distance:.2f})')
                ax.axvspan(idx2, idx2 + window_size, alpha=0.3, color=colors[motif_idx])
            
            ax.set_title(f'Dimension {dim + 1} - Motif Locations')
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary figure
        summary_path = os.path.join(output_dir, 'mstump_motifs_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved motif summary to {summary_path}")
        
        plt.close('all')
    
    def detect_fluss_change_points_multivariate(self, vector_time_series, window_size=10, num_regimes=None, min_segment_length=5, return_arc_curve=False):
        """
        Detects change points in a MULTIVARIATE time series using STUMPY's mSTUMP and FLUSS.
        
        This treats each time point as a vector (e.g., similarity to all clusters) and finds
        regime changes across all dimensions simultaneously.
        
        Args:
            vector_time_series: 2D numpy array (n_timepoints × n_dimensions)
            window_size: Subsequence window size for matrix profile computation
            num_regimes: Number of regimes to segment into (if None, auto-detect)
            min_segment_length: Minimum number of frames between change points
            return_arc_curve: If True, returns (change_points, arc_curve) tuple
            
        Returns:
            List of change point indices, or tuple (change_points, arc_curve) if return_arc_curve=True
        """
        if SEGMENTATION_LIBRARY is None:
            logger.warning("STUMPY library not available. Returning simple segmentation.")
            return [0, len(vector_time_series)]
        
        vector_time_series = np.array(vector_time_series, dtype=np.float64)
        n = len(vector_time_series)
        
        # Need sufficient data points for FLUSS
        if n < 2 * window_size:
            logger.warning(f"Time series too short ({n} points) for window size {window_size}")
            return [0, n]
        
        try:
            # For multivariate time series, use mSTUMP (multivariate STUMP)
            logger.info(f"Computing multivariate matrix profile for {n} timepoints × {vector_time_series.shape[1]} dimensions")
            
            # Transpose for mSTUMP: expects (n_dimensions, n_timepoints)
            T = vector_time_series.T
            
            # Compute multivariate matrix profile
            # mstump returns (P, I) where P is matrix profile and I is indices
            # Each row corresponds to k-dimensional matrix profile (k=1, 2, ..., n_dims)
            mp_distances, mp_indices = stumpy.mstump(T, m=window_size)
            
            # Visualize motifs identified by mSTUMP
            try:
                self.visualize_motifs(mp_distances, mp_indices, T, window_size, vector_time_series)
            except Exception as e:
                logger.warning(f"Could not visualize motifs: {str(e)}")
                import traceback
                logger.warning(traceback.format_exc())
            
            # Compute FLUSS on the multivariate matrix profile
            # Use the full multidimensional matrix profile (last row = all dimensions)
            logger.info("Computing FLUSS arc curve for multivariate regime change detection")
            
            # If num_regimes is None, STUMPY will try to auto-detect but may fail
            # In that case, we still want the arc curve, so we'll catch the error
            try:
                cac, regime_locations = stumpy.fluss(mp_distances[-1], L=window_size, n_regimes=num_regimes, excl_factor=1)
            except (TypeError, ValueError) as e:
                logger.warning(f"FLUSS regime detection failed: {str(e)}. Computing arc curve only.")
                # Compute just the arc curve without regime locations
                cac = stumpy.fluss(mp_distances[-1], L=window_size, n_regimes=1, excl_factor=1)[0]
                regime_locations = None
            
            # regime_locations contains the indices where regimes change
            change_points = [0]
            
            if regime_locations is not None and len(regime_locations) > 0:
                # Add detected regime change points, filtering by min_segment_length
                sorted_locs = sorted(regime_locations)
                for loc in sorted_locs:
                    if 0 < loc < n and (loc - change_points[-1]) >= min_segment_length:
                        change_points.append(int(loc))
            
            # Always end with the last index
            if change_points[-1] != n:
                change_points.append(n)
            
            logger.info(f"Multivariate FLUSS detected {len(change_points) - 1} segments with {len(change_points) - 2} change points")
            
            if return_arc_curve:
                return change_points, cac
            else:
                return change_points
            
        except Exception as e:
            logger.error(f"Error in multivariate FLUSS segmentation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if return_arc_curve:
                return [0, n], None
            else:
                return [0, n]
    
    
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
    
    def segment_cosine_similarity_with_fluss(self, similarities, window_size=10, num_regimes=None, min_segment_length=5):
        """
        Segments cosine similarity time series using STUMPY's FLUSS algorithm.
        
        This method treats the cosine similarities as a MULTIVARIATE time series (vector at each time point)
        and finds global change points where the overall similarity pattern changes using matrix profiles.
        
        Args:
            similarities: Dictionary mapping frame indices to similarity scores
                         (output from compute_cosine_similarities or smooth_cosine_similarities)
            window_size: Subsequence window size for matrix profile (default: 10)
            num_regimes: Number of regimes to segment into (if None, auto-detect)
            min_segment_length: Minimum frames per segment (default: 5)
            
        Returns:
            Dictionary containing segmentation results:
            {
                'method': 'fluss_matrix_profile',
                'segments': [segment_dicts],  # Global segments
                'change_points': [indices],    # Global change points
                'parameters': {parameter_dict}
            }
        """
        if not similarities:
            logger.warning("No similarity data provided for FLUSS segmentation")
            return None
        
        if SEGMENTATION_LIBRARY is None:
            logger.error("STUMPY library not available. Cannot perform FLUSS segmentation.")
            return None
        
        logger.info(f"Starting FLUSS segmentation with window_size={window_size}, "
                   f"num_regimes={num_regimes}, min_segment_length={min_segment_length}")
        
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
        
        # Detect change points using FLUSS on the multivariate time series
        change_points, arc_curve = self.detect_fluss_change_points_multivariate(
            vector_time_series,
            window_size=window_size,
            num_regimes=num_regimes,
            min_segment_length=min_segment_length,
            return_arc_curve=True
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
            'method': 'fluss_matrix_profile',
            'segments': segments,  # Global segments (not per-cluster)
            'change_points': change_points,  # Global change points
            'arc_curve': arc_curve.tolist() if arc_curve is not None else None,  # FLUSS arc curve
            'cluster_ids': cluster_ids,
            'parameters': {
                'window_size': window_size,
                'num_regimes': num_regimes,
                'min_segment_length': min_segment_length,
                'segmentation_library': SEGMENTATION_LIBRARY,
                'vector_dimensions': len(cluster_ids)
            },
            'num_clusters': len(cluster_ids),
            'total_frames': len(frame_numbers)
        }
        
        logger.info(f"FLUSS segmentation completed: {len(segments)} segments")
        
        return results
    
    def segment_by_eventfulness_peaks(self, peak_frames, pose_data):
        """
        Segments video data from peak to peak based on eventfulness data.
        Extracts pose estimation data for each segment.
        
        Args:
            peak_frames: Dictionary mapping peak indices to frame info (from eventfulness detection)
            pose_data: Dictionary of frame-by-frame pose estimation data
            
        Returns:
            List of segment dictionaries, each containing:
            {
                'segment_id': int,
                'start_peak_idx': int,
                'end_peak_idx': int,
                'start_frame': int,
                'end_frame': int,
                'start_time': float,
                'end_time': float,
                'duration': float,
                'num_frames': int,
                'pose_vectors': list of pose vectors for all frames in segment,
                'frame_numbers': list of frame numbers in segment,
                'times': list of timestamps in segment,
                'mean_pose': average pose vector for segment,
                'std_pose': standard deviation of pose vectors
            }
        """
        if not peak_frames or not pose_data:
            logger.warning("No peak frames or pose data provided for segmentation")
            return []
        
        # Sort peaks by frame number
        sorted_peaks = sorted(peak_frames.items(), key=lambda x: x[1]['frame_number'])
        
        if len(sorted_peaks) < 2:
            logger.warning("Need at least 2 peaks for peak-to-peak segmentation")
            return []
        
        logger.info(f"Creating segments from {len(sorted_peaks)} peaks")
        
        segments = []
        
        # Create segments from peak to peak
        for i in range(len(sorted_peaks) - 1):
            peak_idx_start, peak_info_start = sorted_peaks[i]
            peak_idx_end, peak_info_end = sorted_peaks[i + 1]
            
            start_frame = peak_info_start['frame_number']
            end_frame = peak_info_end['frame_number']
            start_time = peak_info_start['time']
            end_time = peak_info_end['time']
            
            # Extract pose data for frames in this segment
            segment_pose_vectors = []
            segment_frame_numbers = []
            segment_times = []
            
            for frame_key, frame_data in pose_data.items():
                frame_num = frame_data['frame_number']
                if start_frame <= frame_num < end_frame:
                    pose_vector = frame_data.get('pose_vector', None)
                    if pose_vector is not None:
                        segment_pose_vectors.append(pose_vector)
                        segment_frame_numbers.append(frame_num)
                        segment_times.append(frame_data['time'])
            
            # Calculate statistics for the segment
            if segment_pose_vectors:
                pose_array = np.array(segment_pose_vectors)
                mean_pose = np.mean(pose_array, axis=0).tolist()
                std_pose = np.std(pose_array, axis=0).tolist()
            else:
                mean_pose = None
                std_pose = None
            
            segment = {
                'segment_id': i,
                'start_peak_idx': peak_idx_start,
                'end_peak_idx': peak_idx_end,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(end_time - start_time),
                'num_frames': len(segment_pose_vectors),
                'pose_vectors': segment_pose_vectors,
                'frame_numbers': segment_frame_numbers,
                'times': segment_times,
                'mean_pose': mean_pose,
                'std_pose': std_pose
            }
            
            segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments from peak-to-peak")
        return segments
    
    def compare_pose_segments(self, segment1, segment2, method='cosine', **kwargs):
        """
        Compares two pose segments using various comparison methods.
        
        Args:
            segment1: First segment dictionary (from segment_by_eventfulness_peaks)
            segment2: Second segment dictionary
            method: Comparison method - 'cosine', 'dtw', 'statistical', 'frechet', or 'all'
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary containing similarity scores and comparison details:
            {
                'method': str,
                'similarity': float (0-1, higher = more similar),
                'distance': float (method-specific distance metric),
                'details': dict (method-specific details)
            }
        """
        if not segment1 or not segment2:
            return None
        
        if segment1.get('mean_pose') is None or segment2.get('mean_pose') is None:
            logger.warning("Segments missing mean_pose, cannot compare")
            return None
        
        results = {}
        
        if method == 'cosine' or method == 'all':
            # Method A: Cosine Similarity (fast, good for normalized data)
            mean1 = np.array(segment1['mean_pose'])
            mean2 = np.array(segment2['mean_pose'])
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity([mean1], [mean2])[0][0]
            
            results['cosine'] = {
                'method': 'cosine',
                'similarity': float(cos_sim),
                'distance': float(1 - cos_sim),
                'details': {
                    'description': 'Cosine similarity between mean pose vectors'
                }
            }
        
        if method == 'dtw' or method == 'all':
            # Method B: Dynamic Time Warping (accounts for temporal alignment)
            try:
                from scipy.spatial.distance import euclidean
                from fastdtw import fastdtw
                
                seq1 = np.array(segment1['pose_vectors'])
                seq2 = np.array(segment2['pose_vectors'])
                
                # Use fastdtw for efficiency
                distance, path = fastdtw(seq1, seq2, dist=euclidean)
                
                # Normalize by sequence length
                normalized_distance = distance / (len(seq1) + len(seq2))
                
                # Convert to similarity (0-1 scale, using exponential decay)
                similarity = np.exp(-normalized_distance)
                
                results['dtw'] = {
                    'method': 'dtw',
                    'similarity': float(similarity),
                    'distance': float(distance),
                    'details': {
                        'normalized_distance': float(normalized_distance),
                        'path_length': len(path),
                        'description': 'Dynamic Time Warping distance'
                    }
                }
            except ImportError:
                logger.warning("fastdtw not available, skipping DTW comparison")
            except Exception as e:
                logger.error(f"Error in DTW comparison: {str(e)}")
        
        if method == 'statistical' or method == 'all':
            # Method C: Statistical Comparison (KS test and t-test)
            from scipy.stats import ks_2samp, ttest_ind
            
            seq1 = np.array(segment1['pose_vectors'])
            seq2 = np.array(segment2['pose_vectors'])
            
            # Perform tests for each pose dimension
            ks_pvalues = []
            ttest_pvalues = []
            
            for dim in range(seq1.shape[1]):
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = ks_2samp(seq1[:, dim], seq2[:, dim])
                ks_pvalues.append(ks_pval)
                
                # t-test
                t_stat, t_pval = ttest_ind(seq1[:, dim], seq2[:, dim])
                ttest_pvalues.append(t_pval)
            
            # Average p-values across dimensions (higher p-value = more similar)
            avg_ks_pval = np.mean(ks_pvalues)
            avg_ttest_pval = np.mean(ttest_pvalues)
            
            # Use average p-value as similarity (0-1 scale)
            similarity = (avg_ks_pval + avg_ttest_pval) / 2
            
            results['statistical'] = {
                'method': 'statistical',
                'similarity': float(similarity),
                'distance': float(1 - similarity),
                'details': {
                    'ks_pvalue': float(avg_ks_pval),
                    'ttest_pvalue': float(avg_ttest_pval),
                    'num_dimensions': seq1.shape[1],
                    'description': 'Statistical tests (KS and t-test) across pose dimensions'
                }
            }
        
        if method == 'frechet' or method == 'all':
            # Method D: Frechet Distance (trajectory similarity)
            try:
                from scipy.spatial.distance import directed_hausdorff
                
                seq1 = np.array(segment1['pose_vectors'])
                seq2 = np.array(segment2['pose_vectors'])
                
                # Use Hausdorff distance as approximation of Frechet distance
                dist_forward = directed_hausdorff(seq1, seq2)[0]
                dist_backward = directed_hausdorff(seq2, seq1)[0]
                distance = max(dist_forward, dist_backward)
                
                # Convert to similarity (0-1 scale)
                similarity = np.exp(-distance)
                
                results['frechet'] = {
                    'method': 'frechet',
                    'similarity': float(similarity),
                    'distance': float(distance),
                    'details': {
                        'hausdorff_forward': float(dist_forward),
                        'hausdorff_backward': float(dist_backward),
                        'description': 'Hausdorff distance (approximation of Frechet)'
                    }
                }
            except Exception as e:
                logger.error(f"Error in Frechet distance comparison: {str(e)}")
        
        # Return single method result or all results
        if method == 'all':
            return results
        else:
            return results.get(method, None)
    
    def calculate_adaptive_threshold(self, similarity_matrix, method='otsu'):
        """
        Calculates an adaptive threshold for segment similarity decisions.
        
        Args:
            similarity_matrix: 2D numpy array of pairwise similarities
            method: Thresholding method - 'otsu', 'percentile', 'kmeans', or 'statistical'
            
        Returns:
            float: Optimal threshold value
        """
        if similarity_matrix is None or similarity_matrix.size == 0:
            return 0.5  # Default threshold
        
        # Flatten and remove diagonal (self-similarities)
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask].flatten()
        
        if len(similarities) == 0:
            return 0.5
        
        if method == 'otsu':
            # Otsu's method for optimal threshold
            # Discretize similarities into bins
            hist, bin_edges = np.histogram(similarities, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate weights and means
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            
            mean1 = np.cumsum(hist * bin_centers) / weight1
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2)[::-1]
            
            # Calculate between-class variance
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            
            # Find threshold that maximizes variance
            idx = np.argmax(variance)
            threshold = bin_centers[idx]
            
            logger.info(f"Otsu threshold: {threshold:.4f}")
            return float(threshold)
        
        elif method == 'percentile':
            # Use median or specific percentile
            threshold = np.percentile(similarities, 50)  # Median
            logger.info(f"Percentile threshold (50th): {threshold:.4f}")
            return float(threshold)
        
        elif method == 'kmeans':
            # Use K-means to separate into similar/dissimilar groups
            from sklearn.cluster import KMeans
            
            similarities_reshaped = similarities.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(similarities_reshaped)
            
            # Threshold is midpoint between cluster centers
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = (centers[0] + centers[1]) / 2
            
            logger.info(f"K-means threshold: {threshold:.4f}")
            return float(threshold)
        
        elif method == 'statistical':
            # Use mean + std as threshold
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim + 0.5 * std_sim
            
            logger.info(f"Statistical threshold (mean + 0.5*std): {threshold:.4f}")
            return float(threshold)
        
        else:
            logger.warning(f"Unknown threshold method: {method}, using median")
            return float(np.median(similarities))
    
    def compute_segment_similarity_matrix(self, segments, comparison_method='cosine'):
        """
        Computes pairwise similarity matrix for all segments.
        
        Args:
            segments: List of segment dictionaries
            comparison_method: Method to use for comparison ('cosine', 'dtw', 'statistical', 'frechet')
            
        Returns:
            Tuple of (similarity_matrix, distance_matrix, comparison_details)
        """
        if not segments:
            return None, None, None
        
        n = len(segments)
        similarity_matrix = np.zeros((n, n))
        distance_matrix = np.zeros((n, n))
        comparison_details = {}
        
        logger.info(f"Computing {n}x{n} similarity matrix using {comparison_method} method")
        
        # Compute pairwise comparisons
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Self-similarity is 1
                    similarity_matrix[i, j] = 1.0
                    distance_matrix[i, j] = 0.0
                else:
                    # Compare segments
                    result = self.compare_pose_segments(
                        segments[i], segments[j], method=comparison_method)
                    
                    if result:
                        similarity = result.get('similarity', 0.0)
                        distance = result.get('distance', 1.0)
                        
                        # Fill symmetric matrix
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
                        distance_matrix[i, j] = distance
                        distance_matrix[j, i] = distance
                        
                        # Store details for first few comparisons
                        if len(comparison_details) < 10:
                            comparison_details[f"seg{i}_vs_seg{j}"] = result
        
        logger.info(f"Computed similarity matrix: mean={np.mean(similarity_matrix):.3f}, "
                   f"std={np.std(similarity_matrix):.3f}")
        
        return similarity_matrix, distance_matrix, comparison_details
    
    def recursive_segment_refinement(self, segments, similarity_matrix, threshold=0.7, 
                                     strategy='merge_similar', max_iterations=10):
        """
        Recursively refines segments by merging similar or splitting different segments.
        
        Args:
            segments: List of initial segment dictionaries
            similarity_matrix: Pairwise similarity matrix
            threshold: Similarity threshold for decisions
            strategy: 'merge_similar' or 'split_different'
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            Dictionary containing:
            {
                'final_segments': list of refined segments,
                'merge_history': list of merge operations,
                'iteration_count': number of iterations performed,
                'strategy': strategy used
            }
        """
        if not segments or similarity_matrix is None:
            return None
        
        logger.info(f"Starting recursive refinement with strategy: {strategy}, threshold: {threshold}")
        
        current_segments = [seg.copy() for seg in segments]
        merge_history = []
        iteration = 0
        
        if strategy == 'merge_similar':
            # Iteratively merge adjacent segments if they are similar
            while iteration < max_iterations:
                merged = False
                new_segments = []
                skip_next = False
                
                for i in range(len(current_segments)):
                    if skip_next:
                        skip_next = False
                        continue
                    
                    # Check if we can merge with next segment
                    if i < len(current_segments) - 1:
                        # Find similarity between adjacent segments
                        seg_i_id = current_segments[i].get('original_id', i)
                        seg_j_id = current_segments[i + 1].get('original_id', i + 1)
                        
                        # For merged segments, use stored similarity or recompute
                        if seg_i_id < len(segments) and seg_j_id < len(segments):
                            sim = similarity_matrix[seg_i_id, seg_j_id]
                        else:
                            # Recompute for merged segments
                            result = self.compare_pose_segments(
                                current_segments[i], current_segments[i + 1], method='cosine')
                            sim = result['similarity'] if result else 0.0
                        
                        if sim >= threshold:
                            # Merge segments
                            merged_segment = self._merge_two_segments(
                                current_segments[i], current_segments[i + 1])
                            new_segments.append(merged_segment)
                            
                            merge_history.append({
                                'iteration': iteration,
                                'action': 'merge',
                                'segments': [i, i + 1],
                                'similarity': float(sim),
                                'reason': f'similarity {sim:.3f} >= threshold {threshold:.3f}'
                            })
                            
                            merged = True
                            skip_next = True
                        else:
                            new_segments.append(current_segments[i])
                    else:
                        new_segments.append(current_segments[i])
                
                current_segments = new_segments
                iteration += 1
                
                logger.info(f"Iteration {iteration}: {len(current_segments)} segments after merging")
                
                if not merged:
                    break
        
        # Renumber segments
        for i, seg in enumerate(current_segments):
            seg['segment_id'] = i
        
        logger.info(f"Refinement completed after {iteration} iterations: "
                   f"{len(segments)} -> {len(current_segments)} segments")
        
        return {
            'final_segments': current_segments,
            'merge_history': merge_history,
            'iteration_count': iteration,
            'strategy': strategy,
            'initial_count': len(segments),
            'final_count': len(current_segments)
        }
    
    def _merge_two_segments(self, seg1, seg2):
        """Helper function to merge two segments."""
        # Combine pose vectors
        combined_poses = seg1['pose_vectors'] + seg2['pose_vectors']
        combined_frames = seg1['frame_numbers'] + seg2['frame_numbers']
        combined_times = seg1['times'] + seg2['times']
        
        # Recalculate statistics
        if combined_poses:
            pose_array = np.array(combined_poses)
            mean_pose = np.mean(pose_array, axis=0).tolist()
            std_pose = np.std(pose_array, axis=0).tolist()
        else:
            mean_pose = None
            std_pose = None
        
        merged = {
            'segment_id': seg1['segment_id'],
            'original_id': seg1.get('original_id', seg1['segment_id']),
            'start_peak_idx': seg1['start_peak_idx'],
            'end_peak_idx': seg2['end_peak_idx'],
            'start_frame': seg1['start_frame'],
            'end_frame': seg2['end_frame'],
            'start_time': seg1['start_time'],
            'end_time': seg2['end_time'],
            'duration': seg2['end_time'] - seg1['start_time'],
            'num_frames': len(combined_poses),
            'pose_vectors': combined_poses,
            'frame_numbers': combined_frames,
            'times': combined_times,
            'mean_pose': mean_pose,
            'std_pose': std_pose,
            'merged_from': [seg1['segment_id'], seg2['segment_id']]
        }
        
        return merged
    
    def get_segments_with_boundaries(self, segments):
        """
        Returns simple segment list with boundaries and scores.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of simplified segment info
        """
        output = []
        for seg in segments:
            output.append({
                'segment_id': seg['segment_id'],
                'start_frame': seg['start_frame'],
                'end_frame': seg['end_frame'],
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'duration': seg['duration'],
                'num_frames': seg['num_frames']
            })
        return output
    
    def get_segments_with_labels(self, segments, similarity_matrix, num_clusters=None):
        """
        Clusters segments and assigns labels based on pose similarity.
        
        Args:
            segments: List of segment dictionaries
            similarity_matrix: Pairwise similarity matrix
            num_clusters: Number of clusters (if None, auto-determine)
            
        Returns:
            List of segments with cluster labels
        """
        if not segments or similarity_matrix is None:
            return []
        
        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Determine optimal number of clusters if not specified
        if num_clusters is None:
            # Use silhouette score to find optimal k
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import AgglomerativeClustering
            
            best_k = 2
            best_score = -1
            
            for k in range(2, min(len(segments), 10)):
                clustering = AgglomerativeClustering(
                    n_clusters=k, metric='precomputed', linkage='average')
                labels = clustering.fit_predict(distance_matrix)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            num_clusters = best_k
            logger.info(f"Auto-determined {num_clusters} clusters for segments")
        
        # Perform clustering
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(distance_matrix)
        
        # Add labels to segments
        output = []
        for i, seg in enumerate(segments):
            seg_with_label = seg.copy()
            seg_with_label['cluster_label'] = int(labels[i])
            seg_with_label['cluster_name'] = f"Pose_Type_{labels[i]}"
            output.append(seg_with_label)
        
        logger.info(f"Assigned {num_clusters} cluster labels to {len(segments)} segments")
        return output
    
    
    def get_change_points(self, segments, pose_data, threshold_percentile=90):
        """
        Identifies frames where significant pose changes occur.
        
        Args:
            segments: List of segment dictionaries
            pose_data: Full pose data dictionary
            threshold_percentile: Percentile for determining significant changes
            
        Returns:
            List of change point dictionaries
        """
        change_points = []
        
        # Calculate frame-to-frame pose differences across entire video
        sorted_frames = sorted(pose_data.items(), 
                              key=lambda x: pose_data[x[0]]['frame_number'])
        
        pose_vectors = []
        frame_numbers = []
        times = []
        
        for frame_key, frame_data in sorted_frames:
            pose_vec = frame_data.get('pose_vector')
            if pose_vec is not None:
                pose_vectors.append(pose_vec)
                frame_numbers.append(frame_data['frame_number'])
                times.append(frame_data['time'])
        
        if len(pose_vectors) < 2:
            return []
        
        # Calculate differences
        pose_array = np.array(pose_vectors)
        diffs = np.linalg.norm(np.diff(pose_array, axis=0), axis=1)
        
        # Find significant changes
        threshold = np.percentile(diffs, threshold_percentile)
        significant_indices = np.where(diffs > threshold)[0]
        
        # Create change point objects
        for idx in significant_indices:
            # Find which segment this belongs to
            frame_num = frame_numbers[idx]
            segment_id = None
            for seg in segments:
                if seg['start_frame'] <= frame_num < seg['end_frame']:
                    segment_id = seg['segment_id']
                    break
            
            change_points.append({
                'frame_number': int(frame_num),
                'time': float(times[idx]),
                'pose_change_magnitude': float(diffs[idx]),
                'segment_id': segment_id,
                'is_segment_boundary': any(
                    seg['start_frame'] == frame_num or seg['end_frame'] == frame_num 
                    for seg in segments
                )
            })
        
        logger.info(f"Identified {len(change_points)} significant pose change points")
        return change_points
    
    def visualize_segment_creation(self, segment, adjacent_segments, comparison_results, 
                                   output_dir=None, video_name='video'):
        """
        Visualizes how a specific segment was created, showing comparisons with adjacent segments.
        
        Args:
            segment: The segment to visualize
            adjacent_segments: Dict with 'prev' and 'next' segments (can be None)
            comparison_results: Dict with comparison results to prev/next segments
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        seg_id = segment['segment_id']
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Pose trajectory for this segment (multiple dimensions)
        ax1 = fig.add_subplot(gs[0:2, 0])
        if segment['pose_vectors']:
            pose_array = np.array(segment['pose_vectors'])
            # Plot first few dimensions
            for dim in range(min(5, pose_array.shape[1])):
                ax1.plot(segment['frame_numbers'], pose_array[:, dim], 
                        label=f'Dim {dim}', alpha=0.7)
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Pose Value')
            ax1.set_title(f'Segment {seg_id}: Pose Trajectory')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Comparison with previous segment
        ax2 = fig.add_subplot(gs[0, 1])
        if adjacent_segments.get('prev') and comparison_results.get('prev'):
            prev_seg = adjacent_segments['prev']
            prev_result = comparison_results['prev']
            
            # Show similarity score
            sim = prev_result.get('similarity', 0)
            ax2.text(0.5, 0.6, f"Similarity: {sim:.3f}", 
                    ha='center', va='center', fontsize=14, weight='bold')
            ax2.text(0.5, 0.4, f"Method: {prev_result.get('method', 'N/A')}", 
                    ha='center', va='center', fontsize=10)
            ax2.text(0.5, 0.2, f"Distance: {prev_result.get('distance', 0):.3f}", 
                    ha='center', va='center', fontsize=10)
            
            # Color code based on similarity
            color = 'green' if sim > 0.7 else 'orange' if sim > 0.4 else 'red'
            ax2.set_facecolor((*plt.cm.colors.to_rgb(color), 0.2))
            
            ax2.set_title(f'vs Segment {prev_seg["segment_id"]} (Previous)')
        else:
            ax2.text(0.5, 0.5, 'No previous segment', ha='center', va='center')
        ax2.axis('off')
        
        # Plot 3: Comparison with next segment
        ax3 = fig.add_subplot(gs[0, 2])
        if adjacent_segments.get('next') and comparison_results.get('next'):
            next_seg = adjacent_segments['next']
            next_result = comparison_results['next']
            
            # Show similarity score
            sim = next_result.get('similarity', 0)
            ax3.text(0.5, 0.6, f"Similarity: {sim:.3f}", 
                    ha='center', va='center', fontsize=14, weight='bold')
            ax3.text(0.5, 0.4, f"Method: {next_result.get('method', 'N/A')}", 
                    ha='center', va='center', fontsize=10)
            ax3.text(0.5, 0.2, f"Distance: {next_result.get('distance', 0):.3f}", 
                    ha='center', va='center', fontsize=10)
            
            # Color code based on similarity
            color = 'green' if sim > 0.7 else 'orange' if sim > 0.4 else 'red'
            ax3.set_facecolor((*plt.cm.colors.to_rgb(color), 0.2))
            
            ax3.set_title(f'vs Segment {next_seg["segment_id"]} (Next)')
        else:
            ax3.text(0.5, 0.5, 'No next segment', ha='center', va='center')
        ax3.axis('off')
        
        # Plot 4: Overlay of mean poses
        ax4 = fig.add_subplot(gs[1, 1:])
        if segment.get('mean_pose'):
            mean_pose = np.array(segment['mean_pose'])
            x_coords = np.arange(len(mean_pose))
            
            ax4.plot(x_coords, mean_pose, 'b-', linewidth=2, label=f'Seg {seg_id}')
            
            if adjacent_segments.get('prev') and adjacent_segments['prev'].get('mean_pose'):
                prev_mean = np.array(adjacent_segments['prev']['mean_pose'])
                ax4.plot(x_coords, prev_mean, 'r--', alpha=0.6, label='Previous')
            
            if adjacent_segments.get('next') and adjacent_segments['next'].get('mean_pose'):
                next_mean = np.array(adjacent_segments['next']['mean_pose'])
                ax4.plot(x_coords, next_mean, 'g--', alpha=0.6, label='Next')
            
            ax4.set_xlabel('Pose Dimension')
            ax4.set_ylabel('Mean Value')
            ax4.set_title('Mean Pose Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Segment statistics
        ax5 = fig.add_subplot(gs[2, :])
        stats_text = f"""
        Segment ID: {seg_id}
        Frames: {segment['start_frame']} - {segment['end_frame']} ({segment['num_frames']} frames)
        Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s (Duration: {segment['duration']:.2f}s)
        Peak Range: {segment.get('start_peak_idx', 'N/A')} - {segment.get('end_peak_idx', 'N/A')}
        """
        
        if segment.get('merged_from'):
            stats_text += f"\nMerged from segments: {segment['merged_from']}"
        
        ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                va='center', transform=ax5.transAxes)
        ax5.axis('off')
        
        # Plot 6: Pose variance within segment
        ax6 = fig.add_subplot(gs[3, :])
        if segment['pose_vectors'] and len(segment['pose_vectors']) > 1:
            pose_array = np.array(segment['pose_vectors'])
            variances = np.var(pose_array, axis=0)
            
            ax6.bar(range(len(variances)), variances, alpha=0.7)
            ax6.set_xlabel('Pose Dimension')
            ax6.set_ylabel('Variance')
            ax6.set_title('Pose Variance Within Segment (higher = more movement)')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Segment {seg_id} Creation Analysis - {video_name}', 
                    fontsize=16, weight='bold')
        
        # Save figure
        output_path = os.path.join(output_dir, f'segment_{seg_id}_creation_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved segment creation visualization to {output_path}")
        return output_path
    
    def plot_segmentation_timeline(self, segments, peak_frames, video_info, 
                                   cluster_labels=None, output_dir=None, video_name='video'):
        """
        Plots all segments on a timeline with color coding and annotations.
        
        Args:
            segments: List of segment dictionaries
            peak_frames: Dictionary of peak frames
            video_info: Video information dictionary
            cluster_labels: Optional cluster labels for color coding
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
        
        # Get color map
        if cluster_labels:
            unique_labels = sorted(set(cluster_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        
        # Plot 1: Segments as horizontal bars
        for i, seg in enumerate(segments):
            start = seg['start_time']
            duration = seg['duration']
            
            if cluster_labels and i < len(cluster_labels):
                color = label_to_color[cluster_labels[i]]
                label = f"Type {cluster_labels[i]}" if i == 0 or cluster_labels[i] != cluster_labels[i-1] else None
            else:
                color = colors[i]
                label = None
            
            ax1.barh(0, duration, left=start, height=0.5, color=color, 
                    edgecolor='black', linewidth=1, label=label, alpha=0.8)
            
            # Add segment ID text
            ax1.text(start + duration/2, 0, str(seg['segment_id']), 
                    ha='center', va='center', fontsize=8, weight='bold')
        
        # Mark segment boundaries
        for seg in segments:
            ax1.axvline(seg['start_time'], color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.set_ylabel('Segments')
        ax1.set_title('Video Segmentation Timeline')
        ax1.set_yticks([])
        ax1.set_xlim(0, video_info.get('duration', segments[-1]['end_time']))
        if cluster_labels:
            ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Eventfulness peaks
        if peak_frames:
            peak_times = [info['time'] for info in peak_frames.values()]
            peak_values = [info.get('peak_value', info.get('eventfulness_value', 1.0)) 
                          for info in peak_frames.values()]
            
            ax2.scatter(peak_times, peak_values, c='red', s=50, alpha=0.6, marker='^', 
                       label='Eventfulness Peaks')
            ax2.plot(peak_times, peak_values, 'r-', alpha=0.3)
            
            # Mark segment boundaries
            for seg in segments:
                ax2.axvline(seg['start_time'], color='blue', linestyle='--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Eventfulness')
        ax2.set_title('Eventfulness Peaks and Segment Boundaries')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'segmentation_timeline_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved segmentation timeline to {output_path}")
        return output_path
    
    def plot_segment_similarity_matrix(self, similarity_matrix, segments, 
                                       output_dir=None, video_name='video'):
        """
        Plots heatmap of segment similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            segments: List of segment dictionaries
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity', rotation=270, labelpad=20)
        
        # Set ticks
        n = len(segments)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([seg['segment_id'] for seg in segments], rotation=45)
        ax.set_yticklabels([seg['segment_id'] for seg in segments])
        
        # Add grid
        ax.set_xticks(np.arange(n) - 0.5, minor=True)
        ax.set_yticks(np.arange(n) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Segment ID')
        ax.set_ylabel('Segment ID')
        ax.set_title(f'Segment Similarity Matrix - {video_name}')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'similarity_matrix_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved similarity matrix to {output_path}")
        return output_path
    
    def plot_segment_comparisons(self, segments, indices_to_compare, 
                                output_dir=None, video_name='video'):
        """
        Creates side-by-side pose trajectory comparisons for selected segments.
        
        Args:
            segments: List of segment dictionaries
            indices_to_compare: List of tuples (seg_idx1, seg_idx2) to compare
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        n_comparisons = len(indices_to_compare)
        fig, axes = plt.subplots(n_comparisons, 2, figsize=(16, 4 * n_comparisons))
        
        if n_comparisons == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx1, idx2) in enumerate(indices_to_compare):
            seg1 = segments[idx1]
            seg2 = segments[idx2]
            
            # Plot segment 1
            ax1 = axes[i, 0]
            if seg1['pose_vectors']:
                pose_array1 = np.array(seg1['pose_vectors'])
                for dim in range(min(5, pose_array1.shape[1])):
                    ax1.plot(seg1['frame_numbers'], pose_array1[:, dim], 
                            label=f'Dim {dim}', alpha=0.7)
            ax1.set_xlabel('Frame Number')
            ax1.set_ylabel('Pose Value')
            ax1.set_title(f'Segment {seg1["segment_id"]} '
                         f'({seg1["start_time"]:.1f}s - {seg1["end_time"]:.1f}s)')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Plot segment 2
            ax2 = axes[i, 1]
            if seg2['pose_vectors']:
                pose_array2 = np.array(seg2['pose_vectors'])
                for dim in range(min(5, pose_array2.shape[1])):
                    ax2.plot(seg2['frame_numbers'], pose_array2[:, dim], 
                            label=f'Dim {dim}', alpha=0.7)
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Pose Value')
            ax2.set_title(f'Segment {seg2["segment_id"]} '
                         f'({seg2["start_time"]:.1f}s - {seg2["end_time"]:.1f}s)')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Segment Pose Trajectory Comparisons - {video_name}', 
                    fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'segment_comparisons_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved segment comparisons to {output_path}")
        return output_path
    
    def visualize_merge_history(self, merge_history, initial_segments, final_segments,
                               output_dir=None, video_name='video'):
        """
        Visualizes the step-by-step merge/split operations.
        
        Args:
            merge_history: List of merge/split operations
            initial_segments: Initial segment list
            final_segments: Final segment list after refinement
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not merge_history:
            logger.info("No merge history to visualize")
            return None
        
        # Create figure showing merge operations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Timeline of merge operations
        for i, operation in enumerate(merge_history):
            iteration = operation.get('iteration', 0)
            action = operation.get('action', 'unknown')
            
            if action == 'merge':
                segments = operation.get('segments', [])
                similarity = operation.get('similarity', 0)
                
                ax1.scatter(iteration, similarity, s=100, alpha=0.6)
                ax1.text(iteration, similarity, f"{segments[0]}-{segments[1]}", 
                        fontsize=8, ha='center', va='bottom')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Similarity Score')
        ax1.set_title('Merge Operations Over Iterations')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Segment count over iterations
        iterations = [op.get('iteration', 0) for op in merge_history]
        if iterations:
            max_iter = max(iterations) + 1
            segment_counts = [len(initial_segments)]
            
            for iter_num in range(max_iter):
                merges_in_iter = sum(1 for op in merge_history 
                                    if op.get('iteration') == iter_num and op.get('action') == 'merge')
                segment_counts.append(segment_counts[-1] - merges_in_iter)
            
            ax2.plot(range(len(segment_counts)), segment_counts, 'bo-', linewidth=2, markersize=8)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Number of Segments')
            ax2.set_title(f'Segment Count: {len(initial_segments)} → {len(final_segments)}')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'merge_history_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved merge history to {output_path}")
        return output_path
    
    def visualize_iterative_segmentation(self, initial_segments, merge_history, strategy,
                                        output_dir=None, video_name='video'):
        """
        Creates a comprehensive visualization showing every single merge/split iteration.
        Each panel shows the state of segments at that iteration.
        
        Args:
            initial_segments: List of initial segment dictionaries
            merge_history: List of merge/split operations with iteration info
            strategy: Name of the strategy used
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
            
        Returns:
            Path to saved visualization
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not merge_history:
            logger.info("No merge history to visualize")
            return None
        
        logger.info(f"Creating iterative segmentation visualization for strategy: {strategy}")
        
        # Reconstruct segment state at each iteration
        iterations_data = []
        
        # Iteration 0: Initial state
        current_segments = [seg.copy() for seg in initial_segments]
        for i, seg in enumerate(current_segments):
            seg['display_id'] = i
            seg['original_id'] = i
        
        iterations_data.append({
            'iteration': 0,
            'segments': [seg.copy() for seg in current_segments],
            'operation': 'Initial segmentation from peaks',
            'count': len(current_segments)
        })
        
        # Group operations by iteration
        max_iter = max(op.get('iteration', 0) for op in merge_history)
        
        for iter_num in range(max_iter + 1):
            iter_ops = [op for op in merge_history if op.get('iteration') == iter_num]
            
            if not iter_ops:
                continue
            
            # Apply operations for this iteration
            for operation in iter_ops:
                action = operation.get('action', 'unknown')
                
                if action == 'merge':
                    seg_indices = operation.get('segments', [])
                    similarity = operation.get('similarity', 0)
                    
                    if len(seg_indices) >= 2:
                        # Find segments to merge
                        seg1_idx = seg_indices[0]
                        seg2_idx = seg_indices[1]
                        
                        # Merge them
                        if seg1_idx < len(current_segments) and seg2_idx < len(current_segments):
                            seg1 = current_segments[seg1_idx]
                            seg2 = current_segments[seg2_idx]
                            
                            # Create merged segment
                            merged = {
                                'display_id': seg1.get('display_id', seg1_idx),
                                'original_id': seg1.get('original_id', seg1_idx),
                                'start_frame': seg1['start_frame'],
                                'end_frame': seg2['end_frame'],
                                'start_time': seg1['start_time'],
                                'end_time': seg2['end_time'],
                                'duration': seg2['end_time'] - seg1['start_time'],
                                'merged_from': [seg1.get('display_id', seg1_idx), 
                                              seg2.get('display_id', seg2_idx)],
                                'similarity': similarity
                            }
                            
                            # Replace first segment with merged, remove second
                            current_segments[seg1_idx] = merged
                            current_segments.pop(seg2_idx)
                
                elif action == 'split':
                    seg_idx = operation.get('segment', 0)
                    split_frame = operation.get('split_frame', 0)
                    
                    # Split logic would go here
                    pass
            
            # Record state after this iteration
            iterations_data.append({
                'iteration': iter_num + 1,
                'segments': [seg.copy() for seg in current_segments],
                'operation': f"{len(iter_ops)} operation(s)",
                'count': len(current_segments)
            })
        
        # Create visualization with one panel per iteration
        n_iterations = len(iterations_data)
        n_cols = min(3, n_iterations)
        n_rows = (n_iterations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
        if n_iterations == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Color map for segments
        colors = plt.cm.tab20(np.linspace(0, 1, len(initial_segments)))
        
        for idx, iter_data in enumerate(iterations_data):
            ax = axes[idx]
            segments = iter_data['segments']
            
            # Draw each segment as a horizontal bar
            for i, seg in enumerate(segments):
                start = seg['start_time']
                duration = seg['duration']
                
                # Use original ID for consistent coloring
                orig_id = seg.get('original_id', i)
                color = colors[orig_id % len(colors)]
                
                # Draw segment
                ax.barh(0, duration, left=start, height=0.8, 
                       color=color, edgecolor='black', linewidth=1.5, alpha=0.8)
                
                # Add segment label
                label_text = str(seg.get('display_id', i))
                if 'merged_from' in seg:
                    label_text = f"{seg['merged_from'][0]}+{seg['merged_from'][1]}"
                
                ax.text(start + duration/2, 0, label_text,
                       ha='center', va='center', fontsize=9, weight='bold',
                       color='white' if np.mean(color[:3]) < 0.5 else 'black')
            
            # Mark boundaries
            for seg in segments:
                ax.axvline(seg['start_time'], color='red', linestyle='--', 
                          alpha=0.3, linewidth=1)
            
            # Styling
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_title(f"Iteration {iter_data['iteration']}: {iter_data['count']} segments\n{iter_data['operation']}", 
                        fontsize=11, weight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Set consistent x-axis limits
            if segments:
                max_time = max(seg['end_time'] for seg in segments)
                ax.set_xlim(0, max_time * 1.05)
        
        # Hide unused subplots
        for idx in range(n_iterations, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Iterative Segmentation Process - {strategy}\n{video_name}', 
                    fontsize=16, weight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'iterative_segmentation_{strategy}_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved iterative segmentation visualization to {output_path}")
        return output_path
    
    def plot_pose_change_detection(self, pose_data, change_points, segments,
                                   output_dir=None, video_name='video'):
        """
        Highlights frames with significant pose changes.
        
        Args:
            pose_data: Full pose data dictionary
            change_points: List of change point dictionaries
            segments: List of segment dictionaries
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate frame-to-frame differences
        sorted_frames = sorted(pose_data.items(), 
                              key=lambda x: pose_data[x[0]]['frame_number'])
        
        pose_vectors = []
        frame_numbers = []
        times = []
        
        for frame_key, frame_data in sorted_frames:
            pose_vec = frame_data.get('pose_vector')
            if pose_vec is not None:
                pose_vectors.append(pose_vec)
                frame_numbers.append(frame_data['frame_number'])
                times.append(frame_data['time'])
        
        if len(pose_vectors) < 2:
            return None
        
        pose_array = np.array(pose_vectors)
        diffs = np.linalg.norm(np.diff(pose_array, axis=0), axis=1)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        
        # Plot 1: Pose change magnitude over time
        ax1.plot(times[:-1], diffs, 'b-', alpha=0.5, linewidth=1)
        ax1.fill_between(times[:-1], diffs, alpha=0.3)
        
        # Mark change points
        if change_points:
            change_times = [cp['time'] for cp in change_points]
            change_mags = [cp['pose_change_magnitude'] for cp in change_points]
            ax1.scatter(change_times, change_mags, c='red', s=100, 
                       marker='*', label='Significant Changes', zorder=5)
        
        # Mark segment boundaries
        for seg in segments:
            ax1.axvline(seg['start_time'], color='green', linestyle='--', 
                       alpha=0.5, linewidth=1.5)
        
        ax1.set_ylabel('Pose Change Magnitude')
        ax1.set_title('Frame-to-Frame Pose Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Segments timeline
        for i, seg in enumerate(segments):
            color = plt.cm.viridis(i / len(segments))
            ax2.barh(0, seg['duration'], left=seg['start_time'], height=0.5, 
                    color=color, edgecolor='black', linewidth=1, alpha=0.8)
            ax2.text(seg['start_time'] + seg['duration']/2, 0, str(seg['segment_id']), 
                    ha='center', va='center', fontsize=8, weight='bold')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Segments')
        ax2.set_yticks([])
        ax2.set_title('Segment Boundaries')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'pose_change_detection_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved pose change detection to {output_path}")
        return output_path
    
    def plot_threshold_analysis(self, similarity_matrix, threshold, method='otsu',
                               output_dir=None, video_name='video'):
        """
        Shows how the adaptive threshold was determined.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            threshold: Calculated threshold value
            method: Method used for threshold calculation
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract similarities (excluding diagonal)
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask].flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Histogram of similarities
        ax1.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.3f}')
        ax1.axvline(np.mean(similarities), color='green', linestyle=':', linewidth=2, 
                   label=f'Mean = {np.mean(similarities):.3f}')
        ax1.axvline(np.median(similarities), color='orange', linestyle=':', linewidth=2, 
                   label=f'Median = {np.median(similarities):.3f}')
        
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Similarity Distribution (Method: {method})')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Cumulative distribution
        sorted_sims = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
        
        ax2.plot(sorted_sims, cumulative, 'b-', linewidth=2)
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.3f}')
        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Mark threshold position on CDF
        threshold_percentile = np.sum(similarities < threshold) / len(similarities) * 100
        ax2.scatter([threshold], [threshold_percentile/100], c='red', s=100, zorder=5)
        ax2.text(threshold, threshold_percentile/100 + 0.05, 
                f'{threshold_percentile:.1f}th percentile', ha='center')
        
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'threshold_analysis_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved threshold analysis to {output_path}")
        return output_path
    
    def plot_segment_statistics(self, segments, output_dir=None, video_name='video'):
        """
        Creates box plots and distribution plots for segment statistics.
        
        Args:
            segments: List of segment dictionaries
            output_dir: Directory to save visualization
            video_name: Name of video for file naming
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Extract statistics
        durations = [seg['duration'] for seg in segments]
        num_frames = [seg['num_frames'] for seg in segments]
        
        # Plot 1: Duration distribution
        axes[0, 0].hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Segment Duration Distribution')
        axes[0, 0].axvline(np.mean(durations), color='red', linestyle='--', 
                          label=f'Mean = {np.mean(durations):.2f}s')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Frame count distribution
        axes[0, 1].hist(num_frames, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Number of Frames')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Segment Frame Count Distribution')
        axes[0, 1].axvline(np.mean(num_frames), color='red', linestyle='--', 
                          label=f'Mean = {np.mean(num_frames):.1f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Box plot of durations
        axes[0, 2].boxplot(durations, vert=True)
        axes[0, 2].set_ylabel('Duration (seconds)')
        axes[0, 2].set_title('Segment Duration Box Plot')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Mean pose variance per segment
        mean_variances = []
        for seg in segments:
            if seg.get('std_pose'):
                mean_var = np.mean(seg['std_pose'])
                mean_variances.append(mean_var)
        
        if mean_variances:
            axes[1, 0].bar(range(len(mean_variances)), mean_variances, alpha=0.7, color='purple')
            axes[1, 0].set_xlabel('Segment ID')
            axes[1, 0].set_ylabel('Mean Pose Variance')
            axes[1, 0].set_title('Pose Variance by Segment')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Segment timeline
        for i, seg in enumerate(segments):
            color = plt.cm.viridis(i / len(segments))
            axes[1, 1].barh(i, seg['duration'], left=seg['start_time'], 
                           color=color, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Segment ID')
        axes[1, 1].set_title('Segment Timeline')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Plot 6: Summary statistics table
        axes[1, 2].axis('off')
        summary_text = f"""
        Summary Statistics
        ==================
        Total Segments: {len(segments)}
        
        Duration:
          Mean: {np.mean(durations):.2f}s
          Std: {np.std(durations):.2f}s
          Min: {np.min(durations):.2f}s
          Max: {np.max(durations):.2f}s
        
        Frame Count:
          Mean: {np.mean(num_frames):.1f}
          Std: {np.std(num_frames):.1f}
          Min: {np.min(num_frames)}
          Max: {np.max(num_frames)}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       va='center', transform=axes[1, 2].transAxes)
        
        plt.suptitle(f'Segment Statistics - {video_name}', fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'segment_statistics_{video_name}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved segment statistics to {output_path}")
        return output_path
    
    def handle_pose_segmentation(self, peak_frames, pose_data, video_info, video_path,
                                 comparison_method='cosine', threshold_strategy='adaptive',
                                 recursive_strategies=None, similarity_threshold=0.7,
                                 output_formats=None, create_visualizations=True,
                                 output_dir=None):
        """
        Main method for pose-based segmentation pipeline.
        
        Segments video from peak to peak, compares segments, and recursively refines.
        
        Args:
            peak_frames: Dictionary of peak frames from eventfulness detection
            pose_data: Dictionary of frame-by-frame pose estimation data
            video_info: Video information dictionary
            video_path: Path to video file
            comparison_method: 'cosine', 'dtw', 'statistical', 'frechet', or 'all'
            threshold_strategy: 'fixed', 'adaptive', 'clustering', or 'statistical_test'
            recursive_strategies: List of strategies to apply (default: ['merge_similar'])
            similarity_threshold: Fixed threshold if using 'fixed' strategy
            output_formats: List of output formats to generate
            create_visualizations: Whether to create visualization plots
            output_dir: Directory for outputs (default: RESULTS_DIR)
            
        Returns:
            Dictionary containing all segmentation results:
            {
                'initial_segments': list,
                'similarity_matrix': array,
                'threshold': float,
                'refined_segments': dict (keyed by strategy),
                'output_formats': dict,
                'visualizations': dict (paths to visualization files)
            }
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        if recursive_strategies is None:
            recursive_strategies = ['merge_similar']
        
        if output_formats is None:
            output_formats = ['boundaries', 'labels', 'change_points']
        
        video_name = os.path.basename(video_path).replace('.mp4', '').replace('.', '_')
        
        logger.info("="*80)
        logger.info("STARTING POSE-BASED SEGMENTATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Video: {video_path}")
        logger.info(f"Comparison method: {comparison_method}")
        logger.info(f"Threshold strategy: {threshold_strategy}")
        logger.info(f"Recursive strategies: {recursive_strategies}")
        logger.info(f"Output formats: {output_formats}")
        
        results = {
            'video_path': video_path,
            'video_name': video_name,
            'parameters': {
                'comparison_method': comparison_method,
                'threshold_strategy': threshold_strategy,
                'recursive_strategies': recursive_strategies,
                'similarity_threshold': similarity_threshold,
                'output_formats': output_formats
            }
        }
        
        # Step 1: Create initial segments from peak to peak
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Creating initial segments from eventfulness peaks")
        logger.info("="*80)
        
        initial_segments = self.segment_by_eventfulness_peaks(peak_frames, pose_data)
        
        if not initial_segments:
            logger.error("Failed to create initial segments")
            return None
        
        logger.info(f"Created {len(initial_segments)} initial segments")
        results['initial_segments'] = initial_segments
        
        # Step 2: Compute similarity matrix
        logger.info("\n" + "="*80)
        logger.info(f"STEP 2: Computing segment similarity matrix using {comparison_method}")
        logger.info("="*80)
        
        similarity_matrix, distance_matrix, comparison_details = \
            self.compute_segment_similarity_matrix(initial_segments, comparison_method)
        
        if similarity_matrix is None:
            logger.error("Failed to compute similarity matrix")
            return None
        
        logger.info(f"Computed {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")
        logger.info(f"Similarity range: [{np.min(similarity_matrix):.3f}, {np.max(similarity_matrix):.3f}]")
        logger.info(f"Mean similarity: {np.mean(similarity_matrix):.3f}")
        
        results['similarity_matrix'] = similarity_matrix
        results['distance_matrix'] = distance_matrix
        results['comparison_details'] = comparison_details
        
        # Step 3: Calculate threshold
        logger.info("\n" + "="*80)
        logger.info(f"STEP 3: Calculating threshold using {threshold_strategy} strategy")
        logger.info("="*80)
        
        if threshold_strategy == 'fixed':
            threshold = similarity_threshold
            logger.info(f"Using fixed threshold: {threshold:.3f}")
        else:
            # Map strategy names
            method_map = {
                'adaptive': 'otsu',
                'clustering': 'kmeans',
                'statistical_test': 'statistical'
            }
            threshold_method = method_map.get(threshold_strategy, 'otsu')
            threshold = self.calculate_adaptive_threshold(similarity_matrix, method=threshold_method)
            logger.info(f"Calculated adaptive threshold: {threshold:.3f}")
        
        results['threshold'] = threshold
        
        # Step 4: Apply recursive refinement strategies
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Applying recursive refinement strategies")
        logger.info("="*80)
        
        refined_results = {}
        
        for strategy in recursive_strategies:
            logger.info(f"\nApplying strategy: {strategy}")
            logger.info("-" * 40)
            
            refinement_result = self.recursive_segment_refinement(
                initial_segments, similarity_matrix, threshold, strategy=strategy)
            
            if refinement_result:
                refined_results[strategy] = refinement_result
                final_segs = refinement_result['final_segments']
                logger.info(f"Strategy '{strategy}': {len(initial_segments)} → {len(final_segs)} segments")
                logger.info(f"Iterations: {refinement_result['iteration_count']}")
                logger.info(f"Merge operations: {len(refinement_result['merge_history'])}")
            else:
                logger.warning(f"Strategy '{strategy}' failed")
        
        results['refined_segments'] = refined_results
        
        # Step 5: Generate output formats
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Generating output formats")
        logger.info("="*80)
        
        output_data = {}
        
        # Use the first refined result, or initial segments if no refinement
        primary_segments = (list(refined_results.values())[0]['final_segments'] 
                          if refined_results else initial_segments)
        
        if 'boundaries' in output_formats:
            logger.info("Generating boundaries output...")
            output_data['boundaries'] = self.get_segments_with_boundaries(primary_segments)
        
        if 'labels' in output_formats:
            logger.info("Generating labels output...")
            output_data['labels'] = self.get_segments_with_labels(
                primary_segments, similarity_matrix)
        
        if 'change_points' in output_formats:
            logger.info("Generating change points output...")
            output_data['change_points'] = self.get_change_points(
                primary_segments, pose_data)
        
        results['output_formats'] = output_data
        
        # Step 6: Create visualizations
        if create_visualizations:
            logger.info("\n" + "="*80)
            logger.info("STEP 6: Creating visualizations")
            logger.info("="*80)
            
            viz_paths = {}
            
            # Timeline visualization
            logger.info("Creating segmentation timeline...")
            cluster_labels = None
            if 'labels' in output_data:
                cluster_labels = [seg.get('cluster_label', 0) for seg in output_data['labels']]
            
            viz_paths['timeline'] = self.plot_segmentation_timeline(
                primary_segments, peak_frames, video_info, cluster_labels, output_dir, video_name)
            
            # Similarity matrix
            logger.info("Creating similarity matrix heatmap...")
            viz_paths['similarity_matrix'] = self.plot_segment_similarity_matrix(
                similarity_matrix, primary_segments, output_dir, video_name)
            
            # Threshold analysis
            logger.info("Creating threshold analysis...")
            viz_paths['threshold_analysis'] = self.plot_threshold_analysis(
                similarity_matrix, threshold, threshold_strategy, output_dir, video_name)
            
            # Segment statistics
            logger.info("Creating segment statistics...")
            viz_paths['statistics'] = self.plot_segment_statistics(
                primary_segments, output_dir, video_name)
            
            # Pose change detection
            if 'change_points' in output_data:
                logger.info("Creating pose change detection plot...")
                viz_paths['pose_changes'] = self.plot_pose_change_detection(
                    pose_data, output_data['change_points'], primary_segments, 
                    output_dir, video_name)
            
            # Merge history and iterative visualization (if available)
            for strategy, refinement in refined_results.items():
                if refinement.get('merge_history'):
                    logger.info(f"Creating merge history for strategy '{strategy}'...")
                    viz_paths[f'merge_history_{strategy}'] = self.visualize_merge_history(
                        refinement['merge_history'], initial_segments, 
                        refinement['final_segments'], output_dir, f"{video_name}_{strategy}")
                    
                    # Create iterative visualization showing each step
                    logger.info(f"Creating iterative segmentation visualization for strategy '{strategy}'...")
                    viz_paths[f'iterative_{strategy}'] = self.visualize_iterative_segmentation(
                        initial_segments, refinement['merge_history'], strategy,
                        output_dir, video_name)
            
            # Segment comparisons (compare a few interesting pairs)
            logger.info("Creating segment comparisons...")
            # Find most similar and most different pairs
            n = len(primary_segments)
            if n > 1:
                # Get upper triangle indices
                triu_indices = np.triu_indices(n, k=1)
                triu_similarities = similarity_matrix[triu_indices]
                
                # Find most similar pair
                most_similar_idx = np.argmax(triu_similarities)
                most_similar_pair = (triu_indices[0][most_similar_idx], 
                                    triu_indices[1][most_similar_idx])
                
                # Find most different pair
                most_different_idx = np.argmin(triu_similarities)
                most_different_pair = (triu_indices[0][most_different_idx], 
                                      triu_indices[1][most_different_idx])
                
                comparison_pairs = [most_similar_pair, most_different_pair]
                
                # Add a few random pairs
                if n > 4:
                    random_indices = np.random.choice(len(triu_similarities), 
                                                     size=min(2, len(triu_similarities)), 
                                                     replace=False)
                    for idx in random_indices:
                        comparison_pairs.append((triu_indices[0][idx], triu_indices[1][idx]))
                
                viz_paths['comparisons'] = self.plot_segment_comparisons(
                    primary_segments, comparison_pairs, output_dir, video_name)
            
            # Individual segment creation visualizations (for first few segments)
            logger.info("Creating individual segment visualizations...")
            for i in range(min(5, len(primary_segments))):
                seg = primary_segments[i]
                
                # Get adjacent segments
                adjacent = {
                    'prev': primary_segments[i-1] if i > 0 else None,
                    'next': primary_segments[i+1] if i < len(primary_segments)-1 else None
                }
                
                # Get comparison results
                comparisons = {}
                if adjacent['prev']:
                    comparisons['prev'] = self.compare_pose_segments(
                        seg, adjacent['prev'], method=comparison_method)
                if adjacent['next']:
                    comparisons['next'] = self.compare_pose_segments(
                        seg, adjacent['next'], method=comparison_method)
                
                viz_paths[f'segment_{i}'] = self.visualize_segment_creation(
                    seg, adjacent, comparisons, output_dir, video_name)
            
            results['visualizations'] = viz_paths
            logger.info(f"Created {len(viz_paths)} visualization files")
        
        # Step 7: Save results to JSON
        logger.info("\n" + "="*80)
        logger.info("STEP 7: Saving results to JSON")
        logger.info("="*80)
        
        # Helper function to convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Prepare JSON-serializable results
        json_results = {
            'video_path': video_path,
            'video_name': video_name,
            'parameters': results['parameters'],
            'initial_segment_count': len(initial_segments),
            'threshold': float(threshold),
            'initial_segments': [
                {k: convert_to_json_serializable(v) for k, v in seg.items() if k not in ['pose_vectors']}
                for seg in initial_segments
            ],
            'refined_segments': {
                strategy: {
                    'final_count': len(ref['final_segments']),
                    'iteration_count': ref['iteration_count'],
                    'merge_history': convert_to_json_serializable(ref['merge_history']),
                    'segments': [
                        {k: convert_to_json_serializable(v) for k, v in seg.items() if k not in ['pose_vectors']}
                        for seg in ref['final_segments']
                    ]
                }
                for strategy, ref in refined_results.items()
            },
            'output_formats': {
                'boundaries': convert_to_json_serializable(output_data.get('boundaries', [])),
                'labels': [
                    {k: convert_to_json_serializable(v) for k, v in seg.items() if k not in ['pose_vectors']}
                    for seg in output_data.get('labels', [])
                ],
                'change_points': convert_to_json_serializable(output_data.get('change_points', []))
            }
        }
        
        if create_visualizations:
            json_results['visualizations'] = results.get('visualizations', {})
        
        json_path = os.path.join(output_dir, f'pose_segmentation_results_{video_name}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved results to {json_path}")
        results['json_path'] = json_path
        
        logger.info("\n" + "="*80)
        logger.info("POSE-BASED SEGMENTATION PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Initial segments: {len(initial_segments)}")
        for strategy, ref in refined_results.items():
            logger.info(f"Final segments ({strategy}): {len(ref['final_segments'])}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*80 + "\n")
        
        return results
    
    def handle_full_video_analysis(self, video_path, peak_frames=None, cluster_assignments=None, num_workers=4, existing_pose_data=None, 
                                   perform_segmentation=True, segmentation_window_size=10, segmentation_num_regimes=None, segmentation_min_segment_length=5):
        """
        Performs full video analysis with pose estimation, clustering, and similarity calculation.
        Uses parallel processing for improved performance.
        
        Args:
            video_path: Path to the video file
            peak_frames: Dictionary of peak frames (optional)
            cluster_assignments: Dictionary of cluster assignments (optional)
            num_workers: Number of parallel workers
            existing_pose_data: Existing pose data to avoid reprocessing the entire video (optional)
            perform_segmentation: Whether to perform FLUSS-based segmentation on similarity time series (default: True)
            segmentation_window_size: Window size for FLUSS segmentation (default: 10)
            segmentation_num_regimes: Number of regimes for FLUSS (if None, auto-detect)
            segmentation_min_segment_length: Minimum segment length (default: 5)
            
        Returns:
            Tuple containing:
            - pose_data: Dictionary of pose data for each frame
            - centroids: Dictionary of cluster centroids
            - similarities: Dictionary of cosine similarities
            - cluster_assignments: Dictionary of cluster assignments
            - fluss_segmentation: Dictionary of FLUSS segmentation results (None if not performed)
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
                    
                    # Optional Step 6: Perform FLUSS-based segmentation
                    fluss_segmentation = None
                    if perform_segmentation and similarities:
                        logger.info("Step 6/6: Performing FLUSS-based segmentation on similarity time series...")
                        fluss_segmentation = self.segment_cosine_similarity_with_fluss(
                            similarities,
                            window_size=segmentation_window_size,
                            num_regimes=segmentation_num_regimes,
                            min_segment_length=segmentation_min_segment_length
                        )
                        if fluss_segmentation:
                            logger.info(f"FLUSS segmentation completed: {fluss_segmentation['num_clusters']} clusters, "
                                      f"{fluss_segmentation['total_frames']} frames")
                        else:
                            logger.warning("FLUSS segmentation failed")
                    else:
                        fluss_segmentation = None
                else:
                    logger.warning(f"Skipping similarity calculation - pose_data: {bool(pose_data)}, centroids: {bool(centroids)}")
                    fluss_segmentation = None
            else:
                fluss_segmentation = None
            
            return pose_data, centroids, similarities, cluster_assignments, fluss_segmentation
            
        except Exception as e:
            logger.error(f"Error in full video analysis: {str(e)}")
            return None, None, None, cluster_assignments, None
    
    def run_complete_analysis(self, video_path, num_workers=4):
        """
        Runs the complete analysis workflow with parallel processing:
        1. Submit eventfulness prediction job (runs in background via SLURM)
        2. Pose estimation on entire video (runs in parallel with eventfulness)
        3. Wait for eventfulness to complete, then load data and extract peak frames
        4. Run clustering and similarity calculation
        5. Run pose-based segmentation from peak to peak
        
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
            - fluss_segmentation: Dictionary of FLUSS segmentation results
            - pose_segmentation_results: Dictionary of pose-based segmentation results
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
                
                # Detect local maxima (peaks) using scipy's find_peaks
                # First detect all potential peaks with minimal filtering
                from scipy.signal import find_peaks
                all_peaks, properties = find_peaks(data, distance=1)
                
                # Calculate peak values
                all_peak_values = [data[p] for p in all_peaks]
                
                # Select top percentage of peaks (default: top 30%)
                peak_percentage = 1.0  # Can be adjusted (0.0 to 1.0)
                num_peaks_to_keep = max(1, int(len(all_peaks) * peak_percentage))
                
                # Sort peaks by their values (heights) in descending order
                peak_value_pairs = list(zip(all_peaks, all_peak_values))
                peak_value_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the top percentage
                top_peaks = peak_value_pairs[:num_peaks_to_keep]
                
                # Sort back by frame index for chronological order
                top_peaks.sort(key=lambda x: x[0])
                
                # Extract final peaks and values
                peaks = [int(p[0]) for p in top_peaks]
                peak_values = [p[1] for p in top_peaks]
                
                logger.info(f"Detected {len(all_peaks)} total peaks, selected top {peak_percentage*100:.0f}% ({len(peaks)} peaks)")
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
                # FLUSS segmentation enabled
                _, centroids, similarities, cluster_assignments, fluss_segmentation = self.handle_full_video_analysis(
                    video_path, peak_frames=peak_frames, num_workers=num_workers, existing_pose_data=pose_data,
                    perform_segmentation=True,  # FLUSS segmentation enabled
                    segmentation_window_size=10,
                    segmentation_num_regimes=None,  # Auto-detect number of regimes
                    segmentation_min_segment_length=100)
                
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
                    
                if fluss_segmentation:
                    num_segments = len(fluss_segmentation.get('segments', []))
                    logger.info(f"FLUSS segmentation completed: {fluss_segmentation['num_clusters']} clusters, "
                              f"{num_segments} global segments")
                else:
                    logger.warning("No FLUSS segmentation results")
                
                # Step 5: Run pose-based segmentation (NEW)
                logger.info("Step 5/5: Running pose-based segmentation from peak to peak...")
                pose_segmentation_results = None
                
                if peak_frames and pose_data:
                    try:
                        pose_segmentation_results = self.handle_pose_segmentation(
                            peak_frames=peak_frames,
                            pose_data=pose_data,
                            video_info=video_info,
                            video_path=video_path,
                            comparison_method='dtw',  # Use all comparison methods
                            threshold_strategy='adaptive',  # Use adaptive thresholding
                            recursive_strategies=['merge_similar'],
                            output_formats=['boundaries', 'labels', 'change_points'],
                            create_visualizations=True
                        )
                        
                        if pose_segmentation_results:
                            logger.info(f"Pose segmentation completed successfully")
                            logger.info(f"Results saved to: {pose_segmentation_results.get('json_path', 'N/A')}")
                            
                            # Add cosine similarity visualization to pose segmentation results
                            if similarities and centroids:
                                logger.info("Creating cosine similarity plots by cluster...")
                                video_name = os.path.basename(video_path).replace('.mp4', '').replace('.', '_')
                                output_dir = os.path.join(RESULTS_DIR, video_name)
                                try:
                                    plot_path = self.plot_cosine_similarities_by_cluster(
                                        similarities, 
                                        cluster_assignments=cluster_assignments,
                                        peak_frames=peak_frames,
                                        output_dir=output_dir,
                                        video_name=video_name,
                                        show_raw=True
                                    )
                                    if plot_path:
                                        logger.info(f"Cosine similarity plot saved to: {plot_path}")
                                        # Add to visualizations dict
                                        if 'visualizations' not in pose_segmentation_results:
                                            pose_segmentation_results['visualizations'] = {}
                                        pose_segmentation_results['visualizations']['cosine_similarities'] = plot_path
                                except Exception as e:
                                    logger.error(f"Error creating cosine similarity plot: {str(e)}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                        else:
                            logger.warning("Pose segmentation returned no results")
                    except Exception as e:
                        logger.error(f"Error in pose segmentation: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.info("Skipping pose segmentation - missing peak_frames or pose_data")
            else:
                logger.info("Skipping clustering and pose segmentation - no peak frames available")
                pose_segmentation_results = None
            
            logger.info("Complete analysis workflow finished")
            return pose_data, eventfulness_data_dict, peak_frames, centroids, similarities, cluster_assignments, fluss_segmentation, pose_segmentation_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis workflow: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None, None, None, None

# Example usage for pose-based segmentation
# 
# The new pose segmentation system segments video from peak to peak based on eventfulness data,
# then compares segments using multiple methods and recursively refines them.
#
# Usage Example 1: Run complete analysis with pose segmentation
# backend = VideoAnalysisBackend()
# results = backend.run_complete_analysis("/path/to/video.mp4", num_workers=4)
# pose_data, eventfulness_data, peak_frames, centroids, similarities, clusters, fluss_seg, pose_seg = results
#
# Usage Example 2: Run pose segmentation separately
# backend = VideoAnalysisBackend()
# pose_seg_results = backend.handle_pose_segmentation(
#     peak_frames=peak_frames,
#     pose_data=pose_data,
#     video_info=video_info,
#     video_path=video_path,
#     comparison_method='cosine',  # or 'dtw', 'statistical', 'frechet', 'all'
#     threshold_strategy='adaptive',  # or 'fixed', 'clustering', 'statistical_test'
#     recursive_strategies=['merge_similar'],
#     similarity_threshold=0.7,  # used if threshold_strategy='fixed'
#     output_formats=['boundaries', 'labels', 'change_points'],
#     create_visualizations=True
# )
#
# The results include:
# - initial_segments: Segments created from peak to peak
# - similarity_matrix: Pairwise similarity scores between segments
# - refined_segments: Segments after recursive refinement (per strategy)
# - output_formats: Various output formats (boundaries, labels, etc.)
# - visualizations: Paths to generated visualization files
#
# Visualization files created:
# - segmentation_timeline: Timeline showing all segments
# - similarity_matrix: Heatmap of segment similarities
# - threshold_analysis: How the threshold was determined
# - segment_statistics: Statistical analysis of segments
# - pose_changes: Frames with significant pose changes
# - merge_history: Step-by-step merge operations
# - segment_N_creation: Individual segment creation analysis
# - segment_comparisons: Side-by-side trajectory comparisons

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