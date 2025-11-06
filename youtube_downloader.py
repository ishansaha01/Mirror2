#!/usr/bin/env python3

import os
import sys
import re
import argparse
import urllib.request
import time
from datetime import datetime
from pytubefix import YouTube
from pytubefix.exceptions import PytubeFixError

def convert_shorts_url(url):
    """
    Convert YouTube Shorts URL to standard YouTube video URL.
    
    Args:
        url (str): YouTube URL which might be a Shorts URL
        
    Returns:
        str: Standard YouTube video URL
    """
    # Check if it's a shorts URL
    shorts_pattern = r'(https?://)?(www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)'
    match = re.match(shorts_pattern, url)
    
    if match:
        video_id = match.group(3)
        return f"https://www.youtube.com/watch?v={video_id}"
    
    return url

def get_video_id(url):
    """
    Extract video ID from YouTube URL.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        str: YouTube video ID
    """
    # First convert shorts URL if needed
    url = convert_shorts_url(url)
    
    # Extract video ID from standard YouTube URL
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})' # Watch
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def download_video(url, output_path=None):
    """
    Download a YouTube video in MP4 format at the highest available resolution.
    
    Args:
        url (str): YouTube video URL
        output_path (str, optional): Directory to save the video. Defaults to dataSets/test_data/results.
    
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    # Set default output path if not provided
    if not output_path:
        output_path = os.path.join(os.getcwd(), "dataSets", "test_data", "val")
    try:
        # Create output directory if it doesn't exist
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Convert shorts URL if needed
        url = convert_shorts_url(url)
        
        # Initialize YouTube object with custom headers to avoid 400 errors
        print(f"Fetching video information from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # Create a request with custom headers
        opener = urllib.request.build_opener()
        opener.addheaders = [(key, value) for key, value in headers.items()]
        urllib.request.install_opener(opener)
        
        # Initialize YouTube object with on_progress_callback
        yt = YouTube(url)
        
        # Get video title for display
        print(f"Title: {yt.title}")
        print(f"Author: {yt.author}")
        print(f"Length: {yt.length} seconds")
        
        # Get video ID for metadata
        video_id = get_video_id(url)
        
        # Create a safe filename from the title
        safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in yt.title])
        safe_title = safe_title.strip().replace(' ', '_')
        
        # Create a folder for this video using the title
        video_folder = os.path.join(output_path, safe_title)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        
        # Get the highest resolution MP4 stream
        print("Finding highest quality MP4 stream...")
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        
        if not stream:
            print("No MP4 stream found. Trying to get highest resolution video and audio separately...")
            # If no progressive stream is available, get the highest resolution video
            video_stream = yt.streams.filter(adaptive=True, file_extension="mp4").order_by("resolution").desc().first()
            if not video_stream:
                print("Error: No suitable video stream found.")
                return None
            
            # Download to the video-specific folder with the title as filename
            filename = f"{safe_title}.mp4"
            file_path = video_stream.download(output_path=video_folder, filename=filename)
            print(f"Downloaded to: {file_path}")
            return file_path
        
        # Download the video to the video-specific folder with the title as filename
        print(f"Downloading: {stream.resolution} ({stream.mime_type})...")
        filename = f"{safe_title}.mp4"
        file_path = stream.download(output_path=video_folder, filename=filename)
        
        print(f"Download complete! Saved to: {file_path}")
        
        # Create a metadata file with video information
        metadata_path = os.path.join(video_folder, "metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Title: {yt.title}\n")
            f.write(f"Author: {yt.author}\n")
            f.write(f"Length: {yt.length} seconds\n")
            f.write(f"Resolution: {stream.resolution}\n")
            f.write(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original URL: {url}\n")
        
        return file_path
        
    except PytubeFixError as e:
        print(f"Error: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download YouTube videos as MP4 files")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", help="Output directory (default: current directory)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Download the video
    download_video(args.url, args.output)

if __name__ == "__main__":
    main()
