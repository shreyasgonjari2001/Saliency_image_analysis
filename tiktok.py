import json
import requests
import tempfile
import cv2
import os


# Load the JSON data (assuming it's a list of ad dictionaries)
with open('tiktok_ads.json', 'r') as f:
    data = json.load(f)

# Extract video URLs from the data (assuming each ad has a 'video_link' key; adjust if needed)
video_urls = [ad.get('url') for ad in data if ad.get('url')]
print(len(video_urls), "video URLs found.")

# If no URLs found, you can hardcode some for testing (uncomment and customize)
# video_urls = [
#     "https://example.com/video1.mp4",
#     "https://example.com/video2.mp4",
#     # Add more URLs here
# ]

if not video_urls:
    print("No video URLs found in the JSON data. Please add 'video_link' to your ads or hardcode them in the script.")
    exit()

# Function to check and play video with user option to skip
def check_and_play_video(url, proxies=None):
    print(f"\nChecking video: {url}")
    try:
        # Check if video is available (HEAD request for efficiency)
        head_response = requests.head(url, allow_redirects=True, timeout=10)
        if head_response.status_code != 200:
            print(f"Video not available (status code: {head_response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error checking video: {e}")
        return False

    # Prompt user to watch or skip
    user_choice = input("Video is available. Enter 'y' to watch, 'n' to skip, or 'q' to quit all: ").lower()
    if user_choice == 'q':
        print("Quitting the script.")
        exit()
    elif user_choice != 'y':
        print("Skipping this video.")
        return False

    # Download the video to a temporary file
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=1024):
                tmp_file.write(chunk)
        video_path = tmp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Failed to download video: {e}")
        return False

    # Play the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        os.unlink(video_path)
        return False

    print("Playing video... Press 'q' to stop playback.")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Video Ad', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    os.unlink(video_path)  # Clean up temporary file
    return True

# Iterate through each video URL
for video_url in video_urls:
    check_and_play_video(video_url)

print("\nAll videos processed.")
