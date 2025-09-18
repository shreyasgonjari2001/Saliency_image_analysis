import os
import requests
import json
import random
import mimetypes
import signal
import sys
from tiktok_data_parser import compute_final_score

# Configurable paths
file_path = "C:/Training/Projects/Saliency/insta_images.json"
api_url = "http://127.0.0.1:8000/analyze-image/"
update_api = "http://127.0.0.1:8000/update_metrics/"

# Signal handler for Ctrl+C to save before exit
def signal_handler(sig, frame):
    print("Interrupt received! Saving current progress...")
    if 'existing_updated' in globals() and existing_updated:  # Use the main list
        save_partial_progress([], output_filename)  # Save the current state (empty batch to trigger write)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def analyze_images_from_json(json_file_path, num_images_to_analyze=None):
    """
    Reads image URLs from a JSON file, shuffles them, sends them
    to the local API for analysis, and saves the analysis to a JSON file.
    """
    try:
        with open(json_file_path, "r") as f:
            ads = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON: {e}")
        return

    image_rows = ads.get("rows", [])
    random.shuffle(image_rows)

    if num_images_to_analyze is not None and num_images_to_analyze > 0:
        images_to_process = image_rows[:num_images_to_analyze]
    else:
        images_to_process = image_rows

    analysis_results = []
    for i, row in enumerate(images_to_process):
        try:
            image_url_suffix = row.get("image_url")
            if not image_url_suffix:
                continue
            full_image_url = f"https://static.poweradspy.com{image_url_suffix}"
            print(f"Processing image {i + 1}/{len(images_to_process)}: {full_image_url}")

            image_response = requests.get(full_image_url, timeout=10)
            image_response.raise_for_status()

            mime_type, _ = mimetypes.guess_type(image_url_suffix)
            if not mime_type:
                mime_type = "image/jpeg"

            files = {"file": (os.path.basename(image_url_suffix), image_response.content, mime_type)}
            api_response = requests.post(api_url, files=files)
            api_response.raise_for_status()

            result_entry = {
                "url": full_image_url,
                "analysis": api_response.json()
            }
            analysis_results.append(result_entry)
            print(f"Successfully processed image {i + 1}.")
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")

    if analysis_results:
        output_filename = "analysis_results.json"
        with open(output_filename, "w") as f:  # Overwrite with single valid JSON
            json.dump(analysis_results, f, indent=4)
        print(f"\nAnalysis complete. Results for {len(analysis_results)} images saved to {output_filename}")

def save_partial_progress(batch, filename):
    """Save batch by extending existing file's list and overwriting with valid JSON."""
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
        else:
            existing_data = []
        existing_data.extend(batch)
        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=4)
        print(f"Saved partial progress to {filename} with total {len(existing_data)} items.")
    except Exception as e:
        print(f"Error saving partial progress: {e}")

def update_dataset(analysis_results_path, output_filename="updated_analysis_results_1.json", batch_size=20):
    """
    Updates the original dataset with analysis results from update API,
    saving intermediate progress after each batch (resumable).
    """
    try:
        with open(analysis_results_path, "r") as f:
            ads = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input JSON: {e}")
        return

    # Load existing output for resuming
    global existing_updated  # For signal handler access
    existing_updated = []
    processed_urls = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                existing_updated = json.load(f)
            if not isinstance(existing_updated, list):
                print("Warning: Existing output is not a list; starting fresh.")
                existing_updated = []
            processed_urls = {item['url'] for item in existing_updated}
        except json.JSONDecodeError as e:
            print(f"Error loading existing output: {e}; starting fresh.")
            existing_updated = []

    batch = []
    for i, ad in enumerate(ads):
        url = ad.get("url")
        if not url or url in processed_urls:  # Skip already processed
            print(f"Skipping already processed {i+1}/{len(ads)}: {url}")
            continue

        try:
            print(f"Updating {i+1}/{len(ads)}: {url}")
            results = single_image(url, full_url=True)
            if "analysis" not in ad:
                ad["analysis"] = {}  # Initialize if missing
            ad["analysis"].update(results)
            batch.append(ad)

            if len(batch) >= batch_size:
                save_partial_progress(batch, output_filename)
                batch = []  # Clear batch
        except Exception as e:
            print(f"Error updating ad {i+1}: {e}")

    # Save any remaining batch
    if batch:
        save_partial_progress(batch, output_filename)

    print(f"Update complete. Saved to {output_filename} with {len(existing_updated)} items.")

def single_image(full_image_url, full_url=False):
    if not full_url:
        full_image_url = f"https://static.poweradspy.com{full_image_url}"

    image_response = requests.get(full_image_url, timeout=10)
    image_response.raise_for_status()

    mime_type, _ = mimetypes.guess_type(full_image_url)
    if not mime_type:
        mime_type = "image/jpeg"

    files = {"file": (os.path.basename(full_image_url), image_response.content, mime_type)}
    api_response = requests.post(update_api, files=files)
    api_response.raise_for_status()

    return api_response.json()


def single_image_local(local_image_path):
    """
    Upload a local image file to the API and return the JSON response.

    Parameters:
    - local_image_path (str): Path to the image file on your computer.

    Returns:
    - dict: JSON response from the API.
    """
    # Read the local image file
    with open(local_image_path, "rb") as f:
        image_content = f.read()

    # Guess MIME type based on file extension
    mime_type, _ = mimetypes.guess_type(local_image_path)
    if not mime_type:
        mime_type = "image/jpeg"  # Default to JPEG if unknown

    # Prepare the file for upload
    files = {"file": (os.path.basename(local_image_path), image_content, mime_type)}

    # Send POST request to the API
    api_response = requests.post(api_url, files=files)  # Assuming 'update_api' is defined elsewhere
    api_response.raise_for_status()  # Raise error if request fails

    return api_response.json()

if __name__ == "__main__":
    # analyze_images_from_json(file_path, num_images_to_analyze=2)
    update_dataset("updated_analysis_results.json")
    # data = {"response": "ok", "analysis": single_image_local(r"C:\Training\Projects\Saliency\ads\zomato1.jpg")}

    # print(compute_final_score(data))
    # print(data["analysis"])
