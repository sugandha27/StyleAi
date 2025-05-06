import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import traceback
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError
import io
import json # For potential pretty printing if needed
from typing import Dict, Optional, List, Union # Ensure all needed types are imported

# Import your existing class
try:
    # Assuming styleRe.py contains the updated FashionRecommender class
    from styleRe import FashionRecommender, DEFAULT_NUM_OUTFIT_OPTIONS
except ImportError as e:
    print(f"Error: Could not import from styleRe.py: {e}")
    exit()
except Exception as e:
    print(f"Error importing or during import of styleRe: {e}"); traceback.print_exc(); exit()

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../img"))
WOMEN_DATA_PATH = os.path.join(IMAGE_DATA_ROOT, "WOMEN")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "fashion_embeddings_cache.pt")

if not os.path.isdir(WOMEN_DATA_PATH): print(f"ERROR: Data path invalid: {WOMEN_DATA_PATH}"); exit()
if not os.path.isdir(IMAGE_DATA_ROOT): print(f"ERROR: Image root invalid: {IMAGE_DATA_ROOT}"); exit()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Allow all origins

# --- Initialize Recommender ---
print("Initializing Fashion Recommender...")
try:
    recommender = FashionRecommender(
        data_path=WOMEN_DATA_PATH,
        embeddings_cache_path=EMBEDDINGS_CACHE_PATH
    )
    print("Fashion Recommender Initialized Successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize FashionRecommender: {e}"); traceback.print_exc(); exit()

# --- Helper Function to Convert Local Paths to URLs ---
def create_image_url(local_path):
    """Converts a local file system path to a URL accessible via the API."""
    if not local_path or not isinstance(local_path, str): return None
    try:
        abs_local_path = os.path.abspath(local_path)
        abs_image_root = os.path.abspath(IMAGE_DATA_ROOT)
        if not abs_local_path.startswith(abs_image_root):
             print(f"Warning: Path '{local_path}' not in IMAGE_DATA_ROOT '{abs_image_root}'.")
             return None
        relative_path = os.path.relpath(abs_local_path, abs_image_root)
        url_path = relative_path.replace("\\", "/")
        if not url_path.startswith('/'): url_path = '/' + url_path
        return f"/images{url_path}"
    except Exception as e:
        print(f"Error creating URL for {local_path}: {e}")
        return None

# --- Helper Function to Process Dictionaries/Lists with Paths ---
def process_paths_to_urls(data: Union[Dict, List]) -> Optional[Union[Dict, List]]:
    """Recursively converts 'path' values in dictionaries or lists of dictionaries to URLs."""
    if isinstance(data, list):
        processed_list = []
        for item in data:
            processed_item = process_paths_to_urls(item) # Recurse for items in list
            if processed_item is None: return None # Propagate failure
            processed_list.append(processed_item)
        return processed_list
    elif isinstance(data, dict):
        processed_data = {}
        valid_structure = True
        skip_recursion_keys = [
            "input_broad_category", "input_subcategory", "input_confidence",
            "predicted_broad_category", "predicted_subcategory", "confidence_score",
            "score", "score_vs_overlay", "score_vs_top", "is_full_outfit",
            "overall_compatibility"
        ]
        for key, value in data.items():
            if key in skip_recursion_keys:
                processed_data[key] = value
                continue
            elif key == "path" and isinstance(value, str):
                url = create_image_url(value)
                if url is None: print(f"Warning: Failed to create URL for path: {value}"); valid_structure = False; break
                else: processed_data[key] = url
            else:
                # Recurse for nested dicts/lists
                processed_value = process_paths_to_urls(value)
                if processed_value is None and value is not None: # Check if processing failed
                     valid_structure = False; break
                processed_data[key] = processed_value
        return processed_data if valid_structure else None
    else:
        # Return non-dict/list types as is
        return data


# --- API Endpoints ---

@app.route('/recommendations', methods=['GET'])
def get_outfit_recommendations_api():
    """API endpoint for random compatible outfit recommendations from the database."""
    try:
        num_recommendations = request.args.get('n', default=3, type=int)
        if num_recommendations < 1: num_recommendations = 1
        print(f"Received request for {num_recommendations} random compatible recommendations.")

        raw_outfits = recommender.get_recommendations(n_recommendations=num_recommendations, use_compatibility=True)

        # Process the list of outfit dictionaries
        processed_outfits = process_paths_to_urls(raw_outfits)
        if processed_outfits is None:
             print("Warning: Failed processing URLs for some random outfits.")
             processed_outfits = [] # Return empty list on failure

        print(f"Returning {len(processed_outfits)} processed recommendations.")
        return jsonify(processed_outfits)

    except Exception as e:
        print(f"Error in /recommendations: {e}"); traceback.print_exc()
        return jsonify({"error": "Failed to generate recommendations", "details": str(e)}), 500


# MODIFIED: /recommend_similar_to_upload - Handle list of outfit options
@app.route('/recommend_similar_to_upload', methods=['POST'])
def recommend_similar_to_upload_api():
    """
    API endpoint for similarity search AND outfit pairing based on UPLOADED image.
    Returns classification, similar items, and a LIST of recommended outfit options.
    """
    print("\nReceived request on /recommend_similar_to_upload")
    try:
        # Get 'n' for similar items, add 'n_outfits' for outfit options
        num_similar = request.form.get('n', default=5, type=int)
        num_outfits = request.form.get('n_outfits', default=DEFAULT_NUM_OUTFIT_OPTIONS, type=int)
        if num_similar < 1: num_similar = 1
        if num_outfits < 1: num_outfits = 1

        if 'image' not in request.files: return jsonify({"error": "Missing 'image' file part"}), 400
        file = request.files['image']
        # ... (rest of file validation) ...
        if file.filename == '': return jsonify({"error": "No selected file"}), 400
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400

        print(f"  Processing uploaded file: {file.filename} (similar={num_similar}, outfits={num_outfits})")
        try:
            file_bytes = file.read()
            img_stream = io.BytesIO(file_bytes)
            received_bytes_count = len(file_bytes)
            print(f"  DEBUG: Received {received_bytes_count} bytes for file {file.filename}")
            if received_bytes_count == 0: return jsonify({"error": "Uploaded file is empty"}), 400

            # --- Unpack FIVE values now, including the LIST of outfit options ---
            predicted_broad_category, predicted_subcategory, confidence, similar_items_raw, recommended_outfits_list_raw = recommender.recommend_similar_to_new_image(
                new_image=img_stream,
                n_recommendations=num_similar,
                n_outfit_options=num_outfits # Pass the number of outfit options
            )
        except UnidentifiedImageError:
             return jsonify({"error": "Cannot identify image file. It might be corrupted or not an image."}), 400
        except Exception as img_proc_err:
            print(f"  Error processing image/getting recommendations: {img_proc_err}"); traceback.print_exc()
            return jsonify({"error": "Failed to process image/get recommendations", "details": str(img_proc_err)}), 500

        # --- Process similar items (List of Tuples) ---
        processed_similar_items = []
        if similar_items_raw:
            for item_path, score in similar_items_raw:
                url = create_image_url(item_path)
                if url: processed_similar_items.append({"url": url, "score": round(score, 4)})
                else: print(f"  Warning: Could not create URL for similar item: {item_path}")

        # --- Process recommended outfit options (List of Dictionaries) ---
        processed_recommended_outfits = []
        if recommended_outfits_list_raw:
             processed_recommended_outfits = process_paths_to_urls(recommended_outfits_list_raw)
             if processed_recommended_outfits is None:
                  print("  Warning: Failed to process URLs for some recommended outfit options.")
                  processed_recommended_outfits = [] # Reset to empty list on failure

        # --- Structure the final JSON response ---
        response_data = {
            "classification": {
                "predicted_broad_category": predicted_broad_category,
                "predicted_subcategory": predicted_subcategory,
                "confidence_score": round(confidence, 4) if confidence is not None else None
            },
            "similar_items": processed_similar_items,
            "recommended_outfits": processed_recommended_outfits # Use the processed list
        }

        print(f"  Returning {len(processed_similar_items)} similar items and {len(processed_recommended_outfits)} recommended outfit options.")
        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /recommend_similar_to_upload: {e}"); traceback.print_exc()
        return jsonify({"error": "Failed to get recommendations", "details": str(e)}), 500


# MODIFIED: /recommend_similar_to_url - Handle list of outfit options
@app.route('/recommend_similar_to_url', methods=['POST'])
def recommend_similar_to_url_api():
    """
    API endpoint for similarity search AND outfit pairing based on an IMAGE URL.
    Returns classification, similar items, and a LIST of recommended outfit options.
    Accepts JSON body: {"image_url": "/images/...", "n": 5, "n_outfits": 3}
    """
    print("\nReceived request on /recommend_similar_to_url")
    try:
        # --- 1. Parse JSON Input ---
        json_data = request.get_json()
        if not json_data: return jsonify({"error": "Invalid or missing JSON body"}), 400

        image_url = json_data.get('image_url')
        num_similar = json_data.get('n', 5) # Default for similar items
        num_outfits = json_data.get('n_outfits', DEFAULT_NUM_OUTFIT_OPTIONS) # Default for outfit options

        if not image_url or not isinstance(image_url, str): return jsonify({"error": "Missing or invalid 'image_url' in JSON body"}), 400
        if not isinstance(num_similar, int) or num_similar < 1: num_similar = 5
        if not isinstance(num_outfits, int) or num_outfits < 1: num_outfits = DEFAULT_NUM_OUTFIT_OPTIONS

        print(f"  Processing image URL: {image_url} (similar={num_similar}, outfits={num_outfits})")

        # --- 2. Convert URL to Local File Path ---
        url_prefix = "/images/"
        if not image_url.startswith(url_prefix): return jsonify({"error": f"Invalid 'image_url': Must start with {url_prefix}"}), 400
        relative_path = image_url[len(url_prefix):].strip('/')
        if not relative_path: return jsonify({"error": "Invalid 'image_url': Path part is empty"}), 400
        local_image_path = os.path.abspath(os.path.join(IMAGE_DATA_ROOT, relative_path))

        if not local_image_path.startswith(os.path.abspath(IMAGE_DATA_ROOT)):
             print(f"  Security Warning: Attempt to access path outside IMAGE_DATA_ROOT: {local_image_path}")
             return jsonify({"error": "Invalid image path specified"}), 400
        if not os.path.exists(local_image_path) or not os.path.isfile(local_image_path):
             print(f"  Error: Image file not found at derived path: {local_image_path}")
             return jsonify({"error": "Image specified by URL not found on server"}), 404
        print(f"  Derived local path: {local_image_path}")

        # --- 3. Call Recommender Logic ---
        try:
            # --- Unpack FIVE values now, including the LIST of outfit options ---
            predicted_broad_category, predicted_subcategory, confidence, similar_items_raw, recommended_outfits_list_raw = recommender.recommend_similar_to_new_image(
                new_image=local_image_path,
                n_recommendations=num_similar,
                n_outfit_options=num_outfits # Pass the number of outfit options
            )
        except UnidentifiedImageError:
             return jsonify({"error": "Cannot identify image file at derived path. It might be corrupted."}), 400
        except Exception as rec_err:
            print(f"  Error calling recommender logic: {rec_err}"); traceback.print_exc()
            return jsonify({"error": "Failed to get recommendations from recommender logic", "details": str(rec_err)}), 500

        # --- 4. Process Results ---
        # Process similar items
        processed_similar_items = []
        if similar_items_raw:
            for item_path, score in similar_items_raw:
                url = create_image_url(item_path)
                if url: processed_similar_items.append({"url": url, "score": round(score, 4)})
                else: print(f"  Warning: Could not create URL for similar item: {item_path}")

        # Process recommended outfit options (List of Dictionaries)
        processed_recommended_outfits = []
        if recommended_outfits_list_raw:
             processed_recommended_outfits = process_paths_to_urls(recommended_outfits_list_raw)
             if processed_recommended_outfits is None:
                  print("  Warning: Failed to process URLs for some recommended outfit options.")
                  processed_recommended_outfits = []

        # --- 5. Structure and Return Response ---
        response_data = {
            "classification": {
                "predicted_broad_category": predicted_broad_category,
                "predicted_subcategory": predicted_subcategory,
                "confidence_score": round(confidence, 4) if confidence is not None else None
            },
            "similar_items": processed_similar_items,
            "recommended_outfits": processed_recommended_outfits # Use the processed list
        }

        print(f"  Returning {len(processed_similar_items)} similar items and {len(processed_recommended_outfits)} recommended outfit options for URL.")
        return jsonify(response_data)

    except Exception as e:
        print(f"Error in /recommend_similar_to_url: {e}"); traceback.print_exc()
        return jsonify({"error": "Failed to process request", "details": str(e)}), 500


# --- /recommend_outfit_for_upload remains the same ---
@app.route('/recommend_outfit_for_upload', methods=['POST'])
def recommend_outfit_for_upload_api():
    """API endpoint to build ONE outfit around an uploaded image using the main dataset.
       Returns classification (broad/sub) of input and the recommended outfit parts.
    """
    # (Code remains the same as previous version)
    print("\nReceived request on /recommend_outfit_for_upload")
    try:
        if 'image' not in request.files: return jsonify({"error": "Missing 'image' file part"}), 400
        file = request.files['image']
        if file.filename == '': return jsonify({"error": "No selected file"}), 400
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400

        print(f"  Processing uploaded file: {file.filename} (for outfit building)")
        try:
            file_bytes = file.read()
            img_stream = io.BytesIO(file_bytes)
            received_bytes_count = len(file_bytes)
            print(f"  DEBUG: Received {received_bytes_count} bytes for file {file.filename}")
            if received_bytes_count == 0: return jsonify({"error": "Uploaded file is empty"}), 400

            # Returns dict including input_broad_category, input_subcategory, input_confidence
            outfit_result_raw = recommender.recommend_outfit_for_image(new_image=img_stream)

        except UnidentifiedImageError:
             return jsonify({"error": "Cannot identify image file. It might be corrupted or not an image."}), 400
        except Exception as outfit_err:
            print(f"  Error during outfit recommendation: {outfit_err}"); traceback.print_exc()
            return jsonify({"error": "Failed to generate outfit recommendation", "details": str(outfit_err)}), 500

        # Process Result
        if outfit_result_raw:
            print(f"  Raw outfit result: {outfit_result_raw}")

            # Extract classification info using the correct keys
            input_broad_cat = outfit_result_raw.get("input_broad_category")
            input_sub_cat = outfit_result_raw.get("input_subcategory")
            input_conf = outfit_result_raw.get("input_confidence")

            # Process the outfit parts for URLs using the generalized helper
            processed_outfit_parts = process_paths_to_urls(outfit_result_raw)

            if processed_outfit_parts is not None:
                # Construct the final response structure
                response_data = {
                    "classification": {
                        "predicted_broad_category": input_broad_cat,
                        "predicted_subcategory": input_sub_cat,
                        "confidence_score": input_conf
                    },
                    # Remove the input keys from the processed parts before sending
                    "recommended_outfit": {k: v for k, v in processed_outfit_parts.items() if k not in ["input_broad_category", "input_subcategory", "input_confidence"]}
                }
                print(f"  Processed outfit result with URLs: {response_data}")
                return jsonify(response_data)
            else:
                 print("  Outfit building failed during URL processing.")
                 return jsonify({"message": "Could not generate complete outfit recommendation (URL failure)."}), 500
        else:
            print("  Outfit building returned no compatible items.")
            return jsonify({"message": "Could not find compatible items to complete the outfit."}), 404

    except Exception as e:
        print(f"Error during outfit recommendation request: {e}"); traceback.print_exc()
        return jsonify({"error": "Failed to process outfit recommendation request", "details": str(e)}), 500


# --- Route to Serve Images ---
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serves image files statically from the IMAGE_DATA_ROOT."""
    try:
        return send_from_directory(IMAGE_DATA_ROOT, filename, as_attachment=False)
    except FileNotFoundError:
        print(f"Error: Image file not found at expected location: {os.path.join(IMAGE_DATA_ROOT, filename)}")
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error serving image {filename}: {e}"); traceback.print_exc()
        return jsonify({"error": "Failed to serve image"}), 500

# --- Run the App ---
if __name__ == '__main__':
    print(f"Starting Flask server on http://0.0.0.0:5001")
    print(f"Serving images from: {IMAGE_DATA_ROOT}")
    print("API Endpoints:")
    print("  GET /recommendations?n=<num> : Get compatible outfits from MAIN dataset (random start)")
    print("  POST /recommend_similar_to_upload (form-data: 'image', 'n', 'n_outfits') : Get similar items AND list of outfit options for UPLOAD") # Updated description
    print("  POST /recommend_similar_to_url (JSON: {'image_url': '/images/...', 'n': 5, 'n_outfits': 3}) : Get similar items AND list of outfit options for URL") # Updated description
    print("  POST /recommend_outfit_for_upload (form-data: 'image') : Build ONE outfit around uploaded image") # Clarified description
    print(f"Image URL format: /images/<category>/<item_folder>/<image_file.jpg>")
    app.run(debug=True, host='0.0.0.0', port=5001)
