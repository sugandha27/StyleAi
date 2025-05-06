import os
import random
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import torch.nn.functional as F
import glob as glob
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import time
import io
import traceback
import json

# --- Constants ---
DEFAULT_EMBEDDINGS_CACHE_FILE = "image_embeddings_cache.pt"
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.3
# --- NEW: Default number of outfit options to generate ---
DEFAULT_NUM_OUTFIT_OPTIONS = 3

class FashionRecommender:
    # --- __init__ and other setup methods remain the same ---
    def __init__(self, data_path: str, embeddings_cache_path: Optional[str] = None):
        """
        Initialize the fashion recommendation system.
        Loads models, precomputes text embeddings, loads/generates image embeddings.
        """
        self.data_path = data_path
        if embeddings_cache_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.embeddings_cache_path = os.path.join(script_dir, DEFAULT_EMBEDDINGS_CACHE_FILE)
        else:
            self.embeddings_cache_path = embeddings_cache_path

        print(f"Using embeddings cache path: {self.embeddings_cache_path}")

        # Define category mappings (Broad Category -> List of Subcategories)
        self.category_mappings = {
            "category1": ["Dresses", "Rompers_Jumpsuits"],
            "category2": ["Pants", "Leggings", "Shorts", "Skirts", "Denim"],
            "category3": ["Blouses_Shirts", "Sweatshirts_Hoodies", "Graphic_Tees", "Tees_Tanks"],
            "category4": ["Jackets_Coats", "Cardigans", "Sweaters"]
        }

        # Define Text Descriptions for EACH Subcategory
        self.subcategory_text_descriptions = {
            # Category 1
            "Dresses": "a photo of a dress, a one-piece garment for the upper and lower body",
            "Rompers_Jumpsuits": "a photo of a romper or jumpsuit, a one-piece garment combining top and shorts or pants",
            # Category 2
            "Pants": "a photo of pants or trousers, worn on the lower body covering both legs separately",
            "Leggings": "a photo of leggings, tight-fitting stretch pants",
            "Shorts": "a photo of shorts, worn on the lower body covering the upper part of the legs",
            "Skirts": "a photo of a skirt, worn on the lower body hanging down from the waist",
            "Denim": "a photo of denim jeans or a denim skirt or denim shorts",
            # Category 3
            "Blouses_Shirts": "a photo of a blouse or button-up shirt, worn on the upper body",
            "Sweatshirts_Hoodies": "a photo of a sweatshirt or hoodie, a heavy upper body garment",
            "Graphic_Tees": "a photo of a graphic t-shirt, a tee with a design or print",
            "Tees_Tanks": "a photo of a t-shirt or tank top, a simple upper body garment",
            # Category 4
            "Jackets_Coats": "a photo of a jacket or coat, outerwear worn over other clothes",
            "Cardigans": "a photo of a cardigan sweater, an open-front knitted garment",
            "Sweaters": "a photo of a sweater or pullover, a knitted upper body garment"
        }
        # Validate that all subcategories in mappings have descriptions
        all_subcats_in_mappings = [sub for subs in self.category_mappings.values() for sub in subs]
        if set(all_subcats_in_mappings) != set(self.subcategory_text_descriptions.keys()):
            missing = set(all_subcats_in_mappings) - set(self.subcategory_text_descriptions.keys())
            extra = set(self.subcategory_text_descriptions.keys()) - set(all_subcats_in_mappings)
            raise ValueError(f"Mismatch between subcategories in mappings and descriptions. Missing: {missing}, Extra: {extra}")

        # Create Reverse Mapping (Subcategory -> Broad Category)
        self.subcategory_to_category_map = {}
        for broad_cat, sub_cats in self.category_mappings.items():
            for sub_cat in sub_cats:
                self.subcategory_to_category_map[sub_cat] = broad_cat

        # Load models
        print("Loading CLIP model...")
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print(f"Models loaded in {time.time() - start_time:.2f} seconds.")

        # Precompute Text Embeddings for Subcategories
        self._precompute_subcategory_text_embeddings()

        # Load or Generate Image Embeddings
        init_start_time = time.time()
        if os.path.exists(self.embeddings_cache_path):
            print(f"Loading image embeddings from cache: {self.embeddings_cache_path}...")
            try:
                cache_data = torch.load(self.embeddings_cache_path, map_location='cpu')
                self.image_database = cache_data['image_database']
                self.image_embeddings = cache_data['image_embeddings']
                print(f"Image embeddings loaded successfully in {time.time() - init_start_time:.2f} seconds.")
                if not self._validate_loaded_data():
                     print("Warning: Loaded data structure seems invalid. Regenerating embeddings.")
                     self._initialize_data_and_embeddings()
            except Exception as e:
                print(f"Error loading image embeddings from cache: {e}. Regenerating...")
                self._initialize_data_and_embeddings()
        else:
            print("Image embeddings cache not found. Generating new embeddings...")
            self._initialize_data_and_embeddings()
            self._save_embeddings()

        print(f"Total initialization time (models, text & image embeddings): {time.time() - start_time:.2f} seconds.")


    def _precompute_subcategory_text_embeddings(self):
        """Generates and stores embeddings for the subcategory text descriptions."""
        print("Precomputing text embeddings for subcategories...")
        start_time = time.time()
        self.subcategory_text_embeddings = {} # Store subcategory embeddings here
        # Get all unique subcategory names in a defined order
        self.ordered_subcategory_names = sorted(list(self.subcategory_text_descriptions.keys()))
        descriptions = [self.subcategory_text_descriptions[name] for name in self.ordered_subcategory_names]

        if not descriptions:
             print("Warning: No subcategory descriptions found to precompute.")
             return

        try:
            inputs = self.clip_processor(text=descriptions, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs) # Shape: [num_subcategories, embedding_dim]

            # Store embeddings on CPU, mapped by subcategory name
            for i, name in enumerate(self.ordered_subcategory_names):
                self.subcategory_text_embeddings[name] = text_features[i].cpu()

            print(f"Subcategory text embeddings computed ({len(self.subcategory_text_embeddings)} subcategories) in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error precomputing subcategory text embeddings: {e}")
            traceback.print_exc()
            self.subcategory_text_embeddings = None # Indicate failure


    def _initialize_data_and_embeddings(self):
        """Loads image database, generates image embeddings (stored on CPU after generation)."""
        start_time = time.time()
        self.image_database = self._load_image_database()
        generated_embeddings = self._generate_embeddings()
        self.image_embeddings = {}
        for cat, subcats in generated_embeddings.items():
            self.image_embeddings[cat] = {}
            for subcat, embeds in subcats.items():
                 self.image_embeddings[cat][subcat] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in embeds.items()}
        print(f"Data loading and image embedding generation (and move to CPU) took {time.time() - start_time:.2f} seconds.")

    def _save_embeddings(self):
        """Saves the image database structure and CPU image embeddings to the cache file."""
        print(f"Saving image embeddings (CPU) to cache: {self.embeddings_cache_path}...")
        start_time = time.time()
        try:
            cache_data = {
                'image_database': self.image_database,
                'image_embeddings': self.image_embeddings
            }
            torch.save(cache_data, self.embeddings_cache_path)
            print(f"Image embeddings saved successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"Error saving image embeddings cache: {e}")
            traceback.print_exc()

    def _validate_loaded_data(self) -> bool:
        """Basic validation of loaded cache data."""
        if not isinstance(self.image_database, dict) or not isinstance(self.image_embeddings, dict):
            print("Validation Error: Loaded data is not dictionary type.")
            return False
        try:
            if self.image_embeddings:
                first_cat = next(iter(self.image_embeddings))
                if self.image_embeddings[first_cat]:
                    first_subcat = next(iter(self.image_embeddings[first_cat]))
                    first_embed_dict = self.image_embeddings[first_cat][first_subcat]
                    if first_embed_dict:
                        first_embed = next(iter(first_embed_dict.values()))
                        if not isinstance(first_embed, torch.Tensor):
                            print("Validation Error: Embeddings in cache are not Torch tensors.")
                            return False
        except StopIteration: pass
        except Exception as e: print(f"Validation Error during embedding check: {e}"); return False
        print("Loaded data basic validation passed.")
        return True

    def _load_image_database(self) -> Dict[str, Dict[str, List[str]]]:
        """Load all images from the dataset into a dictionary."""
        image_db: Dict[str, Dict[str, List[str]]] = {}
        print("Loading image database structure...")
        for category_name, subcategory_folders in self.category_mappings.items():
            image_db[category_name] = {}
            for subcategory_folder in subcategory_folders:
                subcategory_folder_path = os.path.join(self.data_path, subcategory_folder)
                image_db[category_name][subcategory_folder] = []
                if os.path.exists(subcategory_folder_path) and os.path.isdir(subcategory_folder_path):
                    for item_folder_name in os.listdir(subcategory_folder_path):
                        item_folder_path = os.path.join(subcategory_folder_path, item_folder_name)
                        if os.path.isdir(item_folder_path):
                            image_files_in_item_folder = []
                            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                                image_files_in_item_folder.extend(glob.glob(os.path.join(item_folder_path, ext)))
                            if image_files_in_item_folder:
                                image_db[category_name][subcategory_folder].extend(image_files_in_item_folder)
        print("Finished loading image database structure.")
        return image_db

    def _generate_embeddings(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """Generate embeddings for all images in the database using self.device."""
        embeddings: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        print(f"Generating image embeddings on device: {self.device}...")
        total_images = 0; processed_images = 0
        for folders in self.image_database.values():
            for img_paths in folders.values(): total_images += len(img_paths)
        print(f"Found {total_images} images to process.")
        if total_images == 0: return embeddings
        start_gen_time = time.time()
        for category, folders in self.image_database.items():
            embeddings[category] = {}
            for folder, img_paths in folders.items():
                embeddings[category][folder] = {}
                if not img_paths: continue
                for img_path in img_paths:
                    try:
                        with Image.open(img_path) as img: image = img.convert("RGB")
                        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            image_features = self.clip_model.get_image_features(**inputs)
                        embeddings[category][folder][img_path] = image_features.squeeze(0)
                        processed_images += 1
                        if processed_images % 200 == 0 or processed_images == total_images:
                            elapsed = time.time() - start_gen_time
                            rate = processed_images / elapsed if elapsed > 0 else 0
                            print(f"      Processed {processed_images}/{total_images} images... ({elapsed:.1f}s, {rate:.1f} img/s)")
                    except FileNotFoundError: print(f"      Error: Image file not found {img_path}. Skipping.")
                    except UnidentifiedImageError: print(f"      Error: Cannot identify image file {img_path}. Skipping.")
                    except Exception as e: print(f"      Error processing {img_path}: {e}. Skipping.")
        print(f"Finished generating embeddings for {processed_images} images in {(time.time() - start_gen_time):.2f} seconds.")
        if processed_images != total_images: print(f"Warning: Processed count ({processed_images}) != expected total ({total_images}).")
        return embeddings

    def _compute_compatibility_score(self, img1_embedding: torch.Tensor, img2_embedding: torch.Tensor) -> float:
        """Compute compatibility score (cosine similarity mapped to [0, 1]) between two embeddings."""
        emb1 = img1_embedding.squeeze().to(self.device)
        emb2 = img2_embedding.squeeze().to(self.device)
        if emb1.ndim != 1 or emb2.ndim != 1: return 0.0
        try:
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
            similarity = max(-1.0, min(1.0, similarity))
            compatibility = (similarity + 1) / 2
        except Exception as e: print(f"Error calculating compatibility: {e}. Returning 0."); compatibility = 0.0
        return compatibility

    def classify_new_image(self, new_image: Union[str, io.BytesIO, Image.Image]) -> Optional[Tuple[str, str, float]]:
        """
        Classifies a new input image into a specific subcategory and its broad category.
        Returns: (broad_category, sub_category, confidence_score) or None.
        """
        print("\n--- DEBUG: Entering classify_new_image (Subcategory Version) ---") # DEBUG START
        start_time = time.time()

        if not hasattr(self, 'subcategory_text_embeddings') or not self.subcategory_text_embeddings:
            print("  DEBUG (classify): ERROR - subcategory_text_embeddings attribute not found or empty!")
            return None
        if not hasattr(self, 'ordered_subcategory_names') or not self.ordered_subcategory_names:
             print("  DEBUG (classify): ERROR - ordered_subcategory_names attribute not found or empty!")
             return None

        try:
            # --- 1. Load/Preprocess Image ---
            image = None
            if isinstance(new_image, str):
                if not os.path.exists(new_image): print(f"Error: Image path does not exist: {new_image}"); return None
                image = Image.open(new_image).convert("RGB")
                print(f"  DEBUG (classify): Loaded image from path, size: {image.size}")
            elif hasattr(new_image, 'read'):
                 if isinstance(new_image, io.BytesIO): new_image.seek(0)
                 image = Image.open(new_image).convert("RGB")
                 print(f"  DEBUG (classify): Loaded image from stream, size: {image.size}")
            elif isinstance(new_image, Image.Image):
                 image = new_image.convert("RGB")
                 print(f"  DEBUG (classify): Using provided PIL image, size: {image.size}")
            else:
                 print(f"Error: Unsupported input type: {type(new_image)}"); return None

            # --- 2. Generate Image Embedding ---
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_embedding = self.clip_model.get_image_features(**inputs) # Shape: [1, embedding_dim]
            print(f"  DEBUG (classify): Image Embedding Shape: {image_embedding.shape}")
            print(f"  DEBUG (classify): Image Embedding Sum: {image_embedding.sum().item():.4f}")

            # --- 3. Compare with ALL Subcategory Text Embeddings ---
            image_embedding_norm = F.normalize(image_embedding, p=2, dim=1)
            text_embeddings_list = [self.subcategory_text_embeddings[name] for name in self.ordered_subcategory_names]

            print("  DEBUG (classify): Text Embeddings Check (brief):")
            all_text_embeddings_valid = True
            for i, name in enumerate(self.ordered_subcategory_names):
                if text_embeddings_list[i] is None or torch.all(text_embeddings_list[i] == 0):
                    print(f"    ERROR/WARNING: Text embedding for '{name}' is None or all zeros!")
                    all_text_embeddings_valid = False
            if not all_text_embeddings_valid:
                 print("  DEBUG (classify): Aborting due to invalid text embeddings.")
                 return None

            text_embeddings_tensor = torch.stack(text_embeddings_list).to(self.device)
            text_embeddings_norm = F.normalize(text_embeddings_tensor, p=2, dim=1)

            similarities = torch.mm(image_embedding_norm, text_embeddings_norm.T)
            similarities_list = similarities.squeeze(0).cpu().tolist()

            print(f"  DEBUG (classify): Raw Similarities (Image vs Subcategory Text):")
            sorted_sims = sorted(zip(self.ordered_subcategory_names, similarities_list), key=lambda x: x[1], reverse=True)
            for i, (name, score) in enumerate(sorted_sims[:5]):
                print(f"    {i+1}. {name}: {score:.4f}")

            # --- 4. Determine Prediction ---
            best_sim_index = np.argmax(similarities_list)
            predicted_subcategory = self.ordered_subcategory_names[best_sim_index]
            predicted_broad_category = self.subcategory_to_category_map.get(predicted_subcategory, "Unknown")

            probabilities = F.softmax(similarities.squeeze(0), dim=0)
            confidence_score = probabilities[best_sim_index].item()

            print(f"  DEBUG (classify): ArgMax Prediction: {predicted_subcategory} (Broad: {predicted_broad_category}, Index: {best_sim_index})")
            print(f"  DEBUG (classify): Softmax Confidence: {confidence_score:.4f}")

            print(f"--- DEBUG: Exiting classify_new_image (Success) ---")
            return predicted_broad_category, predicted_subcategory, confidence_score

        except UnidentifiedImageError:
            print(f"Error: Cannot identify image from input. It might be corrupted or not an image.")
            print(f"--- DEBUG: Exiting classify_new_image (Error) ---")
            return None
        except Exception as e:
            print(f"Error during classification: {e}"); traceback.print_exc()
            print(f"--- DEBUG: Exiting classify_new_image (Error) ---")
            return None

    # --- NEW HELPER: _find_top_n_compatible_items ---
    def _find_top_n_compatible_items(self,
                                     target_embedding: torch.Tensor,
                                     target_category_key: str,
                                     n: int) -> List[Tuple[str, str, torch.Tensor, float]]:
        """
        Finds the top N items in the target BROAD category most compatible with the target embedding.
        Target embedding is assumed to be on device. Returns list containing (path, subcat, CPU embedding, score).
        """
        all_items = []
        if target_category_key not in self.image_embeddings:
            print(f"Warning: Target category '{target_category_key}' not found in embeddings for top N search.")
            return []

        # target_embedding is already on device
        for subcategory, embeddings_dict in self.image_embeddings[target_category_key].items():
            for img_path, embedding_cpu in embeddings_dict.items(): # Embeddings here are on CPU
                try:
                    # Score function moves embedding_cpu to device for comparison
                    score = self._compute_compatibility_score(target_embedding, embedding_cpu)
                    all_items.append((img_path, subcategory, embedding_cpu, score)) # Store CPU embedding
                except Exception as e:
                    print(f"    Error scoring item {img_path} for top N: {e}")

        # Sort by score (descending) and return top N
        all_items.sort(key=lambda x: x[3], reverse=True)
        return all_items[:n]

    # --- MODIFIED: recommend_similar_to_new_image ---
    def recommend_similar_to_new_image(self,
                                       new_image: Union[str, io.BytesIO, Image.Image],
                                       n_recommendations: int = 5,
                                       n_outfit_options: int = DEFAULT_NUM_OUTFIT_OPTIONS # Add param for outfit options
                                       ) -> Tuple[Optional[str], Optional[str], Optional[float], List[Tuple[str, float]], List[Dict]]: # MODIFIED Return Type
        """
        Recommends items similar to the input image AND a LIST of outfit options.
        - Similarity search is ONLY within the predicted broad category (unless fallback).
        - Outfit recommendations are based on the predicted broad category.

        Returns:
            A tuple containing:
            - Predicted broad category name (str) or None.
            - Predicted sub category name (str) or None.
            - Confidence score (float) or None.
            - List of tuples for similar items: [(similar_item_path, similarity_score), ...]
            - LIST of Dictionaries for recommended outfit options: [{"top": {...}, "bottom": {...}}, ...]
        """
        print(f"\nReceived request for similar items + {n_outfit_options} outfit options for new image (n_similar={n_recommendations}).")
        start_time = time.time(); search_all_categories = False
        predicted_broad_category: Optional[str] = None
        predicted_subcategory: Optional[str] = None
        confidence: Optional[float] = None
        similar_items_raw: List[Tuple[str, float]] = []
        recommended_outfits_list: List[Dict] = [] # Initialize empty LIST for outfit options

        try:
            # --- 1. Load Image & Generate Embedding ---
            img_bytes = None
            if isinstance(new_image, str):
                if not os.path.exists(new_image): print(f"Error: Image path does not exist: {new_image}"); return None, None, None, [], []
                image = Image.open(new_image).convert("RGB")
                print(f"  DEBUG: Loaded image from path, size: {image.size}")
            elif hasattr(new_image, 'read'):
                img_bytes = io.BytesIO(new_image.read())
                if img_bytes.getbuffer().nbytes == 0: print("Error: Input stream is empty in recommender."); return None, None, None, [], []
                img_bytes.seek(0)
                image = Image.open(img_bytes).convert("RGB")
                print(f"  DEBUG: Loaded image from stream, size: {image.size}")
            elif isinstance(new_image, Image.Image):
                image = new_image.convert("RGB")
                print(f"  DEBUG: Using provided PIL image, size: {image.size}")
            else: print(f"Error: Unsupported input type: {type(new_image)}"); return None, None, None, [], []

            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.get('pixel_values')
            if pixel_values is not None: print(f"  DEBUG: Processor output sum: {pixel_values.sum().item():.4f}")

            with torch.no_grad():
                new_image_embedding = self.clip_model.get_image_features(**inputs).squeeze(0) # Keep on device
            print(f"  DEBUG: New image embedding sum: {new_image_embedding.sum().item():.4f}")

            # --- 2. Classify the New Image (gets broad, sub, conf) ---
            print("  Classifying image...")
            if img_bytes: img_bytes.seek(0)
            classification_input = img_bytes if img_bytes else image
            classification_result = self.classify_new_image(classification_input)

            if classification_result:
                predicted_broad_category, predicted_subcategory, confidence = classification_result
                print(f"  Predicted Broad Category: '{predicted_broad_category}', Subcategory: '{predicted_subcategory}' (Confidence: {confidence:.4f})")
                if confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                    print(f"  Confidence low. Similarity search will use all categories.")
                    search_all_categories = True
                elif predicted_broad_category not in self.image_embeddings:
                     print(f"  Predicted broad category '{predicted_broad_category}' not in image embeddings structure. Similarity search will use all.")
                     search_all_categories = True
                     predicted_broad_category = None # Prevent outfit pairing attempt
            else:
                print("  Classification failed. Similarity search will use all categories. Outfit pairing skipped.")
                search_all_categories = True
                predicted_broad_category = None
                predicted_subcategory = None
                confidence = None
            print(f"  DEBUG: Classification Result: Broad={predicted_broad_category}, Sub={predicted_subcategory}, Conf={confidence}, SearchAll={search_all_categories}")

            # --- 3. Perform Similarity Search ---
            print("\n  --- Starting Similarity Search ---")
            similarities = []; total_compared = 0; comparison_start_time = time.time()
            comparison_scores_sample = []

            target_categories = []
            if search_all_categories:
                target_categories = list(self.image_embeddings.keys())
                print("  Comparing similarity against items in ALL categories...")
            elif predicted_broad_category:
                target_categories = [predicted_broad_category]
                print(f"  Comparing similarity ONLY against items in predicted broad category '{predicted_broad_category}'...")
            else:
                print("  Error/Warning: No broad category for similarity search? Falling back to search ALL categories.")
                target_categories = list(self.image_embeddings.keys())
            print(f"  Final target categories for similarity search: {target_categories}")

            for category in target_categories:
                if category not in self.image_embeddings: continue
                for subcategory, embeddings_dict in self.image_embeddings[category].items():
                    for item_path, cached_embedding in embeddings_dict.items():
                        try:
                            score = self._compute_compatibility_score(new_image_embedding, cached_embedding)
                            similarities.append((item_path, score))
                            if len(comparison_scores_sample) < 5: comparison_scores_sample.append(round(score, 4))
                            total_compared += 1
                        except Exception as e: print(f"    Error comparing similarity with {item_path}: {e}")

            print(f"  Compared similarity against {total_compared} items in {time.time() - comparison_start_time:.2f} seconds.")
            print(f"  DEBUG: Sample similarity scores: {comparison_scores_sample}")

            if similarities:
                similarities.sort(key=lambda x: x[1], reverse=True)
                similar_items_raw = similarities[:n_recommendations] # Get top N similar items
                top_scores = [round(s[1], 4) for s in similar_items_raw]
                print(f"  Found {len(similar_items_raw)} similar items. Top scores: {top_scores}")
            else:
                print("  No similar items found.")
                similar_items_raw = []


            # --- 4. Perform Outfit Recommendation Logic (Generate LIST of Options) ---
            print(f"\n  --- Starting Outfit Pairing (finding {n_outfit_options} options) ---")
            if predicted_broad_category and predicted_broad_category in ["category1", "category2", "category3", "category4"]:
                try:
                    if predicted_broad_category == "category3": # Input: Top -> Find N Bottoms + Best Overlay for each
                        print(f"    Finding Top {n_outfit_options} compatible BOTTOMs (category2)...")
                        top_n_bottoms = self._find_top_n_compatible_items(new_image_embedding, "category2", n_outfit_options)
                        if not top_n_bottoms: print("      Could not find any compatible bottoms.")
                        else:
                            print(f"      Found {len(top_n_bottoms)} compatible bottom(s). Finding overlays...")
                            for b_path, b_type, b_embedding_cpu, b_score in top_n_bottoms:
                                outfit_option = {
                                    "recommended_bottom": {"path": b_path, "type": b_type, "score": round(b_score, 4)}
                                }
                                # Find the single best overlay compatible with input top AND this specific bottom
                                best_overlay = self._find_best_compatible_overlay(new_image_embedding, b_embedding_cpu)
                                if best_overlay:
                                    o_path, o_type, _, o_score = best_overlay
                                    outfit_option["recommended_overlay"] = {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                                    print(f"        Found Overlay for Bottom '{os.path.basename(b_path)}': {o_type} (Score: {o_score:.4f})")
                                recommended_outfits_list.append(outfit_option)

                    elif predicted_broad_category == "category2": # Input: Bottom -> Find N Tops + Best Overlay for each
                        print(f"    Finding Top {n_outfit_options} compatible TOPs (category3)...")
                        top_n_tops = self._find_top_n_compatible_items(new_image_embedding, "category3", n_outfit_options)
                        if not top_n_tops: print("      Could not find any compatible tops.")
                        else:
                            print(f"      Found {len(top_n_tops)} compatible top(s). Finding overlays...")
                            for t_path, t_type, t_embedding_cpu, t_score in top_n_tops:
                                outfit_option = {
                                    "recommended_top": {"path": t_path, "type": t_type, "score": round(t_score, 4)}
                                }
                                # Find the single best overlay compatible with input bottom AND this specific top
                                best_overlay = self._find_best_compatible_overlay(t_embedding_cpu, new_image_embedding)
                                if best_overlay:
                                    o_path, o_type, _, o_score = best_overlay
                                    outfit_option["recommended_overlay"] = {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                                    print(f"        Found Overlay for Top '{os.path.basename(t_path)}': {o_type} (Score: {o_score:.4f})")
                                recommended_outfits_list.append(outfit_option)

                    elif predicted_broad_category == "category1": # Input: Full Outfit -> Find N Overlays
                        print(f"    Finding Top {n_outfit_options} compatible OVERLAYs (category4)...")
                        top_n_overlays = self._find_top_n_compatible_items(new_image_embedding, "category4", n_outfit_options)
                        if not top_n_overlays: print("      Could not find any compatible overlays.")
                        else:
                            print(f"      Found {len(top_n_overlays)} compatible overlay(s).")
                            for o_path, o_type, _, o_score in top_n_overlays:
                                outfit_option = {
                                    "recommended_overlay": {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                                }
                                recommended_outfits_list.append(outfit_option)

                    elif predicted_broad_category == "category4": # Input: Overlay -> Find N Full Outfits OR N (Top+Bottom) pairs
                        # Option 1: Find N Full Outfits
                        print(f"    Finding Top {n_outfit_options} compatible FULL OUTFITs (category1)...")
                        top_n_full = self._find_top_n_compatible_items(new_image_embedding, "category1", n_outfit_options)
                        if top_n_full:
                             print(f"      Found {len(top_n_full)} compatible full outfit(s).")
                             for f_path, f_type, _, f_score in top_n_full:
                                  outfit_option = {
                                       "recommended_base_full": {"path": f_path, "type": f_type, "score": round(f_score, 4), "is_full_outfit": True}
                                  }
                                  recommended_outfits_list.append(outfit_option)
                        # Option 2 (Fallback or Alternative): Find N Tops, then best Bottom for each
                        # Limit total options generated if both are found
                        remaining_options = n_outfit_options - len(recommended_outfits_list)
                        if remaining_options > 0:
                             print(f"    Fallback/Also Finding Top {remaining_options} compatible TOPs (category3)...")
                             top_n_tops = self._find_top_n_compatible_items(new_image_embedding, "category3", remaining_options)
                             if not top_n_tops: print("      Could not find any compatible tops.")
                             else:
                                  print(f"      Found {len(top_n_tops)} compatible top(s). Finding bottoms...")
                                  for t_path, t_type, t_embedding_cpu, t_score in top_n_tops:
                                       outfit_option = {
                                            "recommended_top": {"path": t_path, "type": t_type, "score_vs_overlay": round(t_score, 4)}
                                       }
                                       # Find the single best bottom compatible with THIS top
                                       best_bottom = self._find_best_compatible_item(t_embedding_cpu.to(self.device), "category2")
                                       if best_bottom:
                                            b_path, b_type, _, b_score = best_bottom
                                            outfit_option["recommended_bottom"] = {"path": b_path, "type": b_type, "score_vs_top": round(b_score, 4)}
                                            print(f"        Found Bottom for Top '{os.path.basename(t_path)}': {b_type} (Score: {b_score:.4f})")
                                       recommended_outfits_list.append(outfit_option)

                except Exception as e:
                    print(f"    Error during outfit pairing logic: {e}"); traceback.print_exc()
            else:
                print("  Skipping outfit pairing due to failed classification or invalid category.")


            # --- 5. Return Results ---
            print(f"\nTotal time for similarity + outfit pairing: {time.time() - start_time:.2f} seconds.")
            # Return broad_cat, sub_cat, confidence, similar_items list, and LIST of outfit dicts
            return predicted_broad_category, predicted_subcategory, confidence, similar_items_raw, recommended_outfits_list

        except UnidentifiedImageError:
            print(f"Error: Cannot identify image from input. It might be corrupted or not an image.")
            return None, None, None, [], [] # Return empty lists
        except Exception as e:
            print(f"Error during similarity/outfit recommendation: {e}"); traceback.print_exc()
            return None, None, None, [], [] # Return empty lists


    # --- recommend_outfit_for_image remains the same ---
    def recommend_outfit_for_image(self,
                                   new_image: Union[str, io.BytesIO, Image.Image],
                                   include_overlay: bool = True) -> Optional[Dict]:
        """
        Recommends compatible outfit items from the database for a given input image.
        Returns a dictionary including input classification (broad, sub, conf) and recommendations.
        (Code is identical to the previous complete version)
        """
        print(f"\nReceived request to build outfit around new image.")
        start_time = time.time()

        # --- 1. Load Image ---
        img_bytes = None
        try:
            if isinstance(new_image, str):
                if not os.path.exists(new_image): print(f"Error: Image path does not exist: {new_image}"); return None
                image = Image.open(new_image).convert("RGB")
                print(f"  DEBUG: Loaded image from path, size: {image.size}")
            elif hasattr(new_image, 'read'):
                img_bytes = io.BytesIO(new_image.read())
                if img_bytes.getbuffer().nbytes == 0: print("Error: Input stream is empty in recommender."); return None
                img_bytes.seek(0)
                image = Image.open(img_bytes).convert("RGB")
                print(f"  DEBUG: Loaded image from stream, size: {image.size}")
            elif isinstance(new_image, Image.Image):
                 image = new_image.convert("RGB")
                 print(f"  DEBUG: Using provided PIL image, size: {image.size}")
            else: print(f"Error: Unsupported input type: {type(new_image)}"); return None
        except UnidentifiedImageError: print(f"Error: Cannot identify image from input."); return None
        except Exception as e: print(f"Error loading input image: {e}"); traceback.print_exc(); return None

        # --- 2. Classify the Input Image (gets broad, sub, conf) ---
        print("  Classifying input image...")
        if img_bytes: img_bytes.seek(0)
        classification_input = img_bytes if img_bytes else image
        classification_result = self.classify_new_image(classification_input)

        if not classification_result: print("  Classification failed."); return None
        input_broad_category, input_subcategory, confidence = classification_result
        print(f"  Input classified as: Broad='{input_broad_category}', Sub='{input_subcategory}' (Confidence: {confidence:.4f})")
        print(f"  DEBUG: Classification Result: Broad={input_broad_category}, Sub={input_subcategory}, Conf={confidence}")
        if confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            print(f"  Classification confidence too low."); return None

        # --- 3. Generate Embedding for the Input Image ---
        print("  Generating embedding for input image...")
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.get('pixel_values')
            if pixel_values is not None: print(f"  DEBUG: Processor output sum: {pixel_values.sum().item():.4f}")

            with torch.no_grad():
                input_embedding = self.clip_model.get_image_features(**inputs).squeeze(0)
            print(f"  DEBUG: Input embedding sum: {input_embedding.sum().item():.4f}")
        except Exception as e: print(f"Error generating embedding: {e}"); traceback.print_exc(); return None

        # --- 4. Find Compatible Items Based on BROAD Category ---
        outfit_recommendation = {
            "input_broad_category": input_broad_category,
            "input_subcategory": input_subcategory,
            "input_confidence": round(confidence, 4)
        }
        found_compatible = False
        try:
            if input_broad_category == "category3": # Input: Top -> Find Bottom + Overlay
                print("  Finding compatible BOTTOM (category2)...")
                best_bottom = self._find_best_compatible_item(input_embedding, "category2")
                if best_bottom:
                    b_path, b_type, b_embedding_cpu, b_score = best_bottom
                    outfit_recommendation["recommended_bottom"] = {"path": b_path, "type": b_type, "score": round(b_score, 4)}
                    print(f"    Found Bottom: {b_type} (Score: {b_score:.4f})"); found_compatible = True
                    print(f"    DEBUG: Best Bottom Score: {b_score:.4f}")
                    if include_overlay:
                        print("    Finding compatible OVERLAY (category4)...")
                        best_overlay = self._find_best_compatible_overlay(input_embedding, b_embedding_cpu)
                        if best_overlay:
                            o_path, o_type, _, o_score = best_overlay
                            outfit_recommendation["recommended_overlay"] = {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                            print(f"      Found Overlay: {o_type} (Score: {o_score:.4f})")
                            print(f"      DEBUG: Best Overlay Score: {o_score:.4f}")
                else: print("    Could not find compatible bottom.")

            elif input_broad_category == "category2": # Input: Bottom -> Find Top + Overlay
                print("  Finding compatible TOP (category3)...")
                best_top = self._find_best_compatible_item(input_embedding, "category3")
                if best_top:
                    t_path, t_type, t_embedding_cpu, t_score = best_top
                    outfit_recommendation["recommended_top"] = {"path": t_path, "type": t_type, "score": round(t_score, 4)}
                    print(f"    Found Top: {t_type} (Score: {t_score:.4f})"); found_compatible = True
                    print(f"    DEBUG: Best Top Score: {t_score:.4f}")
                    if include_overlay:
                        print("    Finding compatible OVERLAY (category4)...")
                        best_overlay = self._find_best_compatible_overlay(t_embedding_cpu, input_embedding)
                        if best_overlay:
                            o_path, o_type, _, o_score = best_overlay
                            outfit_recommendation["recommended_overlay"] = {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                            print(f"      Found Overlay: {o_type} (Score: {o_score:.4f})")
                            print(f"      DEBUG: Best Overlay Score: {o_score:.4f}")
                else: print("    Could not find compatible top.")

            elif input_broad_category == "category1": # Input: Full Outfit -> Find Overlay
                if include_overlay:
                    print("  Finding compatible OVERLAY (category4)...")
                    best_overlay = self._find_best_compatible_overlay(input_embedding)
                    if best_overlay:
                        o_path, o_type, _, o_score = best_overlay
                        outfit_recommendation["recommended_overlay"] = {"path": o_path, "type": o_type, "score": round(o_score, 4)}
                        print(f"    Found Overlay: {o_type} (Score: {o_score:.4f})"); found_compatible = True
                        print(f"    DEBUG: Best Overlay Score: {o_score:.4f}")
                    else: print("    Could not find compatible overlay.")
                else: print("  Input is Full Outfit, overlay search disabled."); found_compatible = True

            elif input_broad_category == "category4": # Input: Overlay -> Find Full Outfit OR Top+Bottom
                print("  Finding compatible FULL OUTFIT (category1)...")
                best_full = self._find_best_compatible_item(input_embedding, "category1")
                if best_full:
                    f_path, f_type, _, f_score = best_full
                    outfit_recommendation["recommended_base"] = {"path": f_path, "type": f_type, "score": round(f_score, 4), "is_full_outfit": True}
                    print(f"    Found Full Outfit: {f_type} (Score: {f_score:.4f})"); found_compatible = True
                    print(f"    DEBUG: Best Full Outfit Score: {f_score:.4f}")
                else:
                    print("    Fallback: Finding compatible TOP (category3)...")
                    best_top = self._find_best_compatible_item(input_embedding, "category3")
                    if best_top:
                        t_path, t_type, t_embedding_cpu, t_score = best_top
                        print(f"      Found Top: {t_type} (Score vs Overlay: {t_score:.4f})")
                        print(f"      DEBUG: Best Top Score (vs Overlay): {t_score:.4f}")
                        print("      Finding compatible BOTTOM (category2) for the found Top...")
                        best_bottom = self._find_best_compatible_item(t_embedding_cpu.to(self.device), "category2")
                        if best_bottom:
                             b_path, b_type, b_embedding_cpu, b_score = best_bottom
                             print(f"      Found Bottom: {b_type} (Score vs Top: {b_score:.4f})")
                             print(f"      DEBUG: Best Bottom Score (vs Top): {b_score:.4f}")
                             outfit_recommendation["recommended_base"] = {
                                 "top": {"path": t_path, "type": t_type, "score_vs_overlay": round(t_score, 4)},
                                 "bottom": {"path": b_path, "type": b_type, "score_vs_top": round(b_score, 4)},
                                 "is_full_outfit": False
                             }
                             found_compatible = True
                        else: print("      Could not find compatible Bottom.")
                    else: print("    Could not find compatible base (Full Outfit or Top).")
            else: print(f"  Unsupported input broad category: {input_broad_category}")

        except Exception as e: print(f"Error finding compatible items: {e}"); traceback.print_exc(); return None

        print(f"Outfit recommendation process finished in {time.time() - start_time:.2f} seconds.")
        return outfit_recommendation if found_compatible else None


    # --- Helper methods _get_random_item_with_embedding, _find_best_compatible_item,
    #     _find_best_compatible_overlay remain the same ---
    def _get_random_item_with_embedding(self, category_key: str) -> Optional[Tuple[str, str, torch.Tensor]]:
        """Safely gets a random item path, its subcategory, and its CPU embedding. Uses BROAD category key."""
        if category_key not in self.image_embeddings or not self.image_embeddings[category_key]: return None
        valid_subcategories = { sc: embeds for sc, embeds in self.image_embeddings[category_key].items() if embeds }
        if not valid_subcategories: return None
        chosen_subcategory = random.choice(list(valid_subcategories.keys()))
        embeddings_dict = valid_subcategories[chosen_subcategory]
        if not embeddings_dict: return None
        img_path = random.choice(list(embeddings_dict.keys()))
        embedding = embeddings_dict[img_path]
        return img_path, chosen_subcategory, embedding

    def _find_best_compatible_item(self, target_embedding: torch.Tensor, target_category_key: str) -> Optional[Tuple[str, str, torch.Tensor, float]]:
        """Finds the item in the target BROAD category most compatible with the target embedding.
           Target embedding is assumed to be on device. Returns CPU embedding of best item."""
        best_item = None; highest_score = -1.0
        if target_category_key not in self.image_embeddings: return None
        for subcategory, embeddings_dict in self.image_embeddings[target_category_key].items():
            for img_path, embedding_cpu in embeddings_dict.items():
                score = self._compute_compatibility_score(target_embedding, embedding_cpu)
                if score > highest_score:
                    highest_score = score
                    best_item = (img_path, subcategory, embedding_cpu, score)
        return best_item

    def _find_best_compatible_overlay(self, top_embedding: torch.Tensor, bottom_embedding: Optional[torch.Tensor] = None) -> Optional[Tuple[str, str, torch.Tensor, float]]:
        """Finds the best overlay (category4) compatible with top and optionally bottom.
           Input embeddings can be on CPU or device. Returns CPU embedding of best overlay."""
        best_overlay = None; highest_avg_score = -1.0
        overlay_category_key = "category4"
        if overlay_category_key not in self.image_embeddings or not self.image_embeddings[overlay_category_key]: return None
        for subcategory, embeddings_dict in self.image_embeddings[overlay_category_key].items():
            for img_path, overlay_embedding_cpu in embeddings_dict.items():
                score_top = self._compute_compatibility_score(top_embedding, overlay_embedding_cpu)
                if bottom_embedding is not None:
                    score_bottom = self._compute_compatibility_score(bottom_embedding, overlay_embedding_cpu)
                    current_score = (score_top + score_bottom) / 2
                else: current_score = score_top
                if current_score > highest_avg_score:
                    highest_avg_score = current_score
                    best_overlay = (img_path, subcategory, overlay_embedding_cpu, highest_avg_score)
        return best_overlay

    # --- generate_compatible_outfits, display_outfit_recommendation, get_recommendations,
    #     recommend_similar_items remain the same ---
    def generate_compatible_outfits(self, n_recommendations: int = 3) -> List[Dict]:
        """Generate outfit recommendations starting randomly."""
        outfit_recommendations = []; attempts = 0; max_attempts = n_recommendations * 15
        has_cat1 = "category1" in self.image_embeddings and any(self.image_embeddings["category1"].values())
        has_cat2 = "category2" in self.image_embeddings and any(self.image_embeddings["category2"].values())
        has_cat3 = "category3" in self.image_embeddings and any(self.image_embeddings["category3"].values())
        has_cat4 = "category4" in self.image_embeddings and any(self.image_embeddings["category4"].values())
        if not (has_cat1 or (has_cat2 and has_cat3)): print("Error: Need Cat 1 or (Cat 2 + Cat 3) with embeddings."); return []
        while len(outfit_recommendations) < n_recommendations and attempts < max_attempts:
            attempts += 1; outfit = {}
            use_category1 = has_cat1 and (not has_cat2 or not has_cat3 or random.choice([True, False]))
            if use_category1:
                cat1_item = self._get_random_item_with_embedding("category1")
                if not cat1_item: continue
                fp, ft, fe_cpu = cat1_item
                outfit = {"full_outfit": {"path": fp, "type": ft}}
                if has_cat4 and random.choice([True, False]):
                    best_overlay = self._find_best_compatible_overlay(fe_cpu.to(self.device))
                    if best_overlay: op, ot, _, os = best_overlay; outfit["overlay"] = {"path": op, "type": ot, "score": round(os, 4)}
                outfit["overall_compatibility"] = outfit.get("overlay", {}).get("score", 1.0)
            elif has_cat2 and has_cat3:
                cat3_item = self._get_random_item_with_embedding("category3")
                if not cat3_item: continue
                tp, tt, te_cpu = cat3_item
                best_bottom = self._find_best_compatible_item(te_cpu.to(self.device), "category2")
                if not best_bottom: continue
                bp, bt, be_cpu, bs = best_bottom
                outfit = {"top": {"path": tp, "type": tt}, "bottom": {"path": bp, "type": bt, "score": round(bs, 4)}}
                if has_cat4 and random.choice([True, False]):
                    best_overlay = self._find_best_compatible_overlay(te_cpu, be_cpu)
                    if best_overlay: op, ot, _, os = best_overlay; outfit["overlay"] = {"path": op, "type": ot, "score": round(os, 4)}
                scores = [s for s in [outfit.get("bottom", {}).get("score"), outfit.get("overlay", {}).get("score")] if s is not None]
                outfit["overall_compatibility"] = round(sum(scores) / len(scores), 4) if scores else 0.0
            else: continue
            outfit_recommendations.append(outfit)
        if len(outfit_recommendations) < n_recommendations: print(f"Warning: Only generated {len(outfit_recommendations)}/{n_recommendations} outfits.")
        return outfit_recommendations

    def display_outfit_recommendation(self, outfit: Dict):
        """Display an outfit recommendation using matplotlib."""
        items_to_plot = []
        if outfit.get("full_outfit"): items_to_plot.append((outfit["full_outfit"]["path"], f"Full: {outfit['full_outfit']['type']}", None))
        if outfit.get("top"): items_to_plot.append((outfit["top"]["path"], f"Top: {outfit['top']['type']}", None))
        if outfit.get("bottom"): items_to_plot.append((outfit["bottom"]["path"], f"Bottom: {outfit['bottom']['type']}", outfit["bottom"].get("score")))
        if outfit.get("overlay"): items_to_plot.append((outfit["overlay"]["path"], f"Overlay: {outfit['overlay']['type']}", outfit["overlay"].get("score")))
        if not items_to_plot: print("Cannot display empty outfit."); return
        num_items = len(items_to_plot)
        fig, axes = plt.subplots(1, num_items, figsize=(5 * num_items, 6))
        if num_items == 1: axes = [axes]
        for i, (path, title_base, score) in enumerate(items_to_plot):
            try:
                img = Image.open(path).convert("RGB"); axes[i].imshow(img)
                title = title_base + (f"\n(Score: {score:.2f})" if score is not None else "")
                axes[i].set_title(title); axes[i].axis('off')
            except Exception as e: axes[i].set_title(f"{title_base}\n(Error: {e})"); axes[i].axis('off')
        plt.suptitle(f"Outfit Recommendation (Overall Score: {outfit.get('overall_compatibility', 'N/A'):.2f})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    def get_recommendations(self, n_recommendations: int = 3, use_compatibility: bool = True) -> List[Dict]:
        """Generate outfit recommendations (currently only compatibility based)."""
        if use_compatibility: return self.generate_compatible_outfits(n_recommendations)
        else: print("Random generation not implemented."); return []

    def recommend_similar_items(self, item_path: str, n_recommendations: int = 3) -> List[Tuple[str, float]]:
        """Recommend similar items to a given item *already in the database*."""
        if not os.path.exists(item_path): print(f"Error: Query path does not exist: {item_path}"); return []
        query_embedding_cpu, query_category, query_subcategory = None, None, None
        for category, subcategories in self.image_embeddings.items():
            for subcategory_name, embeddings_dict in subcategories.items():
                if item_path in embeddings_dict:
                    query_embedding_cpu, query_category, query_subcategory = embeddings_dict[item_path], category, subcategory_name; break
            if query_embedding_cpu is not None: break
        if query_embedding_cpu is None: print(f"Error: Embedding not found for query: {item_path}."); return []

        similarities = []
        candidate_embeddings = self.image_embeddings.get(query_category, {}).get(query_subcategory, {})
        for candidate_path, candidate_embedding_cpu in candidate_embeddings.items():
            if candidate_path != item_path:
                try:
                    similarity = self._compute_compatibility_score(query_embedding_cpu, candidate_embedding_cpu)
                    similarities.append((candidate_path, similarity))
                except Exception as e: print(f"Error computing similarity with {candidate_path}: {e}")
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]


# --- __main__ block remains the same ---
if __name__ == "__main__":
    # --- Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, "../img/WOMEN"))
    cache_path = os.path.join(script_dir, "fashion_embeddings_cache.pt")

    print(f"Data Path: {data_path}")
    print(f"Cache Path: {cache_path}")
    if not os.path.exists(data_path): print(f"ERROR: Data path not found: {data_path}"); exit()

    # --- Initialize ---
    recommender = FashionRecommender(data_path=data_path, embeddings_cache_path=cache_path)

    # --- Find a test item ---
    test_item_path = None; test_item_broad_cat = None; test_item_sub_cat = None
    search_cats = ["category3", "category2", "category1", "category4"]
    for cat_key in search_cats:
        if cat_key in recommender.image_embeddings and recommender.image_embeddings[cat_key]:
            for subcat, embeds in recommender.image_embeddings[cat_key].items():
                if embeds:
                    try:
                        test_item_path = list(embeds.keys())[0]
                        test_item_broad_cat = cat_key
                        test_item_sub_cat = subcat
                        break
                    except IndexError: continue
            if test_item_path: break

    if not test_item_path:
        print("\nERROR: Could not find any item in embeddings to use for testing.")
        exit()
    print(f"\nUsing test item (Broad: {test_item_broad_cat}, Sub: {test_item_sub_cat}): {os.path.basename(test_item_path)}")

    # --- Test Outfit Building Feature ---
    print("\n--- Testing Outfit Building (around test item) ---")
    outfit_result = recommender.recommend_outfit_for_image(new_image=test_item_path)
    if outfit_result:
        print("\nOutfit Recommendation Result:")
        print(json.dumps(outfit_result, indent=2))
    else:
        print("\nCould not generate an outfit recommendation for the test item.")

    # --- Test Similarity Search + Outfit Pairing (New Image) ---
    print("\n--- Testing Similarity Search + Outfit Pairing (using test item as new image) ---")
    # Unpack FIVE values now
    pred_broad_cat_sim, pred_sub_cat_sim, conf_sim, similar_to_new, outfit_options_list = recommender.recommend_similar_to_new_image(
        new_image=test_item_path,
        n_recommendations=4,
        n_outfit_options=DEFAULT_NUM_OUTFIT_OPTIONS # Use default or specify
    )

    print(f"\nClassification during Similarity/Outfit Search:")
    if pred_broad_cat_sim:
        print(f"  - Predicted Broad Category: {pred_broad_cat_sim}")
        print(f"  - Predicted Sub Category: {pred_sub_cat_sim}")
        print(f"  - Confidence: {conf_sim:.4f}")
    else:
        print("  - Classification failed or confidence too low.")

    if similar_to_new:
        print("\nSimilar Items Found (for new image):")
        for path, score in similar_to_new: print(f"  - {os.path.basename(path)} (Score: {score:.4f})")
    else:
        print("No similar items found for the new test image.")

    if outfit_options_list:
         print(f"\nRecommended Outfit Options ({len(outfit_options_list)} found):")
         # Print first few options for brevity
         for i, option in enumerate(outfit_options_list[:3]):
              print(f"--- Option {i+1} ---")
              print(json.dumps(option, indent=2))
    else:
         print("No outfit options recommended.")


    # --- Test Random Outfit Generation ---
    print("\n--- Testing Random Outfit Generation ---")
    compatible_recommendations = recommender.get_recommendations(n_recommendations=1, use_compatibility=True)
    if compatible_recommendations:
        print("\nGenerated Random Compatible Outfit:")
        print(json.dumps(compatible_recommendations[0], indent=2))
    else: print("Could not generate compatible recommendations.")

    print("\n--- Testing Finished ---")
