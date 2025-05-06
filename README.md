# StyleAi

**StyleAi** is an intelligent Flask-based web API designed to power a sophisticated fashion recommendation system. It leverages cutting-edge machine learning models (like OpenAI's CLIP) to understand visual and textual fashion concepts, enabling powerful features like image-based similarity search, new item classification, and automated outfit generation. The system is architected to source its image dataset directly from Google Drive, making data management flexible and scalable.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-%234285F4.svg?style=for-the-badge&logo=googledrive&logoColor=white)](https://www.google.com/drive/)

---

## Table of Contents

*   [Overview](#overview)
*   [Features](#features)
*   [How It Works](#how-it-works)
*   [Tech Stack](#tech-stack)
*   [System Architecture](#system-architecture)
*   [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Configuration](#configuration)
    *   [Running the Application](#running-the-application)
*   [API Endpoints](#api-endpoints)
*   [Data Structure on Google Drive](#data-structure-on-google-drive)
*   [Future Enhancements](#future-enhancements)

---

## Overview

StyleAi provides a backend service for fashion applications requiring intelligent recommendations. Users can upload an image of a clothing item, and the system will:
1.  Classify the item into predefined categories (e.g., "Dresses," "Tops," "Pants").
2.  Find visually similar items from the existing fashion catalog.
3.  Suggest complete, compatible outfits based on the new item or an existing catalog item.

The core recommendation logic is encapsulated in `styleRe.py`, which handles model loading, embedding generation, similarity calculations, and outfit logic. The Flask application (`FashionRecommender.py`) exposes this functionality via a RESTful API.

## Features

*   **Image-Based Similarity:** Find fashion items visually similar to a query image.
*   **New Item Classification:** Automatically categorize uploaded fashion items.
*   **Outfit Generation:**
    *   Recommend complete outfits for a newly uploaded item.
    *   Generate outfits around an existing item from the catalog.
    *   Provide multiple outfit options.
*   **Google Drive Integration:** Loads and processes image datasets directly from a specified Google Drive folder.
*   **CLIP-Powered Embeddings:** Utilizes OpenAI's CLIP model for rich image and text embeddings, enabling nuanced understanding of fashion styles.
*   **Caching:** Caches image embeddings for faster subsequent lookups and recommendations.
*   **Configurable Categories:** Easily define and map broad fashion categories to specific subcategories.

## How It Works

1.  **Data Loading:** On startup, StyleAi connects to Google Drive, scans the specified dataset folder, and identifies all fashion item images.
2.  **Embedding Generation:** For each image, it generates a feature vector (embedding) using the CLIP image encoder. These embeddings capture the visual essence of the items. Text embeddings for category names are also precomputed.
3.  **Caching:** Image embeddings are cached locally to speed up future initializations.
4.  **New Image Processing:** When a new image is uploaded via the API:
    *   It's classified by comparing its embedding to the text embeddings of predefined fashion subcategories.
    *   Its image embedding is generated.
5.  **Similarity Search:** To find similar items, the embedding of the query image is compared against all embeddings in the relevant subcategory using cosine similarity.
6.  **Outfit Generation:** Outfits are constructed by selecting compatible items from different categories based on pre-defined rules and compatibility scores (which can also leverage embedding similarity or other heuristics).

## Tech Stack

*   **Backend Framework:** Flask
*   **Machine Learning:** PyTorch, Transformers (for CLIP)
*   **Image Processing:** Pillow
*   **Data Storage/Source:** Google Drive (via Google Drive API)
*   **Core Language:** Python 3.8+

## System Architecture

```
User/Client Application
        |
        v
  Flask API (FashionRecommender.py)
  - Handles HTTP requests
  - Serves image URLs (proxied from Drive)
  - Validates input
        |
        v
  Recommendation Engine (styleRe.py)
  - CLIP Model (Image/Text Encoding)
  - Image Classification
  - Similarity Search
  - Outfit Generation Logic
  - Embedding Cache Management
        |
        v
  Google Drive API Client
  - Lists files/folders
  - Downloads image data
        |
        v
  Google Drive (Image Dataset)
```

## Setup and Installation

### Prerequisites

*   Python 3.8 or higher.
*   `pip` for package management.
*   A Google Cloud Platform project with the **Google Drive API enabled**.
*   OAuth 2.0 credentials (`credentials.json`) downloaded from your Google Cloud Project.
*   Your fashion image dataset uploaded to Google Drive, following the expected folder structure (see Data Structure).

### Configuration

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd backend # Or your project root
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Google Drive Credentials:**
    *   Place your downloaded `credentials.json` file in the `/Users/sugandhab/Downloads/backend/` directory.
    *   Upon first run, the application will guide you through an OAuth flow in your browser to authorize access. A `token.json` will be generated and stored in the same directory for future use.
5.  **Environment Variables / Configuration in `FashionRecommender.py`:**
    *   Open `/Users/sugandhab/Downloads/backend/FashionRecommender.py`.
    *   Set `GOOGLE_DRIVE_WOMEN_FOLDER_ID` to the ID of your main "WOMEN" data folder on Google Drive.
    *   Ensure `EMBEDDINGS_CACHE_PATH`, `GOOGLE_DRIVE_CREDENTIALS_PATH`, and `GOOGLE_DRIVE_TOKEN_PATH` are correctly pointing to your files.

### Running the Application

1.  **Start the Flask development server:**
    ```bash
    python FashionRecommender.py
    ```
2.  **Initial Embedding Generation:** The first time you run the server (or if the cache is cleared/data changes significantly), it will download images from Google Drive and generate embeddings. This can take some time depending on the dataset size. Subsequent startups will be faster if a valid cache exists.

## API Endpoints

*   **`POST /recommend/similar-to-new`**:
    *   Upload a new image to find similar items and get outfit recommendations.
    *   **Request:** `multipart/form-data` with an image file under the key `image`.
    *   **Response:** JSON with classified category, confidence, list of similar items (with image URLs and scores), and outfit options.
*   **`POST /recommend/outfit-for-image`**:
    *   Upload a new image to get outfit recommendations for it.
    *   **Request:** `multipart/form-data` with an image file under the key `image`.
    *   **Response:** JSON with outfit options.
*   **`GET /recommend/similar-to-existing/<drive_file_id>`**:
    *   Get items similar to an existing item in the database (identified by its Google Drive File ID).
    *   **Response:** JSON list of similar items (with image URLs and scores).


## Data Structure on Google Drive

Your image dataset on Google Drive should be organized as follows, under a root folder (e.g., "WOMEN") whose ID you provide in the configuration:

```
<GOOGLE_DRIVE_WOMEN_FOLDER_ID>/  (e.g., "WOMEN" folder)
├── Dresses/
│   ├── Item_Dress_001/
│   │   ├── image1.jpg
│   │   └── image2.png
│   └── Item_Dress_002/
│       └── front_view.jpeg
├── Pants/
│   ├── Item_Pants_001/
│   │   └── image.jpg
│   └── ...
├── Tops/
│   └── ...
└── (Other subcategory folders as defined in category_mappings in styleRe.py)
```

## Future Enhancements

*   Support for more complex outfit generation rules and user preferences.
*   Integration with textual descriptions for items to enhance search and recommendation.
*   Fine-tuning CLIP or using domain-specific models for even better fashion understanding.
*   More robust error handling and logging.
*   Scalability improvements for very large datasets (e.g., distributed embedding generation).
*   Interactive API documentation (e.g., Swagger/OpenAPI).

---


