"""
Utility functions for PCA Face Analysis System
Provides common helper functions and error handling
"""

import numpy as np
import sys
import os


def load_pca_data(eigenfaces_path, mean_faces_path, top_k=None):
    """
    Load precomputed eigenfaces and mean face data with error handling.
    
    Args:
        eigenfaces_path (str): Path to the eigenfaces numpy file
        mean_faces_path (str): Path to the mean faces numpy file
        top_k (int, optional): Number of top eigenfaces to use. If None, uses all.
    
    Returns:
        tuple: (eigenfaces, mean_face) numpy arrays
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If loaded data has invalid shape or format
    """
    try:
        # Check if files exist
        if not os.path.exists(eigenfaces_path):
            raise FileNotFoundError(
                f"Eigenfaces file not found: {eigenfaces_path}\n"
                "Please ensure you have generated the PCA data using pca.ipynb"
            )
        
        if not os.path.exists(mean_faces_path):
            raise FileNotFoundError(
                f"Mean faces file not found: {mean_faces_path}\n"
                "Please ensure you have generated the PCA data using pca.ipynb"
            )
        
        # Load eigenfaces
        eigenfaces = np.load(eigenfaces_path).astype(np.float32)
        
        # Validate eigenfaces shape
        if eigenfaces.ndim != 2:
            raise ValueError(
                f"Eigenfaces should be 2D array, got shape: {eigenfaces.shape}"
            )
        
        # Use only top k eigenfaces if specified
        if top_k is not None:
            if top_k > eigenfaces.shape[1]:
                print(f"Warning: Requested {top_k} eigenfaces but only {eigenfaces.shape[1]} available. Using all.")
                top_k = eigenfaces.shape[1]
            eigenfaces = eigenfaces[:, :top_k]
        
        # Load mean face
        mean_face = np.load(mean_faces_path)
        
        # Validate mean face shape
        if mean_face.ndim != 1:
            raise ValueError(
                f"Mean face should be 1D array, got shape: {mean_face.shape}"
            )
        
        # Validate compatibility
        if eigenfaces.shape[0] != mean_face.shape[0]:
            raise ValueError(
                f"Eigenfaces and mean face dimensions mismatch: "
                f"{eigenfaces.shape[0]} vs {mean_face.shape[0]}"
            )
        
        print(f"✓ Loaded eigenfaces: {eigenfaces.shape}")
        print(f"✓ Loaded mean face: {mean_face.shape}")
        
        return eigenfaces, mean_face
        
    except Exception as e:
        print(f"Error loading PCA data: {e}", file=sys.stderr)
        raise


def load_yolo_model(model_path):
    """
    Load YOLOv8 face detection model with error handling.
    
    Args:
        model_path (str): Path to the YOLO model file
        
    Returns:
        YOLO: Loaded YOLO model object
        
    Raises:
        FileNotFoundError: If model file is not found
        ImportError: If ultralytics package is not installed
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package not installed. "
            "Install it using: pip install ultralytics"
        )
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model file not found: {model_path}\n"
                "Please ensure the model file is in the correct location"
            )
        
        # Load the model
        model = YOLO(model_path)
        print(f"✓ Loaded YOLO model from: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}", file=sys.stderr)
        raise


def initialize_camera(camera_index=0):
    """
    Initialize camera with error handling.
    
    Args:
        camera_index (int): Camera device index (default: 0)
        
    Returns:
        cv2.VideoCapture: Initialized camera object
        
    Raises:
        RuntimeError: If camera cannot be opened
    """
    import cv2
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            "Please check if camera is connected and not in use by another application."
        )
    
    print(f"✓ Camera initialized at index {camera_index}")
    return cap


def validate_face_region(face, min_size=20):
    """
    Validate that a detected face region is usable.
    
    Args:
        face (numpy.ndarray): Face region image
        min_size (int): Minimum acceptable dimension
        
    Returns:
        bool: True if face is valid, False otherwise
    """
    if face is None or face.size == 0:
        return False
    
    h, w = face.shape[:2]
    if h < min_size or w < min_size:
        return False
    
    return True


def validate_coordinates(x1, y1, x2, y2, frame_height, frame_width):
    """
    Validate and clip bounding box coordinates to frame boundaries.
    
    Args:
        x1, y1, x2, y2 (int): Bounding box coordinates
        frame_height (int): Frame height
        frame_width (int): Frame width
        
    Returns:
        tuple: (x1, y1, x2, y2) clipped coordinates
    """
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    
    # Ensure x2 > x1 and y2 > y1
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    
    return x1, y1, x2, y2


def compress_face_pca(face_vector, eigenfaces, mean_face):
    """
    Compress a face using PCA projection.
    
    Args:
        face_vector (numpy.ndarray): Flattened face vector
        eigenfaces (numpy.ndarray): Eigenfaces matrix
        mean_face (numpy.ndarray): Mean face vector
        
    Returns:
        numpy.ndarray: Compressed representation (PCA coefficients)
    """
    face_normalized = face_vector - mean_face
    compressed = eigenfaces.T @ face_normalized
    return compressed


def reconstruct_face_pca(compressed_representation, eigenfaces, mean_face, target_shape):
    """
    Reconstruct a face from PCA compressed representation.
    
    Args:
        compressed_representation (numpy.ndarray): PCA coefficients
        eigenfaces (numpy.ndarray): Eigenfaces matrix
        mean_face (numpy.ndarray): Mean face vector
        target_shape (tuple): Target image shape (height, width)
        
    Returns:
        numpy.ndarray: Reconstructed face image
    """
    reconstructed = eigenfaces @ compressed_representation + mean_face
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    reconstructed = reconstructed.reshape(target_shape)
    return reconstructed


def calculate_compression_stats(original_size, compressed_size):
    """
    Calculate compression percentage and ratio.
    
    Args:
        original_size (int): Original data size in bytes
        compressed_size (int): Compressed data size in bytes
        
    Returns:
        tuple: (compression_percentage, compression_ratio)
    """
    if original_size <= 0:
        return 0.0, 1.0
    
    compression_percentage = (1 - (compressed_size / original_size)) * 100
    compression_percentage = max(0, min(compression_percentage, 100))
    
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    return compression_percentage, compression_ratio
