#ALL RIGHTS RESERVED
#HUMAIR MUNIR
#humairmunirawan@gmail.com





import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import argparse
from datetime import datetime
import logging
import time
from scipy.ndimage import rotate, shift
from sklearn.utils import class_weight
import pandas as pd
from tqdm import tqdm

# Configure logging
def setup_logger(log_dir="./logs"):
    """Set up a logger with file and console handlers"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('neural_network')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# One-hot encoding function
def one_hot_encode(labels):
    """Convert integer labels to one-hot encoded vectors"""
    # Make sure labels is a 1D array of integers
    if isinstance(labels, np.ndarray) and labels.ndim > 1:
        # If multi-dimensional, flatten to 1D
        labels = np.ravel(labels)
    
    # Convert to regular Python integers
    labels = np.asarray(labels).astype(int)
    
    unique_labels = np.unique(labels)
    n_labels = len(labels)
    n_unique_labels = len(unique_labels)
    
    # Create a mapping from label to index
    label_to_idx = {int(label): i for i, label in enumerate(unique_labels)}
    
    # Create one-hot encoded matrix
    one_hot = np.zeros((n_labels, n_unique_labels))
    for i, label in enumerate(labels):
        one_hot[i, label_to_idx[int(label)]] = 1
        
    return one_hot

# Data augmentation functions
def apply_augmentation(image, img_size=(28, 28)):
    """Apply various augmentations to an image"""
    # Reshape to proper image format
    channels = 3  # RGB
    img = image.reshape(img_size[0], img_size[1], channels)
    
    # Randomly choose an augmentation
    aug_type = np.random.choice([
        'rotate', 'shift', 'flip_horizontal', 'brightness', 'none'
    ])
    
    if aug_type == 'rotate':
        # Random rotation between -15 and 15 degrees
        angle = np.random.uniform(-15, 15)
        # Rotate each channel
        augmented = np.zeros_like(img)
        for c in range(channels):
            augmented[:,:,c] = rotate(img[:,:,c], angle, reshape=False)
    
    elif aug_type == 'shift':
        # Random shift up to 10% in each direction
        dx = np.random.uniform(-2, 2)
        dy = np.random.uniform(-2, 2)
        # Shift each channel
        augmented = np.zeros_like(img)
        for c in range(channels):
            augmented[:,:,c] = shift(img[:,:,c], [dy, dx], mode='nearest')
    
    elif aug_type == 'flip_horizontal':
        # Horizontal flip
        augmented = img[:, ::-1, :]
    
    elif aug_type == 'brightness':
        # Random brightness adjustment (0.8 to 1.2)
        factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(img * factor, 0, 1)
    
    else:  # 'none'
        augmented = img
    
    # Flatten and return
    return augmented.reshape(-1)

# Generate synthetic samples using SMOTE-like approach
def generate_synthetic_samples(minority_samples, n_synthetic):
    """Generate synthetic samples for minority classes"""
    n_samples, n_features = minority_samples.shape
    synthetic_samples = np.zeros((n_synthetic, n_features))
    
    for i in range(n_synthetic):
        # Pick a random sample
        idx = np.random.randint(0, n_samples)
        sample = minority_samples[idx]
        
        # Pick another random sample
        idx2 = np.random.randint(0, n_samples)
        while idx2 == idx:  # Ensure we pick a different sample
            idx2 = np.random.randint(0, n_samples)
        sample2 = minority_samples[idx2]
        
        # Generate a synthetic sample between them
        alpha = np.random.random()
        synthetic_samples[i] = sample*alpha + sample2*(1-alpha)
    
    return synthetic_samples

# Modified load and preprocess function with balance checking and data augmentation
def load_data(data_dir=None, use_pickle=True, pickle_path="./rgb_data.pkl", img_size=(28, 28), 
              check_balance=True, augment_data=False, logger=None):
    """
    Load data either from a pickle file or from image directories, supporting RGB images
    Options to check class balance and augment data
    """
    if logger is None:
        logger = logging.getLogger('neural_network')
    
    if use_pickle:
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        logger.info(f"Loading data from pickle file: {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                
            # Support different pickle formats
            if isinstance(data, tuple) and len(data) == 4:
                X_train, X_test, y_train, y_test = data
            else:
                raise ValueError("Unexpected pickle data format")
                
        except Exception as e:
            raise Exception(f"Failed to load pickle file: {e}")
    else:
        if not data_dir or not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        logger.info(f"Processing images from directory: {data_dir}")
        X, y = [], []
        
        # Count valid directories to determine number of classes
        class_dirs = [d for d in sorted(os.listdir(data_dir)) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"No valid class directories found in {data_dir}")
        
        # Track class distribution for balancing
        class_counts = {}
            
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            
            try:
                class_label = int(class_dir)  # Assuming directory names are digits
            except ValueError:
                logger.warning(f"Skipping non-numeric directory: {class_dir}")
                continue
                
            logger.info(f"Processing class: {class_label}")
            
            # Look for image files
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not img_files:
                logger.warning(f"No image files found in {class_path}")
                continue
            
            # Update class count
            class_counts[class_label] = len(img_files)
                
            # Process images with a progress indicator
            for i, img_file in enumerate(tqdm(img_files, desc=f"Class {class_label}")):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Using PIL for loading and processing - now keeping RGB
                    img = Image.open(img_path).convert('RGB')  # Convert to RGB
                    img = img.resize(img_size)  # Resize to specified size
                    img_array = np.array(img) / 255.0  # Convert to numpy array and normalize
                    
                    # Reshape to flatten the image while preserving RGB channels
                    X.append(img_array.reshape(-1))  # Flatten the image
                    y.append(class_label)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
        
        if not X:
            raise ValueError("No valid images were processed")
            
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Split the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    
    # Check class balance
    if check_balance:
        # Count occurrences of each class in training data
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        
        # Log class distribution
        logger.info("Class distribution in training data:")
        for cls, count in zip(unique_classes, class_counts):
            logger.info(f"Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
        
        # Calculate imbalance ratio (max count / min count)
        imbalance_ratio = np.max(class_counts) / np.min(class_counts)
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        # Create class weights for weighted loss
        cls_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train
        )
        
        # Convert to dictionary for easier lookup
        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, cls_weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Perform data augmentation and balancing if requested
        if augment_data and imbalance_ratio > 1.5:  # Only if significant imbalance
            logger.info("Performing data augmentation and class balancing...")
            
            # Get the majority class count
            max_samples = np.max(class_counts)
            
            # Create a balanced dataset with augmentation
            X_balanced = []
            y_balanced = []
            
            for cls in unique_classes:
                # Get samples for this class
                class_indices = np.where(y_train == cls)[0]
                class_samples = X_train[class_indices]
                samples_needed = max_samples - len(class_indices)
                
                # Add original samples
                X_balanced.extend(class_samples)
                y_balanced.extend([cls] * len(class_samples))
                
                # If we need more samples for this class
                if samples_needed > 0:
                    logger.info(f"Generating {samples_needed} additional samples for class {cls}")
                    
                    if len(class_indices) > 5:  # Enough samples for SMOTE-like approach
                        # Generate some synthetic samples
                        synthetic_portion = min(0.3, samples_needed / 2)  # Up to 30% synthetic
                        n_synthetic = int(samples_needed * synthetic_portion)
                        if n_synthetic > 0:
                            synthetic_samples = generate_synthetic_samples(class_samples, n_synthetic)
                            X_balanced.extend(synthetic_samples)
                            y_balanced.extend([cls] * n_synthetic)
                            samples_needed -= n_synthetic
                    
                    # Generate the rest using augmentation
                    if samples_needed > 0:
                        augmented_samples = []
                        for _ in range(samples_needed):
                            # Randomly select a sample to augment
                            idx = np.random.choice(class_indices)
                            sample = X_train[idx]
                            
                            # Apply augmentation
                            augmented = apply_augmentation(sample, img_size)
                            augmented_samples.append(augmented)
                        
                        # Add augmented samples
                        X_balanced.extend(augmented_samples)
                        y_balanced.extend([cls] * samples_needed)
            
            # Replace training data with balanced dataset
            X_train = np.array(X_balanced)
            y_train = np.array(y_balanced)
            
            # Log new class distribution
            unique_classes, new_counts = np.unique(y_train, return_counts=True)
            logger.info("New class distribution after balancing:")
            for cls, count in zip(unique_classes, new_counts):
                logger.info(f"Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
    else:
        class_weight_dict = None
    
    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    num_classes = y_train_encoded.shape[1]
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    logger.info(f"Input dimension: {X_train.shape[1]} (RGB flattened)")
    logger.info(f"Number of classes: {num_classes}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, num_classes, class_weight_dict

# Function to save processed data
def save_processed_data(X_train, X_test, y_train, y_test, output_path="processed_data.pkl", logger=None):
    """Save the processed data to a pickle file for future use"""
    if logger is None:
        logger = logging.getLogger('neural_network')
        
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    logger.info(f"Data saved to {output_path}")

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # More stable softmax implementation
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize Weights
def initialize_weights(input_size, hidden_size1, hidden_size2, output_size, seed=42, logger=None):
    """Initialize weights using Xavier/Glorot initialization for better convergence"""
    if logger is None:
        logger = logging.getLogger('neural_network')
        
    np.random.seed(seed)
    logger.info(f"Initializing network with architecture: {input_size}->{hidden_size1}->{hidden_size2}->{output_size}")
    
    # Xavier initialization for better gradient flow
    weights_input_hidden1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / (input_size + hidden_size1))
    bias_hidden1 = np.zeros((1, hidden_size1))
    
    weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / (hidden_size1 + hidden_size2))
    bias_hidden2 = np.zeros((1, hidden_size2))
    
    weights_hidden2_output = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / (hidden_size2 + output_size))
    bias_output = np.zeros((1, output_size))
    
    return (weights_input_hidden1, bias_hidden1, 
            weights_hidden1_hidden2, bias_hidden2, 
            weights_hidden2_output, bias_output)

# Forward Propagation
def forward(X, weights, is_training=False, dropout_rate=0.2):
    """
    Forward pass with proper training/inference handling for dropout
    """
    weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output = weights

    # First hidden layer
    hidden_layer1_input = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden_layer1 = relu(hidden_layer1_input)
    
    # Apply dropout only during training
    dropout_mask1 = None
    if is_training and dropout_rate > 0:
        dropout_mask1 = np.random.rand(*hidden_layer1.shape) > dropout_rate
        hidden_layer1 *= dropout_mask1
        hidden_layer1 /= (1 - dropout_rate)  # Scale to maintain expected value

    # Second hidden layer
    hidden_layer2_input = np.dot(hidden_layer1, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer2 = relu(hidden_layer2_input)
    
    # Apply dropout only during training
    dropout_mask2 = None
    if is_training and dropout_rate > 0:
        dropout_mask2 = np.random.rand(*hidden_layer2.shape) > dropout_rate
        hidden_layer2 *= dropout_mask2
        hidden_layer2 /= (1 - dropout_rate)  # Scale to maintain expected value

    # Output layer
    output_layer_input = np.dot(hidden_layer2, weights_hidden2_output) + bias_output
    output_layer = softmax(output_layer_input)
    
    # Store activations for backpropagation
    cache = (hidden_layer1, hidden_layer2, dropout_mask1, dropout_mask2)
    
    return output_layer, cache

# Backward Propagation with class weights
def backward(X, y, cache, output, weights, learning_rate, gradient_clip_value, dropout_rate=0.2, class_weights=None):
    """Backward pass with class weighting and gradient clipping"""
    weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output = weights
    hidden_layer1, hidden_layer2, dropout_mask1, dropout_mask2 = cache

    # Output layer error with optional class weighting
    output_error = output - y  # Cross-entropy derivative with softmax
    
    # Apply class weights if provided
    if class_weights is not None:
        # Need to convert one-hot encoded y back to class indices
        class_indices = np.argmax(y, axis=1)
        # Get weights for each sample
        sample_weights = np.array([class_weights[idx] for idx in class_indices])
        # Apply weights to error (reshape for broadcasting)
        output_error = output_error * sample_weights.reshape(-1, 1)
    
    # Gradients for output layer
    d_weights_hidden2_output = np.dot(hidden_layer2.T, output_error)
    d_bias_output = np.sum(output_error, axis=0, keepdims=True)

    # Hidden layer 2 error
    hidden2_error = np.dot(output_error, weights_hidden2_output.T) * relu_derivative(hidden_layer2)
    if dropout_mask2 is not None:
        hidden2_error *= dropout_mask2 / (1 - dropout_rate)  # Apply dropout mask

    # Gradients for hidden layer 2
    d_weights_hidden1_hidden2 = np.dot(hidden_layer1.T, hidden2_error)
    d_bias_hidden2 = np.sum(hidden2_error, axis=0, keepdims=True)

    # Hidden layer 1 error
    hidden1_error = np.dot(hidden2_error, weights_hidden1_hidden2.T) * relu_derivative(hidden_layer1)
    if dropout_mask1 is not None:
        hidden1_error *= dropout_mask1 / (1 - dropout_rate)  # Apply dropout mask

    # Gradients for hidden layer 1
    d_weights_input_hidden1 = np.dot(X.T, hidden1_error)
    d_bias_hidden1 = np.sum(hidden1_error, axis=0, keepdims=True)

    # Apply gradient clipping to all gradients
    gradients = [d_weights_input_hidden1, d_bias_hidden1, 
                d_weights_hidden1_hidden2, d_bias_hidden2, 
                d_weights_hidden2_output, d_bias_output]
    
    for i in range(len(gradients)):
        norm = np.sqrt(np.sum(gradients[i]**2))
        if norm > gradient_clip_value:
            gradients[i] = gradients[i] * (gradient_clip_value / norm)
    
    # Update weights with clipped gradients
    weights_input_hidden1 -= learning_rate * gradients[0]
    bias_hidden1 -= learning_rate * gradients[1]
    weights_hidden1_hidden2 -= learning_rate * gradients[2]
    bias_hidden2 -= learning_rate * gradients[3]
    weights_hidden2_output -= learning_rate * gradients[4]
    bias_output -= learning_rate * gradients[5]

    return (weights_input_hidden1, bias_hidden1, 
            weights_hidden1_hidden2, bias_hidden2, 
            weights_hidden2_output, bias_output)

# Compute Loss with class weights
def compute_loss(y_true, y_pred, epsilon=1e-15, class_weights=None):
    """Compute cross-entropy loss with class weighting"""
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if class_weights is not None:
        # Convert one-hot y_true to class indices
        class_indices = np.argmax(y_true, axis=1)
        # Get sample weights
        sample_weights = np.array([class_weights[idx] for idx in class_indices])
        # Weighted cross-entropy
        loss = -np.sum(y_true * np.log(y_pred) * sample_weights.reshape(-1, 1)) / len(y_true)
    else:
        # Standard cross-entropy
        loss = -np.mean(y_true * np.log(y_pred))
    
    return loss

# Training with improved logging and metrics
def train(X_train, y_train, X_val, y_val, weights, epochs, batch_size, learning_rate, 
          gradient_clip_value, dropout_rate=0.2, patience=5, min_delta=0.001, 
          class_weights=None, logger=None):
    """Train the model with detailed logging and class weighting"""
    if logger is None:
        logger = logging.getLogger('neural_network')
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    # Training metrics tracking
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
    if class_weights is not None:
        logger.info(f"Using class weights: {class_weights}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        # Mini-batch training with progress bar
        batch_losses = []
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))
        
        for i in tqdm(range(0, X_train.shape[0], batch_size), total=n_batches, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass in training mode (with dropout)
            output_batch, cache = forward(X_batch, weights, is_training=True, dropout_rate=dropout_rate)
            
            # Compute batch loss for monitoring
            batch_loss = compute_loss(y_batch, output_batch, class_weights=class_weights)
            batch_losses.append(batch_loss)
            
            # Backward pass and update weights
            weights = backward(X_batch, y_batch, cache, output_batch, weights, 
                              learning_rate, gradient_clip_value, dropout_rate, class_weights)

        # Calculate average batch loss
        avg_batch_loss = np.mean(batch_losses)

        # Compute and store losses and accuracies (in inference mode, no dropout)
        output_train, _ = forward(X_train, weights, is_training=False)
        output_val, _ = forward(X_val, weights, is_training=False)
        
        train_loss = compute_loss(y_train, output_train, class_weights=class_weights)
        val_loss = compute_loss(y_val, output_val, class_weights=class_weights)
        
        # Calculate accuracies
        train_preds = np.argmax(output_train, axis=1)
        train_true = np.argmax(y_train, axis=1)
        train_acc = np.mean(train_preds == train_true) * 100
        
        val_preds = np.argmax(output_val, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = np.mean(val_preds == val_true) * 100
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update metrics history
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['epoch_time'].append(epoch_time)

        # Log detailed metrics
        logger.info(f'Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - '
                   f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                   f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Log per-class metrics on validation set (every 5 epochs or last epoch)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Calculate per-class metrics
            class_metrics = {}
            val_class_indices = np.argmax(y_val, axis=1)
            val_pred_indices = np.argmax(output_val, axis=1)
            
            unique_classes = np.unique(val_class_indices)
            
            for cls in unique_classes:
                # Find samples of this class
                cls_indices = val_class_indices == cls
                
                # Calculate metrics for this class
                cls_correct = np.sum((val_pred_indices == cls) & cls_indices)
                cls_total = np.sum(cls_indices)
                cls_precision = cls_correct / np.sum(val_pred_indices == cls) if np.sum(val_pred_indices == cls) > 0 else 0
                cls_recall = cls_correct / cls_total if cls_total > 0 else 0
                cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
                
                class_metrics[cls] = {
                    'accuracy': cls_correct / cls_total * 100 if cls_total > 0 else 0,
                    'precision': cls_precision * 100,
                    'recall': cls_recall * 100,
                    'f1': cls_f1 * 100
                }
            
            # Log class metrics
            logger.info("Per-class validation metrics:")
            for cls, metrics in class_metrics.items():
                logger.info(f"Class {cls}: Acc={metrics['accuracy']:.2f}%, "
                           f"Prec={metrics['precision']:.2f}%, "
                           f"Rec={metrics['recall']:.2f}%, "
                           f"F1={metrics['f1']:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_weights = tuple(w.copy() for w in weights)  # Deep copy of weights
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Create a DataFrame for metrics history
    metrics_df = pd.DataFrame(metrics_history)
    
    # Return the best weights and metrics
    return best_weights or weights, train_losses, val_losses, train_accuracies, val_accuracies, metrics_df

# Save and load model functions
def save_model(weights, model_path="./output/model_weights.pkl", logger=None):
    """Save the model weights to a file"""
    if logger is None:
        logger = logging.getLogger('neural_network')
        
    output_dir = os.path.dirname(model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)
    logger.info(f"Model saved to {model_path}")

def load_model(model_path, logger=None):
    """Load a saved model from a file"""
    if logger is None:
        logger = logging.getLogger('neural_network')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return weights

# Evaluate model function
def evaluate(X, y, weights, logger=None):
    """Evaluate the model on the given data"""
    if logger is None:
        logger = logging.getLogger('neural_network')
        
    # Forward pass in inference mode (no dropout)
    output, _ = forward(X, weights, is_training=False)
    
    # Calculate loss
    loss = compute_loss(y, output)
    
    # Calculate accuracy
    predictions = np.argmax(output, axis=1)
    ground_truth = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == ground_truth) * 100
    
    # Calculate per-class metrics
    class_metrics = {}
    unique_classes = np.unique(ground_truth)
    
    for cls in unique_classes:
        # Find samples of this class
        cls_indices = ground_truth == cls
        
        # Calculate metrics for this class
        cls_correct = np.sum((predictions == cls) & cls_indices)
        cls_total = np.sum(cls_indices)
        cls_precision = cls_correct / np.sum(predictions == cls) if np.sum(predictions == cls) > 0 else 0
        cls_recall = cls_correct / cls_total if cls_total > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        
        class_metrics[cls] = {
            'accuracy': cls_correct / cls_total * 100 if cls_total > 0 else 0,
            'precision': cls_precision * 100,
            'recall': cls_recall * 100,
            'f1': cls_f1 * 100
        }
    
    # Log metrics
    logger.info(f"Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    logger.info("Per-class evaluation metrics:")
    for cls, metrics in class_metrics.items():
        logger.info(f"Class {cls}: Acc={metrics['accuracy']:.2f}%, "
                   f"Prec={metrics['precision']:.2f}%, "
                   f"Rec={metrics['recall']:.2f}%, "
                   f"F1={metrics['f1']:.2f}%")
    
    # Calculate confusion matrix
    n_classes = len(unique_classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(ground_truth)):
        true_class = ground_truth[i]
        pred_class = predictions[i]
        confusion_matrix[true_class, pred_class] += 1
    
    return loss, accuracy, class_metrics, confusion_matrix, predictions

# Visualization functions
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, output_dir="./output"):
    """Plot training and validation losses and accuracies"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names=None, output_dir="./output"):
    """Plot confusion matrix"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names is None:
        class_names = [str(i) for i in range(confusion_matrix.shape[0])]
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network for image classification")
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing training data')
    parser.add_argument('--use_pickle', action='store_true', help='Use pickle data instead of processing images')
    parser.add_argument('--pickle_path', type=str, default='./data/processed_data.pkl', help='Path to pickle file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for model and visualizations')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size1', type=int, default=128, help='Size of first hidden layer')
    parser.add_argument('--hidden_size2', type=int, default=64, help='Size of second hidden layer')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    parser.add_argument('--gradient_clip', type=float, default=5.0, help='Gradient clipping value')
    parser.add_argument('--augment_data', action='store_true', help='Apply data augmentation')
    parser.add_argument('--check_balance', action='store_true', help='Check class balance')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate a saved model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model for evaluation')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger()
    logger.info("Starting Neural Network Training Pipeline")
    logger.info(f"Arguments: {args}")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load data
    if args.evaluate_only and args.model_path:
        # For evaluation only, still need to load data
        X_train, X_test, y_train, y_test, num_classes, class_weight_dict = load_data(
            data_dir=args.data_dir,
            use_pickle=args.use_pickle,
            pickle_path=args.pickle_path,
            check_balance=False,  # Don't need to check balance for evaluation
            augment_data=False,   # Don't need to augment data for evaluation
            logger=logger
        )
        
        # Load model
        weights = load_model(args.model_path, logger)
        
        # Evaluate model
        logger.info("Evaluating model on test data")
        loss, accuracy, class_metrics, confusion_matrix, predictions = evaluate(X_test, y_test, weights, logger)
        
        # Plot confusion matrix
        plot_confusion_matrix(confusion_matrix, output_dir=args.output_dir)
        
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
    else:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, num_classes, class_weight_dict = load_data(
            data_dir=args.data_dir,
            use_pickle=args.use_pickle,
            pickle_path=args.pickle_path,
            check_balance=args.check_balance,
            augment_data=args.augment_data,
            logger=logger
        )
        
        # Save processed data
        save_processed_data(X_train, X_test, y_train, y_test, 
                           os.path.join(args.output_dir, 'processed_data.pkl'), logger)
        
        # Initialize model weights
        input_size = X_train.shape[1]
        weights = initialize_weights(
            input_size=input_size,
            hidden_size1=args.hidden_size1,
            hidden_size2=args.hidden_size2,
            output_size=num_classes,
            seed=args.random_seed,
            logger=logger
        )
        
        # Train model
        logger.info("Starting model training")
        weights, train_losses, val_losses, train_accuracies, val_accuracies, metrics_df = train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,  # Using test data as validation data
            y_val=y_test,
            weights=weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_clip_value=args.gradient_clip,
            dropout_rate=args.dropout_rate,
            patience=args.patience,
            min_delta=args.min_delta,
            class_weights=class_weight_dict,
            logger=logger
        )
        
        # Save model
        save_model(weights, os.path.join(args.output_dir, 'model_weights.pkl'), logger)
        
        # Save training metrics
        metrics_df.to_csv(os.path.join(args.output_dir, 'training_metrics.csv'), index=False)
        
        # Plot training history
        plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, args.output_dir)
        
        # Evaluate model
        logger.info("Evaluating model on test data")
        loss, accuracy, class_metrics, confusion_matrix, predictions = evaluate(X_test, y_test, weights, logger)
        
        # Plot confusion matrix
        plot_confusion_matrix(confusion_matrix, output_dir=args.output_dir)
        
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    logger.info("Neural Network Training Pipeline completed")

if __name__ == "__main__":
    main()
