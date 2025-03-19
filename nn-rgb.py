import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import argparse
from datetime import datetime

# One-hot encoding function remains the same
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

# Modified Load and Preprocess function for RGB images
def load_data(data_dir=None, use_pickle=True, pickle_path="./rgb_data.pkl", img_size=(28, 28)):
    """
    Load data either from a pickle file or from image directories, supporting RGB images
    """
    if use_pickle:
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        print(f"Loading data from pickle file: {pickle_path}")
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
            
        print(f"Processing images from directory: {data_dir}")
        X, y = [], []
        
        # Count valid directories to determine number of classes
        class_dirs = [d for d in sorted(os.listdir(data_dir)) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        if not class_dirs:
            raise ValueError(f"No valid class directories found in {data_dir}")
            
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            
            try:
                class_label = int(class_dir)  # Assuming directory names are digits
            except ValueError:
                print(f"Skipping non-numeric directory: {class_dir}")
                continue
                
            print(f"Processing class: {class_label}")
            
            # Look for image files
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not img_files:
                print(f"No image files found in {class_path}")
                continue
                
            # Process images with a progress indicator
            for i, img_file in enumerate(img_files):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Using PIL for loading and processing - now keeping RGB
                    img = Image.open(img_path).convert('RGB')  # Convert to RGB
                    img = img.resize(img_size)  # Resize to specified size
                    img = np.array(img) / 255.0  # Convert to numpy array and normalize
                    
                    # Reshape to flatten the image while preserving RGB channels
                    X.append(img.reshape(-1))  # Flatten the image
                    y.append(class_label)
                    
                    # Show progress
                    if (i+1) % 100 == 0:
                        print(f"  Processed {i+1}/{len(img_files)} images")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
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
    
    # No need to reshape here as we're already handling the flattened RGB data
    
    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    
    num_classes = y_train_encoded.shape[1]
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Input dimension: {X_train.shape[1]} (RGB flattened)")
    print(f"Number of classes: {num_classes}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, num_classes

# Function to save processed data remains the same
def save_processed_data(X_train, X_test, y_train, y_test, output_path="processed_data.pkl"):
    """Save the processed data to a pickle file for future use"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    print(f"Data saved to {output_path}")

# Activation Functions remain the same
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # More stable softmax implementation
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Initialize Weights remains the same
def initialize_weights(input_size, hidden_size1, hidden_size2, output_size, seed=42):
    """Initialize weights using Xavier/Glorot initialization for better convergence"""
    np.random.seed(seed)
    
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

# Forward Propagation remains the same
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

# Backward Propagation remains the same
def backward(X, y, cache, output, weights, learning_rate, gradient_clip_value, dropout_rate=0.2):
    """Backward pass with proper gradient clipping"""
    weights_input_hidden1, bias_hidden1, weights_hidden1_hidden2, bias_hidden2, weights_hidden2_output, bias_output = weights
    hidden_layer1, hidden_layer2, dropout_mask1, dropout_mask2 = cache

    # Output layer error
    output_error = output - y  # Cross-entropy derivative with softmax
    
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

# Compute Loss remains the same
def compute_loss(y_true, y_pred, epsilon=1e-15):
    """Compute cross-entropy loss with better numerical stability"""
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))

# Training the Model remains the same
def train(X_train, y_train, X_val, y_val, weights, epochs, batch_size, learning_rate, 
          gradient_clip_value, dropout_rate=0.2, patience=5, min_delta=0.001):
    """Train the model with early stopping based on validation loss"""
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

        # Mini-batch training
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass in training mode (with dropout)
            output_batch, cache = forward(X_batch, weights, is_training=True, dropout_rate=dropout_rate)
            
            # Backward pass and update weights
            weights = backward(X_batch, y_batch, cache, output_batch, weights, 
                              learning_rate, gradient_clip_value, dropout_rate)

        # Compute and store losses (in inference mode, no dropout)
        output_train, _ = forward(X_train, weights, is_training=False)
        output_val, _ = forward(X_val, weights, is_training=False)
        
        train_loss = compute_loss(y_train, output_train)
        val_loss = compute_loss(y_val, output_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_weights = tuple(w.copy() for w in weights)  # Deep copy of weights
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Return the best weights instead of the latest ones
    return best_weights or weights, train_losses, val_losses

# Save and Load Model functions remain the same
def save_model(weights, model_path="./output/model_weights.pkl"):
    """Save the model weights to a file"""
    output_dir = os.path.dirname(model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Model saved to {model_path}")

def load_model(model_path="./output/model_weights.pkl"):
    """Load model weights from a file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return weights

# Plot Losses remains the same
def plot_loss(train_losses, val_losses, save_path=None):
    """Plot and optionally save the training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
        
    plt.show()

# Modified Show Predictions function for RGB images
def show_predictions(X, y, weights, num_samples=20, index=None, save_fig=False, fig_path=None, img_size=(28, 28)):
    """Show model predictions on sample images"""
    output, _ = forward(X, weights)
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Reshape images for display - adapt for RGB
    channels = 3  # RGB
    reshaped_images = X.reshape(-1, img_size[0], img_size[1], channels)

    if index is not None:
        # Show a single prediction
        plt.figure(figsize=(4, 4))
        plt.imshow(reshaped_images[index])  # No need for cmap with RGB
        plt.title(f'Pred: {predictions[index]}, True: {labels[index]}')
        plt.axis('off')
        if save_fig and fig_path:
            plt.savefig(fig_path)
        plt.show()
    else:
        # Show multiple predictions
        n = min(num_samples, len(X))
        rows = int(np.ceil(n / 5))
        plt.figure(figsize=(12, rows * 2.5))
        
        for i in range(n):
            plt.subplot(rows, 5, i + 1)
            plt.imshow(reshaped_images[i])  # No need for cmap with RGB
            
            # Add color to title based on correctness
            if predictions[i] == labels[i]:
                color = 'green'
            else:
                color = 'red'
                
            plt.title(f'P: {predictions[i]}, T: {labels[i]}', color=color)
            plt.axis('off')
            
        plt.tight_layout()
        if save_fig and fig_path:
            output_dir = os.path.dirname(fig_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(fig_path)
            print(f"Predictions saved to {fig_path}")
        plt.show()

# Modified Predict Image function for RGB images
def predict_image(image_path, weights, img_size=(28, 28)):
    """Predict the class of a single RGB image"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess the image with PIL
        img = Image.open(image_path).convert('RGB')  # Convert to RGB
        img = img.resize(img_size)  # Resize with PIL
        img = np.array(img) / 255.0  # Convert to numpy array and normalize
        
        # Reshape for prediction
        img = img.reshape(1, -1)
        
        # Predict
        output, _ = forward(img, weights)
        prediction = np.argmax(output, axis=1)[0]
        confidence = np.max(output) * 100
        
        # Display the image and prediction
        plt.figure(figsize=(5, 5))
        plt.imshow(img.reshape(img_size[0], img_size[1], 3))
        plt.title(f'Prediction: {prediction} (Confidence: {confidence:.2f}%)')
        plt.axis('off')
        plt.savefig('/home/humair/myenv/prediction.png')
        plt.show()
        
        return prediction, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Modified Process Directory function for RGB images
def process_directory(directory_path, weights, img_size=(28, 28)):
    """Process all images in a directory and show predictions"""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    results = {}
    
    # Find all image files
    img_files = [f for f in os.listdir(directory_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not img_files:
        print(f"No image files found in {directory_path}")
        return results
        
    for img_file in img_files:
        img_path = os.path.join(directory_path, img_file)
            
        print(f"Processing {img_file}...")
        prediction, confidence = predict_image(img_path, weights, img_size)
        
        if prediction is not None:
            results[img_file] = (prediction, confidence)
    
    return results

# Modified Main function with RGB image size parameter
def main():
    """Main function with command-line arguments"""
    parser = argparse.ArgumentParser(description="Neural Network for Image Classification (RGB)")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default=None, 
                        help="Directory containing subdirectories of class folders")
    parser.add_argument("--use_pickle", action="store_true", 
                        help="Use pickle file instead of processing images")
    parser.add_argument("--pickle_path", type=str, default="./rgb_processed_data.pkl", 
                        help="Path to pickle file if use_pickle is True")
    parser.add_argument("--save_data", action="store_true",
                        help="Save processed data to pickle file")
    parser.add_argument("--processed_data_path", type=str, default="./processed_data.pkl",
                        help="Path to save processed data")
    parser.add_argument("--img_width", type=int, default=28,
                        help="Width to resize images to")
    parser.add_argument("--img_height", type=int, default=28,
                        help="Height to resize images to")
    
    # Model parameters
    parser.add_argument("--hidden_size1", type=int, default=256, 
                        help="Size of first hidden layer")
    parser.add_argument("--hidden_size2", type=int, default=128, 
                        help="Size of second hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--gradient_clip", type=float, default=1.0, 
                        help="Gradient clipping value")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.2, 
                        help="Dropout rate")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Early stopping patience")
    parser.add_argument("--load_model", action="store_true",
                        help="Load pre-trained model weights")
    parser.add_argument("--model_path", type=str, default="./rgb_model_weights.pkl",
                        help="Path to load or save model weights")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./rgb_output", 
                        help="Directory to save model and results")
    parser.add_argument("--random_seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Prediction parameters
    parser.add_argument("--predict", action="store_true",
                        help="Run prediction mode instead of training")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to single image for prediction")
    parser.add_argument("--predict_dir", type=str, default=None,
                        help="Directory of images for prediction")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    img_size = (args.img_width, args.img_height)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set output paths with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.output_dir, f"rgb_model_{timestamp}.pkl") if not args.model_path else args.model_path
    loss_plot_path = os.path.join(args.output_dir, f"rgb_loss_curve_{timestamp}.png")
    predictions_plot_path = os.path.join(args.output_dir, f"predictions_{timestamp}.png")
    
    try:
        # Prediction mode
        if args.predict:
            if not args.load_model:
                raise ValueError("Must specify --load_model and --model_path for prediction mode")
                
            # Load the model
            weights = load_model(args.model_path)
            
            # Single image prediction
            if args.image_path:
                print(f"Predicting single image: {args.image_path}")
                predict_image(args.image_path, weights, img_size)
                
            # Directory prediction
            elif args.predict_dir:
                print(f"Processing directory: {args.predict_dir}")
                results = process_directory(args.predict_dir, weights, img_size)
                
                # Print summary of results
                if results:
                    print("\nPrediction Results:")
                    for img_file, (pred, conf) in results.items():
                        print(f"{img_file}: Predicted {pred} with {conf:.2f}% confidence")
            else:
                print("No prediction inputs specified. Use --image_path or --predict_dir")
                
        # Training mode
        else:
            # Load data
            X_train, X_test, y_train, y_test, output_size = load_data(
                data_dir=args.data_dir, 
                use_pickle=args.use_pickle, 
                pickle_path=args.pickle_path,
                img_size=img_size
            )
            
            # Save processed data if requested
            if args.save_data:
                save_processed_data(X_train, X_test, y_train, y_test, args.processed_data_path)
            
            # Load or initialize model weights
            if args.load_model:
                weights = load_model(args.model_path)
            else:
                # Initialize the model
                input_size = X_train.shape[1]
                weights = initialize_weights(
                    input_size, args.hidden_size1, args.hidden_size2, output_size, 
                    seed=args.random_seed
                )
            
            # Split training data into train and validation
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            split_index = int((1 - args.val_split) * len(indices))
            train_indices = indices[:split_index]
            val_indices = indices[split_index:]
            
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            
            print(f"Training set: {X_train_split.shape[0]} samples")
            print(f"Validation set: {X_val.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")

            # Train the model
            weights, train_losses, val_losses = train(
                X_train_split, y_train_split, X_val, y_val, 
                weights, args.epochs, args.batch_size, args.learning_rate, 
                args.gradient_clip, args.dropout_rate, args.patience
            )
            
            # Save the trained model
            save_model(weights, model_save_path)
            
            # Evaluate the model
            print("Evaluating model...")
            output_train, _ = forward(X_train, weights)
            output_test, _ = forward(X_test, weights)
            
            train_accuracy = np.mean(np.argmax(output_train, axis=1) == 
                                   np.argmax(y_train, axis=1)) * 100
            test_accuracy = np.mean(np.argmax(output_test, axis=1) == 
                                  np.argmax(y_test, axis=1)) * 100
            
            print(f'Training Accuracy: {train_accuracy:.2f}%')
            print(f'Test Accuracy: {test_accuracy:.2f}%')

            # Plot the training and validation losses
            plot_loss(train_losses, val_losses, loss_plot_path)
            
            # Show some example predictions
            print("Showing sample predictions...")
            show_predictions(X_test, y_test, weights, save_fig=True, 
                            fig_path=predictions_plot_path, img_size=img_size)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
