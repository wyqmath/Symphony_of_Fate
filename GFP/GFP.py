import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# --- Setup ---
# Ensure the output directory exists
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
# Load data from the CSV file
file_path = 'GFP.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it is in the correct directory.")
    exit()

# Extract sequence data and conv(%) values
sequences = df['Sequence'].tolist()
conv_values = df['conv(%)'].tolist()

# Normalize conv(%) to a [0, 1] scale
scaler = MinMaxScaler()
conv_values_normalized = scaler.fit_transform(np.array(conv_values).reshape(-1, 1)).flatten()

# Define the amino acid vocabulary
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
char_to_index = {char: idx for idx, char in enumerate(amino_acids)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

# --- Conditional Diffusion Model ---
def add_noise_to_sequence_with_model(sequence, conv_value, model, noise_level=0.1):
    """
    Adds noise to a sequence using a one-hot representation and adjusts the noise
    based on a conditional model's prediction to guide the generation process.
    """
    # Convert sequence to one-hot matrix
    onehot_matrix = np.zeros((len(sequence), len(amino_acids)))
    for i, char in enumerate(sequence):
        if char in char_to_index:
            onehot_matrix[i, char_to_index[char]] = 1

    # Add Gaussian noise
    noisy_matrix = onehot_matrix + noise_level * np.random.normal(0, 1, size=onehot_matrix.shape)
    
    # Create input tensor for the model from the noisiest representation
    noisy_indices = np.argmax(noisy_matrix, axis=1)
    input_tensor = torch.tensor(noisy_indices).long().unsqueeze(0).to(device)

    # Predict the conv(%) from the noisy sequence
    with torch.no_grad():
        target_conv = model(input_tensor).item()

    # Rescale the predicted conv(%) back to its original range
    target_conv_rescaled = scaler.inverse_transform([[target_conv]])[0, 0]

    # Calculate a conditioning factor based on the difference between the goal and prediction
    # This factor will amplify or reduce noise to steer the sequence towards the target conv(%)
    condition_factor = 1 + (conv_value - target_conv_rescaled) / 100.0
    noisy_matrix *= condition_factor

    # Apply softmax to get probabilities
    exp_matrix = np.exp(noisy_matrix)
    softmax_matrix = exp_matrix / exp_matrix.sum(axis=1, keepdims=True)

    # Reconstruct the sequence from the adjusted probabilities
    noisy_sequence = "".join(index_to_char[np.argmax(softmax_matrix[i])] for i in range(len(sequence)))

    return noisy_sequence

# --- Conv(%) Predictor Model ---
class ConvPredictorTransformer(torch.nn.Module):
    def __init__(self, transformer_model, hidden_dim):
        super(ConvPredictorTransformer, self).__init__()
        self.transformer = transformer_model
        self.fc1 = torch.nn.Linear(self.transformer.config.n_embd, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        transformer_output = self.transformer(input_ids=x).last_hidden_state.mean(dim=1)
        x = self.fc1(transformer_output)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize GPT-2 model and tokenizer for the predictor
conv_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if conv_tokenizer.pad_token is None:
    conv_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

conv_transformer_model = GPT2LMHeadModel.from_pretrained("gpt2")
conv_transformer_model.resize_token_embeddings(len(conv_tokenizer))

# Initialize the full predictor model and move it to the correct device
conv_predictor = ConvPredictorTransformer(conv_transformer_model.transformer, hidden_dim=128)
conv_predictor.to(device)
optimizer = torch.optim.Adam(conv_predictor.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# --- Model Training ---
def train_conv_predictor(sequences, conv_values, epochs=5, batch_size=8):
    """Trains the neural network to predict conv(%) from a sequence."""
    print("Starting conv(%) predictor model training...")
    conv_values_tensor = torch.tensor(conv_values).float().to(device)
    conv_predictor.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_conv_values = conv_values_tensor[i:i + batch_size]

            encoded_inputs = conv_tokenizer(
                batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            input_ids = encoded_inputs['input_ids'].to(device)

            optimizer.zero_grad()
            pred_conv = conv_predictor(input_ids).squeeze()

            if pred_conv.ndim == 0:
                pred_conv = pred_conv.unsqueeze(0)

            loss = criterion(pred_conv, batch_conv_values[:len(pred_conv)])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(sequences) / batch_size):.4f}")
    conv_predictor.eval() # Set model to evaluation mode after training

train_conv_predictor(sequences, conv_values_normalized, epochs=5)

# --- Sequence Generation ---
def generate_sequences_with_conditional_diffusion(base_sequences, conv_values, num_sequences=100, noise_level=0.1):
    """Generates new sequences using the conditional diffusion model."""
    generated_sequences = []
    for i in range(num_sequences):
        idx = np.random.randint(len(base_sequences))
        base_sequence = base_sequences[idx]
        conv_value = conv_values[idx]
        noisy_sequence = add_noise_to_sequence_with_model(base_sequence, conv_value, conv_predictor, noise_level)
        generated_sequences.append(noisy_sequence)
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{num_sequences} sequences...")
    return generated_sequences

print("\nGenerating new sequences with conditional diffusion model...")
generated_sequences_diffusion = generate_sequences_with_conditional_diffusion(
    sequences, conv_values_normalized, num_sequences=100, noise_level=0.1
)

# Save generated sequences to a CSV file
diffusion_df = pd.DataFrame({'Generated_Sequence': generated_sequences_diffusion})
diffusion_csv_path = os.path.join(output_dir, "generated_sequences_diffusion.csv")
diffusion_df.to_csv(diffusion_csv_path, index=False)
print(f"\nGenerated sequences saved to {diffusion_csv_path}")

# --- Feature Extraction and Visualization ---
def extract_features(sequences):
    """Extracts basic biochemical features from sequences for visualization."""
    features = []
    for seq in sequences:
        length = len(seq)
        counts = [seq.count(aa) for aa in amino_acids]
        proportions = [count / length for count in counts] if length > 0 else [0] * len(amino_acids)
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
        features.append(proportions + [length, entropy])
    return np.array(features)

print("\nExtracting features for visualization...")
original_features = extract_features(sequences)
generated_features_diffusion = extract_features(generated_sequences_diffusion)

all_features = np.vstack([original_features, generated_features_diffusion])
labels = (["Original"] * len(sequences) + ["Generated"] * len(generated_sequences_diffusion))

# Reduce dimensionality using PCA
print("Performing PCA for visualization...")
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# Visualize the feature space
plt.figure(figsize=(10, 8))
for label_name in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label_name]
    plt.scatter(
        reduced_features[indices, 0],
        reduced_features[indices, 1],
        label=label_name,
        alpha=0.7
    )

# Highlight the wild-type sequence
plt.scatter(reduced_features[0, 0], reduced_features[0, 1], color='red', s=100, edgecolors='black', label='Wild-Type')

plt.legend()
plt.title("Feature Space Visualization (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
visualization_path = os.path.join(output_dir, "feature_space_PCA.png")
plt.savefig(visualization_path)
print(f"Feature space visualization saved to {visualization_path}")
plt.show()


# --- Find and Display Farthest Sequences ---
print("\nIdentifying generated sequences farthest from the wild-type...")

# The wild-type is the first sequence, its reduced features are at index 0
wild_type_reduced_features = reduced_features[0]

# The generated sequences' features start after the original ones
num_original_seqs = len(sequences)
generated_reduced_features = reduced_features[num_original_seqs:]

# Calculate Euclidean distance in the 2D PCA space for each generated sequence
distances = np.linalg.norm(generated_reduced_features - wild_type_reduced_features, axis=1)

# Pair each generated sequence with its calculated distance
generated_with_distances = list(zip(generated_sequences_diffusion, distances))

# Sort the pairs by distance in descending order (farthest first)
sorted_generated = sorted(generated_with_distances, key=lambda x: x[1], reverse=True)

# Print the top 3 farthest sequences
print("\n--- Top 3 Farthest Generated Sequences from Wild-Type ---")
for i in range(min(3, len(sorted_generated))):
    sequence, distance = sorted_generated[i]
    print(f"\nRank {i+1} (Distance: {distance:.4f}):")
    print(sequence)
print("-" * 50)