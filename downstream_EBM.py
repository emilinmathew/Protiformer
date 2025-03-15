import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.decomposition import PCA

# Define joint projection head
class JointProjectionHead(nn.Module):
    def __init__(self, input_dim=768 * 2, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_emb, struct_emb):
        joint_emb = torch.cat([text_emb, struct_emb], dim=-1)  # Concatenate along feature dim
        return self.projection(joint_emb)

# Load model & move to correct device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = JointProjectionHead().to(device)

# Load best model weights
model.eval()

# Load new text & structure embeddings
text_embeddings_data = torch.load("cleaned_filtered_output_embedding.pt")
new_text_embeddings = text_embeddings_data["embeddings"].to(torch.float32).to(device)
text_ids = text_embeddings_data["ids"]  # Assuming the IDs are stored under the key "ids"

structure_embeddings_data = torch.load("cleaned_structure_embeddings_final.pt")
new_structure_embeddings_dict = structure_embeddings_data["embeddings"].to(torch.float32)
new_structure_embeddings = torch.stack([v.squeeze(0) for v in new_structure_embeddings_dict]).to(device)
structure_ids = structure_embeddings_data["ids"]  # Assuming the IDs are stored under the key "ids"

with torch.no_grad():
    projected_text = model(new_text_embeddings, new_structure_embeddings)
    projected_struct = model(new_structure_embeddings, new_text_embeddings)

print(f" Embedding projection completed")

# Save projected embeddings with IDs
torch.save({"ids": text_ids, "projected_text": projected_text}, "projected_text.pt")
torch.save({"ids": structure_ids, "projected_struct": projected_struct}, "projected_struct.pt")

print(" Projected embeddings saved!")

# Task 1 - Compute similarity matrix
# Convert embeddings to numpy

text_repr = projected_text.clone().detach()
struct_repr = projected_struct.clone().detach()

# Normalize embeddings
text_repr = F.normalize(text_repr, p=2, dim=-1)
struct_repr = F.normalize(struct_repr, p=2, dim=-1)

# Task 2 - Visualize embeddings

# Reduce dimensionality

# **Reduce dimensionality with t-SNE**

print(" Loading embeddings")
text_repr_np = text_repr.cpu().numpy()
struct_repr_np = struct_repr.cpu().numpy()

subset_size = 100
indices = np.random.choice(len(text_repr_np), subset_size, replace=False)  # Randomly select 100 samples

# Extract only the subset
text_subset = text_repr_np[indices]
struct_subset = struct_repr_np[indices]

# Reduce dimensions first with PCA
pca = PCA(n_components=50)  # Reduce to 50 dimensions first
text_struct_combined = np.vstack([text_subset, struct_subset])  # Stack only the subset
pca_embeddings = pca.fit_transform(text_struct_combined)  # Apply PCA only on the subset

# Initialize t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, method="barnes_hut", angle=0.5)

# Apply t-SNE to the PCA-reduced subset
reduced_embeddings = tsne.fit_transform(pca_embeddings)
print(text_subset.shape, struct_subset.shape, reduced_embeddings.shape)

print(" t-SNE completed on subset")

plt.figure(figsize=(8, 6))

plt.scatter(reduced_embeddings[:100, 0], reduced_embeddings[:100, 1], 
            label="Text Embeddings", alpha=0.5, color="blue", s=10)
plt.scatter(reduced_embeddings[100:, 0], reduced_embeddings[100:, 1], 
            label="Structure Embeddings", alpha=0.5, color="orange", s=10)

plt.xlabel("t-SNE Dimension 1")  # X-axis label
plt.ylabel("t-SNE Dimension 2")  # Y-axis label
plt.title("t-SNE Visualization of Text vs Structure Embeddings (LogSigmoidNCE Loss)")
plt.legend()
plt.show()

def fast_pairwise_similarity(text_emb, struct_emb, num_samples=1000):
    """
    Computes cosine similarity for a random subset of (text, structure) pairs.
    Much faster than full NxN similarity matrices.
    """
    indices = np.random.choice(len(text_emb), num_samples, replace=False)
    text_subset = text_emb[indices]
    struct_subset = struct_emb[indices]
    
    similarity_scores = (text_subset * struct_subset).sum(dim=-1).cpu().numpy()  # Compute cosine similarity
    return similarity_scores

print(" Computing pairwise similarity for a subset...")
pairwise_similarities = fast_pairwise_similarity(projected_text, projected_struct, num_samples=1000)

mean_similarity = np.mean(pairwise_similarities)
print(f" Mean Cosine Similarity: {mean_similarity:.4f}")

# Plot histogram of similarity scores
plt.figure(figsize=(6, 4))
plt.hist(pairwise_similarities, bins=30, alpha=0.7, color="blue", edgecolor="black")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Text-Structure Cosine Similarities")
plt.show()

eval_results = {
    "mean_cosine_similarity": mean_similarity
}

torch.save(eval_results, "fast_contrastive_eval_results.pt")

