import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse

class CustomLRAdapter:
    def __init__(self, optimizer, initial_lr, min_lr=1e-6, factor=0.5, patience=2):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.best_val_loss = float('inf')
        self.num_bad_epochs = 0

    def step(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group['lr'] * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate to {new_lr}")

class EditAwareGAT(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edit_emb):
        if edit_emb.shape[1] != x.shape[0]:  
            edit_emb = edit_emb.reshape(-1, edit_emb.shape[-1])
        x = x + edit_emb
        x = super().forward(x, edge_index)
        return x


class ProteinEditDecoder(nn.Module):
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', structure_emb_dim=768, 
                 text_emb_dim=128, hidden_dim=256, num_heads=4):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.edit_encoder = AutoModel.from_pretrained(model_name)
        self.edit_dim = self.edit_encoder.config.hidden_size
        

        self.structure_proj = nn.Linear(structure_emb_dim, hidden_dim)
        self.text_proj = nn.Linear(text_emb_dim, hidden_dim)
        self.edit_proj = nn.Linear(self.edit_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        self.gnn_layers = nn.ModuleList([
            EditAwareGAT(hidden_dim, hidden_dim, heads=4, concat=False),
            EditAwareGAT(hidden_dim, hidden_dim, heads=4, concat=False)
        ])

        self.node_projection = nn.Linear(hidden_dim * 3, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.coord_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  
        )
        
        self.aa_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20) 
        )
    
    def forward(self, structure_emb, text_emb, node_features, edge_index, edit_instructions=None, node_mask=None):
        batch_size, num_nodes, _ = structure_emb.shape
        
       
        struct_features = self.structure_proj(structure_emb)
        text_features = self.text_proj(text_emb)
        
        with torch.no_grad():
            encoded_inputs = self.tokenizer(edit_instructions, padding=True, truncation=True, return_tensors="pt")
            encoded_inputs = {key: value.to(text_emb.device) for key, value in encoded_inputs.items()}
            edit_emb = self.edit_encoder(**encoded_inputs).last_hidden_state[:, 0, :]

        edit_emb = self.edit_proj(edit_emb)
        edit_emb = edit_emb.unsqueeze(1)


        text_features = text_features.unsqueeze(1).expand(-1, num_nodes, -1)
        edit_features = edit_emb.expand(-1, num_nodes, -1)
        
        if node_mask is None:
            node_mask = (structure_emb.abs().sum(dim=-1) > 0)

        combined = torch.cat([struct_features, text_features, edit_features], dim=-1)
        node_embeddings = self.node_projection(combined)
        node_embeddings = node_embeddings * node_mask.unsqueeze(-1)
        node_embeddings = node_embeddings.view(-1, self.hidden_dim)

        valid_edges = (edge_index >= 0).all(dim=0)
        edge_index = edge_index[:, valid_edges]
        edge_index = edge_index[:, edge_index.min(dim=0)[0] >= 0]

        for gnn_layer in self.gnn_layers:
            node_embeddings = gnn_layer(node_embeddings, edge_index, edit_features)
            node_embeddings = F.relu(node_embeddings)

        node_embeddings = node_embeddings.view(batch_size, num_nodes, -1)
        

        text_ebm = self.text_proj(text_emb)
        global_text_edit = torch.cat([text_ebm.unsqueeze(1), edit_emb], dim=1)
        
        global_attn_scores = torch.bmm(node_embeddings, global_text_edit.transpose(1, 2))
        attention_mask = (1.0 - node_mask.float().unsqueeze(-1)) * -10000.0
        global_attn_scores = global_attn_scores + attention_mask
        global_attn_weights = F.softmax(global_attn_scores, dim=2)
        weighted_global = torch.bmm(global_attn_weights, global_text_edit)

        struct_attn_output, _ = self.cross_attn(
            node_embeddings.permute(1, 0, 2),
            text_features.permute(1, 0, 2),
            text_features.permute(1, 0, 2)
        )
        struct_attn_output = struct_attn_output.permute(1, 0, 2)

        combined_features = node_embeddings + weighted_global + struct_attn_output
        
        coords = self.coord_pred(combined_features)
        aa_logits = self.aa_pred(combined_features)
        
        coords = coords * node_mask.unsqueeze(-1)
        aa_logits = aa_logits * node_mask.unsqueeze(-1)
        
        aa_probs = torch.zeros_like(aa_logits)
        for i in range(batch_size):
            valid_idx = node_mask[i]
            if valid_idx.sum() > 0:
                aa_probs[i, valid_idx] = F.softmax(aa_logits[i, valid_idx], dim=-1)
        
        return coords, aa_probs


def consistency_loss(pred_coords, struct_embeddings, node_mask):
    """Ensures that the predicted coordinates are consistent with the node embeddings."""
    batch_size, num_nodes, _ = pred_coords.shape
    
    struct_embeddings = struct_embeddings[:, :num_nodes, :]
    node_mask = node_mask[:, :num_nodes]
    
    pred_coords = pred_coords * node_mask.unsqueeze(-1)
    struct_embeddings = struct_embeddings * node_mask.unsqueeze(-1)
    
    coord_dists = torch.cdist(pred_coords, pred_coords, p=2)
    embed_norm = F.normalize(struct_embeddings, p=2, dim=-1)
    embed_sims = torch.bmm(embed_norm, embed_norm.transpose(1, 2))
    
    loss_matrix = coord_dists * embed_sims
    mask = torch.eye(loss_matrix.shape[1], device=loss_matrix.device).bool().unsqueeze(0)
    loss_matrix = loss_matrix.masked_fill(mask, 0)
    
    valid_counts = node_mask.sum(dim=-1).float() - 1
    loss = loss_matrix.sum(dim=[1, 2]) / (valid_counts * valid_counts.clamp(min=1))
    
    return loss.mean()


def rmsd_loss(pred_coords, node_mask):
    """Calculates RMSD loss for predicted coordinates."""
    batch_size = pred_coords.shape[0]
    loss = torch.tensor(0.0, device=pred_coords.device)
    
    for b in range(batch_size):
        valid_coords = pred_coords[b, node_mask[b]]
        if valid_coords.shape[0] <= 1:
            continue
        
        centered_coords = valid_coords - valid_coords.mean(dim=0, keepdim=True)
        # Calculate RMSD
        rmsd = torch.sqrt(torch.mean(torch.sum(centered_coords**2, dim=1)) + 1e-8)
        loss += rmsd
    
    return loss / batch_size if batch_size > 0 else torch.tensor(0.0, device=pred_coords.device)


def smoothness_loss(pred_coords, node_mask):
    """Calculates smoothness loss for predicted coordinates."""
    batch_size = pred_coords.shape[0]
    total_loss = torch.tensor(0.0, device=pred_coords.device)
    valid_batch_count = 0
    
    for b in range(batch_size):
        valid_mask = node_mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count <= 1:
            continue
        
        valid_coords = pred_coords[b, valid_mask]
        
        diffs = valid_coords.unsqueeze(1) - valid_coords.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)
        
        # Create mask 
        diag_mask = ~torch.eye(valid_count, dtype=torch.bool, device=distances.device)
        total_loss += torch.sum(distances * diag_mask.float()) / (diag_mask.float().sum() + 1e-8)
        valid_batch_count += 1
    
    return total_loss / valid_batch_count if valid_batch_count > 0 else torch.tensor(0.0, device=pred_coords.device)


def plot_training_results(history, results_dir, epoch=None):
    """Generate and save plots of training metrics."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Training and validation loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Component losses
    axes[0, 1].plot(epochs, history['rmsd_loss'], 'g-', label='RMSD Loss')
    axes[0, 1].plot(epochs, history['smoothness_loss'], 'c-', label='Smoothness Loss')
    axes[0, 1].plot(epochs, history['consistency_loss'], 'm-', label='Consistency Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Validation component losses
    axes[1, 0].plot(epochs, history['val_rmsd_loss'], 'g--', label='Val RMSD Loss')
    axes[1, 0].plot(epochs, history['val_smoothness_loss'], 'c--', label='Val Smoothness Loss')
    axes[1, 0].plot(epochs, history['val_consistency_loss'], 'm--', label='Val Consistency Loss')
    axes[1, 0].set_title('Validation Component Losses')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Learning rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'k-')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True)

    plt.tight_layout()

    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    fig_path = Path(results_dir) / f"training_plots{epoch_str}.png"
    plt.savefig(fig_path)
    plt.close()

    print(f"Training plots saved to {fig_path}")

def main():
    
    parser = argparse.ArgumentParser(description='Train ProteinEditDecoder')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--dataset_path', type=str, default='pdb_text_embeddings_with_graphs.pt', 
                        help='Path to the dataset file')
    args = parser.parse_args()
    
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    

    print(f"Loading dataset from {args.dataset_path}")
    data = torch.load(args.dataset_path, weights_only=False)
    print(f"Dataset loaded with {len(data)} entries")
    
    dataset = []
    for pdb_key, entry in data.items():
        graph_data = entry["graph_data"]
        
        structure_emb = torch.tensor(graph_data["node_embeddings"], dtype=torch.float32)
        
        text_emb = torch.tensor(entry["text_embedding"], dtype=torch.float32).clone().detach()
        
        node_features = torch.tensor(graph_data["node_features"], dtype=torch.float32)
        
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
        
        dataset.append((structure_emb, text_emb, node_features, edge_index, pdb_key))
    
    print(f"Prepared {len(dataset)} samples for training")

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    
    print(f"Dataset Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    

    batch_size = args.batch_size
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, 
                             collate_fn=lambda x: list(zip(*x)))
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, 
                           collate_fn=lambda x: list(zip(*x)))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda x: list(zip(*x)))
        
    print(f"Dataset Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    

    device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    
    model = ProteinEditDecoder(
            structure_emb_dim=768,
            text_emb_dim=128,
            hidden_dim=256,
            num_heads=4,
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_adapter = CustomLRAdapter(optimizer, initial_lr=args.lr, min_lr=1e-6, factor=0.5, patience=2)

    print("Starting training...")

    history = {
        'train_loss': [], 'val_loss': [],
        'rmsd_loss': [], 'smoothness_loss': [], 'consistency_loss': [],
        'val_rmsd_loss': [], 'val_smoothness_loss': [], 'val_consistency_loss': [],
        'learning_rate': []
    }

    best_val_loss = float("inf")
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_rmsd_loss = 0
        total_smoothness_loss = 0
        total_consistency_loss = 0
    
    print("Starting training...")
    
    history = {
        'train_loss': [], 'val_loss': [],
        'rmsd_loss': [], 'smoothness_loss': [], 'consistency_loss': [],
        'val_rmsd_loss': [], 'val_smoothness_loss': [], 'val_consistency_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float("inf")
    num_epochs = args.num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_rmsd_loss = 0
        total_smoothness_loss = 0
        total_consistency_loss = 0
        
        for batch in train_loader:
            structure_embs, text_embs, node_features_list, edge_indices, pdb_keys = batch
            batch_size = len(structure_embs)
            
            max_edges = max(ei.shape[1] for ei in edge_indices)

            structure_embs = pad_sequence(structure_embs, batch_first=True, padding_value=0).to(device)

            node_features = pad_sequence(node_features_list, batch_first=True, padding_value=0).to(device)

            edge_indices_padded = [
                torch.cat([ei, torch.full((2, max_edges - ei.shape[1]), -1, dtype=torch.long)], dim=1).to(device)
                if ei.shape[1] < max_edges else ei.to(device)
                for ei in edge_indices
            ]
            edge_indices = torch.stack(edge_indices_padded, dim=0).to(device)

            node_mask = (structure_embs.abs().sum(dim=-1) > 0).to(device)

            text_embs = torch.stack(text_embs).to(device)

            original_instructions = [
            "Increase alpha helices", 
            "Decrease alpha helices", 
            "Increase beta sheets", 
            "Decrease beta sheets", 
            "Increase ordered regions", 
            "Decrease ordered regions", 
            "Increase Vilin stability", 
            "Decrease Vilin stability", 
            "Increase Pin1 stability", 
            "Decrease Pin1 stability", 
            "Increase binding affinity", 
            "Decrease binding affinity"
            ]

            edit_instructions = [original_instructions[i % len(original_instructions)] for i in range(batch_size)]

            pred_coords, aa_probs = model(structure_embs, text_embs, node_features, edge_indices, edit_instructions, node_mask)

            loss_rmsd = rmsd_loss(pred_coords, node_mask)
            loss_smoothness = smoothness_loss(pred_coords, node_mask)
            loss_consistency = consistency_loss(pred_coords, structure_embs, node_mask)
            
            loss_consistency = loss_consistency / (loss_consistency.detach() + 1e-8)
            
            rmsd_weight = 1.0
            smoothness_weight = 0.5
            consistency_weight = 0.5
            
            loss = rmsd_weight * loss_rmsd + smoothness_weight * loss_smoothness + consistency_weight * loss_consistency
            
            total_loss += loss.item()
            total_rmsd_loss += loss_rmsd.item()
            total_smoothness_loss += loss_smoothness.item()
            total_consistency_loss += loss_consistency.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        avg_rmsd_loss = total_rmsd_loss / len(train_loader)
        avg_smoothness_loss = total_smoothness_loss / len(train_loader)
        avg_consistency_loss = total_consistency_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        total_val_rmsd_loss = 0
        total_val_smoothness_loss = 0
        total_val_consistency_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                structure_embs, text_embs, node_features_list, edge_indices, pdb_keys = batch
                batch_size = len(structure_embs)
                
                max_edges = max(ei.shape[1] for ei in edge_indices)
                
                structure_embs = pad_sequence(structure_embs, batch_first=True, padding_value=0).to(device)

                node_features = pad_sequence(node_features_list, batch_first=True, padding_value=0).to(device)

                edge_indices_padded = [
                    torch.cat([ei, torch.full((2, max_edges - ei.shape[1]), -1, dtype=torch.long)], dim=1).to(device)
                    if ei.shape[1] < max_edges else ei.to(device)
                    for ei in edge_indices
                ]
                edge_indices = torch.stack(edge_indices_padded, dim=0).to(device)
                
                # Create node mask
                node_mask = (structure_embs.abs().sum(dim=-1) > 0).to(device)
                
                # Convert text embeddings
                text_embs = torch.stack(text_embs).to(device)

                original_instructions = [
                "Increase alpha helices", 
                "Decrease alpha helices", 
                "Increase beta sheets", 
                "Decrease beta sheets", 
                "Increase ordered regions", 
                "Decrease ordered regions", 
                "Increase Vilin stability", 
                "Decrease Vilin stability", 
                "Increase Pin1 stability", 
                "Decrease Pin1 stability", 
                "Increase binding affinity", 
                "Decrease binding affinity"
                 ]

                edit_instructions = [original_instructions[i % len(original_instructions)] for i in range(batch_size)]

                
                # Forward pass
                pred_coords, aa_probs = model(structure_embs, text_embs, node_features, edge_indices, edit_instructions, node_mask)
                
                # Calculate losses
                val_rmsd_loss = rmsd_loss(pred_coords, node_mask)
                val_smoothness_loss = smoothness_loss(pred_coords, node_mask)
                val_consistency_loss = consistency_loss(pred_coords, structure_embs, node_mask)
                
                # Normalize consistency loss
                val_consistency_loss = val_consistency_loss / (val_consistency_loss.detach() + 1e-8)
                
                # Calculate total loss
                val_loss = rmsd_weight * val_rmsd_loss + smoothness_weight * val_smoothness_loss + consistency_weight * val_consistency_loss
                
                # Track losses
                total_val_loss += val_loss.item()
                total_val_rmsd_loss += val_rmsd_loss.item()
                total_val_smoothness_loss += val_smoothness_loss.item()
                total_val_consistency_loss += val_consistency_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_rmsd_loss = total_val_rmsd_loss / len(val_loader)
        avg_val_smoothness_loss = total_val_smoothness_loss / len(val_loader)
        avg_val_consistency_loss = total_val_consistency_loss / len(val_loader)
        
        # Learning rate scheduling
        lr_adapter.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['rmsd_loss'].append(avg_rmsd_loss)
        history['smoothness_loss'].append(avg_smoothness_loss)
        history['consistency_loss'].append(avg_consistency_loss)
        history['val_rmsd_loss'].append(avg_val_rmsd_loss)
        history['val_smoothness_loss'].append(avg_val_smoothness_loss)
        history['val_consistency_loss'].append(avg_val_consistency_loss)
        history['learning_rate'].append(current_lr)
        

        # Add this line to generate and save the plots after each epoch
        plot_training_results(history, results_dir, epoch=epoch+1)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(results_dir / "best_protein_decoder.pth"))
            print("Saved best model")
         
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"  RMSD: {avg_rmsd_loss:.4f} | Smooth: {avg_smoothness_loss:.4f} | "
              f"Consist: {avg_consistency_loss:.4f}")
    
    print(" Training Complete")
    plot_training_results(history, results_dir)

    print("Testing best model...")
    model.load_state_dict(torch.load(str(results_dir / "best_protein_decoder.pth")))
    model.eval()
    

    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            structure_embs, text_embs, node_features_list, edge_indices, pdb_keys = batch
            batch_size = len(structure_embs)
            
            if batch_size == 0:
                continue

            max_edges = max(ei.shape[1] for ei in edge_indices)
            structure_embs = pad_sequence(structure_embs, batch_first=True, padding_value=0).to(device)
            node_features = pad_sequence(node_features_list, batch_first=True, padding_value=0).to(device)
            
            edge_indices_padded = [
                torch.cat([ei, torch.full((2, max_edges - ei.shape[1]), -1, dtype=torch.long)], dim=1).to(device)
                if ei.shape[1] < max_edges else ei.to(device)
                for ei in edge_indices
            ]
            edge_indices = torch.stack(edge_indices_padded, dim=0).to(device)
            
            node_mask = (structure_embs.abs().sum(dim=-1) > 0).to(device)
            text_embs = torch.stack(text_embs).to(device)
            
            original_instructions = [
            "Increase alpha helices", 
            "Decrease alpha helices", 
            "Increase beta sheets", 
            "Decrease beta sheets", 
            "Increase ordered regions", 
            "Decrease ordered regions", 
            "Increase Vilin stability", 
            "Decrease Vilin stability", 
            "Increase Pin1 stability", 
            "Decrease Pin1 stability", 
            "Increase binding affinity", 
            "Decrease binding affinity"
            ]

            edit_instructions = [original_instructions[i % len(original_instructions)] for i in range(batch_size)]


            pred_coords, aa_probs = model(structure_embs, text_embs, node_features, edge_indices, edit_instructions, node_mask)
            
            test_rmsd = rmsd_loss(pred_coords, node_mask)
            test_smoothness = smoothness_loss(pred_coords, node_mask)
            test_consistency = consistency_loss(pred_coords, structure_embs, node_mask)
            
            test_loss = rmsd_weight * test_rmsd + smoothness_weight * test_smoothness + consistency_weight * test_consistency
            total_test_loss += test_loss.item()
    
    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("Testing complete!")


if __name__ == "__main__":
    main()

# Define the generate_pdb function
def generate_pdb(coords, aa_probs, chain_id='A', output_file="output.pdb"):
    """
    Generate a PDB file from predicted coordinates and amino acid probabilities.

    Args:
        coords (torch.Tensor): Predicted 3D coordinates (num_nodes, 3).
        aa_probs (torch.Tensor): Predicted amino acid probabilities (num_nodes, 20).
        chain_id (str): Chain identifier (default: 'A').
        output_file (str): Path to save the PDB file.
    """
    # Convert probabilities to amino acid names
    aa_indices = torch.argmax(aa_probs, dim=-1)  # (num_nodes,)
    amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                   "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    predicted_aa = [amino_acids[idx] for idx in aa_indices]

    # Write PDB file
    with open(output_file, "w") as f:
        # Write header
        f.write("HEADER    Generated by ProteinEditDecoder\n")
        f.write("REMARK    Predicted structure\n")
        f.write("REMARK    Coordinates are in Angstroms\n")
        
        #  atom records
        atom_number = 1
        residue_number = 1
        for i, (aa, (x, y, z)) in enumerate(zip(predicted_aa, coords)):
            #ATOM record for the backbone CA atom
            f.write(f"ATOM  {atom_number:5d}  CA  {aa:3s} {chain_id}{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            atom_number += 1
            residue_number += 1
        
        #TER record to mark the end of the chain
        f.write("TER\n")
        f.write("END\n")

    print(f"PDB file saved to {output_file}")

