import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import atexit
import optuna  # Import Optuna for Bayesian Optimization

device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),  
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)

# Load text & structure embeddings
text_embeddings = torch.load("cleaned_filtered_output_embedding.pt")["embeddings"].to(torch.float32)
structure_embeddings_dict = torch.load("cleaned_structure_embeddings_final.pt")["embeddings"].to(torch.float32)
structure_embeddings = torch.stack([v.squeeze(0) for v in structure_embeddings_dict])

text_embeddings = text_embeddings.to(device)
structure_embeddings = structure_embeddings.to(device)

lr = 1e-4
batch_size = 512
temperature = 0.07
neg_samples = 5
patience = 10
best_val_loss = float("inf")
best_model_path = "best_model.pth"

dataset = TensorDataset(text_embeddings, structure_embeddings)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

writer = SummaryWriter(log_dir=f"runs/lr{lr}_bs{batch_size}_temp{temperature}_neg{neg_samples}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize separate projection heads
text_projection_model = ProjectionHead().to(device)
structure_projection_model = ProjectionHead().to(device)

optimizer_text = optim.AdamW(text_projection_model.parameters(), lr=lr, weight_decay=5e-4)
optimizer_structure = optim.AdamW(structure_projection_model.parameters(), lr=lr, weight_decay=5e-4)

def ebm_nce_loss(X, Y, temperature, neg_samples):
    criterion = nn.BCEWithLogitsLoss()

    batch_size = X.shape[0]  
    
    
    sim_matrix = torch.mm(Y, Y.T)  
    

    _, hard_neg_indices = torch.topk(sim_matrix, neg_samples + 1, largest=True) 
    hard_neg_indices = hard_neg_indices[:, 1:] 
    neg_Y = Y[hard_neg_indices.flatten()].view(batch_size, neg_samples, -1)  
    
  
    neg_indices_t = torch.randint(0, len(Y), (batch_size * neg_samples,), device=Y.device)
    neg_indices_p = torch.randint(0, len(X), (batch_size * neg_samples,), device=X.device)
    
    neg_Y_t = Y[neg_indices_t].view(batch_size, neg_samples, -1)  
    neg_X_p = X[neg_indices_p].view(batch_size, neg_samples, -1)  
    
   
    pred_pos = F.cosine_similarity(X, Y, dim=1) / temperature
    
  
    pred_neg_t = F.cosine_similarity(X.unsqueeze(1), neg_Y_t, dim=2) / temperature  # Text negatives
    pred_neg_p = F.cosine_similarity(neg_X_p, Y.unsqueeze(1), dim=2) / temperature  # Protein negatives
    
    loss_pos = criterion(pred_pos, torch.ones_like(pred_pos, device=X.device))
    loss_neg_t = criterion(pred_neg_t, torch.zeros_like(pred_neg_t, device=X.device)).mean()
    loss_neg_p = criterion(pred_neg_p, torch.zeros_like(pred_neg_p, device=X.device)).mean()
    
    loss = 0.5 * (loss_pos + loss_neg_t + loss_neg_p) / (1 + neg_samples)
    
    accuracy = ((pred_pos > 0.5).sum().float() + (pred_neg_t < -0.5).sum().float() +
                (pred_neg_p < -0.5).sum().float()) / (len(pred_pos) + len(pred_neg_t.flatten()) + len(pred_neg_p.flatten()))
    
    return loss, accuracy.detach()


def train_ebm(text_model, struct_model, train_loader, val_loader, optimizer_text, optimizer_struct, temperature, neg_samples, patience=10, num_epochs=10):

    text_model.requires_grad_(True)
    struct_model.requires_grad_(True)
    
    text_model.train()
    struct_model.train()

    best_loss = float("inf")
    early_stopping = EarlyStopping(patience)
    loss_log = []

    writer = SummaryWriter(log_dir="logs")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for text_batch, struct_batch in train_loader:
            step = (epoch * len(train_loader)) + num_batches

           
            projected_text = text_model(text_batch)  
            projected_struct = struct_model(struct_batch) 
           
            sim_scores = torch.cosine_similarity(projected_text.unsqueeze(1), projected_struct.unsqueeze(0), dim=-1)
            loss_ebm, acc = ebm_nce_loss(projected_text, projected_struct, temperature, int(neg_samples))

            
            loss_log.append(loss_ebm.item())

          
            optimizer_text.zero_grad()
            optimizer_struct.zero_grad()
            
            with torch.autograd.set_detect_anomaly(True):
                loss_ebm.backward()
                torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(struct_model.parameters(), max_norm=1.0)
                
                optimizer_text.step()
                optimizer_struct.step()

            epoch_loss += loss_ebm.item()
            num_batches += 1

           
            if loss_ebm.item() < best_loss:
                best_loss = loss_ebm.item()
                print(f"New best model saved with loss: {best_loss:.4f}")

        epoch_loss /= num_batches  # Get average loss for the epoch

        text_model.eval()
        struct_model.eval()
        
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for text_batch, struct_batch in val_loader:
                projected_text = text_model(text_batch)
                projected_struct = struct_model(struct_batch)

                # Compute contrastive similarity loss
                sim_scores = torch.cosine_similarity(projected_text.unsqueeze(1), projected_struct.unsqueeze(0), dim=-1)
                loss_ebm, acc = ebm_nce_loss(projected_text, projected_struct, temperature, int(neg_samples))
    
                val_loss += loss_ebm.item()
                num_val_batches += 1

        val_loss /= num_val_batches  # Average validation loss
        writer.add_scalar("Loss/val", val_loss, epoch)  

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopping(val_loss):
            print(f"Early stopping activated at epoch {epoch}. Stopping training!")
            break
        
        text_model.train()
        struct_model.train()

    print("Training complete")
    atexit.register(writer.close)

val_loss = train_ebm(text_projection_model, structure_projection_model, train_loader, val_loader, optimizer_text, optimizer_structure, temperature, neg_samples)


model_path_text = f"model_lr{lr}_bs{batch_size}_temp{temperature}_neg{neg_samples}_text.pth"
model_path_struct = f"model_lr{lr}_bs{batch_size}_temp{temperature}_neg{neg_samples}_struct.pth"

torch.save(text_projection_model.state_dict(), f"{model_path_text}")
torch.save(structure_projection_model.state_dict(), f"{model_path_struct}")

