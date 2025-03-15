import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast
import gc 
class TextProcessor:
    def __init__(self, model_name='allenai/scibert_scivocab_uncased', emb_dim=256, checkpoint_dir=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_dim = 768  # SciBERT output dimension
        self.emb_dim = emb_dim

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../data/temp_pretrained_SciBert")
        self.text_model = AutoModel.from_pretrained(model_name, cache_dir="../data/temp_pretrained_SciBert")
        
        # Linear transformation layer
        self.text2latent = nn.Linear(self.text_dim, self.emb_dim)

        # Load pre-trained weights if provided
        if checkpoint_dir:
            text_ckpt_path = "text_model.pth"
            text2latent_ckpt_path = "text2latent_model.pth"
            
            # Load text model with strict=False to ignore unexpected keys
            state_dict = torch.load(text_ckpt_path, map_location=self.device)
            self.text_model.load_state_dict(state_dict, strict=False)
            
            self.text2latent.load_state_dict(torch.load(text2latent_ckpt_path, map_location=self.device))

        # Move to device
        self.text_model.to(self.device)
        self.text2latent.to(self.device)

        # Set to eval mode
        self.text_model.train()
        self.text2latent.train()

    def encode_text(self, text):
        """Tokenizes and encodes text into an embedding."""
        print(f"Tokenizing text: {text[:]}...") 
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with autocast():  # Enable mixed precision
                text_embeddings = self.text_model(**inputs).last_hidden_state[:, 0, :]  # CLS token embedding
        
        # Free up memory
        del inputs
        torch.cuda.empty_cache()
        
        return text_embeddings


# Initialize text processor
text_processor = TextProcessor(checkpoint_dir="/Users/divya/Downloads/", device=None, emb_dim=256)

# Read from file
input_file = "filtered_text_sequence.txt"
output_file = "filtered_output_embeddings.pt"

embeddings_list = []
ids_list = []  # Store matching IDs

# Reduce batch size to avoid CUDA out of memory errors
chunk_size = 2  # Process 2 texts at a time

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i in range(0, len(lines), chunk_size * 2):
        chunk = lines[i:i + chunk_size * 2]
        batch_texts = []
        batch_ids = []
        for j in range(0, len(chunk), 2):
            if j + 1 >= len(chunk):
                break
                
            current_id = chunk[j].strip()
            description = chunk[j + 1].strip()
            batch_texts.append(description)
            batch_ids.append(current_id)
        
        # Process the batch of texts
        embeddings = text_processor.encode_text(batch_texts)
        
        # Move embeddings to CPU before clearing GPU memory
        embeddings_cpu = embeddings.cpu()
        embeddings_list.append(embeddings_cpu)
        ids_list.extend(batch_ids)

        # Free up GPU memory
        del embeddings
        del batch_texts
        del batch_ids
        gc.collect()
        torch.cuda.empty_cache()

# Save filtered embeddings and IDs to a file
if embeddings_list:
    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    data_dict = {"ids": ids_list, "embeddings": embeddings_tensor}
    
    torch.save(data_dict, output_file)
    print(f"Filtered embeddings and IDs saved to {output_file}")
else:
    print("No matching text data found.")
