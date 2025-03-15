import torch.nn as nn


class ProteinTextModel(nn.Module):
    def __init__(self, text2latent_model, structure2latent_model):
        super().__init__()
        #self.text_model = text_model
        self.text2latent_model = text2latent_model
        #self.structure_model = structure_model
        self.structure2latent_model = structure2latent_model
        return
    
    def forward(self, text_embeddings, structure_embeddings):
        structure_repr = self.structure2latent_model(structure_embeddings)
        description_repr = self.text2latent_model(text_embeddings)
        
        return structure_repr, description_repr
