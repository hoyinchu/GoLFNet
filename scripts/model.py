import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, EsmModel


class ESMCombinedModel(nn.Module):
    def __init__(self, esm_model_name, num_additional_features, out_features):
        super(ESMCombinedModel, self).__init__()
        self.esm_model = AutoModel.from_pretrained(esm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

        # Assuming the last layer's hidden size for ESM model
        esm_hidden_size = self.esm_model.config.hidden_size

        # Define additional layers
        self.linear = nn.Linear(esm_hidden_size + num_additional_features, out_features)

    def forward(self, amino_acid_sequences, additional_features):
        # Tokenize the sequences
        inputs = self.tokenizer(amino_acid_sequences, return_tensors="pt", padding=True, truncation=True)
        for key in inputs:
            inputs[key] = inputs[key].to(additional_features.device)
        with torch.no_grad():  # No need to track gradients for the pretrained model
            outputs = self.esm_model(**inputs)
            esm_embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling the outputs
            esm_embeddings = esm_embeddings.to(additional_features.device)

        # Check and adjust dimensions if necessary
        if len(esm_embeddings.shape) == 1:
            esm_embeddings = esm_embeddings.unsqueeze(0)
        if len(additional_features.shape) == 1:
            additional_features = additional_features.unsqueeze(0)

        combined_features = torch.cat((esm_embeddings, additional_features), dim=1)   
        # Pass through the linear layer
        output = self.linear(combined_features)
        return output
