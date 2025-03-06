import pickle
import os

import pandas as pd
import torch.nn as nn

# Define the model
class Model(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_func, dropout):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(nn.BatchNorm1d(input_dim))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_layers[i]))
                layers.append(getattr(nn, activation_func)())
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                layers.append(getattr(nn, activation_func)())
                layers.append(nn.BatchNorm1d(hidden_layers[i]))
                if dropout is not None:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
general_path = os.path.join(os.getcwd(), "data_genomit\\no_symptoms")

all_scores = pd.read_pickle(os.path.join(general_path, "all_scores.pkl"))
all_models = pd.read_pickle(os.path.join(general_path, "all_models.pkl"))
all_configs = pd.read_pickle(os.path.join(general_path, "all_configs.pkl"))

# Sort the scores in descending order based on the F1-score (first value in tuple), keeping corresponding accuracy (second value)
sorted_scores_with_indices = sorted(enumerate(all_scores), key=lambda x: x[1][0], reverse=True)

# Sort the corresponding other files based on the sorted scores
all_scores = [all_scores[idx] for idx, _ in sorted_scores_with_indices]
all_models = [all_models[idx] for idx, _ in sorted_scores_with_indices]
all_configs = [all_configs[idx] for idx, _ in sorted_scores_with_indices]

# Ask user to choose the best model for testing
print("\nChoose the best model configuration for the test set:")
for idx, (original_idx, (f1_training, accuracy_training)) in enumerate(sorted_scores_with_indices):
    print(f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}")

model_idx = int(input("Enter the index number of the model to evaluate on the test set: ")) - 1