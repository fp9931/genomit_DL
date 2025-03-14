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
    

# symp ADASYN path
data_path = "D:\\Dottorato\\Progetti\\Genomit\\Results\\no_symptoms\\ADASYN"
result_path = "D:\\Dottorato\\Progetti\\Genomit\\Results\\Complete\\no_symptoms\\ADASYN"
if not os.path.exists(result_path):
    os.makedirs(result_path)

all_scores_symp_ADASYN = []
all_models_symp_ADASYN = []
all_configs_symp_ADASYN = []
for k, folder in enumerate(os.listdir(data_path)):
    print(folder)
    folder_path = os.path.join(data_path, folder)
    all_scores = pd.read_pickle(os.path.join(folder_path, "all_scores.pkl"))
    #all_models = pd.read_pickle(os.path.join(folder_path, "all_models.pkl"))   # Promblem with loading the model [pkl truncated]
    all_configs = pd.read_pickle(os.path.join(folder_path, "all_configs.pkl"))

    for i, (f1_training, accuracy_training) in enumerate(all_scores):
        all_scores_symp_ADASYN.append((f1_training, accuracy_training))
        all_configs_symp_ADASYN.append(all_configs[i])
    
    # for (best_model) in all_models[:-1]:
    #     all_models_symp_ADASYN.append((best_model))

with open(os.path.join(result_path, "all_scores.pkl"), 'wb') as file:
    pickle.dump(all_scores_symp_ADASYN, file)
# with open(os.path.join(symp_ADASYN_path, "all_models.pkl"), 'wb') as file:
#     pickle.dump(all_models_symp_ADASYN, file)
with open(os.path.join(result_path, "all_configs.pkl"), 'wb') as file:
    pickle.dump(all_configs_symp_ADASYN, file)

# # # Trova il massimo f1-score
# original_idx = all_scores_symp_ADASYN.index(max(all_scores_symp_ADASYN, key=lambda x: x[0]))
# print(original_idx)
# f1_training = all_scores_symp_ADASYN[original_idx][0]
# accuracy_training = all_scores_symp_ADASYN[original_idx][1]
# print("Best model Symp ADASYN:")
# print(f"F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}")
# print(f"Confision matrix: {all_configs_symp_ADASYN[original_idx]['confusion_matrix']}")
# print('Number of features: ',len(all_configs_symp_ADASYN[original_idx]['feature set']), '\nNumber of hidden layers: ', all_configs_symp_ADASYN[original_idx]['hyperparameters']['hidden layer'], '\nActivation function: ', all_configs_symp_ADASYN[original_idx]['hyperparameters']['activation function'], '\nLearning rate: ', all_configs_symp_ADASYN[original_idx]['hyperparameters']['learning rate'], '\nDropout: ', all_configs_symp_ADASYN[original_idx]['hyperparameters']['dropout'], '\nBatch size: ', all_configs_symp_ADASYN[original_idx]['hyperparameters']['batch_size'])


sorted_scores_with_indices = sorted(enumerate(all_scores_symp_ADASYN), key=lambda x: x[1][0], reverse=True)

# # # Sort the corresponding other files based on the sorted scores
all_scores = [all_scores_symp_ADASYN[idx] for idx, _ in sorted_scores_with_indices]
# all_models = [all_models[idx] for idx, _ in sorted_scores_with_indices]
all_configs = [all_configs_symp_ADASYN[idx] for idx, _ in sorted_scores_with_indices]

# # Ask user to choose the best model for testing
for idx, (original_idx, (f1_training, accuracy_training)) in enumerate(sorted_scores_with_indices):
    if idx == 0:
        print("Best model Symp ADASYN:")
        print(f"{original_idx + 1}: F1-score = {f1_training:.3f}, Accuracy = {accuracy_training:.3f}")
        print(f"Confision matrix: {all_configs[idx]['confusion_matrix']}")
        print('Number of features: ',len(all_configs[idx]['feature set']), '\nNumber of hidden layers: ', all_configs[idx]['hyperparameters']['hidden layer'], '\nActivation function: ', all_configs[idx]['hyperparameters']['activation function'], '\nLearning rate: ', all_configs[idx]['hyperparameters']['learning rate'], '\nDropout: ', all_configs[idx]['hyperparameters']['dropout'], '\nBatch size: ', all_configs[idx]['hyperparameters']['batch_size'])

