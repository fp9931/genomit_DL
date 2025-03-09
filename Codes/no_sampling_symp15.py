import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif

import pandas as pd
import numpy as np
import pickle
import random
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set a random seed for reproducibility
random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set path
general_path = os.path.dirname(os.getcwd())
data_path = os.path.join(general_path, "Data/data_genomit")
result_path = os.path.join(general_path, "Results/results_genomit/symptoms_15/no_sampling")
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Load the dataset
df = pd.read_csv(os.path.join(data_path, "df_symp.csv"))

# Swap the values of 'gendna_type' (0 -> 1 and 1 -> 0) to consider mtDNA as the positive class
df['gendna_type'] = df['gendna_type'].apply(lambda x: 1 if x == 0 else 0)

# Remove unnecessary columns
df.drop(['Unnamed: 0', 'subjid'], axis=1, inplace=True)

# Split the data
df_train = df[df['test'] == 0].reset_index(drop=True)
df_test = df[df['test'] == 1].reset_index(drop=True)

# Separate features and target
X_train_full = df_train.drop(['gendna_type', 'test'], axis=1)
y_train_full = df_train['gendna_type']
X_test = df_test.drop(['gendna_type', 'test'], axis=1)
y_test = df_test['gendna_type']

print("ALL data retrieved")

# Compute missing value proportions
missing_ratios = (X_train_full == -998).mean()  # Proportion of missing values per feature
penalty_factors = 1 - missing_ratios  # Create a penalty factor

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_train_scaled_df = (pd.DataFrame(X_train_scaled, columns=X_train_full.columns)) * penalty_factors
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = (pd.DataFrame(X_test_scaled, columns=X_test.columns)) * penalty_factors

# Apply missing value penalty in MRMR feature selection
num_features = X_train_full.shape[1]
selected_features = mrmr_classif(X_train_scaled_df, y_train_full, K=num_features)

# Create feature sets in increasing length order
feature_sets = [selected_features[:i] for i in range(1, len(selected_features) + 1)]

# Define resampling strategies
samplers = {
    "no_resampling": None,
    #"SMOTE": SMOTE(random_state=random_state),
    #"ADASYN": ADASYN(random_state=random_state)
}

params_grid = {
    'activation function': ['ReLU', 'Tanh', 'Sigmoid', 'SiLU', 'GELU', 'LeakyReLU'],
    'hidden layer': [(8,), (16,), (32,), (64,), (16,8), (32,16), (64,32), (32, 16, 8), (64, 32, 16), (16, 32, 16)],
    'learning rate': [0.0001, 0.001, 0.01, 0.1],
    'dropout': [None, 0.25],
    'batch_size': [32, 64]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# Store all models and configurations
all_scores = []
all_models = []
all_configs = []

# Total iterations count
total_iterations = len(samplers) * len(feature_sets)
iteration_count = 0

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

class EarlyStopping:
    def __init__(self, patience=10, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_model_state = None
        self.epochs_since_improvement = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_since_improvement = 0  # Reset patience counter
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.patience:
            #print(f"Early stopping triggered after {self.patience} epochs!")
            if self.restore_best_weights and self.best_model_state:
                model.load_state_dict(self.best_model_state)
            return True  # Stop training

        return False  # Continue training

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor=0.2, patience=5, min_lr=0.0001):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_since_improvement = 0  # Reset patience counter
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                if param_group['lr'] > new_lr:
                    #print(f"Reducing learning rate from {param_group['lr']} to {new_lr}")
                    param_group['lr'] = new_lr
            self.epochs_since_improvement = 0  # Reset patience counter

    
# Loop through feature sets and sampling methods
for feature_set in feature_sets[15:]:
    X_train_subset = X_train_scaled_df[feature_set]
    X_test_subset = X_test_scaled_df[feature_set]
    input_dim = X_train_subset.shape[1]

    for sampling_name, sampler in samplers.items():
        iteration_count += 1
        print(f"\nIteration {iteration_count}/{total_iterations} | Features: {len(feature_set)} | Sampling: {sampling_name}")

        # Apply resampling
        X_resampled, y_resampled = X_train_subset, y_train_full
        if sampler:
            X_resampled, y_resampled = sampler.fit_resample(X_train_subset, y_train_full)

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
        class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

        # Create a model and grid search
        for activation_func in params_grid['activation function']:
            for hidden_layers in params_grid['hidden layer']:
                for learning_rate in params_grid['learning rate']:
                    for dropout in params_grid['dropout']:
                        for batch_size in params_grid['batch_size']:
                            print(f"{activation_func}, {hidden_layers}, {learning_rate}, {dropout}, {batch_size}")

                            model = Model(input_dim, hidden_layers, activation_func, dropout)
                            model = nn.DataParallel(model)
                            model.to(device)

                            criterion = nn.BCELoss()
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
                            reduce_lr = ReduceLROnPlateau(optimizer, factor=0.2, patience=5, min_lr=0.0001)

                            fold_scores = []
                            best_model_validation = model
                            max_f1_score = 0
                            train_loss_history = []
                            val_loss_history = []

                            for train_idx, val_idx in cv.split(X_resampled, y_resampled):
                                X_train_split = torch.tensor(X_resampled.iloc[train_idx].values, dtype=torch.float32).to(device)
                                y_train_split = torch.tensor(y_resampled.iloc[train_idx].values, dtype=torch.float32).to(device)
                                X_val_split = torch.tensor(X_resampled.iloc[val_idx].values, dtype=torch.float32).to(device)
                                y_val_split = torch.tensor(y_resampled.iloc[val_idx].values, dtype=torch.float32).to(device)
                                
                                train_loader = DataLoader(TensorDataset(X_train_split, y_train_split), batch_size=batch_size, shuffle=True)
                                val_loader = DataLoader(TensorDataset(X_val_split, y_val_split), batch_size=batch_size)

                                val_loss = []
                                train_loss = []
                                for epoch in range(100):
                                    model.train()
                                    training_loss = 0.0
                                    for X_batch, y_batch in train_loader:
                                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                                        optimizer.zero_grad()
                                        outputs = model(X_batch).squeeze()
                                        loss = criterion(outputs, y_batch.float())
                                        loss.backward()
                                        optimizer.step()
                                        training_loss += loss.item()
                                    training_loss /= len(train_loader)
                                    train_loss.append(training_loss)

                                    # Validation
                                    model.eval()
                                    y_true, y_pred = [], []
                                    validation_loss = 0.0
                                    with torch.no_grad():
                                        for X_val, y_val in val_loader:
                                            X_val, y_val = X_val.to(device), y_val.to(device)
                                            outputs = model(X_val).squeeze()
                                            loss = criterion(outputs, y_val.float())
                                            preds = (outputs > 0.5).int()
                                            y_true.extend(y_val.cpu().numpy())
                                            y_pred.extend(preds.cpu().numpy())
                                            validation_loss += loss.item()                                    
                                    validation_loss /= len(val_loader)
                                    val_loss.append(validation_loss)
                                    f1 = f1_score(y_true, y_pred)

                                    #print(f"Epoch {epoch+1}/{100} | Validation F1-score: {f1:.3f}")

                                    reduce_lr(validation_loss)
                                    if early_stopping(validation_loss, model):
                                        break
                                
                                # best for this fold
                                if f1 >= max_f1_score:
                                    best_model_validation = model
                                    max_f1_score = f1
                                    train_loss_history.append(train_loss)
                                    val_loss_history.append(val_loss)
                            
                            # Evaluate the model on the trsin set
                            best_model_validation.eval()
                            best_model_train_loss = train_loss_history[-1]
                            best_model_val_loss = val_loss_history[-1]
                            with torch.no_grad():
                                y_train_pred = (best_model_validation(torch.tensor(X_resampled.values, dtype=torch.float32).to(device)).squeeze() > 0.5).int().cpu().numpy()
                                accuracy_train = accuracy_score(y_resampled, y_train_pred)
                                f1_score_train = f1_score(y_resampled, y_train_pred)
                                conf_matrix_train = confusion_matrix(y_resampled, y_train_pred)
                                # print(conf_matrix_train)
                                # print(f"Training Accuracy: {accuracy_train:.3f}")
                                # print(f"F1-score: {f1_score_train:.3f}")

                                # Save all models and configurations
                                all_scores.append((f1_score_train, accuracy_train))
                                all_models.append(best_model_validation)
                                all_configs.append({
                                    "feature set": feature_set,
                                    "features": len(feature_set),
                                    "sampling": sampling_name,
                                    "confusion_matrix": conf_matrix_train.tolist(),
                                    "train_loss_history": best_model_train_loss,
                                    "val_loss_history": best_model_val_loss,
                                    "hyperparameters": {
                                        "activation function": activation_func,
                                        "hidden layer": hidden_layers,
                                        "learning rate": learning_rate,
                                        "dropout": dropout,
                                        "batch_size": batch_size
                                    }
                                })
                            # # Sort the scores in descending order based on the F1-score (first value in tuple), keeping corresponding accuracy (second value)
                            # sorted_scores_with_indices = sorted(enumerate(all_scores), key=lambda x: x[1][0], reverse=True)

                            # # Sort the corresponding other files based on the sorted scores
                            # all_scores = [all_scores[idx] for idx, _ in sorted_scores_with_indices]
                            # all_models = [all_models[idx] for idx, _ in sorted_scores_with_indices]
                            # all_configs = [all_configs[idx] for idx, _ in sorted_scores_with_indices]

                            # Save all models and configurations
                            with open(os.path.join(result_path, "all_scores_2.pkl"), 'wb') as file:
                                pickle.dump(all_scores, file)
                            with open(os.path.join(result_path, "all_configs_2.pkl"), 'wb') as file:
                                pickle.dump(all_configs, file)
                            with open(os.path.join(result_path, "all_models_2.pkl"), 'wb') as file:
                                pickle.dump(all_models, file)

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

# Evaluate selected model
selected_model = all_models[model_idx]
selected_config = all_configs[model_idx]
print("\nEvaluating Best Model on Test Set:")

selected_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled_df[selected_config['feature set']].values, dtype=torch.float32).to(device)
    y_pred = (selected_model(X_test_tensor).squeeze() > 0.5).int().cpu().numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest Set Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["nDNA", "mtDNA"], yticklabels=["nDNA", "mtDNA"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of Selected Model")
plt.show()

# Assuming that mtDNA is the positive class, compute sensitivity and specificity
spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sens = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
print(f"Sensitivity: {sens:.3f}")
print(f"Specificity: {spec:.3f}")