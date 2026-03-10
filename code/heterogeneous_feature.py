"""Heterogeneous feature extraction pipeline.

Trains an autoencoder on concatenated drug similarity embeddings (SMILE, target,
enzyme) to produce compressed 512D heterogeneous features per drug. Then evaluates
DDI prediction using a DNN classifier on concatenated drug-pair embeddings with
5-fold cross-validation.

The autoencoder loss combines cross-entropy on DDI event classification with
MSE reconstruction loss to preserve the original similarity information.

Output: embeddings/hetbranch_embedding.txt (572 drugs x 512D, best fold by AUROC).
"""

import numpy as np
import matplotlib
import torch
import torch.nn as nn
matplotlib.use('agg')
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from torch_geometric.nn.models import GAE, InnerProductDecoder
import os
import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from utils import *
from encoder import *

# --- Hyperparameters ---
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
HIDDEN_DIM_3 = 170       # Single drug network hidden size
DROPOUT_RATE = 0.5
EVENT_NUM = 65            # Number of DDI event types
NODE_FEATURE_DIM = 548
AE_BOTTLENECK_DIM = 512   # Autoencoder bottleneck dimensionality


class DNN(torch.nn.Module):
    """Deep neural network classifier for DDI event type prediction.

    Architecture: Linear -> ReLU -> BN -> Dropout -> Linear -> ReLU -> BN -> Dropout -> Linear.

    Args:
        input_dim: Dimensionality of input features (concatenated drug pair embeddings).
        event_num: Number of output classes (DDI event types).
        dropout_rate: Dropout probability.
    """

    def __init__(self, input_dim, event_num, dropout_rate=0.5):
        super(DNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(256, event_num),
        )

    def forward(self, x):
        return self.network(x)


def train_dnn(x_train, y_train, x_val, y_val, input_dim, event_num, dropout_rate=0.5,
              batch_size=128, epochs=100, patience=10):
    """Train a DNN classifier and return softmax predictions on the validation set.

    Uses Adam optimizer with cross-entropy loss and early stopping based on
    validation loss.

    Args:
        x_train: Training features, numpy array of shape (num_train, input_dim).
        y_train: Training labels, numpy array of integer class indices.
        x_val: Validation features, numpy array of shape (num_val, input_dim).
        y_val: Validation labels, numpy array of integer class indices.
        input_dim: Feature dimensionality.
        event_num: Number of output classes.
        dropout_rate: Dropout probability.
        batch_size: Mini-batch size for training.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).

    Returns:
        Numpy array of shape (num_val, event_num) with softmax prediction scores.
    """
    dnn = DNN(input_dim, event_num, dropout_rate)
    optimizer = torch.optim.Adam(dnn.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None
    for epoch in range(epochs):
        dnn.train()
        permutation = torch.randperm(x_train_t.size(0))
        for i in range(0, x_train_t.size(0), batch_size):
            batch_indices = permutation[i:i+batch_size]
            output = dnn(x_train_t[batch_indices])
            loss = criterion(output, y_train_t[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dnn.eval()
        with torch.no_grad():
            val_output = dnn(x_val_t)
            val_loss = criterion(val_output, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in dnn.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        dnn.load_state_dict(best_state)
    dnn.eval()
    with torch.no_grad():
        predictions = torch.softmax(dnn(x_val_t), dim=1).numpy()
    return predictions


def evaluate(predicted_types, predicted_scores, true_labels, event_num):
    """Compute classification metrics for DDI event type prediction.

    Args:
        predicted_types: Predicted class indices, numpy array.
        predicted_scores: Predicted score matrix of shape (num_samples, event_num).
        true_labels: Ground truth class indices, numpy array.
        event_num: Number of event classes.

    Returns:
        Numpy array of shape (6, 1) with [accuracy, AUPRC, AUROC, F1, precision, recall].
    """
    NUM_METRICS = 6
    results = np.zeros((NUM_METRICS, 1), dtype=float)
    true_one_hot = label_binarize(true_labels, classes=np.arange(event_num))
    results[0] = accuracy_score(true_labels, predicted_types)
    results[1] = roc_aupr_score(true_one_hot, predicted_scores, average='micro')
    results[2] = roc_auc_score(true_one_hot, predicted_scores, average='micro')
    results[3] = f1_score(true_labels, predicted_types, average='macro')
    results[4] = precision_score(true_labels, predicted_types, average='macro')
    results[5] = recall_score(true_labels, predicted_types, average='macro')
    return results


class HeterogeneousDenseDecoder(torch.nn.Module):
    """Dense decoder for the heterogeneous branch GAE, predicting 65 event types.

    Architecture: Linear(input_dim -> 512) -> Dropout(0.1) -> BN -> ReLU
                  -> Linear(512 -> 256) -> Dropout(0.1) -> BN -> ReLU
                  -> Linear(256 -> 65) -> Dropout(0.1) -> Sigmoid.

    Args:
        input_dim: Dimensionality of input features.
    """

    def __init__(self, input_dim):
        super(HeterogeneousDenseDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 65),
            torch.nn.Dropout(p=0.1),
            torch.nn.Sigmoid(),
        )

    def forward(self, feature):
        return self.network(feature)


def roc_aupr_score(y_true, y_score, average="macro"):
    """Compute Area Under the Precision-Recall Curve (AUPRC).

    Supports binary, micro, and macro averaging modes.

    Args:
        y_true: True binary labels (one-hot encoded for multi-class).
        y_score: Predicted scores.
        average: Averaging method - 'binary', 'micro', or 'macro'.

    Returns:
        AUPRC score as a float.
    """
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        num_classes = y_score.shape[1]
        scores = np.zeros((num_classes,))
        for c in range(num_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            scores[c] = binary_metric(y_true_c, y_score_c)
        return np.average(scores)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


class DrugSimilarityAutoencoder(torch.nn.Module):
    """Autoencoder for compressing concatenated drug similarity embeddings.

    Encodes concatenated similarity features (SMILE + target + enzyme = 1716D)
    down to a bottleneck representation, then reconstructs the original input.
    The bottleneck output serves as the heterogeneous drug embedding.

    Architecture: Linear(input -> 1024) -> BN -> ReLU -> Dropout
                  -> Linear(1024 -> bottleneck) [encoder output]
                  -> Linear(bottleneck -> 1024) -> BN -> ReLU -> Dropout
                  -> Linear(1024 -> input) [reconstruction output]

    Args:
        input_dim: Dimensionality of the concatenated similarity features.
    """

    def __init__(self, input_dim):
        super(DrugSimilarityAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.dropout_rate = 0.1
        self.encoder_hidden = torch.nn.Linear(self.input_dim, 1024)
        self.encoder_bn = torch.nn.BatchNorm1d(1024)
        self.bottleneck = torch.nn.Linear(1024, AE_BOTTLENECK_DIM)
        self.decoder_hidden = torch.nn.Linear(AE_BOTTLENECK_DIM, 1024)
        self.decoder_bn = torch.nn.BatchNorm1d(1024)
        self.decoder_output = torch.nn.Linear(1024, self.input_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """Encode input to bottleneck and reconstruct.

        Args:
            x: Input tensor of shape (num_drugs, input_dim).

        Returns:
            Tuple of (bottleneck_embedding, reconstruction):
                - bottleneck_embedding: Compressed features of shape (num_drugs, AE_BOTTLENECK_DIM).
                - reconstruction: Reconstructed input of shape (num_drugs, input_dim).
        """
        encoded = self.dropout(self.encoder_bn(self.relu(self.encoder_hidden(x))))
        bottleneck_embedding = self.bottleneck(encoded)
        decoded = self.dropout(self.decoder_bn(self.relu(self.decoder_hidden(bottleneck_embedding))))
        reconstruction = self.decoder_output(decoded)
        return bottleneck_embedding, reconstruction


def get_fold_indices(labels, event_num, seed, num_folds):
    """Assign each sample to a cross-validation fold, stratified by event type.

    Args:
        labels: Array of event type labels for all samples.
        event_num: Number of distinct event types.
        seed: Random seed for reproducible fold assignment.
        num_folds: Number of cross-validation folds.

    Returns:
        Array of fold indices (0 to num_folds-1) for each sample.
    """
    fold_assignments = np.zeros(len(labels))
    for event_type in range(event_num):
        event_indices = np.where(labels == event_type)
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_num = 0
        for train_idx, test_idx in kf.split(range(len(event_indices[0]))):
            fold_assignments[event_indices[0][test_idx]] = fold_num
            fold_num += 1
    return fold_assignments


def save_result(result_type, result):
    """Save evaluation results to a CSV file.

    Args:
        result_type: Prefix for the output filename (saves as '{result_type}_.csv').
        result: Results array to write row by row.
    """
    with open(result_type + '_' + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)


def prepare_data(fold, num_cross_val):
    """Load drug/event data and split into train/test sets for a given fold.

    Reads drug metadata, SMILES strings, event pairs, and labels from ../data/.
    Splits by assigning every num_cross_val-th sample to the test set.

    Args:
        fold: Current fold index (0 to num_cross_val-1).
        num_cross_val: Total number of cross-validation folds.

    Returns:
        Tuple of (smiles_list, train_pairs, test_pairs, train_labels,
                  test_labels, all_pairs, all_labels).
    """
    drug = pd.read_csv('../data/drug572.csv')
    event = pd.read_csv('../data/event.csv')
    smiles_list = []
    with open("../data/smile572.txt") as f:
        for line in f:
            line = line.rstrip()
            smiles_list.append(line)
    drug_id_to_index = dict([(drug_id, idx) for drug_id, idx in zip(drug['id'], drug['index'])])
    drug1_indices = [drug_id_to_index[drug_id] for drug_id in event['id1']]
    drug2_indices = [drug_id_to_index[drug_id] for drug_id in event['id2']]
    all_pairs = [[d1, d2] for d1, d2 in zip(drug1_indices, drug2_indices)]
    all_labels = np.loadtxt("../data/type572.txt", dtype=float, delimiter=" ")
    train_labels = np.array([x for i, x in enumerate(all_labels) if i % num_cross_val != fold])
    test_labels = np.array([x for i, x in enumerate(all_labels) if i % num_cross_val == fold])
    train_pairs = np.array([x for i, x in enumerate(all_pairs) if i % num_cross_val != fold])
    test_pairs = np.array([x for i, x in enumerate(all_pairs) if i % num_cross_val == fold])
    return smiles_list, train_pairs, test_pairs, train_labels, test_labels, all_pairs, all_labels


# =============================================================================
# Main training loop: autoencoder training + DNN evaluation per fold
# =============================================================================

num_cross_val = 5
best_auroc = 0.1
seed = 0

# Load three drug similarity embedding sources and concatenate (572 x 1716)
smile_embedding = np.loadtxt("../data/smile_embedding.txt", dtype=float, delimiter=" ")
smile_embedding = torch.tensor(smile_embedding, dtype=torch.float32)
target_embedding = np.loadtxt("../data/target_embedding.txt", dtype=float, delimiter=" ")
target_embedding = torch.tensor(target_embedding, dtype=torch.float32)
enzyme_embedding = np.loadtxt("../data/enzyme_embedding.txt", dtype=float, delimiter=" ")
enzyme_embedding = torch.tensor(enzyme_embedding, dtype=torch.float32)
concatenated_similarity = torch.cat((smile_embedding, target_embedding, enzyme_embedding), dim=1)
print(concatenated_similarity.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
concatenated_similarity = concatenated_similarity.to(device)
model = GAE(DrugSimilarityAutoencoder(1716), HeterogeneousDenseDecoder(1024)).to(device)

for fold in range(5):
    smiles_list, train_pairs, test_pairs, train_labels, test_labels, all_pairs, all_labels = \
        prepare_data(fold, num_cross_val)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)

    for epoch in range(800):
        print(epoch)

        # --- Training phase ---
        model.train()
        optimizer.zero_grad()
        embedding, ae_reconstruction = model.encoder(concatenated_similarity)

        drug1_train_emb = embedding[train_pairs[:, 0], :]
        drug2_train_emb = embedding[train_pairs[:, 1], :]
        pair_features_train = torch.cat((drug1_train_emb, drug2_train_emb), 1)

        logits_train = model.decoder(pair_features_train)
        pred_scores_train = logits_train.detach().cpu().numpy()
        pred_types_train = np.argmax(pred_scores_train, axis=1)

        classification_loss = nn.CrossEntropyLoss()
        reconstruction_loss = torch.nn.MSELoss()
        loss = classification_loss(logits_train, train_labels_tensor) + \
            reconstruction_loss(concatenated_similarity, ae_reconstruction)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.tolist())

        # Compute training metrics
        true_one_hot_train = label_binarize(train_labels, classes=np.arange(65))
        train_results = np.zeros((6, 1), dtype=float)
        train_results[0] = accuracy_score(train_labels, pred_types_train)
        train_results[1] = roc_aupr_score(true_one_hot_train, pred_scores_train, average='micro')
        train_results[2] = roc_auc_score(true_one_hot_train, pred_scores_train, average='micro')
        train_results[3] = f1_score(train_labels, pred_types_train, average='macro')
        train_results[4] = precision_score(train_labels, pred_types_train, average='macro')
        train_results[5] = recall_score(train_labels, pred_types_train, average='macro')
        print('Training set')
        print(train_results)

        # --- Evaluation phase ---
        model.eval()
        drug1_test_emb = embedding[test_pairs[:, 0], :]
        drug2_test_emb = embedding[test_pairs[:, 1], :]
        pair_features_test = torch.cat((drug1_test_emb, drug2_test_emb), 1)

        logits_test = model.decoder(pair_features_test)
        pred_scores_test = logits_test.detach().cpu().numpy()
        pred_types_test = np.argmax(pred_scores_test, axis=1)

        # Compute test metrics
        true_one_hot_test = label_binarize(test_labels, classes=np.arange(65))
        test_results = np.zeros((6, 1), dtype=float)
        test_results[0] = accuracy_score(test_labels, pred_types_test)
        test_results[1] = roc_aupr_score(true_one_hot_test, pred_scores_test, average='micro')
        test_results[2] = roc_auc_score(true_one_hot_test, pred_scores_test, average='micro')
        test_results[3] = f1_score(test_labels, pred_types_test, average='macro')
        test_results[4] = precision_score(test_labels, pred_types_test, average='macro')
        test_results[5] = recall_score(test_labels, pred_types_test, average='macro')
        print('Test set')
        print(test_results)

        # Save embedding if this is the best AUROC so far
        if test_results[2] > best_auroc:
            best_auroc = test_results[2]
            best_embedding = embedding.detach().cpu().numpy()
            np.savetxt("hetbranch_embedding.txt", best_embedding, fmt="%6.4f")

    print("Optimization Finished!")

    # =====================================================================
    # DNN evaluation: predict DDI events using saved heterogeneous embeddings
    # =====================================================================
    hetbranch_embedding = np.loadtxt("hetbranch_embedding.txt", dtype=float, delimiter=" ")
    hetbranch_embedding = torch.tensor(hetbranch_embedding)
    print(hetbranch_embedding.shape)
    all_pairs_array = np.array(all_pairs)

    drug1_emb = hetbranch_embedding[all_pairs_array[:, 0], :]
    drug2_emb = hetbranch_embedding[all_pairs_array[:, 1], :]
    pair_features_all = torch.cat((drug1_emb, drug2_emb), 1)

    fold_assignments = get_fold_indices(all_labels, EVENT_NUM, seed, num_cross_val)
    train_mask = np.where(fold_assignments != fold)
    test_mask = np.where(fold_assignments == fold)

    x_train = pair_features_all[train_mask].detach().numpy()
    x_test = pair_features_all[test_mask].detach().numpy()
    y_train = all_labels[train_mask]
    y_test = all_labels[test_mask]

    dnn_predictions = train_dnn(
        x_train, y_train.astype(int), x_test, y_test.astype(int),
        input_dim=1024, event_num=EVENT_NUM, dropout_rate=DROPOUT_RATE,
        batch_size=128, epochs=100, patience=10)

    pred_types_dnn = np.argmax(dnn_predictions, axis=1)
    fold_results = evaluate(pred_types_dnn, dnn_predictions, y_test, EVENT_NUM)
    print('Final evaluation')
    print(fold_results)
    save_result('heterogeneous_result', fold_results)
