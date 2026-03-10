"""Multi-source feature fusion pipeline.

Combines pre-computed sequence embeddings (160D from seqbranch_embedding.txt) and
heterogeneous embeddings (512D from hetbranch_embedding.txt) into a fused 672D
representation using self-attention, then trains a dense decoder to predict
65 DDI event types. Evaluates with 5-fold cross-validation.

Output: embeddings/sequ_hete_embedding.txt (572 drugs x 672D, best fold by AUROC).
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
import os
import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from model import *
from utils import *
from encoder import *

# --- Hyperparameters ---
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
HIDDEN_DIM_3 = 170       # Single drug network hidden size
DROPOUT_RATE = 0.5
EVENT_NUM = 65            # Number of DDI event types
NODE_FEATURE_DIM = 548


class DNN(torch.nn.Module):
    """Deep neural network classifier for DDI event type prediction (fusion stage).

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
              batch_size=68, epochs=100, patience=10):
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
    """Compute overall and per-event classification metrics for DDI prediction.

    Args:
        predicted_types: Predicted class indices, numpy array.
        predicted_scores: Predicted score matrix of shape (num_samples, event_num).
        true_labels: Ground truth class indices, numpy array.
        event_num: Number of event classes.

    Returns:
        Tuple of (overall_results, per_event_results):
            - overall_results: Array of shape (6, 1) with
              [accuracy, AUPRC, AUROC, F1, precision, recall].
            - per_event_results: Array of shape (event_num, 6) with the same
              metrics computed per event type (binary).
    """
    NUM_METRICS = 6
    overall_results = np.zeros((NUM_METRICS, 1), dtype=float)
    per_event_results = np.zeros((event_num, NUM_METRICS), dtype=float)
    true_one_hot = label_binarize(true_labels, classes=np.arange(event_num))
    pred_one_hot = label_binarize(predicted_types, classes=np.arange(event_num))

    overall_results[0] = accuracy_score(true_labels, predicted_types)
    overall_results[1] = roc_aupr_score(true_one_hot, predicted_scores, average='micro')
    overall_results[2] = roc_auc_score(true_one_hot, predicted_scores, average='micro')
    overall_results[3] = f1_score(true_labels, predicted_types, average='macro')
    overall_results[4] = precision_score(true_labels, predicted_types, average='macro')
    overall_results[5] = recall_score(true_labels, predicted_types, average='macro')

    for event_idx in range(event_num):
        true_binary = true_one_hot.take([event_idx], axis=1).ravel()
        pred_binary = pred_one_hot.take([event_idx], axis=1).ravel()
        per_event_results[event_idx, 0] = accuracy_score(true_binary, pred_binary)
        per_event_results[event_idx, 1] = roc_aupr_score(true_binary, pred_binary, average=None)
        per_event_results[event_idx, 2] = roc_auc_score(true_binary, pred_binary, average=None)
        per_event_results[event_idx, 3] = f1_score(true_binary, pred_binary, average='binary')
        per_event_results[event_idx, 4] = precision_score(true_binary, pred_binary, average='binary')
        per_event_results[event_idx, 5] = recall_score(true_binary, pred_binary, average='binary')

    return overall_results, per_event_results


class FusionDenseDecoder(torch.nn.Module):
    """Dense decoder for the fusion branch, predicting 65 event types.

    Architecture: Linear(input_dim -> 1024) -> Dropout(0.1) -> BN -> ReLU
                  -> Linear(1024 -> 512) -> Dropout(0.1) -> BN -> ReLU
                  -> Linear(512 -> 65) -> Dropout(0.1) -> Sigmoid.

    Args:
        input_dim: Dimensionality of input features.
    """

    def __init__(self, input_dim):
        super(FusionDenseDecoder, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.Dropout(p=0.1),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 65),
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


class FusionSelfAttention(torch.nn.Module):
    """Multi-head self-attention for fusing drug embeddings.

    Computes self-attention over the feature dimension of each drug independently.
    Projects input through query, key, and value matrices, applies scaled
    dot-product attention across heads, then concatenates head outputs.

    Args:
        input_dim: Dimensionality of input features.
        n_heads: Number of attention heads. input_dim must be divisible by n_heads.
        output_dim: Output dimensionality (defaults to input_dim if not specified).
    """

    def __init__(self, input_dim, n_heads, output_dim=None):
        super(FusionSelfAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        print(self.d_k)
        print(self.d_v)
        self.n_heads = n_heads
        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.output_dim, bias=False)

    def forward(self, x):
        """Apply multi-head self-attention.

        Args:
            x: Input tensor of shape (seq_len, input_dim).

        Returns:
            Tensor of shape (seq_len, n_heads * d_v) with attended features.
        """
        # (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        query = self.W_Q(x).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        key = self.W_K(x).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        value = self.W_V(x).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scale = np.sqrt(self.d_k)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / scale
        attention_weights = torch.nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_weights, value)
        # Concatenate heads: (seq_len, n_heads * d_v)
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        return context


class SelfAttentionEncoder(torch.nn.Module):
    """Self-attention encoder that processes each drug embedding independently.

    Wraps FusionSelfAttention to handle batched drug embeddings by treating
    each drug as a single-token sequence.

    Args:
        input_dim: Dimensionality of drug embeddings.
        n_heads: Number of attention heads.
    """

    def __init__(self, input_dim, n_heads):
        super(SelfAttentionEncoder, self).__init__()
        self.attention = self.attn = FusionSelfAttention(input_dim, n_heads)

    def forward(self, embedding):
        """Encode drug embeddings through self-attention.

        Args:
            embedding: Tensor of shape (num_drugs, input_dim).

        Returns:
            Tensor of shape (num_drugs, n_heads * d_v) with attention-encoded features.
        """
        num_drugs = embedding.shape[0]
        attn = self.attn
        # Each drug is a single-token sequence; batch all drugs at once
        x = embedding.unsqueeze(1)  # (N, 1, D)
        query = attn.W_Q(x).view(num_drugs, 1, attn.n_heads, attn.d_k).transpose(1, 2)
        key = attn.W_K(x).view(num_drugs, 1, attn.n_heads, attn.d_k).transpose(1, 2)
        value = attn.W_V(x).view(num_drugs, 1, attn.n_heads, attn.d_v).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(attn.d_k)
        context = torch.matmul(torch.nn.Softmax(dim=-1)(scores), value)
        encoded_features = context.transpose(1, 2).reshape(num_drugs, attn.n_heads * attn.d_v)
        return encoded_features


class FusionEncoderDecoder(torch.nn.Module):
    """End-to-end fusion model: self-attention encoder + dense decoder.

    Encodes concatenated drug embeddings (672D = 160D seq + 512D het) through
    self-attention, then predicts DDI event types for given drug pairs.
    """

    def __init__(self):
        super(FusionEncoderDecoder, self).__init__()
        self.encoder = SelfAttentionEncoder(672, 1)
        self.decoder = FusionDenseDecoder(1344)

    def forward(self, embedding, pair_indices):
        """Encode drugs and predict interaction scores for given pairs.

        Args:
            embedding: Drug embedding tensor of shape (num_drugs, 672).
            pair_indices: Array of shape (num_pairs, 2) with [drug1_idx, drug2_idx].

        Returns:
            Tuple of (prediction_scores, encoded_embedding):
                - prediction_scores: Tensor of shape (num_pairs, 65).
                - encoded_embedding: Tensor of shape (num_drugs, 672).
        """
        encoded_embedding = self.encoder(embedding)
        drug1_emb = encoded_embedding[pair_indices[:, 0], :]
        drug2_emb = encoded_embedding[pair_indices[:, 1], :]
        pair_features = torch.cat((drug1_emb, drug2_emb), 1)
        prediction_scores = self.decoder(pair_features)
        return prediction_scores, encoded_embedding


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
# Main training loop: fusion encoder-decoder training + DNN evaluation per fold
# =============================================================================

num_cross_val = 5
best_auroc = 0.1
seed = 0

# Load pre-computed embeddings from Stage 1 and concatenate (572 x 672)
sequence_embedding = np.loadtxt("seqbranch_embedding.txt", dtype=float, delimiter=" ")
sequence_embedding = torch.tensor(sequence_embedding, dtype=torch.float32)
print(sequence_embedding.shape)
heterogeneous_embedding = np.loadtxt("hetbranch_embedding.txt", dtype=float, delimiter=" ")
heterogeneous_embedding = torch.tensor(heterogeneous_embedding, dtype=torch.float32)
print(heterogeneous_embedding.shape)
fused_embedding = torch.cat((sequence_embedding, heterogeneous_embedding), dim=1)
print(fused_embedding.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fused_embedding = fused_embedding.to(device)
model = FusionEncoderDecoder().to(device)

for fold in range(5):
    smiles_list, train_pairs, test_pairs, train_labels, test_labels, all_pairs, all_labels = \
        prepare_data(fold, num_cross_val)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)

    for epoch in range(500):
        print(epoch)

        # --- Training phase ---
        model.train()
        optimizer.zero_grad()
        logits_train, encoded_embedding = model(fused_embedding, train_pairs)
        pred_scores_train = logits_train.detach().cpu().numpy()
        pred_types_train = np.argmax(pred_scores_train, axis=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits_train, train_labels_tensor)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.tolist())

        # Compute training metrics
        true_one_hot_train = label_binarize(train_labels, classes=np.arange(65))
        pred_one_hot_train = label_binarize(pred_types_train, classes=np.arange(65))
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
        logits_test, encoded_embedding = model(fused_embedding, test_pairs)
        pred_scores_test = logits_test.detach().cpu().numpy()
        pred_types_test = np.argmax(pred_scores_test, axis=1)

        # Compute test metrics
        true_one_hot_test = label_binarize(test_labels, classes=np.arange(65))
        pred_one_hot_test = label_binarize(pred_types_test, classes=np.arange(65))
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
            best_fused_embedding = fused_embedding.detach().cpu().numpy()
            np.savetxt("sequ_hete_embedding.txt", best_fused_embedding, fmt="%6.4f")

    print("Optimization Finished!")

    # =====================================================================
    # DNN evaluation: predict DDI events using saved fused embeddings
    # =====================================================================
    fused_drug_embedding = np.loadtxt("sequ_hete_embedding.txt", dtype=float, delimiter=" ")
    fused_drug_embedding = torch.tensor(fused_drug_embedding)
    print(fused_drug_embedding.shape)
    all_pairs_array = np.array(all_pairs)

    drug1_emb = fused_drug_embedding[all_pairs_array[:, 0], :]
    drug2_emb = fused_drug_embedding[all_pairs_array[:, 1], :]
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
        input_dim=1344, event_num=EVENT_NUM, dropout_rate=DROPOUT_RATE,
        batch_size=68, epochs=100, patience=10)

    pred_types_dnn = np.argmax(dnn_predictions, axis=1)
    overall_results, per_event_results = evaluate(pred_types_dnn, dnn_predictions, y_test, EVENT_NUM)
    print('Final evaluation')
    print(overall_results)
    save_result('sequ_hete_all_result', overall_results)
    save_result('sequ_hete_eve_result', per_event_results)
