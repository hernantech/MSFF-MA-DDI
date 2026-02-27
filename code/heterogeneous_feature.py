import numpy as np
import matplotlib
import torch
import torch.nn as nn
matplotlib.use('agg')
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc,average_precision_score
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from torch_geometric.nn.models import GAE,InnerProductDecoder
import os
import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from utils import *
from encoder import *

hid1 = 256
hid2 = 128
# 单个药物网络
hid3 = 170
# 两个
# hid3 = 512
droprate = 0.5
event_num = 65
node_feature = 548

class DNN(torch.nn.Module):
    def __init__(self, input_dim, event_num, droprate=0.5):
        super(DNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=droprate),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=droprate),
            torch.nn.Linear(256, event_num),
        )
    def forward(self, x):
        return self.net(x)

def train_dnn(x_train, y_train, x_val, y_val, input_dim, event_num, droprate=0.5,
              batch_size=128, epochs=100, patience=10):
    dnn = DNN(input_dim, event_num, droprate)
    optimizer = torch.optim.Adam(dnn.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_val_loss = float('inf')
    wait = 0
    best_state = None
    for epoch in range(epochs):
        dnn.train()
        perm = torch.randperm(x_train_t.size(0))
        for i in range(0, x_train_t.size(0), batch_size):
            idx = perm[i:i+batch_size]
            out = dnn(x_train_t[idx])
            loss = criterion(out, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dnn.eval()
        with torch.no_grad():
            val_out = dnn(x_val_t)
            val_loss = criterion(val_out, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_state = {k: v.clone() for k, v in dnn.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        dnn.load_state_dict(best_state)
    dnn.eval()
    with torch.no_grad():
        pred = torch.softmax(dnn(x_val_t), dim=1).numpy()
    return pred

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')
    return result_all
class DenseDecoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(DenseDecoder, self).__init__()
        self.fullynet = torch.nn.Sequential(
            torch.nn.Linear(input_dim,512),
            # 526
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
        outputs = self.fullynet(feature)
        return outputs
def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
class AE(torch.nn.Module):  # Joining together
    def __init__(self, vector_size):
        super(AE, self).__init__()
        self.vector_size = vector_size
        self.drop_out_rating = 0.1
        self.l1 = torch.nn.Linear(self.vector_size, 1024)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.l2 = torch.nn.Linear(1024, len_after_AE)
        self.l3 = torch.nn.Linear(len_after_AE, 1024)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.l4 = torch.nn.Linear(1024, self.vector_size)
        self.dr = torch.nn.Dropout(self.drop_out_rating)
        self.ac = torch.nn.ReLU()
    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.l2(X)
        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))
        X_AE = self.l4(X_AE)
        return X,X_AE
def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
    return index_all_class
def save_result(result_type, result):
    with open(result_type + '_' + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0
def prepare_data1(fold,num_cross_val):
    drug = pd.read_csv('../data/drug572.csv')
    event = pd.read_csv('../data/event.csv')
    # print(drug)
    smiles = []
    with open("../data/smile572.txt") as f:
        for line in f:
            line = line.rstrip()
            smiles.append(line)
    # print(smiles)
    drug_smile_dict = dict([(drug_id, id) for drug_id, id in zip(drug['id'], drug['index'])])
    # print(drug_smile_dict)
    index1 = []
    index2 = []
    index_pair = []
    for i in event['id1']:
        index1.append(drug_smile_dict[i])
    for j in event['id2']:
        index2.append(drug_smile_dict[j])
    for i in range(0, len(index1)):
        index_pair.append([index1[i], index2[i]])
    label = np.loadtxt("../data/type572.txt", dtype=float, delimiter=" ")
    # print(label.shape)
    train_label = np.array([x for i, x in enumerate(label) if i % num_cross_val != fold])
    test_label = np.array([x for i, x in enumerate(label) if i % num_cross_val == fold])
    train_index = np.array([x for i, x in enumerate(index_pair) if i % num_cross_val != fold])
    test_index = np.array([x for i, x in enumerate(index_pair) if i % num_cross_val == fold])
    return smiles,train_index,test_index,train_label,test_label,index_pair,label

num_cross_val = 5
max_auc = 0.1
seed = 0
smile_embedding = np.loadtxt("../data/smile_embedding.txt", dtype=float, delimiter=" ")
smile_embedding = torch.tensor(smile_embedding, dtype=torch.float32)
target_embedding= np.loadtxt("../data/target_embedding.txt", dtype=float, delimiter=" ")
target_embedding = torch.tensor(target_embedding, dtype=torch.float32)
enzyme_embedding= np.loadtxt("../data/enzyme_embedding.txt", dtype=float, delimiter=" ")
enzyme_embedding = torch.tensor(enzyme_embedding, dtype=torch.float32)
drugsim_feature = torch.cat((smile_embedding,target_embedding,enzyme_embedding), dim=1)
print(drugsim_feature.shape)
len_after_AE = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drugsim_feature = drugsim_feature.to(device)
model = GAE(AE(1716),  DenseDecoder(1024)).to(device)
for fold in range(5):
    smiles,train_index,test_index,train_label,test_label,index_pair,label= prepare_data1(fold, num_cross_val)
    train_label_t = torch.tensor(train_label, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
    for epoch in range(800):
        print(epoch)
        model.train()
        optimizer.zero_grad()
        embedding,AE_out = model.encoder(drugsim_feature)
        # print(embedding.shape)
        drug1_train = train_index[:,0]
        drug2_train = train_index[:,1]
        drug1_emb_train = embedding[drug1_train, :]
        drug2_emb_train = embedding[drug2_train, :]
        drug_data = torch.cat((drug1_emb_train, drug2_emb_train), 1)
        pred_score = model.decoder(drug_data)
        l_pred_score = pred_score
        pred_score = pred_score.detach().cpu().numpy()
        pred_type = np.argmax(pred_score,axis=1)
        criterion = nn.CrossEntropyLoss()
        criterion1 = torch.nn.MSELoss()
        loss = criterion(l_pred_score, train_label_t) + criterion1(drugsim_feature, AE_out)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.tolist())
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.zeros((0, 65), dtype=float)
        y_true = np.hstack((y_true, train_label))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        all_eval_type = 6
        result_all = np.zeros((all_eval_type, 1), dtype=float)
        y_test = y_true
        y_one_hot = label_binarize(y_test, classes=np.arange(65))
        # print(y_one_hot)
        pred_one_hot = label_binarize(pred_type, classes=np.arange(65))
        # print(pred_one_hot.shape)
        result_all[0] = accuracy_score(y_test, pred_type)
        result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
        result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
        result_all[3] = f1_score(y_test, pred_type, average='macro')
        result_all[4] = precision_score(y_test, pred_type, average='macro')
        result_all[5] = recall_score(y_test, pred_type, average='macro')
        print('训练集')
        print(result_all)
        model.eval()
        drug1_test = test_index[:, 0]
        drug2_test = test_index[:, 1]
        drug1_emb_test = embedding[drug1_test, :]
        drug2_emb_test = embedding[drug2_test, :]
        drug_data = torch.cat((drug1_emb_test, drug2_emb_test), 1)
        pred_score = model.decoder(drug_data)
        l_pred_score = pred_score
        pred_score = pred_score.detach().cpu().numpy()
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.zeros((0, 65), dtype=float)
        y_true = np.hstack((y_true, test_label))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))
        all_eval_type = 6
        result_all = np.zeros((all_eval_type, 1), dtype=float)
        y_test = y_true
        y_one_hot = label_binarize(y_test, classes=np.arange(65))
        # print(y_one_hot)
        pred_one_hot = label_binarize(pred_type, classes=np.arange(65))
        # print(pred_one_hot.shape)
        result_all[0] = accuracy_score(y_test, pred_type)
        result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
        result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
        result_all[3] = f1_score(y_test, pred_type, average='macro')
        result_all[4] = precision_score(y_test, pred_type, average='macro')
        result_all[5] = recall_score(y_test, pred_type, average='macro')
        print('测试集')
        print(result_all)
        if result_all[2] > max_auc:
            max_auc = result_all[2]
            heterogeneous_embedding = embedding
            heterogeneous_embedding = heterogeneous_embedding.detach().cpu().numpy()
            np.savetxt("hetbranch_embedding.txt",  heterogeneous_embedding, fmt="%6.4f")
    print("Optimization Finished!")
####=====================deep learning model to predict =============================
    hetbranch_embedding = np.loadtxt("hetbranch_embedding.txt", dtype=float, delimiter=" ")
    hetbranch_embedding = torch.tensor((hetbranch_embedding))
    print(hetbranch_embedding.shape)
    index_pair = np.array(index_pair)
    drug1 = index_pair[:, 0]
    drug2 = index_pair[:, 1]
    drug1_emb = hetbranch_embedding[drug1, :]
    drug2_emb = hetbranch_embedding[drug2, :]
    drug_data = torch.cat((drug1_emb, drug2_emb), 1)

    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label, event_num, seed, num_cross_val)
    train_index = np.where(index_all_class != fold)
    test_index = np.where(index_all_class == fold)
    pred = np.zeros((len(test_index[0]), event_num), dtype=float)
    x_train = drug_data[train_index].detach().numpy()
    x_test = drug_data[test_index].detach().numpy()
    y_train = label[train_index]
    y_test = label[test_index]
    pred += train_dnn(x_train, y_train.astype(int), x_test, y_test.astype(int),
                      input_dim=1024, event_num=event_num, droprate=droprate,
                      batch_size=128, epochs=100, patience=10)
    pred_score = pred
    pred_type = np.argmax(pred_score, axis=1)
    y_true = np.hstack((y_true, y_test))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score))
    result_all = evaluate(y_pred, y_score, y_true, event_num)
    print('最后')
    print(result_all)
    save_result('heterogeneous_result', result_all)
