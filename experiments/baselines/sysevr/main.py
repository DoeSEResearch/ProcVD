import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from sklearn.metrics import roc_curve
import numpy as np

def calculate_vds(pred_probs, labels, target_fpr=0.005):
    fpr, tpr, thresholds = roc_curve(labels, pred_probs)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        threshold = thresholds[np.abs(fpr - target_fpr).argmin()]
    else:
        threshold = thresholds[valid[-1]]
    
    binary_preds = [1 if p >= threshold else 0 for p in pred_probs]
    fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_preds[i] == 0])
    tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_preds[i] == 1])
    vds = fn / (fn + tp) if (fn + tp) > 0 else 0
    return vds, threshold

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def tokenize(code):
    return re.findall(r'\b\w+\b', code)

def build_vocab(data, max_vocab_size=10000):
    counter = Counter()
    for item in data:
        tokens = tokenize(item['func'])
        counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab

def encode(code, vocab, max_len=200):
    tokens = tokenize(code)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

class CodeDataset(Dataset):
    def __init__(self, data, vocab, max_len=200):
        self.samples = [(encode(d['func'], vocab, max_len), d['target']) for d in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_tensor = torch.tensor(self.samples[idx][0], dtype=torch.long)
        label_tensor = torch.tensor(self.samples[idx][1], dtype=torch.float)
        return code_tensor, label_tensor

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        _, (hn, _) = self.lstm(emb)
        out = self.fc(hn[-1])
        return self.sigmoid(out).squeeze()

def train_model(model, train_loader, valid_loader, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

    return model

def calc_pair_metrics(predictions, labels):
    assert len(predictions) == len(labels), "Not paired"
    assert len(predictions) % 2 == 0, "Samples must be even."

    p_c = p_v = p_b = p_r = 0
    total_pairs = len(predictions) // 2

    for i in range(0, len(predictions), 2):
        pred1, pred2 = predictions[i], predictions[i + 1]
        label1, label2 = labels[i], labels[i + 1]

        if pred1 == label1 and pred2 == label2:
            p_c += 1
        else:
            if label1 == 1 and label2 == 0:
                if pred1 == 1 and pred2 == 1:
                    p_v += 1
                elif pred1 == 0 and pred2 == 0:
                    p_b += 1
            if pred1 != label1 and pred2 != label2:
                p_r += 1

    assert p_c + p_v + p_b + p_r == total_pairs, f"Not Paired: {p_c + p_v + p_b + p_r} vs {total_pairs}"

    return {
        "P-C": p_c / total_pairs,
        "P-V": p_v / total_pairs,
        "P-B": p_b / total_pairs,
        "P-R": p_r / total_pairs,
        "Total Pairs": total_pairs
    }

def evaluate_model(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.view(-1).numpy())

    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, bin_preds)
    precision = precision_score(all_labels, bin_preds, zero_division=0)
    recall = recall_score(all_labels, bin_preds, zero_division=0)
    f1 = f1_score(all_labels, bin_preds, zero_division=0)
    vds, thr = calculate_vds(all_preds, all_labels, target_fpr=0.005)
    pair_metrics = calc_pair_metrics(bin_preds, all_labels)

    print("\n=== Report ===")
    print(f"- Accuracy : {accuracy:.16f}")
    print(f"- Precision: {precision:.16f}")
    print(f"- Recall   : {recall:.16f}")
    print(f"- F1-score : {f1:.16f}")
    print(f"- VD-S     : {vds:.16f}  (FPR ≤ 0.005, threshold = {thr:.4f})")
    print(f"- P-C      : {pair_metrics['P-C']:.16f}")
    print(f"- P-V      : {pair_metrics['P-V']:.16f}")
    print(f"- P-B      : {pair_metrics['P-B']:.16f}")
    print(f"- P-R      : {pair_metrics['P-R']:.16f}")
    print(f"- Total Pairs: {pair_metrics['Total Pairs']}")

def main():
    print("Loading...")
    train_data = load_jsonl("train.jsonl")
    valid_data = load_jsonl("valid.jsonl")
    test_data  = load_jsonl("test.jsonl")

    vocab = build_vocab(train_data + valid_data)

    train_set = CodeDataset(train_data, vocab)
    valid_set = CodeDataset(valid_data, vocab)
    test_set  = CodeDataset(test_data,  vocab)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32)
    test_loader  = DataLoader(test_set,  batch_size=32)

    model = LSTMClassifier(vocab_size=len(vocab))
    model = train_model(model, train_loader, valid_loader)

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()