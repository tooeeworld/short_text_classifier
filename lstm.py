import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
import re
import time

# Load data
def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    labels, sentences = [], []
    for line in lines:
        print(line)
        label, sentence = line.strip().split('\t')
        labels.append(int(label))
        sentences.append(sentence)
    return labels, sentences

train_labels, train_sentences = load_data('./dataset/sst_train.txt')
test_labels, test_sentences = load_data('./dataset/sst_test.txt')

# Tokenize sentences
def tokenize(sentence):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.lower().split()

# Filter out empty lists of tokens and their corresponding labels
train_tokens, train_labels = zip(*[(tokens, label) for tokens, label in zip([tokenize(sentence) for sentence in train_sentences], train_labels) if tokens])
test_tokens, test_labels = zip(*[(tokens, label) for tokens, label in zip([tokenize(sentence) for sentence in test_sentences], test_labels) if tokens])

# Create vocabulary
word_counts = Counter([word for sentence in train_tokens for word in sentence])
vocab = {word: i + 2 for i, word in enumerate(word_counts.keys())}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

# Convert tokens to indices
def tokens_to_indices(tokens_list, vocab):
    return [[vocab.get(token, vocab['<unk>']) for token in tokens] for tokens in tokens_list]

train_indices = tokens_to_indices(train_tokens, vocab)
test_indices = tokens_to_indices(test_tokens, vocab)

# Create Dataset
class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

train_dataset = TextDataset(train_indices, train_labels)
test_dataset = TextDataset(test_indices, test_labels)

# Create DataLoader
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(yy), torch.tensor(x_lens)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=pad_collate)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate)

# Define model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, x_lens):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.squeeze(0))
        return out

# Instantiate model
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 64
num_classes = len(set(train_labels))
print(num_classes)

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Add a separate device for inference
inference_device = torch.device("cpu")

best_accuracy = 0.0
cumulative_time = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.perf_counter()

    for batch_idx, (data, labels, lengths) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_time = time.perf_counter() - start_time
    cumulative_time += train_time
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Train time: {train_time:.5f}s, Cumulative time: {cumulative_time:.5f}s")

    # Evaluate on test set
    model.eval()
    predictions, true_labels = [], []
    inference_start = time.perf_counter()

    # Move model to the inference_device for inference
    model.to(inference_device)

    with torch.no_grad():
        for data, labels, lengths in test_loader:
            data, labels = data.to(inference_device), labels.to(inference_device)
            outputs = model(data, lengths)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())

    # Move model back to the original device after inference
    model.to(device)

    inference_time = time.perf_counter() - inference_start

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch + 1
    print(f"Epoch {epoch+1}/{num_epochs}, Test accuracy: {accuracy}, Inference time: {inference_time:.5f}s, Best accuracy: {best_accuracy} (Epoch {best_epoch})")
