import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import re

# Step 1: Load the dataset
with open('poems.txt', 'r', encoding='utf-8') as f:
    poems = f.read().splitlines()

# Step 2: Tokenize the text
word_list = []
for poem in poems:
    # Split into words and punctuation, convert to lowercase
    words = re.findall(r'\w+|[^\w\s]', poem.lower())
    word_list.extend(words)

# Step 3: Create vocabulary
vocab = list(set(word_list))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

# Step 4: Create sequences
seq_length = 5  # Adjust this if you want longer/shorter sequences
sequences = []
for i in range(len(word_list) - seq_length):
    seq = word_list[i:i + seq_length]
    target = word_list[i + seq_length]
    sequences.append((seq, target))

# Step 5: Define the Dataset class
class PoemDataset(Dataset):
    def __init__(self, sequences, word_to_idx, one_hot=False):
        self.sequences = sequences
        self.word_to_idx = word_to_idx
        self.one_hot = one_hot
        self.vocab_size = len(word_to_idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        seq_indices = [self.word_to_idx[word] for word in seq]
        target_idx = self.word_to_idx[target]
        if self.one_hot:
            seq_one_hot = torch.zeros(len(seq), self.vocab_size)
            for i, idx in enumerate(seq_indices):
                seq_one_hot[i, idx] = 1
            return seq_one_hot, target_idx
        else:
            return torch.tensor(seq_indices), target_idx

# Step 6: Define the Models
class OneHotLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(OneHotLSTM, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(EmbeddingLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Step 7: Training Function
def train_model(model, dataset, epochs=10):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for seq, target in dataloader:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    training_time = time.time() - start_time
    return training_time, avg_loss

# Step 8: Generation Functions
def generate_text_one_hot(model, start_seq, word_to_idx, idx_to_word, length=20):
    model.eval()
    current_seq = start_seq.copy()
    generated = current_seq.copy()
    for _ in range(length):
        seq_indices = [word_to_idx[word] for word in current_seq]
        seq_one_hot = torch.zeros(len(current_seq), len(word_to_idx))
        for i, idx in enumerate(seq_indices):
            seq_one_hot[i, idx] = 1
        seq_one_hot = seq_one_hot.unsqueeze(0)
        with torch.no_grad():
            output = model(seq_one_hot)
        next_word_idx = torch.argmax(output).item()
        next_word = idx_to_word[next_word_idx]
        generated.append(next_word)
        current_seq = current_seq[1:] + [next_word]
    return ' '.join(generated)

def generate_text_embedding(model, start_seq, word_to_idx, idx_to_word, length=20):
    model.eval()
    current_seq = start_seq.copy()
    generated = current_seq.copy()
    for _ in range(length):
        seq_indices = [word_to_idx[word] for word in current_seq]
        seq_tensor = torch.tensor(seq_indices).unsqueeze(0)
        with torch.no_grad():
            output = model(seq_tensor)
        next_word_idx = torch.argmax(output).item()
        next_word = idx_to_word[next_word_idx]
        generated.append(next_word)
        current_seq = current_seq[1:] + [next_word]
    return ' '.join(generated)

# Step 9: Set Hyperparameters
hidden_size = 128
embedding_dim = 100
epochs = 10

# Step 10: One-Hot Encoding Approach
print("Training One-Hot Encoding Model...")
one_hot_dataset = PoemDataset(sequences, word_to_idx, one_hot=True)
one_hot_model = OneHotLSTM(vocab_size, hidden_size)
one_hot_time, one_hot_loss = train_model(one_hot_model, one_hot_dataset, epochs)
start_seq = word_list[:seq_length]  # Use first 5 words as starting sequence
one_hot_generated = generate_text_one_hot(one_hot_model, start_seq, word_to_idx, idx_to_word)

# Step 11: Trainable Embeddings Approach
print("\nTraining Trainable Embeddings Model...")
embedding_dataset = PoemDataset(sequences, word_to_idx, one_hot=False)
embedding_model = EmbeddingLSTM(vocab_size, embedding_dim, hidden_size)
embedding_time, embedding_loss = train_model(embedding_model, embedding_dataset, epochs)
embedding_generated = generate_text_embedding(embedding_model, start_seq, word_to_idx, idx_to_word)

# Step 12: Compare Results
print(f"\nOne-Hot Encoding: Training Time: {one_hot_time:.2f} seconds, Final Loss: {one_hot_loss:.4f}")
print(f"Trainable Embeddings: Training Time: {embedding_time:.2f} seconds, Final Loss: {embedding_loss:.4f}")
print("\nGenerated Text with One-Hot Encoding:")
print(one_hot_generated)
print("\nGenerated Text with Trainable Embeddings:")
print(embedding_generated)

# Step 13: Quick Analysis
print("\nQuick Comparison:")
print("- One-Hot: Simple but uses more memory, may not capture word meanings well.")
print("- Embeddings: Faster, learns word relationships, likely generates better text.")
