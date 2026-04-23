import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import re

if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'

# --- 1. DATA PREPROCESSING ---
def preprocess_text(file_path):
    with open(file_path, 'r') as f:
        text = f.read().lower()
    
    # Separate punctuation so "death!" becomes ["death", "!"]
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    words = text.split()
    
    vocab = sorted(list(set(words)))
    word_to_int = {w: i for i, w in enumerate(vocab)}
    int_to_word = {i: w for i, w in enumerate(vocab)}
    
    return words, vocab, word_to_int, int_to_word


# Custom Dataset for sliding windows
class ShakespeareDataset(Dataset):
    def __init__(self, words, word_to_int, seq_length):
        self.words_int = [word_to_int[w] for w in words]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.words_int) - self.seq_length

    def __getitem__(self, index):
        # Input: words [0,1,2], Target: words [1,2,3]
        return (
            torch.tensor(self.words_int[index : index + self.seq_length]),
            torch.tensor(self.words_int[index + 1 : index + self.seq_length + 1])
        )


# LSTM Model 
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # Ask about the meaning of this more, especially embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_state=None):
        x = self.embedding(x)

        out, state = self.lstm(x, prev_state)

        out = self.fc(out.reshape(-1, out.size(2)))

        return out, state


# Training Config
seq_length = 20
batch_size = 64
hidden_size = 512
embed_size = 256

words, vocab, word_to_int, int_to_word = preprocess_text(
    r"\Users\mattm\Documents\py_projects\Neural_Networks\LSTM\training_data\tiny_shakespear.txt"
)
dataset = ShakespeareDataset(words, word_to_int, seq_length)
dataloader = DataLoader(dataset, batch_size, shuffle=True)


model = TextLSTM(len(vocab), embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
print(f"Training on {device}...")
for epoch in range(15):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output, _ = model(x)
        loss = criterion(output, y.view(-1))
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
   
    print(f"Epoch: {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'shakespeare.pth')


def generate_response(prompt, gen_length=50, temp=0.7, top_k=5):
    model.eval()
    words = prompt.lower().split()
    current_seq = [word_to_int[w] for w in words if w in word_to_int]
    state = None

    with torch.no_grad():

        for _ in range(gen_length):
            x = torch.tensor([current_seq]).to(device)
            logits, state = model(x, state) # We use raw logits here
        
            # 1. Apply Temperature 
            # (output[-1] is the last word predicted)
            logits = logits[-1] / temp
            
            # 2. Apply Top-K Filtering
            # This ignores everything except the 5 best words
            v, i = torch.topk(logits, top_k)
            probs = torch.softmax(v, dim=0)

            # Explain this
            next_idx = i[torch.multinomial(probs, num_samples=1)].item()

            words.append(int_to_word[next_idx])

            current_seq = [next_idx]

        return " ".join(words)
    

print("Model Response...")
print(generate_response("the robot said : n", temp=0.5))