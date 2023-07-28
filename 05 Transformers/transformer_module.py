from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tiktoken

path = "data/opus100_croped_10"
opus100 = load_from_disk(path)

class Opus100Dataset(Dataset):
    def __init__(self, dataset, source_language, target_language, tokenizer, start_token, end_token, padding_token, max_length):
        self.dataset = dataset
        self.source_language = source_language
        self.target_language = target_language
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def encode(self, text):
        encoded = self.tokenizer(text)
        encoded = self.start_token + encoded + self.end_token
        if len(encoded) > self.max_length:  # Truncate if too long
            encoded = encoded[:self.max_length]
        else:  # Pad if too short
            encoded = encoded + self.padding_token * (self.max_length - len(encoded))
        return torch.tensor(encoded)
    
    def __getitem__(self, idx):
        source = self.dataset[idx]["translation"][self.source_language]
        source = self.encode(source)

        target = self.dataset[idx]["translation"][self.target_language]
        target = self.encode(target)
        return source, target

encoder = tiktoken.get_encoding("cl100k_base")
start_token = encoder.encode(chr(1))
end_token = encoder.encode(chr(2))
padding_token = encoder.encode(chr(3))

max_secuence_length = 10+2 #104
train_dataset = Opus100Dataset(opus100["train"], "en", "es", encoder.encode, start_token, end_token, padding_token, max_secuence_length)
validation_dataset = Opus100Dataset(opus100["validation"], "en", "es", encoder.encode, start_token, end_token, padding_token, max_secuence_length)
test_dataset = Opus100Dataset(opus100["test"], "en", "es", encoder.encode, start_token, end_token, padding_token, max_secuence_length)

# ! Quitar
import numpy as np
def subsample_dataset(dataset, new_size):
    indices = np.random.choice(len(dataset), new_size, replace=False)
    indices = indices.tolist()  # Convert numpy.int64 indices to native int
    return torch.utils.data.Subset(dataset, indices)
train_dataset = subsample_dataset(train_dataset, 1000)
validation_dataset = subsample_dataset(validation_dataset, 100)
test_dataset = subsample_dataset(test_dataset, 100)
print(f"train_dataset: {len(train_dataset)}, validation_dataset: {len(validation_dataset)}, test_dataset: {len(test_dataset)}")
# ! Quitar

BS = 4
train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BS, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

vocab_size = encoder.n_vocab
dim_embedding = 512
max_sequence_len = 104
heads = 8
Nx = 6
prob_dropout = 0.1
transformer = nn.Transformer(d_model=dim_embedding, 
                             nhead=heads, 
                             num_encoder_layers=Nx, 
                             num_decoder_layers=Nx, 
                             dim_feedforward=2048, 
                             dropout=prob_dropout, 
                             activation="relu", 
                             custom_encoder=None, 
                             custom_decoder=None)

loss_function = nn.CrossEntropyLoss(ignore_index=padding_token[0])
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_len, embedding_model_dim):
        super().__init__()
        self.embedding_dim = embedding_model_dim
        positional_encoding = torch.zeros(max_sequence_len, self.embedding_dim)
        for pos in range(max_sequence_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i]     = torch.sin(torch.tensor(pos / (10000 ** ((2 * i) / self.embedding_dim))))
                positional_encoding[pos, i + 1] = torch.cos(torch.tensor(pos / (10000 ** ((2 * (i+1)) / self.embedding_dim))))
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.embedding_dim))
        sequence_len = x.size(1)
        x = x + self.positional_encoding[:,:sequence_len]
        return x
class Linear(nn.Module):
    def __init__(self, dim_embedding, vocab_size):
        super().__init__()
        self.linear = nn.Linear(dim_embedding, vocab_size)
        
    def forward(self, x):
        x = self.linear(x)
        return x
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.softmax(x)
        return x
embedding = Embedding(vocab_size, dim_embedding).to(device)
positionalEncoding = PositionalEncoding(max_sequence_len, dim_embedding).to(device)
linear = Linear(dim_embedding, vocab_size).to(device)
softmax = Softmax().to(device)

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        src = embedding(src)
        src = positionalEncoding(src).permute(1, 0, 2)
        trg_to_loss = trg
        trg = embedding(trg)
        trg = positionalEncoding(trg).permute(1, 0, 2)
        # print(f"src shape: {src.shape}, trg shape: {trg.shape}, trg_to_loss shape: {trg_to_loss.shape}")

        pred = model(src, trg)
        pred = linear(pred)
        pred = softmax(pred)
        # print(f"pred.permute(1, 2, 0) shape: {pred.permute(1, 2, 0).shape}, trg_to_loss shape: {trg_to_loss.shape}")
        loss = loss_fn(pred.permute(1, 2, 0), trg_to_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def validation_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            src = embedding(src)
            src = positionalEncoding(src).permute(1, 0, 2)
            trg_to_loss = trg
            trg = embedding(trg)
            trg = positionalEncoding(trg).permute(1, 0, 2)

            pred = model(src, trg)
            pred = linear(pred)
            pred = softmax(pred)
            test_loss += loss_fn(pred.permute(1, 2, 0), trg_to_loss).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

transformer = transformer.to(device)
for t in range(10):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, transformer, loss_function, optimizer, device)
    validation_loop(validation_dataloader, transformer, loss_function, device)
print("Done!")
