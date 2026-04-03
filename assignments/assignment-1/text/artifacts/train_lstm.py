import pandas as pd
import numpy as np
import torch, wandb, os, shutil, time, zipfile, re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import urllib.request
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torch import tensor, nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding, DistilBertForSequenceClassification, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("GPU available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

wandb.login(key="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")

# load dataset from hugging face
dataset = load_dataset("SetFit/20_newsgroups")
df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)

### truncate and clean input corpus
MAX_WORDS = 400
def truncate_text(text, max_words=MAX_WORDS):
    words = str(text).split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Skip quoted text
        stripped = line.strip()
        if stripped.startswith('>') or stripped.startswith(':') or stripped.startswith('|'):
            continue
        # Skip email headers
        if any(line.startswith(h) for h in ['From:', 'Subject:', 'Organization:',
                                            'Lines:', 'Reply-To:', 'NNTP-Posting-Host:']):
            continue
        # Skip "--"
        if line.strip() == '--':
            break
        cleaned.append(line)
    clean_text = ' '.join(cleaned).strip()
    clean_text = re.sub(r'http\S+|ftp\S+|www\.\S+', '', text)               # URLs
    clean_text = re.sub(r'\S+@\S+', '', clean_text)                         # emails
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()                    # collapse whitespace
    clean_text = re.sub(r'\[.*?deletia.*?\]', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\[.*?snip.*?\]', '', clean_text, flags=re.IGNORECASE)
    return clean_text

train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
train_df['text_clean'] = train_df['text'].apply(clean_text).apply(truncate_text)
test_df['text_clean'] = test_df['text'].apply(clean_text).apply(truncate_text)

# handle len(text) < min length
min_length = 100

train_df = train_df[train_df['text_clean'].str.len() >= min_length]
test_df = test_df[test_df['text_clean'].str.len() >= min_length]

# Split train -> train + val (80/20)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df['label']
)

# tokenize texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LENGTH = 512
def tokenize_fn(df):
    return tokenizer(
        df['text_clean'],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_ds = Dataset.from_pandas(train_df[['text_clean', 'label']].reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df[['text_clean', 'label']].reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df[['text_clean', 'label']].reset_index(drop=True))

print("Tokenizing...")
train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=256).rename_column("label", "labels")
val_ds   = val_ds.map(tokenize_fn, batched=True, batch_size=256).rename_column("label", "labels")
test_ds  = test_ds.map(tokenize_fn, batched=True, batch_size=256).rename_column("label", "labels")

# final process ds
if 'text_clean' in train_ds.column_names:
    train_ds = train_ds.remove_columns(['text_clean'])
if 'text_clean' in val_ds.column_names:
    val_ds   = val_ds.remove_columns(['text_clean'])
if 'text_clean' in test_ds.column_names:
    test_ds  = test_ds.remove_columns(['text_clean'])

# rm token_type_ids: no need in classification task
if 'token_type_ids' in train_ds.column_names:
    train_ds = train_ds.remove_columns(['token_type_ids'])
if 'token_type_ids' in val_ds.column_names:
    val_ds   = val_ds.remove_columns(['token_type_ids'])
if 'token_type_ids' in test_ds.column_names:
    test_ds  = test_ds.remove_columns(['token_type_ids'])

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        return focal_loss.mean()
    
train_labels = torch.tensor(train_ds['labels'])

class_counts = torch.bincount(train_labels)
num_classes = len(class_counts)
total_samples = len(train_labels)

# class weights = N / num_labels * n_i
class_weights = total_samples / (num_classes * class_counts)
print("Class weights:", class_weights)

# init FocalLoss
criterion = FocalLoss(gamma=2.0, weight=class_weights.float().to(device))

def download_glove(glove_dir='DL/glove', dim=100):
    os.makedirs(glove_dir, exist_ok=True)
    glove_path = f'{glove_dir}/glove.6B.{dim}d.txt'

    if os.path.exists(glove_path):
        print(f"GloVe {dim}d already exists at {glove_path}")
        return glove_path

    zip_path = f'{glove_dir}/glove.6B.zip'
    if not os.path.exists(zip_path):
        print("Downloading GloVe 6B (822MB)... this may take a few minutes")
        urllib.request.urlretrieve(
            'https://nlp.stanford.edu/data/glove.6B.zip', zip_path
        )
        print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract(f'glove.6B.{dim}d.txt', glove_dir)
    print(f"GloVe {dim}d ready at {glove_path}")
    return glove_path

# ============ Attention Layer ============
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim * 2)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # (batch, hidden_dim * 2)
        return context, attn_weights


# ============ LSTM Model with Attention ============
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, tfidf_scores,
                num_layers=1, dropout=0.3, pretrained_embeddings=None, freeze_embeddings=False,
                use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_attention = use_attention

        # Load pretrained embeddings
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen")
            else:
                print("Embeddings will be fine-tuned")

        if tfidf_scores is not None:
            self.register_buffer('tfidf_scores', tfidf_scores)
        else:
            self.tfidf_scores = None

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layer
        if use_attention:
            self.attention = Attention(hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        if self.tfidf_scores is not None:
            weights = self.tfidf_scores[x]           # (batch, seq_len)
            weights = weights.unsqueeze(-1)          # (batch, seq_len, 1)
            embedded = embedded * weights            # broadcast multiply
        embedded = self.dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_dim * 2)

        if self.use_attention:
            # Use attention over all timesteps
            context, attn_weights = self.attention(lstm_out)
            output = self.dropout(context)
        else:
            # Use final hidden state (original behavior)
            hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            output = self.dropout(hidden_cat)

        return self.fc(output)


# ============ Dataset ============
class TextDataset(TorchDataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = text_to_indices(text, self.vocab, self.max_len)
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ============ Training & Eval ============
def training_lstm(model, loader, optimizer, scheduler, device, class_weights, max_grad_norm=5.0):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc="Training", leave=True)
    for batch in progress:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


def evaluate_lstm(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return accuracy, all_preds, all_labels


def build_vocab(texts, min_freq=2, max_vocab=30000):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in counter.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def text_to_indices(text, vocab, max_len=256):
    tokens = text.lower().split()[:max_len]
    return [vocab.get(t, vocab['<UNK>']) for t in tokens]

train_texts, val_texts, test_texts = train_df['text_clean'].tolist(), val_df['text_clean'].tolist(), test_df['text_clean'].tolist()
train_labels, val_labels, test_labels = train_df['label'].tolist(), val_df['label'].tolist(), test_df['label'].tolist()

# build vocab dict
vocab = build_vocab(train_texts, min_freq=1, max_vocab=50000)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Create datasets
MAX_LEN = 256
train_ds_lstm = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
val_ds_lstm = TextDataset(val_texts, val_labels, vocab, MAX_LEN)
test_ds_lstm = TextDataset(test_texts, test_labels, vocab, MAX_LEN)

print(f"Train: {len(train_ds_lstm)}, Val: {len(val_ds_lstm)}, Test: {len(test_ds_lstm)}") 



# Compute TF-IDF on training data
tfidf = TfidfVectorizer(vocabulary=vocab, max_features=len(vocab), lowercase=False)
tfidf.fit(train_texts)

# Get TF-IDF scores for each word in vocab
tfidf_scores = np.zeros(len(vocab))
for word, idx in vocab.items():
    if word in ('<PAD>', '<UNK>'):
        continue
    if word in tfidf.vocabulary_:
        tfidf_scores[idx] = tfidf.idf_[tfidf.vocabulary_[word]]

tfidf_scores = torch.tensor(tfidf_scores, dtype=torch.float32).to(device)

# Train 
train_texts = train_df['text_clean'].tolist()
val_texts = val_df['text_clean'].tolist()
test_texts = test_df['text_clean'].tolist()
train_labels = train_df['label'].tolist()
val_labels = val_df['label'].tolist()
test_labels = test_df['label'].tolist()

vocab = build_vocab(train_texts, min_freq=1, max_vocab=50000)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

MAX_LEN = 256
train_ds_lstm = TextDataset(train_texts, train_labels, vocab, MAX_LEN)
val_ds_lstm = TextDataset(val_texts, val_labels, vocab, MAX_LEN)
test_ds_lstm = TextDataset(test_texts, test_labels, vocab, MAX_LEN)
print(f"Train: {len(train_ds_lstm)}, Val: {len(val_ds_lstm)}, Test: {len(test_ds_lstm)}")


# ============ Load GloVe ============
HIDDEN_DIM = 128
NUM_LAYERS = 2

lstm_model = LSTMClassifier(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    tfidf_scores=tfidf_scores,
    num_classes=20,
    dropout=0.3,
    pretrained_embeddings=pretrained_emb,
    freeze_embeddings=False       # fine-tune embeddings
).to(device)

total_params = sum(p.numel() for p in lstm_model.parameters())
trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")

# ============ Dataloaders ============
BATCH_SIZE = 32
train_loader_lstm = DataLoader(train_ds_lstm, batch_size=BATCH_SIZE//2, shuffle=True)
val_loader_lstm = DataLoader(val_ds_lstm, batch_size=BATCH_SIZE, shuffle=False)
test_loader_lstm = DataLoader(test_ds_lstm, batch_size=BATCH_SIZE, shuffle=False)

# ============ Training config ============
EPOCHS = 20
LR = 1e-3               # from-scratch LR, not fine-tuning LR
MAX_GRAD_NORM = 5.0

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)  # Adam, no weight decay
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader_lstm),
    num_training_steps=len(train_loader_lstm) * EPOCHS
)

# ============ Train ============
wandb.init(project="20-newsgroups-lstm", name="lstm_glove_100d", reinit=True)
wandb.config.update({
    "model": "BiLSTM + GloVe",
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "epochs": EPOCHS,
    "lr": LR,
})

best_acc = 0.0
best_model_path = './best_model_lstm.pt'
print(f"\nTraining LSTM on {device}...")

start_time = time.time()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = training_lstm(lstm_model, train_loader_lstm, optimizer, scheduler, device, class_weights, MAX_GRAD_NORM)
    val_acc, _, _ = evaluate_lstm(lstm_model, val_loader_lstm, device)

    wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_accuracy': val_acc,
                'learning_rate': scheduler.get_last_lr()[0]})

    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(lstm_model.state_dict(), best_model_path)
        print(f"  saved")

train_time = time.time() - start_time

lstm_model.load_state_dict(torch.load(best_model_path))
test_acc, test_preds, test_labels_out = evaluate_lstm(lstm_model, test_loader_lstm, device)

wandb.log({'test_accuracy': test_acc})
wandb.finish()

print(f"\nLSTM Training Complete")
print(f"Best Val Accuracy: {best_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Time: {train_time:.0f}s")

