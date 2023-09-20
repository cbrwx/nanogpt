import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
from bs4 import BeautifulSoup
import re
import os

def classify_title(title):
    if all(word[0].isupper() for word in title.split()):
        return 'name'
    
    place_suffixes = ["ville", "town", "city", "land", "stan", "burg", "shire", "mountain", "hill"]
    for suffix in place_suffixes:
        if title.lower().endswith(suffix):
            return 'place'
    
    place_keywords = ["mount ", "lake ", "river ", "ocean ", "sea ", "forest "]
    for keyword in place_keywords:
        if keyword in title.lower():
            return 'place'

    return 'item'

def formulate_question(title):
    category = classify_title(title)
    
    if not title.startswith("The ") and not title.split()[0][0].isupper():
        title = "the " + title
    
    if category == 'name':
        return f"Who is {title}?"
    elif category == 'place':
        return f"Where is {title}?"
    else:
        return f"What is {title}?"

def is_relevant_sentence(sentence):
    defining_keywords = ["is a", "are a", "was a", "were a"]
    return any(keyword in sentence for keyword in defining_keywords)

def clean_content(title, content):
    content = content.replace(title, '', 1)
    
    remove_phrases = [
        "This article needs additional citations", 
        "This article does not", 
        "Find sources:"
    ]
    for phrase in remove_phrases:
        content = content.replace(phrase, "")
    
    split_phrases = ["See also", "External links", "References"]
    for phrase in split_phrases:
        content = content.split(phrase)[0]
    
    content_lines = content.split('\n')
    metadata_phrases = ["Act of Parliament", "Territorial extent", "Dates", "Repeals/revokes"]
    cleaned_lines = [line for line in content_lines if not any(phrase in line for phrase in metadata_phrases) and
                     "wiki" not in line.lower()]

    cleaned_lines = [line for line in cleaned_lines if len(line.split()) > 5 or is_relevant_sentence(line)]
    
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'\n+', '\n', cleaned_content).strip()

    return cleaned_content

def get_random_wikipedia_articles(num_articles=10):
    articles = []
    batch_size = 10  # Number of articles to fetch in a single batch
    num_batches = num_articles // batch_size
    
    for batch in range(num_batches):
        print(f"Fetching batch {batch+1}/{num_batches}...")  # Print the current batch
        
        for _ in range(batch_size):
            while True:
                response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find("h1", {"id": "firstHeading"}).get_text()
                content_div = soup.find("div", {"id": "mw-content-text"}).find("div", {"class": "mw-parser-output"})

                for tag in ["a", "blockquote", "sup"]:
                    for match in content_div.findAll(tag):
                        match.extract()

                content = content_div.get_text()
                content = re.sub(r'\[.*?\]', '', content)
                content = clean_content(title, content)
                
                if not content or len(content) < 50:
                    continue

                question = formulate_question(title)
                result = f'user: {question}\njelpod: {content}\n'
                articles.append(result)
                break
        
        if batch < num_batches - 1:  # If it's not the last batch, wait for 1 second
            print(f"Pausing for 1 second before fetching the next batch...")
            time.sleep(1)
    
    return '\n\n'.join(articles)

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Integrated Training Loop
loop_iterations = 10
for loop_iter in range(loop_iterations):
    print(f"\n### Loop Iteration {loop_iter+1}/{loop_iterations} ###\n")  # Print the current loop iteration

    # Load model from previous epoch if it exists
    if loop_iter > 0 and os.path.exists(f"model_epoch_{loop_iter-1}.pt"):
        print(f"Loading model from epoch {loop_iter-1}...")
        model = torch.load(f"model_epoch_{loop_iter-1}.pt")
        model.to(device)  # Ensure the model is transferred to the correct device

    # Fetch articles and save to 'input.txt'
    articles = get_random_wikipedia_articles()
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(articles)
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    model = LanguageModel() if loop_iter == 0 else model  # Only create a new model on the first loop
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"Loop {loop_iter+1}/{loop_iterations}, Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    
    # If input.txt exists from a previous epoch, delete it
    if os.path.exists('input.txt'):
        os.remove('input.txt')   

    # Save the model at the end of each epoch
    torch.save(model, f"model_epoch_{loop_iter}.pt")   
