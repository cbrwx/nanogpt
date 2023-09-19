# gpty2_wiki.py

# Transformer-based Language Model for Wikipedia Article Summarization
This repository contains a language model inspired by the Transformer architecture. It dynamically fetches and cleans Wikipedia articles, formulates a question based on the title of the article, and pairs it with concise content. The model is then trained on this Q&A format, allowing for the generation of similar sequences.

## Features
- Uses the Transformer architecture for the language model.
- Dynamically fetches random Wikipedia articles for fresh and diverse training data.
- Cleans and processes the Wikipedia content to generate a Q&A format for training.
- Built upon the example provided by Andrej Karpathy, titled "nanogpt".
## Prerequisites
- Python 3.x
- PyTorch
- Beautiful Soup 4
## Model Architecture
The model utilizes the self-attention mechanism, a hallmark of the Transformer architecture. Specifically, the architecture is comprised of:

- Multi-head self-attention mechanism.
- Positional embeddings to understand the order of the sequence.
- Feed-forward neural networks.
- Layer normalization and dropout for regularization.
## Data Processing
To simulate a summarization task, the following steps are taken:

- Fetch random Wikipedia articles.
- Formulate a question based on the title of the article (e.g., "Who is Albert Einstein?" or "What is Quantum Mechanics?").
- Clean and process the content of the article to generate a concise answer.
- Train the model on this Q&A format.
## Training
Training parameters are easily configurable. The current settings are:

- Batch size: 64
- Block size: 256
- Max iterations: 5000
- Learning rate: 3e-4
During training, evaluation is performed at regular intervals to keep track of the model's performance on the training and validation sets.

## Usage
Ensure all required libraries are installed.
Clone this repository.
Run the main script to train the model. The script will:
- Fetch and process Wikipedia articles.
- Train the model on the generated Q&A format.
- Generate text at the end of each epoch.
- Save the model's state.

# Model Components:
## Token Embeddings:

- self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
The model begins by embedding tokens into a continuous vector space. The vocab_size determines the number of unique tokens, and n_embd is the embedding size. Each token is transformed into a n_embd-dimensional vector.
## Positional Embeddings:

- self.position_embedding_table = nn.Embedding(block_size, n_embd)
Given that the Transformer does not have inherent sequence awareness, positional embeddings are added to the token embeddings to give the model information about the position of a token in a sequence.
## Blocks (Transformer Layers):

- self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
The model comprises multiple transformer blocks, each containing multi-head self-attention mechanisms and feed-forward networks.
## Layer Normalization and Linear Head:

- self.ln_f = nn.LayerNorm(n_embd)
- self.lm_head = nn.Linear(n_embd, vocab_size)
After passing through all transformer blocks, the output goes through layer normalization. The lm_head is the final linear layer that projects the output back to the vocabulary size, effectively allowing the model to predict the next token in a sequence.
## Weight Initialization:

- self.apply(self._init_weights)
The weights of the model are initialized using the _init_weights method. This ensures that the weights have values that are conducive to training. Specifically:
- Linear layers are initialized with a normal distribution with a mean of 0 and a standard deviation of 0.02.
- Biases in linear layers are initialized with zeros.
- Embedding weights are initialized with a normal distribution with a mean of 0 and a standard deviation of 0.02. 

When data is passed through the model (forward method), the following steps occur:

- Token and positional embeddings are fetched and summed.
- The combined embeddings pass through the transformer blocks.
- The output is normalized and passed through the final linear layer.
- If targets are provided, the loss is computed; otherwise, only the logits are returned.

This implementation was inspired and built upon the example provided by Andrej Karpathy, specifically the "nanogpt" example. It incorporates the Transformer architecture to model the probability distribution of sequences of words for text generation.

.cbrwx

<---> original readme below <--->

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT
