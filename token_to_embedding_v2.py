import torch

from gpt_dataset_v1 import create_dataloader_v1

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# Input / Target tensors erzeugen
max_length = 4
dataloader_v1 = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader_v1)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("Targets IDs:\n", targets)

# Für jedes token in 256-dimensionales Embedding erzeugen
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)  # [0, 1, 2, 3, 4]

# Wenn das Token 2 zweimal in der input Sequenz vorkommt, hat es zweimal das
# selbe Embedding. Dadurch, dass es z.B. an Stelle 0 und an Stelle 2 vorkommt,
# führt die Addition der pos_embeddings erzeugen zu zwei unterschiedlichen
# Input-Embeddings.
input_embeddings = token_embeddings + pos_embeddings
