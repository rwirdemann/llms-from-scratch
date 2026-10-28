import torch

input_ids = torch.tensor([2, 3, 5, 1, 2])
print("Inputs:\n", input_ids)

vocab_size = 6
output_dim = 3  # embedding size

torch.manual_seed(129)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print("Embedding layer:\n", embedding_layer.weight)

# gib mir das embedding für token-ID 3 (das ist der Eintrag an Index 3 des
# embedding layers)
print("Embedding index 3\n", embedding_layer(torch.tensor([3])))
