from importlib.metadata import version

import tiktoken

# tiktoken implementiert Byte Pair Encoding (BPE). Unbekannte Wörter werden bis
# auf einzelne Zeichen herunter gebrochen. Und das Vokabular enthält für jedes
# Zeichen ein Token.
print("tiktoken version:", version("tiktoken"))

tik_tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, world. Is this-- a test? Akwirw"

integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tik_tokenizer.decode(integers)
print(strings)
