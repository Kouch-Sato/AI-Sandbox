import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the kouch."
text = " <|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
print(tokenizer.decode([479]))
print(tokenizer.decode([7673]))

# 2-4のSimpleTokenizerV2を使うと、<|unk|>はそのまま出力される。
# tiktokenを使う場合は、BPE(Byte Pair Encoding)によって未知の単語も出力される。
# ex: "kouch" -> ["k", "ouch"] に分解された
