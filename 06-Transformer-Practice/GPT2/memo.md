# parameter数を可視化する
```
for p in model.parameters():
    print(p.shape, p.numel())
```
合計：163009536 1.6億パラメーター！

torch.Size([50257, 768]) 38597376 # num_emb
torch.Size([1024, 768]) 786432 # postion_emb

## transformer 1個分 × 12個
### attention
torch.Size([768, 768]) 589824 # w_q
torch.Size([768, 768]) 589824 # w_k
torch.Size([768, 768]) 589824 # w_v
torch.Size([768, 768]) 589824 # out_proj wight
torch.Size([768]) 768.        # out_proj bias

### FeedForward
torch.Size([3072, 768]) 2359296 # Linear weight
torch.Size([3072]) 3072.        # Linear bias
torch.Size([768, 3072]) 2359296 # Linear weight
torch.Size([768]) 768           # Linear bias
torch.Size([768]) 768           # LayerNorm1 scale
torch.Size([768]) 768           # LayerNorm1 bias
torch.Size([768]) 768           # LayerNorm2 scale
torch.Size([768]) 768           # LayerNorm2 bias

## FinalNorm
torch.Size([768]) 768 # scale
torch.Size([768]) 768 # bias

## out_head
torch.Size([50257, 768]) 38597376 # out_head weight

# GPTモデルの変遷
## 入力した文字列
txt1 = "I love Jobs because"
txt2 = "Today, we are"

## 4章直後、ランダムな重み
I love Jobs because theatre unfolds Dice architect 59Ocean
Today, we are 342 IT!/rill architectures Ridley
The meaning of life Aeiman Byeswickattribute argue

## ５−５でGPT2-smallの重みをinstallしたとき
I love Jobs because he's a great guy. He's a great
Today, we are going to be doing a lot of work on the
The meaning of life is not the same as the meaning of death.