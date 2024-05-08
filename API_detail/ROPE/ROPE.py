import torch

base = 10000
dim = 8
max_seq_len_cached = 5
print(torch.arange(0, dim, 2).float() / dim)
print(base ** (torch.arange(0, dim, 2).float() / dim))
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))            # YoungL：1 / (10000 ** (2*i / d))
print(inv_freq)
list1 = inv_freq.tolist()
# for i in range(len(list1)-1):
#     print(list1[i] / list1[i+1])
breakpoint()

t = torch.arange(max_seq_len_cached, dtype=inv_freq.dtype)         # YoungL：生成[0-max_seq_len_cached)的一维数组

freqs = torch.einsum("i,j->ij", t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1) 
print(t.shape, inv_freq.shape, freqs.shape, emb.shape)
print(freqs.sin())

print(emb[..., : emb.shape[-1] // 2].shape)

freqs1 = freqs[..., : freqs.shape[-1] // 2] 
freqs2 = freqs[..., freqs.shape[-1] // 2 :]  
a = torch.cat((-freqs2, freqs1), dim=-1)
print(a)