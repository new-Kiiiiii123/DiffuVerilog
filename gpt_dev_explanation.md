**`gpt_dev.ipynb` 文件说明文档**

这个Jupyter Notebook是[Zero To Hero](https://karpathy.ai/zero-to-hero.html)关于GPT视频的配套笔记本，旨在逐步构建一个GPT模型。

---

**代码块解析：**

**1. Markdown 单元格：标题和简介**

```markdown
## Building a GPT

Companion notebook to the [Zero To Hero](https://karpathy.ai/zero-to-hero.html) video on GPT.
```

*   **功能**：这是Notebook的标题和简介部分，说明了Notebook的目的——构建一个GPT模型，并指出它是Andre Karpathy的"Zero To Hero"系列视频的配套资源。

**2. 代码单元格：下载数据集**

```python
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

*   **功能**：此代码块使用`wget`命令从GitHub下载一个名为`input.txt`的文本文件。这个文件是“tiny shakespeare”数据集，包含了莎士比亚作品的文本，将作为训练GPT模型的数据集。

**3. 代码单元格：读取数据集**

```python
# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

*   **功能**：此代码块打开刚刚下载的`input.txt`文件，并将其全部内容读取到一个名为`text`的字符串变量中。`encoding='utf-8'`确保正确处理文件中的各种字符。

**4. 代码单元格：打印数据集长度**

```python
print("length of dataset in characters: ", len(text))
```

*   **功能**：此代码块打印出`text`字符串（即整个数据集）的字符长度，让用户了解数据集的大小。

**5. 代码单元格：查看数据集前1000个字符**

```python
# let's look at the first 1000 characters
print(text[:1000])
```

*   **功能**：此代码块打印出数据集的前1000个字符，以便用户可以初步检查数据集的内容和格式。

**6. 代码单元格：提取唯一字符和词汇表大小**

```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
```

*   **功能**：
    *   `set(text)`：创建一个包含`text`中所有唯一字符的集合。
    *   `list(...)`：将集合转换为列表。
    *   `sorted(...)`：对列表中的字符进行排序，确保顺序一致性。
    *   `chars`：存储排序后的唯一字符列表。
    *   `vocab_size`：计算唯一字符的数量，即词汇表的大小。
    *   最后打印出所有唯一字符和词汇表大小。

**7. 代码单元格：创建字符与整数的映射**

```python
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))
```

*   **功能**：
    *   `stoi` (string to integer)：创建一个字典，将每个字符映射到一个唯一的整数ID。
    *   `itos` (integer to string)：创建一个字典，将每个整数ID映射回对应的字符。
    *   `encode`：一个lambda函数，用于将字符串转换为整数ID列表（编码器）。
    *   `decode`：一个lambda函数，用于将整数ID列表转换回字符串（解码器）。
    *   最后，通过示例字符串"hii there"展示了编码和解码的功能。

**8. 代码单元格：将文本数据集编码为PyTorch张量**

```python
# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this
```

*   **功能**：
    *   导入PyTorch库。
    *   使用之前定义的`encode`函数将整个`text`数据集编码为整数列表。
    *   将编码后的整数列表转换为PyTorch的`torch.tensor`，数据类型设置为`torch.long`，这是PyTorch中用于索引和分类任务的常用整数类型。
    *   打印出张量的形状和数据类型，并展示前1000个编码后的整数，与原始文本的前1000个字符进行对比。

**9. 代码单元格：划分训练集和验证集**

```python
# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
```

*   **功能**：将整个数据集`data`划分为训练集和验证集。
    *   `n`：计算数据集的90%作为训练集的大小。
    *   `train_data`：包含数据集的前90%。
    *   `val_data`：包含数据集的后10%。

**10. 代码单元格：展示`block_size`和训练数据切片**

```python
block_size = 8
train_data[:block_size+1]
```

*   **功能**：
    *   定义`block_size`为8，这代表了模型在进行预测时考虑的最大上下文长度。
    *   打印出训练数据的前`block_size + 1`个元素，用于演示数据块的结构。

**11. 代码单元格：演示上下文和目标**

```python
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
```

*   **功能**：此代码块演示了语言模型训练中“上下文”和“目标”的概念。
    *   `x`：作为输入序列，取`train_data`的前`block_size`个元素。
    *   `y`：作为目标序列，取`train_data`从第二个元素开始的`block_size`个元素。
    *   循环遍历`block_size`，每次迭代都展示了当输入是当前上下文时，模型应该预测的下一个目标字符。这模拟了自回归语言模型的训练过程。

**12. 代码单元格：批量数据加载函数`get_batch`**

```python
torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
```

*   **功能**：
    *   设置随机种子以确保结果可复现。
    *   定义`batch_size`（并行处理的独立序列数量）和`block_size`（预测的最大上下文长度）。
    *   `get_batch(split)`函数：
        *   根据`split`参数（'train'或'val'）选择相应的数据集。
        *   随机选择`batch_size`个起始索引`ix`。
        *   从这些索引开始，提取长度为`block_size`的输入序列`x`和对应的目标序列`y`。
        *   返回`x`和`y`。
    *   调用`get_batch('train')`获取一个训练批次的数据，并打印出输入`xb`和目标`yb`的形状和内容。
    *   最后，通过嵌套循环再次演示了批次中每个序列的上下文和目标关系。

**13. 代码单元格：打印输入张量`xb`**

```python
print(xb) # our input to the transformer
```

*   **功能**：简单地打印出上一个代码块中生成的输入张量`xb`，强调这是将要输入到Transformer模型中的数据。

**14. 代码单元格：Bigram语言模型定义和测试**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
```

*   **功能**：
    *   导入必要的PyTorch模块。
    *   定义了一个简单的`BigramLanguageModel`类，这是一个基于大字的语言模型。
        *   `__init__`：初始化模型，主要包含一个`nn.Embedding`层，它将每个token映射到一个与词汇表大小相同的向量，这个向量直接作为下一个token的logits。
        *   `forward`：定义模型的前向传播。接收输入`idx`和可选的目标`targets`。它通过嵌入层获取logits，如果提供了目标，则计算交叉熵损失。
        *   `generate`：定义文本生成方法。给定一个起始序列`idx`和要生成的最大token数量`max_new_tokens`，模型会循环生成下一个token，直到达到指定数量。它通过获取当前序列的logits，对最后一个时间步的logits应用softmax，然后从概率分布中采样下一个token，并将其添加到序列中。
    *   实例化`BigramLanguageModel`。
    *   使用之前生成的`xb`和`yb`进行一次前向传播，并打印logits的形状和计算出的损失。
    *   使用模型生成100个新token，并解码打印出来，展示未经训练的模型生成的结果。

**15. 代码单元格：创建PyTorch优化器**

```python
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

*   **功能**：此代码块创建了一个PyTorch优化器`torch.optim.AdamW`。
    *   `m.parameters()`：指定要优化的模型参数。
    *   `lr=1e-3`：设置学习率为0.001。`AdamW`是一种常用的优化算法，适用于深度学习模型。

**16. 代码单元格：模型训练循环**

```python
batch_size = 32
for steps in range(100): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
```

*   **功能**：此代码块执行模型的训练循环。
    *   重新设置`batch_size`为32。
    *   循环100步（在实际训练中通常需要更多步）。
    *   在每次迭代中：
        *   调用`get_batch('train')`获取一个训练批次的数据。
        *   将输入`xb`和目标`yb`传递给模型`m`进行前向传播，获取logits和损失。
        *   `optimizer.zero_grad(set_to_none=True)`：清除之前计算的梯度。
        *   `loss.backward()`：执行反向传播，计算每个参数的梯度。
        *   `optimizer.step()`：根据计算出的梯度更新模型参数。
    *   训练循环结束后，打印出最后一次迭代的损失值。

**17. 代码单元格：训练后生成文本**

```python
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
```

*   **功能**：在模型经过100步训练后，再次使用模型生成500个新token，并解码打印出来。与训练前的生成结果对比，可以观察到模型学习到的一些模式。

**18. Markdown 单元格：自注意力机制的数学技巧**

```markdown
## The mathematical trick in self-attention
```

*   **功能**：这是一个标题，引入了Notebook的下一个主要部分——自注意力机制的数学原理。

**19. 代码单元格：加权聚合的玩具示例（矩阵乘法）**

```python
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
```

*   **功能**：此代码块通过一个玩具示例演示了如何使用矩阵乘法实现“加权聚合”。
    *   `a`：创建一个下三角矩阵，并对其行进行归一化，使其每行和为1。这模拟了注意力权重。
    *   `b`：创建一个随机整数矩阵。
    *   `c = a @ b`：执行矩阵乘法，展示了`a`中的权重如何应用于`b`中的值，实现加权平均。

**20. 代码单元格：初始化用于自注意力演示的张量**

```python
# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape
```

*   **功能**：初始化一个随机张量`x`，其形状为`(B, T, C)`，分别代表批次大小、时间步长（序列长度）和通道数（特征维度）。这是后续自注意力机制演示的输入。

**21. 代码单元格：手动实现平均聚合**

```python
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)
```

*   **功能**：此代码块手动实现了一个“平均聚合”操作。对于每个时间步`t`，它计算当前时间步及其之前所有时间步的特征的平均值。这是一种简单的上下文聚合方式，但效率不高。

**22. 代码单元格：使用矩阵乘法实现平均聚合（版本2）**

```python
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
```

*   **功能**：此代码块展示了如何使用矩阵乘法更高效地实现与上一个代码块相同的平均聚合。
    *   `wei`：创建一个下三角矩阵并归一化，使其每行表示对过去和当前时间步的均匀权重。
    *   `xbow2 = wei @ x`：通过矩阵乘法实现加权聚合。
    *   `torch.allclose(xbow, xbow2)`：验证两种实现方式的结果是否一致，证明矩阵乘法的效率。

**23. 代码单元格：使用Softmax实现平均聚合（版本3）**

```python
# version 3: use Softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
```

*   **功能**：此代码块演示了如何使用Softmax函数实现加权聚合，这是自注意力机制中的关键一步。
    *   `tril`：创建一个下三角矩阵。
    *   `wei.masked_fill(tril == 0, float('-inf'))`：将下三角矩阵中为0（即上三角部分）的元素填充为负无穷大。
    *   `F.softmax(wei, dim=-1)`：对`wei`应用Softmax。由于负无穷大的存在，Softmax会将上三角部分的权重变为0，从而实现只关注过去和当前时间步的效果。
    *   `xbow3 = wei @ x`：通过矩阵乘法进行加权聚合。
    *   `torch.allclose(xbow, xbow3)`：验证结果的一致性。

**24. 代码单元格：自注意力机制的实现（版本4）**

```python
# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape
```

*   **功能**：这是自注意力机制的核心实现。
    *   重新初始化`x`，并定义`head_size`。
    *   `key`, `query`, `value`：三个独立的线性层，用于将输入`x`投影到键（key）、查询（query）和值（value）空间。
    *   `k = key(x)`, `q = query(x)`：计算键和查询。
    *   `wei = q @ k.transpose(-2, -1)`：计算注意力分数（affinity），即查询与键的点积。这衡量了每个位置对其他位置的关注程度。
    *   `wei = wei.masked_fill(tril == 0, float('-inf'))`：应用因果掩码（causal mask），确保每个位置只能关注其自身和之前的位置。
    *   `wei = F.softmax(wei, dim=-1)`：对注意力分数应用Softmax，将其转换为概率分布，表示权重。
    *   `v = value(x)`：计算值。
    *   `out = wei @ v`：将注意力权重应用于值，进行加权聚合，得到自注意力层的输出。
    *   打印输出的形状。

**25. 代码单元格：打印注意力权重矩阵**

```python
wei[0]
```

*   **功能**：打印出批次中第一个样本的注意力权重矩阵`wei`，可以观察到因果掩码的效果（上三角部分为0或接近0）。

**26. Markdown 单元格：自注意力机制的注意事项**

```markdown
Notes:
- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides `wei` by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below
```

*   **功能**：此Markdown单元格提供了关于自注意力机制的重要说明和概念解释，包括：
    *   注意力是一种通信机制。
    *   注意力没有空间概念，需要位置编码。
    *   批次中的每个样本独立处理。
    *   编码器注意力与解码器注意力的区别（掩码）。
    *   自注意力与交叉注意力的区别。
    *   缩放注意力（Scaled Attention）的重要性，以及为什么需要除以`sqrt(head_size)`。

**27. 代码单元格：缩放注意力演示（方差检查）**

```python
k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5
```

*   **功能**：此代码块演示了缩放注意力（Scaled Attention）中的缩放因子。它重新生成了随机的`k`和`q`，并计算了注意力分数`wei`，其中包含了`head_size**-0.5`的缩放因子。

**28. 代码单元格：打印`k`的方差**

```python
k.var()
```

*   **功能**：打印随机生成的`k`张量的方差。

**29. 代码单元格：打印`q`的方差**

```python
q.var()
```

*   **功能**：打印随机生成的`q`张量的方差。

**30. 代码单元格：打印`wei`的方差**

```python
wei.var()
```

*   **功能**：打印计算出的注意力分数`wei`的方差。通过与`k`和`q`的方差对比，可以观察到缩放因子有助于保持`wei`的方差在合理范围内，防止Softmax饱和。

**31. 代码单元格：Softmax饱和演示**

```python
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)
```

*   **功能**：此代码块演示了Softmax函数在输入值范围较小时的输出。

**32. 代码单元格：Softmax饱和演示（乘以8）**

```python
torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1) # gets too peaky, converges to one-hot
```

*   **功能**：此代码块演示了当Softmax的输入值被放大时（乘以8），输出会变得“尖锐”（peaky），即概率分布会趋向于one-hot编码。这说明了为什么在自注意力中需要进行缩放，以避免Softmax过早饱和。

**33. 代码单元格：自定义LayerNorm1d实现**

```python
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape
```

*   **功能**：此代码块实现了一个自定义的`LayerNorm1d`（层归一化）模块。
    *   `__init__`：初始化`gamma`（缩放参数）和`beta`（偏移参数），以及一个小的epsilon值用于数值稳定性。
    *   `__call__`：定义前向传播。它计算输入`x`的均值和方差，然后对`x`进行归一化，使其具有单位均值和方差，最后通过`gamma`和`beta`进行缩放和偏移。
    *   `parameters`：返回可学习的参数`gamma`和`beta`。
    *   实例化`LayerNorm1d`并用随机张量进行测试，打印输出形状。

**34. 代码单元格：检查LayerNorm1d输出的均值和标准差（按特征）**

```python
x[:,0].mean(), x[:,0].std() # mean,std of one feature across all batch inputs
```

*   **功能**：此代码块检查经过`LayerNorm1d`处理后，第一个特征在所有批次输入上的均值和标准差。理想情况下，均值应接近0，标准差应接近1。

**35. 代码单元格：检查LayerNorm1d输出的均值和标准差（按单个输入）**

```python
x[0,:].mean(), x[0,:].std() # mean,std of a single input from the batch, of its features
```

*   **功能**：此代码块检查经过`LayerNorm1d`处理后，批次中第一个输入的均值和标准差（在其所有特征上）。层归一化是在特征维度上进行归一化，因此每个样本的特征均值应接近0，标准差应接近1。

**36. Markdown 单元格：法英翻译示例**

```markdown
# French to English translation example:

# <--------- ENCODE ------------------><--------------- DECODE ----------------->
# les réseaux de neurones sont géniaux! <START> neural networks are awesome!<END>
```

*   **功能**：这是一个Markdown单元格，提供了一个法英翻译的示例，用于说明编码器-解码器架构中输入和输出的结构。

**37. Markdown 单元格：完整代码参考**

```markdown
### Full finished code, for reference

You may want to refer directly to the git repo instead though.
```

*   **功能**：这是一个标题，指示接下来的代码块是完整的、最终版本的GPT模型实现，并建议用户也可以直接参考GitHub仓库。

**38. 代码单元格：完整的GPT模型实现和训练**

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

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
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
```