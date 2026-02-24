# Transformer解码器工作流程详解

## 1. 解码器的输入：Shifted Right序列

在训练阶段，解码器的输入是目标语言序列的**右移版本**（shifted right）。例如，对于目标序列 `[i, like, eating]`，右移后的输入为 `[<s>, i, like]`，其中 `<s>` 是起始标记。

## 2. 解码器完整工作流程

### 2.1 输入处理：嵌入和位置编码

```python
# 解码器输入嵌入和位置编码
self.target_embberding = nn.Sequential(
    Emberddings(tag_vocab, d_model), PositionnalEncoding(d_model)
)

# 在decoder方法中调用
x = self.target_embberding(target)  # target是shifted right序列
```

1. **词嵌入**：将输入的token（`<s>`, `i`, `like`）转换为高维向量
2. **位置编码**：为每个token添加位置信息，确保模型理解序列顺序

### 2.2 自注意力层（带掩码）

```python
# 解码器层中的自注意力
out1 = self.sublayer[0](x, lambda y1: self.self_attn(x, x, x, tag_mask))
```

自注意力层允许解码器关注生成序列中之前的token，但使用**下三角掩码**（tag_mask）防止关注未来的token：
- 对于第一个token `<s>`：只能关注自身
- 对于第二个token `i`：可以关注 `<s>` 和自身
- 对于第三个token `like`：可以关注 `<s>`, `i` 和自身

这种掩码机制确保了自回归生成特性（只能基于已生成的内容预测下一个token）。

### 2.3 交叉注意力层

```python
# 解码器层中的交叉注意力
out2 = self.sublayer[1](out1, lambda y2: self.cross_attn(out1, memory, memory))
```

交叉注意力层允许解码器关注编码器的输出（memory）：
- **Q**（查询）来自解码器自注意力层的输出
- **K**（键）和 **V**（值）来自编码器的输出
- 这一步使解码器能够访问源语言的完整上下文信息

### 2.4 前馈神经网络

```python
# 解码器层中的前馈神经网络
out3 = self.sublayer[2](out2, lambda y3: self.feed_forward(out2))
```

前馈神经网络对每个位置的表示进行独立的非线性变换，进一步处理特征。

### 2.5 输出层

```python
# Transformer类中的输出层
self.out = nn.Linear(d_model, tag_vocab)

# 在forward方法中调用
return self.out(out)  # out是解码器的最终输出
```

输出层将解码器的输出转换为目标语言词汇表的概率分布：
- 使用线性变换将d_model维度映射到tag_vocab大小
- 通常后接softmax函数获取概率分布

## 3. 训练阶段的完整流程

假设我们有：
- 源序列：`[我, 喜欢, 吃饭]`
- 目标序列：`[i, like, eating]`
- 解码器输入（shifted right）：`[<s>, i, like]`

### 3.1 编码器处理源序列

```python
# 编码器处理源序列
memory = self.encoder(src, src_mask)
```

1. 源序列经过词嵌入和位置编码
2. 6层编码器依次处理，生成上下文表示（memory）

### 3.2 解码器处理shifted right输入

```python
# 解码器处理shifted right输入
out = self.decoder(target, memory, src_mask, target_mask)
```

1. shifted right序列 `[<s>, i, like]` 经过嵌入和位置编码
2. 6层解码器依次处理：
   - 每层包含自注意力（带掩码）、交叉注意力、前馈神经网络
   - 最终生成解码器输出表示

### 3.3 输出层生成预测

```python
# 输出层生成词汇表概率分布
logits = self.out(out)
```

对于输入 `[<s>, i, like]`，解码器输出3个位置的预测：
- 位置1：预测 `<s>` 之后的词，期望输出 `i`
- 位置2：预测 `i` 之后的词，期望输出 `like`
- 位置3：预测 `like` 之后的词，期望输出 `eating`

### 3.4 计算损失并优化

使用交叉熵损失函数比较预测结果与真实目标序列 `[i, like, eating]`，然后通过反向传播调整模型参数。

## 4. 推理阶段的工作流程

在推理阶段，解码器逐词生成输出：

1. **初始输入**：仅包含起始标记 `[<s>]`
2. **第一次预测**：解码器处理 `[<s>]`，预测下一个词为 `i`
3. **第二次输入**：`[<s>, i]`，预测下一个词为 `like`
4. **第三次输入**：`[<s>, i, like]`，预测下一个词为 `eating`
5. **第四次输入**：`[<s>, i, like, eating]`，预测下一个词为 `<e>`（结束标记）
6. **终止条件**：当预测到结束标记或达到最大长度时停止

## 5. 代码中的完整调用链

```python
# Transformer的forward方法
def forward(self, src, target, src_mask=None, target_mask=None):
    memory = self.encoder(src, src_mask)  # 编码器生成上下文表示
    out = self.decoder(target, memory, src_mask, target_mask)  # 解码器处理输入
    return self.out(out)  # 输出层生成预测
```

## 总结

Transformer解码器从shifted right输入开始，通过以下步骤工作：
1. **输入处理**：嵌入和位置编码
2. **自注意力**：带掩码的自注意力 + 残差连接 + 层归一化
   - 带掩码的自注意力，只能关注已生成的token
   - 层归一化：对自注意力层的输入进行归一化，稳定训练
   - 残差连接：将原始输入与自注意力层输出相加，解决梯度消失问题
3. **交叉注意力**：关注编码器的上下文表示 + 残差连接 + 层归一化
   - 关注编码器的上下文表示
   - 层归一化：对交叉注意力层的输入进行归一化
   - 残差连接：将自注意力层输出与交叉注意力层输出相加
4. **前馈网络**：进一步处理特征 + 残差连接 + 层归一化
   - 进一步处理特征
   - 层归一化：对前馈神经网络的输入进行归一化
   - 残差连接：将交叉注意力层输出与前馈神经网络输出相加
5. **输出层**：生成词汇表概率分布

这种设计确保了解码器能够生成连贯、准确的目标语言序列，同时保持自回归特性。