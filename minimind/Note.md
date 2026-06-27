项目来源https://github.com/jingyaogong/minimind



# 模型推理

在自己体验训练乐趣之前，可以先体验一下这个项目作者提供的训练好的模型效果。



推理引擎和框架作用：极限加速与显存管理，高并发处理，硬件适配与压缩降级，服务化与接口封装

主流的有：`vLLM`、`Ollama`、`llama.cpp` 

该项目中的 `eval_llm.py` 可以直接用，支持使用 Transformers 格式模型和基于 PyTorch 模型权重格式

此外作者使用 Streamlit 编写了 `web_demo.py`，可以支持WebUI展示



分词器：**是一个“处理工具”或“程序模块”。**

它的核心任务有两个：

- **训练（Train）**：在海量文本上统计，生成一个“词表”（Vocab）和“合并规则”（合并规则主要针对BPE等子词算法）。
- **推理（分词/编码）**：将输入的自然语言句子，按照词表和规则切分成一个**Token序列**。

**必须提前训练，且通常是在大规模通用语料上进行的。**

- 在训练LLM之前，工程师会先单独运行分词器的训练算法（如BPE、WordPiece或SentencePiece）。
- 它会在海量文本中统计字符、子词出现的频率，最终生成那个我们上一轮提到的“词表”和“合并规则”。
- **注意**：这个训练过程与后续的LLM训练是**完全独立**的，它不需要用到深度学习显卡（GPU），只在CPU上跑就行。而且大多数情况，这个分词器也不是一个神经网络模型，而是一些其他算法构建的模型



随机种子：让回答多样性

输入内容的拼装

生成时的KV Cache

```python
def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
    input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
    attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
    past_key_values = kwargs.pop("past_key_values", None)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    if streamer: streamer.put(input_ids.cpu())
    for _ in range(max_new_tokens):
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
        logits = outputs.logits[:, -1, :] / temperature
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        if top_k > 0: 
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        past_key_values = outputs.past_key_values if use_cache else None
        if streamer: streamer.put(next_token.cpu())
        if eos_token_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all(): break
    if streamer: streamer.end()
    if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
    return input_ids
```





随机权重的模型尝试

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    # 强制初始化未训练的模型结构，去掉了所有 if...else 判断和权重加载逻辑
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer
```



















# 模型训练



## 预训练

预训练一般训练1-2个epoch



### 参数意义

`--hidden_size` (默认: 768)：这个参数一般就是指Transformer中的d_model这个参数，决定了模型单次能处理和记忆的信息量。数值越大，模型表征能力越强，但也意味着参数量呈平方级增长，显存消耗和计算时间都会大幅增加。

`--num_hidden_layers` (默认: 8)：Transformer块数，minimind用的是解码器结构，因此就是解码器的transformer块数，决定了模型的逻辑推理和抽象能力。网络越深，模型越能理解复杂的上下文关系；但层数过多容易导致梯度消失或爆炸，且推理延迟会变高。

`--max_seq_len` (默认: 340)：Transformer的注意力机制计算量随序列长度呈**平方级增长**（$O(N^2)$）。340 意味着模型只能处理短文本对话，如果强行输入更长的文本会被截断。那么训练和推理时输入输出也就在这个限度内表现比较好。

`--use_moe` (默认: 0)：MOE架构，模型可以在不显著增加单次推理计算量的情况下，大幅增加总参数量。这是一种“用显存换智商”的高阶技巧



`--batch_size` (默认: 24)

`--accumulation_steps` (默认: 16)

--learning_rate

训练中batch_size*accumulation_steps如果变大，一般会缩小learning_rate，保持最终训出来的模型差不多

--epochs



`--grad_clip` (默认: 1.0)：当某一步计算出的梯度过大时，强制按比例将其缩放到 1.0，从而**有效防止梯度爆炸**，避免模型训练突然崩溃

BF16相比传统的 FP32，它能**节省一半显存并加速计算**；相比 FP16，它的数值范围更大，不容易发生上溢出（Overflow），非常适合对数值稳定性要求极高的LLM。

`--use_compile` (默认: 0)：开启后，PyTorch 会在底层融合计算图（类似于给代码做预编译）。能在不改变数学等价性的前提下，**免费获得 10%~30% 的训练加速**。代价是每次刚开始训练时需要花几分钟时间编译。





### 训练技术细节



#### 爆显存问题

在小机器上，显存不够，只能降低batch_size和`max_seq_len`。

##### 降低batch_size

单纯减小 `batch_size` 会导致梯度更新过于频繁和剧烈，模型难以收敛。因此，必须配合梯度累积（Gradient Accumulation）技术：让模型在前向传播几次后再统一更新一次权重。

- 原脚本默认：`batch_size=32`, `accumulation_steps=8` （等效全局 batch 为 256）
- **4060 修改方案**：`batch_size=8`, `accumulation_steps=32` （等效全局 batch 依然是 256，但显存占用瞬间暴降！）

但是能用大batch_size尽量用大的，原因是**大显存带来的大 Batch Size，其最特别的优势在于**极其恐怖的并行计算效率和访存收益。

##### 降低 `max_seq_len`

Transformer 架构中，Self-Attention 机制的计算复杂度（和显存占用）是随着序列长度呈二次方（平方）增长的。如果把句子截断得稍微短一点，能挤出大量显存。

- 在 `minimind-3` 的 README 中，针对 mini 数据集推荐的长度是 768，但这对于 8GB 显存可能有点勉强。
- **你的 4060 修改方案**：将其降至 `340` 或 `256`。

但是`max_seq_len`（最大序列长度）本质上定义了模型在一次前向传播中能拥有的“最大工作记忆窗口”。将它调小虽然能立竿见影地拯救显存，但不可避免地会在模型能力上付出代价。





#### 多卡分布式训练

同步更新与分布式训练



1. `batch_size` 参数是“对单张卡”而言的，因此多卡时会导致算的总有效batch_size是单卡的N倍，这个N是卡数
1. 因此也进而导致多卡并行时和单卡的更新次数不一致，因为 4 张卡每次更新的步数少了（只有 25 次），但每次更新参考的数据变多了（400条），算出来的梯度方向会比单卡（100条）**更加准确、更少噪音**。 所以在实际炼丹时，当我们增加显卡数量，通常也会**适当把学习率（Learning Rate）调大一点**，让模型在每一次更新时“步子迈得更大一些”，以此来弥补总更新次数变少带来的影响。





#### Wandb的监控方法









#### 分词器

利用了 HuggingFace 的 `AutoTokenizer` 类，从本地路径（默认 `../model`）加载已经训练好的词表和分词规则

**Hugging Face 并不是 PyTorch 的一部分。** 它们是两个完全独立、但又在深度学习生态中紧密合作的实体。

Hugging Face 的工具库（特别是 `transformers` 库）是**建立在 PyTorch 之上**的。

- 在我们之前分析的代码中，你可以看到 `AutoTokenizer` 是从 Hugging Face 的 `transformers` 库里导出的，它负责把文本变成数字。
- 但这个模型最终的参数矩阵、前向传播、反向求导、多卡分布式训练（DDP），完全是依赖 **PyTorch** 来完成的。
- （注：Hugging Face 其实也支持 TensorFlow 和 JAX，只是目前在学术界和开源大模型界，PyTorch + Hugging Face 的组合占据了绝对的主流。）



#### 精度数值缩放

在大模型训练中，为了省显存和加速，通常不会用 32 位浮点数（float32），而是用半精度（float16 或 bfloat16）。

但 `float16` 有个致命缺点：**数值表示范围较小**。在反向传播时，计算出的梯度往往非常非常小（比如 0.0000001），很容易超出 float16 的表示下限，变成 0，这叫“梯度下溢”（Underflow）。

`GradScaler` 就是为了解决这个问题。当开启 `float16` 时，它会在反向传播前先把 Loss 乘以一个极大的缩放因子（把它放大），算出梯度后，再在更新参数前把梯度除以相同的因子缩放回来。这样就完美避免了下溢问题。

如果使用的是 `bfloat16`（数值范围比 float16 大，不易下溢），这个功能就会被 `enabled=False` 关掉。





#### 优化器选择

目前，**AdamW** 是LLM训练中应用最广泛、也最受认可的首选优化器。这背后是其在性能、稳定性和工程适用性上的综合优势。

AdamW是Adam的改进版，它将**权重衰减（Weight Decay）** 与梯度更新解耦，有效提升了模型的泛化能力。它之所以成为主流，主要因为它解决了LLM训练中的几个核心痛点：

- **强大的自适应学习率**：AdamW为每个参数维护独立的自适应学习率。这对于参数规模巨大、各层梯度差异显著的Transformer模型至关重要，能有效应对损失函数中崎岖不平的地形。
- **超参数鲁棒性高**：相比对学习率极其敏感的SGD，AdamW在较宽的超参数范围内都能稳定训练，这对于需要花费巨资进行调参的大模型来说非常宝贵。
- **理论和实践验证**：大量研究和实践（如GPT、LLaMA等）都证明了AdamW的可靠性和有效性。一项研究表明，即使像Lion、Sophia这样的新优化器，其最优性能也只能做到与调优后的AdamW相近，或仅在特定场景下略有优势



#### 模型架构层面细节

##### 注意力机制优化：MQA 与 GQA（分组查询注意力）

**背景**：标准的 Transformer 使用多头注意力（MHA），每个查询头（Q）都有自己独立的键（K）和值（V）头。这在推理（生成回答）时会导致显存被历史状态（KV Cache）瞬间撑爆。

**技术细节**：为了节省显存，诞生了 MQA（所有 Q 共享 1 个 KV）和 GQA（几个 Q 共享 1 个 KV）。

**面试考点**：

- *问：为什么训练时不觉得 KV Cache 占显存，推理时却很占？* （答：训练是并行计算的，不需要缓存历史 Token；推理是自回归的，一个个吐字，必须把前面的 KV 存下来避免重复计算）。
- *代码印证*：在 MiniMind 的 config 中，`num_attention_heads=8` 而 `num_key_value_heads=4`，这就是标准的 **GQA（分组查询注意力）** 配置。

##### 位置编码：RoPE（旋转位置编码）

- **背景**：模型需要知道词语的前后顺序。绝对位置编码（如正弦波或可学习参数）泛化能力差。
- **技术细节**：RoPE 通过矩阵旋转的方式，在绝对位置的乘法计算中优雅地融入了**相对位置信息**。
- **面试考点**：
  - *问：RoPE 相比于绝对位置编码有什么优势？* （答：更好地捕捉相对距离，且具备一定的**长度外推能力**，即训练时只见过 2K 长度，推理时有希望泛化到 4K 甚至更长）。
  - *问：如何解决超长文本的外推问题？* （答：YaRN、PI 等动态缩放 RoPE 角度的算法，MiniMind 代码中包含的 `rope_scaling` 就是做这个的）。



#####  归一化：RMSNorm 与 Pre-Norm

- **背景**：为了防止梯度消失/爆炸，需要对每层的数据做归一化。
- **面试考点**：
  - *问：为什么现在的大模型（如 LLaMA, MiniMind）用 RMSNorm 替代了原来的 LayerNorm？* （答：LayerNorm 需要计算均值和方差，把数据减去均值。RMSNorm 发现“减去均值”这一步对模型表现影响不大，直接去掉这一步（只除以均方根），能减少计算开销，提升约 10% 的训练速度）。
  - *问：Pre-Norm 和 Post-Norm 的区别？* （答：现在的 LLM 都用 Pre-Norm，即在 Attention 和 MLP *之前*做归一化，这能让残差连接直接贯通到底，极大地提升了深层网络的训练稳定性）。

此外区分：深度学习中除了**层归一化（Layer Normalization, LN）\**之外，最经典、最常被拿来和它对比的通常是\**批归一化（Batch Normalization, BN）**。



#### 训练稳定性与优化策略

##### 1. 学习率策略：Warmup（预热） + 余弦退火（Cosine Annealing）

- **技术细节**：MiniMind 代码中使用了 `get_lr` 实现了余弦退火。但在很多大型预训练中，还需要加上 **Warmup**。
- **面试考点**：
  - *问：为什么要用 Warmup（让学习率从 0 慢慢涨到最大值）？*
  - 答：模型刚初始化时，权重是随机的，如果一开始就给很大的学习率，会导致梯度过大，瞬间把某些神经元“击穿”（变成死神经元）或者导致 Loss 变成 NaN（梯度爆炸）。Warmup 让模型在最初的几千步先“试探”性地走，等权重分布稍微稳定后，再以最大学习率全速学习。



##### 2. 梯度裁剪（Gradient Clipping）

- **技术细节**：代码中有一句 `torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)`。
- **面试考点**：
  - 这是防止梯度爆炸的最后一道防线。当计算出的全局梯度范数（L2-norm）超过设定的阈值（如 1.0）时，强制将梯度按比例缩小。这在长文本训练和混合精度训练中极其重要。



##### 3. 权重衰减（Adam vs AdamW）

- **面试考点**：
  - *问：AdamW 的 'W' 代表什么？为什么不用 Adam？*
  - 答：W 代表 Weight Decay（权重衰减，即 L2 正则化）。在标准的 Adam 中，L2 正则化项被混在了自适应学习率的计算里，导致正则化效果大打折扣。AdamW 将权重衰减**解耦**出来，直接在更新参数时减去一点点自身的值，这大大提升了模型的泛化能力，避免过拟合。

####  数据处理层面的细节



##### 1. Loss 计算的屏蔽（Ignore Index）

- **技术细节**：在组装 Batch 时，短句子会被 Pad 填补充齐（比如用 0 填充）。
- **面试考点**：
  - *问：模型在计算 Loss 的时候，会不会把预测 Padding 的误差也算进去？*
  - 答：**绝对不能**。在 PyTorch 中，必须把 Padding 位置对应的 `labels` 设置为 **-100**（MiniMind 的 `PretrainDataset` 里正是这么做的）。PyTorch 的 `CrossEntropyLoss` 默认带有 `ignore_index=-100` 的属性，遇到 -100 的位置会直接忽略，不产生梯度。

##### 2. Sequence Packing（序列拼接与打包）

- **背景**：如果我们有大量几十个字的短句，即使 Padding 到 1024 长度去训练，显卡大部分算力都在算无意义的 Pad 符号，极其浪费。
- **面试考点（进阶）**：
  - *问：如何提高预训练的数据吞吐率？*
  - 答：使用 **Packing** 技术。把多个短句子用 `<EOS>`（结束符）首尾相连，拼成一个正好长度为 1024 的长序列丢给模型。
  - *追问：拼在一起后，前一个句子的注意力会不会跑到后一个句子上去（交叉污染）？*
  - 答：标准的预训练通常允许这种跨文档注意力（模型能自己学会 `<EOS>` 代表重新开始）；如果在指令微调阶段，为了严格隔离，会修改 Attention Mask，做成**块对角矩阵（Block Diagonal Mask）**，让不同句子之间互相看不见。























