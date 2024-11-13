

<div align="center">
  <img src="./images/Hungry_Hippos_Readme_Header.png" alt="Hungry Hippos Header" width="500"/>
</div>

# Presentation: Hungry Hungry Hippos: Towards Language Modeling with State Space Models
By Sarah Auch

## Introduction

Welcome to an introduction on "Hungry Hungry Hippos: Towards Language Modeling with State Space Models." This research investigated whether the attention mechanisms in transformers, which are central to models like ChatGPT, Claude, and Co-Pilot, can be replaced by alternative computational methods. Specifically, the study examines the viability of using State Space Models (SSMs) as a substitute, aiming to maintain or even enhance transformers' performance in tasks that require long-range dependencies.

### "We are interested in seeing if we can start to replace attention with some other primitives that do not grow quadratically in the sequence length" - Daniel Y. Fu

This project addresses several computational bottlenecks presented by the attention mechanism. Transformers have revolutionized natural language processing, yet they face significant challenges, especially with high computational and memory costs that grow quadratically with sequence length. Each additional token in a sequence demands increased processing power and memory, slowing performance and escalating operational expenses. Transformers also struggle with long contexts, often forcing truncation and risking the loss of crucial information. Additionally, the high energy demands of transformers raise concerns about environmental and accessibility implications.

The H3 project proposes SSMs as an alternative to attention. Unlike attention mechanisms, SSMs are well-suited for efficiently handling continuous sequential data, especially in fields like time series analysis and control systems, without the same computational overhead. By implementing SSMs, the H3 project seeks to preserve the language modeling capabilities of transformers while significantly reducing computational costs.

## Motivation

The aim to expand the context length of our model. However, increasing it twofold—for example, from 32k to 64k tokens—is not simply twice as expensive; it’s actually four times as costly.

### Why? 
In a transformer model, each token in the context window doesn’t operate independently—it "pays attention" to all other tokens to understand relationships and context. This means each token must connect with every other token in the window, creating a dense network of interactions that grows with the number of tokens. Doubling the number of tokens doesn’t just double these interactions; it quadruples them. For instance, increasing the tokens from 4 to 8 results in interactions jumping from 16 to 64, thereby quadrupling the processing cost. This exponential increase in interactions is what drives up computational costs significantly as we expand context length.

### Illistration of the Quadratic Complexity Problem:

<div align="center">
  <img src="./images/arch.drawio.png" alt="Hungry Hippos Header" width="500"/>
</div>

Figure 1: Transformer Langage Models from #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models

<div align="center">
  <img src="./images/Attention.png" alt="Hungry Hippos Header" width="500"/>
</div>
Figure 2: Attention: Quadratic Context Bottleneck from #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models


Take a second to think about what doubling n would look like in this image

## What is the solution?

### State Space Models (SSMs)

State Space Models (SSMs) have proven effective for modeling sequences like audio data and time series, demonstrating their potential for sequential tasks. While SSMs offer attractive properties like linear scaling and infinite context during generation, previous SSM architectures faced two key challenges:

Pros:
- During training: SSMs scale with 
O(NlogN) in sequence length, instead of O(N^2) like attention – that makes them promising for long sequence modeling.
- During Generation: There’s no fixed context window, since SSMs admit a completely recurrent view

Cons:
Performance Gap:
- Underperform compared to Transformers on language tasks
- Example: Several perplexity points worse than attention on language modeling
- Existing SSMs (like S4D, GSS) struggled with token comparison and recall

Speed Issues:
- Despite theoretical efficiency, slower than Transformers in practice
- Poor hardware utilization on modern GPUs
- Particularly inefficient for shorter sequences
- FFT operations don't leverage specialized hardware (like tensor cores)

### Addressing the Performance Gap with the H3 layer

State Space Models have traditionally failed to perform as well on language modeling tasks. The research examined two specific synthetic tasks designed to test certain capabilities:

1. **Ability to Remember Tokens After an Event**  
   - **Task**: "Induction Head"  
   - **Example**: Given a sequence where a special token appears, the model needs to recall what token came immediately after that special token when it appeared earlier in the sequence
   - **Purpose**: This tests the model's ability to "log" or remember specific tokens based on their relationship to a trigger event.
  
2. **Ability to Compare Tokens Across a Sequence**  
   - **Task**: "Associative Recall"  
   - **Example**: Given a sequence of key-value pairs (e.g., "a 2 c 4 b 3 d 1") and then a key, the model must recall the corresponding value.  
   - **Purpose**: This tests the model's ability to:
     - Compare tokens (to find matching keys)
     - Remember associations between token pairs
     - Retrieve the correct value when the key is seen again

### How Attention Mechanisms Naturally Address These Capabilities
Attention has inherent mechanisms to manage both tasks:

- **Token Comparison**: Achieved through the quadratic attention matrix \( QK^T \)
- **Direct Recall**: Accomplished by multiplying \( softmax(QK^T) \) with \( V \)

### Performance of Traditional SSMs
Traditional SSMs struggled on these tasks:
- S4D achieved **35.6% on Induction Head** and **86% on Associative Recall**.

## Introducing the H3 Model
The H3 model was designed to address these challenges, achieving:
- **100% on Induction Head**
- **99.8% on Associative Recall**

### H3 Layer

<div align="center">
  <img src="./images/H3..png" alt="Hungry Hippos Header" width="500"/>
</div>
Figure 3: Hunger Hunger Hippos:Design for Associative Recall from #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models


### Shifft SSM

Acts like a sliding window over recent tokens, providing short-term, local memory.

#### Properties
- **Uses shift matrix A**: Moves elements down by 1
- **Example**: `[a,b,c] -> [0,a,b]`
- **Creates local "memory"**: Retains a record of recent tokens

#### Purpose
- Looks at **local context**
- Tracks **recently appeared tokens**
- Functions as **short-term memory**

<div align="center">
  <img src="./images/Shift.png" alt="Hungry Hippos Header">
</div>

Figure 4: The shift Remembers Prevous Token from #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models

### Diagonal SSM

Acts as persistent memory across the entire sequence, providing long-term, global memory.

#### Properties
- **Uses diagonal matrix A**: Maintains information over long distances
- **Can remember tokens**: From the beginning of the sequence

#### Purpose
- Tracks **global context**
- Provides **long-term memory storage**
- Retains information across the **entire sequence length**

### Combining Both Types of Memory

1. **Shift SSM (Local)**:
   - "I just saw token X"
   - Short-term pattern detection

2. **Diagonal SSM (Global)**:
   - "I remember seeing X earlier"
   - Long-term information storage

#### This Combination Enables
- **Short-term token tracking**
- **Long-term memory retention**
- **Both local and global pattern recognition**


### Multiplicative Interactions Between SSMs (K⊙V)

#### 1. First Multiplicative Interaction: SSMshift(K) ⊙ V
- **Purpose**: Gates which values get stored in memory
  - Functions like saying "when I see key K, store value V"
  - Enables local pattern detection and value storage

- **Example for "a 2 b 3"**:
  - **SSMshift(K)**: Tracks when we see 'a'
  - **Multiplication with V**: Stores '2' when 'a' appears

#### 2. After Diagonal SSM (Q⊙)
- **Purpose**: Compares current token with stored memory
  - Functions like asking "does the current token match what's in memory?"
  - Controls when to output stored values

- **Example for "a 2 b 3 a ?"**:
  - **Q**: Represents the current token 'a'
  - **Multiplication**: Checks if 'a' matches stored keys
  - If a match is found, outputs the corresponding stored value '2'

### Together in H3

#### Complete Flow:
1. **SSMshift(K) ⊙ V** - Stores values when keys appear
2. **SSMdiag(...)** - Maintains stored values in memory
3. **Q ⊙ [output]** - Retrieves values when matching keys appear

#### Final Output:
- `Final output = Q ⊙ SSMdiag(SSMshift(K) ⊙ V)`


### A simple example in action

<div align="center">
  <img src="./images/H3-example.png" alt="Hungry Hippos Header">
</div>
<div align="center">
  <img src="./images/example_full.drawio.png" alt="Hungry Hippos Header">
</div>

Figure 4: How H3 Can Solve Associative Recall from #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models



### Pseudocode

```
# Input: x ∈ ℝ^{d×N} (sequence length N, dimension d)
Q = WQ @ x  # Query projection: ℝ^{d×d} @ ℝ^{d×N} -> ℝ^{d×N}
K = WK @ x  # Key projection
V = WV @ x  # Value projection

#Query (Q): Represents what we're looking for
#Key (K): Represents what we're matching against
#Value (V): Represents the information to be retrieved

# Split into H heads of dimension dh = d/H
def split_heads(x, H):
    # x: ℝ^{d×N} -> H × ℝ^{dh×N}
    return x.reshape(H, -1, N)

# Split into H heads of dimension dh = d/H
def split_heads(x, H):
    # x: ℝ^{d×N} -> H × ℝ^{dh×N}
    return x.reshape(H, -1, N)

Q_heads = split_heads(Q, H)
K_heads = split_heads(K, H)
V_heads = split_heads(V, H)

#Multi-head processing enables
#1. Parallel processing of different features
#2. Multiple representation subspaces
#3. Better modeling of different types of relationships


def shift_ssm(K, A_shift, B_shift, C_shift):
    """
    Applies shift SSM to capture sequential relationships
    Input: K ∈ ℝ^{dh×N}
    Output: K' ∈ ℝ^{dh×N}

    A_shift: Shifts state vector elements down by one position
    B_shift: Projects input into state space
    C_shift: Projects state back to output space
    """
    x_t = 0  # Initial state
    outputs = []
    for t in range(N):
        x_t = A_shift @ x_t + B_shift @ K[:, t]
        y_t = C_shift @ x_t
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)

# For each head h:
K_shifted = shift_ssm(K_heads[h], A_shift, B_shift, C_shift)
# Matrix multiplication for similarity computation
S = K_shifted @ V_heads[h].transpose(-2, -1)  # ℝ^{dh×N} @ ℝ^{N×dh} -> ℝ^{dh×dh}

# For each head h:
K_shifted = shift_ssm(K_heads[h], A_shift, B_shift, C_shift)
# Matrix multiplication for similarity computation
S = K_shifted @ V_heads[h].transpose(-2, -1)  # ℝ^{dh×N} @ ℝ^{N×dh} -> ℝ^{dh×dh}

# Combine heads
O_combined = torch.cat([O_h for O_h in O_heads], dim=0)  # H×ℝ^{dh×N} -> ℝ^{d×N}
# Final projection
y = WO @ O_combined  # ℝ^{d×d} @ ℝ^{d×N} -> ℝ^{d×N}



```



## Evaluating 

Evaluation of 2-layer models on synthetic language tasks.

| Task               | Random | S4D  | Gated State Spaces | H3   | Attention |
|--------------------|--------|------|---------------------|------|-----------|
| Induction Head    | 5.0    | 35.6 | 6.8                 | 100.0| 100.0     |
| Associative Recall| 25.0   | 86.0 | 78.0                | 99.8 | 100.0     |

Tabel 1: From Hungry Hungry Hippos: Towards Language Modeling with State Space Models page 4

### Perplexity of SSM Variants Compared to Transformers on OpenWebText

All models have 12 layers, with a size around 125M, and are trained with the same hyperparameters for 50B tokens.

| Model                | Perplexity |
|----------------------|------------|
| H3                   | 21.0       |
| H3 Hybrid (2 Attn)   | 19.6       |
| S4D                  | 24.9       |
| GSS                  | 24.0       |
| GSS Hybrid (2 Attn)  | 19.8       |
| Transformer          | 20.6       |
Tabel 2: From Hungry Hungry Hippos: Towards Language Modeling with State Space Models page 6


### Hybrid H3-Attention Language Models

Key takeaway: Hybrid H3-Attention Language Models found evidence that attention is not required at every layer to perform similarly to or even outperform traditional transformer models

<div align="center">
  <img src="./images/Hybrid.drawio.png" alt="Hungry Hippos Header">
</div>
Figure 5: Hybrid H3-Attention Layers


## What about the Speed Issues?
The core challenge is that while SSMs theoretically scale better than attention (linear vs quadratic), naive implementations are still slow due to poor hardware utilization. FlashConv addresses this through two main components:

For "shorter" sequences (up to 8K):

- Uses fused block FFT that combines operations into a single kernel to minimize memory transfers
- Breaks down FFT computation into specialized matrix multiplications that can leverage hardware accelerators like tensor cores
- Trades slightly more FLOPs for much better hardware efficiency

For longer sequences (>8K):


- Novel state-passing algorithm that processes sequence in chunks
- Maintains recurrent state between chunks to preserve sequence continuity
- Enables processing arbitrarily long sequences while keeping linear complexity

### Evalution: Comparing H3 Hybrid Model with Transformer
- 2x speedup on Long Range Arena
- 4-8x faster training for long sequences
- 2.4x faster text generation
- Maintains near-linear scaling



## What does a H3 model have to do with the classic childhood game Hungry Hungry Hippos?
Just like those hungry hippos snapping up marbles in a nonstop stream, the H3 model is built to "devour" long sequences of text with impressive speed and efficiency. Imagine playing Hungry Hungry Hippos—you don’t need to carefully compare each marble to every other marble (which would be painfully slow!); the hippos just keep chomping away in a smooth, continuous motion.

Similarly, while traditional Transformers get bogged down by trying to compare every word to every other word in a sequence (picture having to pair up each marble one by one), H3 tackles text in a more streamlined, seamless way, just like our marble-munching hippos. And just as those hippos never seem to get full, H3 can handle much longer sequences of text without getting computationally “stuffed,” making it a truly hungry model for processing language!



## Analysis

While models up to 2.7B parameters were tested, a deeper analysis of scaling behavior would be valuable, such as examining how pure H3 models vs. hybrid models scale with size.

## Impact

Mamba extends the foundational concepts of H3 by incorporating input-dependent dynamics and hardware optimizations, resulting in a more versatile and efficient sequence modeling architecture.

<div align="center">
  <img src="./images/Mamba.png" alt="Hungry Hippos Header">
</div>

Figure 5: Mamba use of H3 Concepts from Mamba: Linear-Time Sequence Modeling with Selective State Spaces

"Mamba is the first attention-free model to match the performance of a very strong Transformer recipe (Transformer++) that has now become standard, particularlyas the sequence length grows" - Page 11 of Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Sources:

Fu, D. Y. (2023, May 1). #4 - Hungry Hungry Hippos: Towards Language Modeling with State Space Models. YouTube. https://www.youtube.com/watch?v=TkOSKrlpnU4 

Fu, D. Y., Dao, T., Saabz, K. K., Thomas, A. W., Rudra, A., & Re, C. (2023, April 29). Hungry Hungry Hippos: Towards Language Modeling with State Space Models. https://arxiv.org/pdf/2212.14052 

Gu, A., & Dao, T. (2024, May 31). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/pdf/2312.00752 






