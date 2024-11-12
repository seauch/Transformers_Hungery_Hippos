
<div align="center">
  <img src="./images/Hungry_Hippos_Readme_Header.png" alt="Hungry Hippos Header" width="500"/>
</div>

# Presentation: Hungry Hungry Hippos: Towards Language Modeling with State Space Models
By Sarah Auch

## Introduction

Welcome to an introduction on "Hungry Hungry Hippos: Towards Language Modeling with State Space Models." This research investigates whether the attention mechanisms in transformers, which are central to models like ChatGPT, Claude, and Co-Pilot, can be replaced by alternative computational methods. Specifically, the study examines the viability of using State Space Models (SSMs) as a substitute, aiming to maintain or even enhance transformers' performance in tasks that require long-range dependencies.

### "We are interested in seeing if we can start to replace attention with some other primatives that does not grow quadratically in the sequence length" - Daniel Y. Fu

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

<div align="center">
  <img src="./images/Attention.drawio.png" alt="Hungry Hippos Header" width="500"/>
</div>

### ... This is just one layer...

### Self-Attention Complexity Analysis

- **Sequence length**: \( n \) tokens
- **Self-attention complexity**: \( O(n^2) \)

If there are \( h \) attention heads:
- **Each head**: \( O(n^2) \)
- **Total per layer**: \( h * O(n^2) \)

### Across All Layers
- **Number of layers**: \( L \)
- **Total complexity**: \( L * h * O(n^2) \)

Memory requirements grow with \( L * h * n^2 \).


## What is the solution?

### State Space Models (SSMs)

State Space Models (SSMs) have proven effective for modeling sequences like audio data and time series, demonstrating their potential for sequential tasks. While SSMs offer attractive properties like linear scaling and infinite context during generation, previous SSM architectures faced two key challenges:


Pros:
During training: O(NlogN) in sequence length
During Generation: No need to process the whole input and no constraight to Context Length

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


Recurrent View:
For Generation:
Continuous-time SSM:
```
# Differential equations:
ẋ(t) = Ax(t) + Bu(t)    # State equation
y(t) = Cx(t) + Du(t)    # Output equation

Where:
- x(t) is state variable (dimension m)
- u(t) is input signal
- y(t) is output signal  
- t represents continuous time
```

Discrete-time SSM
```
# Difference equations:
xi = Axi-1 + Bui    # State update
yi = Cxi + Dui      # Output computation

Where:
- xi is state at step i
- ui is input at step i
- yi is output at step i
```

Basic Structure:
- Usually run d parallel SSMs (one per hidden dimension)
- Each SSM learns its own A, B, C, D matrices
- State variable x acts as memory, tracking sequence history

```
Convolutional View:
For Trainning:
# SSM as Convolution:
f = [CB, CAB, CA²B, ..., CA^(N-1)B]  # Create filter
y = f * u + Du  # Convolve filter with input

# Efficient Implementation:
- Use Fast Fourier Transforms (FFT)
- Complexity: O(N log N) vs O(N²)
- Steps:
  1. Take FFT of filter f
  2. Take FFT of input u
  3. Multiply results pointwise
  4. Take inverse FFT
```

### Addressing the Performance Gap with the H3 layer

State Space Models have traditionally failed to perform as well on language modeling tasks. The research examined two specific synthetic tasks designed to test certain capabilities:

1. **Ability to Remember Tokens After an Event**  
   - **Task**: "Induction Head"  
   - **Example**: Given a sequence where a special token ` appears, the model needs to recall what token came immediately after that special token when it appeared earlier in the sequence
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
  <img src="./images/Attention.drawio.png" alt="Hungry Hippos Header" width="500"/>
</div>





"The shift SSM can detect when a particular event occurs, and the diagonal SSM can remember a token afterwards for the rest of the sequence."




1. H3 layer
2. FlashConv

### Illistration H3 Layer


"The shift SSM can detect when a particular event occurs, and the diagonal SSM can remember a token afterwards for the rest of the sequence."

### Mechanisms for Associative Recall Task
In the Associative Recall task, the model's components function as follows:

- The **shift SSM** and **first multiplicative interaction** act as a gate to control when to pass a value to the diagonal SSM.
- The **diagonal SSM** stores the value in memory and continually outputs it.
- The **final multiplicative interaction** determines whether to pass the diagonal SSM's output based on the current input token.





## Evaluating Perplexity on OpenWebText




## Hybrid H3-Attention Language Models

Key takeaway: Hybrid H3-Attention Language Models found evidence that attention is not required at every layer to perform similarly to or even outperform traditional transformer models

<div align="center">
  <img src="./images/Hybrid.drawio.png" alt="Hungry Hippos Header">
</div>


## H3 Layer

### Illistration


### Shift SSM



```
Time step 1:  [A] -> [_] -> [_] -> [_]
Time step 2:  [B] -> [A] -> [_] -> [_]
Time step 3:  [C] -> [B] -> [A] -> [_]
Time step 4:  [D] -> [C] -> [B] -> [A]
```

```
Original sequence (K):
"Barack"    [1.0, 0.5]  # Original values
"and"    [0.8, 0.7]
"Michelle" [1.2, 0.3]
"Obama's"   [0.6, 0.9]

After Shift SSM (K̄):
"Barack"    [0.0, 0.0]  # No previous context
"and"    [0.8, 0.4]  # Influenced by "Barack"
"Michelle" [0.7, 0.6]  # Influenced by "Barack" and "and"
"Obama's"   [0.9, 0.3]  # Influenced by all previous words
```


### Multiplicative Interactions

### Diagonal SSM

### Multiplicative Interactions







## What does the childhood game Hungry Hungry Hippos have to do with replacing the attention mechanism with state space models?







## What does H3 have to do with the classic childhood game Hungry Hungry Hippos?
Just like those hungry hippos snapping up marbles in a nonstop stream, the H3 model is built to "devour" long sequences of text with impressive speed and efficiency. Imagine playing Hungry Hungry Hippos—you don’t need to carefully compare each marble to every other marble (which would be painfully slow!); the hippos just keep chomping away in a smooth, continuous motion.

Similarly, while traditional Transformers get bogged down by trying to compare every word to every other word in a sequence (picture having to pair up each marble one by one), H3 tackles text in a more streamlined, seamless way, just like our marble-munching hippos. And just as those hippos never seem to get full, H3 can handle much longer sequences of text without getting computationally “stuffed,” making it a truly hungry model for processing language!











Sources:

https://www.youtube.com/watch?v=TkOSKrlpnU4





