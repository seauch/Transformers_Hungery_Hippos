
<div align="center">
  <img src="./images/Hungry_Hippos_Readme_Header.png" alt="Hungry Hippos Header" width="500"/>
</div>


# Presentation: Hungry Hungry Hippos: Towards Language Modeling with State Space Models
By Sarah Auch

## Introduction

Welcome to an introduction on "Hungry Hungry Hippos: Towards Language Modeling with State Space Models." This research investigates whether the attention mechanisms in transformers, which are central to models like ChatGPT, Claude, and Co-Pilot, can be replaced by alternative computational methods. Specifically, the study examines the viability of using State Space Models (SSMs) as a substitute, aiming to maintain or even enhance transformers' performance in tasks that require long-range dependencies.

### "We are interested in seeing if we can start to replace attention with some other primatives that does not grow quadratically in the sequence length" - Daniel Y. Fu

This project, referred to as H3, addresses several computational bottlenecks presented by the attention mechanism. Transformers have revolutionized natural language processing, yet they face significant challenges, especially with high computational and memory costs that grow quadratically with sequence length. Each additional token in a sequence demands increased processing power and memory, slowing performance and escalating operational expenses. Transformers also struggle with long contexts, often forcing truncation and risking the loss of crucial information. Additionally, the high energy demands of transformers raise concerns about environmental and accessibility implications.

The H3 project proposes SSMs as an alternative to attention. Unlike attention mechanisms, SSMs are well-suited for efficiently handling continuous sequential data, especially in fields like time series analysis and control systems, without the same computational overhead. By implementing SSMs, the H3 project seeks to preserve the language modeling capabilities of transformers while significantly reducing computational costs.

## Motivation

We aim to expand the context length of our model. However, increasing it twofold—for example, from 32k to 64k tokens—is not simply twice as expensive; it’s actually four times as costly

### Why? 
In a transformer model, each token in the context window doesn’t operate independently—it "pays attention" to all other tokens to understand relationships and context. This means each token must connect with every other token in the window, creating a dense network of interactions that grows with the number of tokens. Doubling the number of tokens doesn’t just double these interactions; it quadruples them. For instance, increasing the tokens from 4 to 8 results in interactions jumping from 16 to 64, thereby quadrupling the processing cost. This exponential increase in interactions is what drives up computational costs significantly as we expand context length.

<div align="center">
  <img src="./images/Attention.png" alt="Hungry Hippos Header" width="500"/>
</div>


## What did they do?

The H3 layers, designed with state space models (SSMs) in place of self-attention, address several challenges inherent to Transformer architectures:

Scalability: Unlike traditional attention mechanisms, which scale quadratically, H3 layers scale linearly with context length, making them more efficient for processing longer sequences.

Flexible Context Handling: H3 layers operate without a fixed context window, allowing for more adaptable context management compared to conventional Transformer layers.
Minimal Attention Usage: Nearly all attention layers in the Transformer model are replaced by H3 layers, except for two, significantly reducing the model's reliance on attention.

Enhanced Performance: This approach leads to improved performance in language modeling tasks, achieving superior metrics like perplexity (PPL) and outperforming traditional Transformers in various downstream evaluations.

Accelerated Speed and Scalability: With the implementation of FlashConv to optimize SSMs, H3 layers support scalability up to 2.7 billion parameters, significantly enhancing the model's speed and capacity.

These features allow the H3 model to efficiently manage long sequences and large-scale data with enhanced performance and reduced computational requirements compared to traditional Transformers.







## Self-attention vs. State Space Models



## What does H3 have to do with the classic childhood game Hungry Hungry Hippos?
Just like those hungry hippos snapping up marbles in a nonstop stream, the H3 model is built to "devour" long sequences of text with impressive speed and efficiency. Imagine playing Hungry Hungry Hippos—you don’t need to carefully compare each marble to every other marble (which would be painfully slow!); the hippos just keep chomping away in a smooth, continuous motion.

Similarly, while traditional Transformers get bogged down by trying to compare every word to every other word in a sequence (picture having to pair up each marble one by one), H3 tackles text in a more streamlined, seamless way, just like our marble-munching hippos. And just as those hippos never seem to get full, H3 can handle much longer sequences of text without getting computationally “stuffed,” making it a truly hungry model for processing language!


## What is the the Attention Problem?

Transformers are "computationally intensive" because they require substantial processing power and memory, particularly as input sequences grow longer. This intensive demand stems from the self-attention mechanism, which calculates relationships between every pair of tokens in an input sequence. This calculation necessitates O(N^2) operations, meaning the computational requirements increase quadratically with sequence length.

For instance, if the sequence length doubles from 500 to 1,000 tokens, the computational load doesn’t just double—it quadruples, increasing from 250,000 to 1,000,000 operations. This quadratic growth imposes a heavy burden on memory and processing resources, especially when handling lengthy sequences. Such demand can slow processing speed, increase energy consumption, and limit accessibility, particularly for organizations with fewer resources. As a result, this scaling issue restricts transformers’ ability to manage tasks that require extended context or long-form data processing efficiently.

## Why State Space Models?

State Space Models (SSMs) offer an alternative by scaling linearly, requiring 
O(n) operations instead of O(N^2). This linear scaling means that computational costs grow at the same rate as the sequence length, making SSMs significantly more efficient for longer sequences.

For example:

For 1,000 tokens, an SSM requires approximately 1,000 operations.
For 2,000 tokens, it requires around 2,000 operations.

By growing linearly, SSMs avoid the exponential increase in computational costs that transformers face. This makes them a promising approach for maintaining the benefits of transformers while mitigating the high resource demands, allowing for efficient handling of lengthy sequences without overwhelming memory or processing capacity.


## The Goal of Hungry Hungry Hippos
The goal of the Hungry Hungry Hippos project is to replace attention mechanisms in transformers with alternative methods that do not grow quadratically in sequence length. These new mechanisms should scale better with longer sequences, maintaining efficiency as input size increases. Additionally, they should provide similar modeling capabilities, ensuring that the model’s performance is not compromised. Finally, these methods should be designed to use less memory and computation, making the model more resource-efficient and accessible for broader applications. 



## How H3 Achieves This?


Instead of comparing all tokens with all others (attention approach)
Uses two types of SSMs:

Shift SSM: remembers recent tokens
Diagonal SSM: maintains long-term information


Combines them with multiplicative interactions to enable token comparison










## Questions

Sources:

https://www.youtube.com/watch?v=TkOSKrlpnU4





