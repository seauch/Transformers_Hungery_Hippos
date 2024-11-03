

<img src="./images/Hungry_Hippos_Readme_Header.png" alt="Hungry Hippos Header" width="500"/>


# Presentation: Hungry Hungry Hippos: Towards Language Modeling with State Space Models
By Sarah Auch

## Overview

Welcome to an introduction to the H3 (Hungry Hungry Hippos) layer and FlashConv algorithm for State Space Models. With the release of models like ChatGPT, Claude, and Co-Pilot, transformers have become widely discussed and used by the general public for natural language processing. However, transformers are computationally intensive, especially for processing long sequences due to their quadratic time complexity. This has sparked interest in research to explore methods for achieving the benefits of transformers without the high computational costs. H3 adapts State Space Models (SSMs), well-regarded for modeling continuous sequential data and excelling in fields like time series analysis and control systems engineering to solve the problems seen with Transformers. 

SSMs offer the potential for lower computational expense, as they operate with linear complexity, and can manage longer context lengths effectively. Innovations like the H3 architecture are enabling SSMs to expand into natural language processing, providing a more efficient alternative to transformers for handling long text sequences while maintaining strong performance, coming within 0.4 PPL of Transformers on OpenWebText, and even outperforming them by 1.0 PPL in hybrid configurations. Specifically, the H3 architecture incorporates the FlashConv algorithm, which enhances hardware utilization, and an H3 layer, which addresses SSMs' limitations in token recall and cross-sequence token comparison, two crucial capabilities for language modeling.

### What does it mean for transofrmers to be computationally intensive? 

Transformers are "computationally intensive" because they demand significant processing power and memory, especially as input sequences get longer. This high demand primarily arises from the self-attention mechanism, which calculates relationships between every pair of tokens in an input sequence. This requires 
O(n^2) operations, meaning that the computational needs grow quadratically with the length of the sequence. For example, if the sequence length doubles (from 500 to 1,000 tokens), the computational load increases fourfold (from 250,000 to 1,000,000 operations). This quadratic growth puts significant strain on memory and processing resources when handling long sequences.


## Why is it problematic? 

Each new token in a Transformer model introduces substantial computational demand, causing the model to slow down significantly with longer inputs. This impacts both training speed and real-time inference, making Transformers impractical for tasks that require quick responses. Additionally, handling longer sequences increases overall training time and costs, making large Transformer-based models (e.g., GPT, BERT) expensive to train. Training these models on extensive datasets with long sequences requires significant resources and can take weeks on specialized hardware, which limits accessibility for smaller organizations or research teams.

Tasks like document summarization, language translation of long passages, and analysis of legal or technical documents require contextual understanding across many sentences or even pages. Standard Transformers struggle here since input sequences must be kept relatively short to manage computational demands. Often, sequences must be truncated, leading to a loss of crucial context, which can hinder performance on tasks requiring a full-document understanding.

The high computational and memory demands of Transformers also translate to increased energy consumption, with significant environmental impacts. Running large language models contributes to high carbon footprints, driven by the power required for both processing and cooling. Additionally, the cost of operating these models restricts their accessibility, potentially widening the gap between well-funded tech organizations and smaller labs, universities, or companies.



## Questions

How does a beloved childhood game relate to Natural Language Modeling? What are State Space Models? 

## Introduction



