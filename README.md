# Visualizing the learned space-time attention 

This repository contains implementations of __Attention Rollout__ for __TimeSformer__ model. 

Attention Rollout was introduced in paper [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928). It is a method to use attention weights to understand how a self-attention network works, and provides valuable insights into which part of the input is the most important when generating the output. 

## Implementating Attention Rollout for TimeSformer

For divided space-time attention, each token has `2` dimensions,  let's denote the token as `z(p,t)`, where `p` is spatial dimension and  `t` is the time dimension; 


we can calculate the combined space time attention `W` as 
```python
W[i,j,p,q] = S[i,j,p]* T[p,j,q]
```

note that the classification token did not participate in the time attention layer - it was removed from the input before it enter the time attention layer and added back before passing to the space attention layer. This means it only attends to itself during time attention computation, we use an identity matrix to account for this. Since classification did not participate in time attention computation, all the tokens will only be able to attend to classification token from same frame, to address this limitation, in TimeSformer implementation, the `cls_token` output is averaged across all frames at end of each space-time attention block, so that it will be able to carry information from other frames, we also need to average its attention to all input tokens when we compute the combined space time attention

## Usage

Here is a notebook demostrate how to use attention rollout to visualize space time attention learnt from TimeSformer


[a colab notebook: Visualizing learned space time attention with attention rollout](https://colab.research.google.com/github/yiyixuxu/TimesFormer_rolled_attention/blob/main/visualizing_space_time_attention.ipynb)

## Visualizing the learned space time attention

This is the example used in the TimeSformer paper to demonstrate that the model can learn to attend to the
relevant regions in the video in order to perform complex spatiotemporal reasoning. we can see that
the model focuses on the configuration of the hand when visible and the object-only when not visible.
![alt text](https://github.com/yiyixuxu/TimesFormer_rolled_attention/blob/6f3bce9fdb35ab6178b15a27b1d7b493ae69d9aa/img.png?raw=true)
![alt text](https://github.com/yiyixuxu/TimesFormer_rolled_attention/blob/6f3bce9fdb35ab6178b15a27b1d7b493ae69d9aa/mask.png?raw=true)




## References
memo
