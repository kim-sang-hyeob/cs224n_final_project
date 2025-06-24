from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """

    # input - residual connection
    # output - return value of multi-head attention or MLP
    # dense_layer - ffn for each sublayer
    # dropout - apply dropout after sublayer

    new_out = dense_layer(output)
    dropped = dropout(new_out)
    return dropped + input

    


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """

    # Masked Multi-Headed Self-Attention
    lnorm_for_attn = self.attention_layer_norm(hidden_states)
    attention_result = self.self_attention(lnorm_for_attn, attention_mask)
    attn_layer_out = self.add(hidden_states, attention_result, self.attention_dense, self.attention_dropout)

    # MLP
    lnorm_for_mlp = self.out_layer_norm(attn_layer_out)
    mlp_result = self.interm_af(self.interm_dense(lnorm_for_mlp))
    mlp_layer_out = self.add(attn_layer_out, mlp_result, self.out_dense, self.out_dropout)

    return mlp_layer_out

    

