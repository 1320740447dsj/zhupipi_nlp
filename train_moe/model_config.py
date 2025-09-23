from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "moe_model"

    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=16,
                 num_key_value_heads=8,
                 flash_attn=True,
                 attention_bias=False,
                 max_seq_len=512,
                 intermediate_size=2048,
                 mlp_bias=False,
                 vocab_size=6400,
                 n_layers=8,
                 dropout=0.0,
                 expert_num=4,
                 topk=2,
                 output_router_logits=True,
                 aux_loss_coef=0.01,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)