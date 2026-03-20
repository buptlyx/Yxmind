from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
from typing import Optional
class YxMindConfig(PretrainedConfig):
    model_type = "yxmind"

    def __init__(
            self,
            dropout: float = 0.0,#训练时随机屏蔽掉一些神经元，防止过拟合。dropout为0，表示默认不开启dropout。
            bos_token_id: int = 1,#句首/句尾的特殊token的id
            eos_token_id: int = 2,
            hidden_act: str = 'silu',#silu激活函数
            hidden_size: int = 512,#每个token的向量维度
            intermediate_size: int = None,#网络的中间层维度，一也就是网络扩张开的大小。
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,#query的数量
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,#key和value的数量，这里使用了mqa机制，减少了计算量
            vocab_size: int = 6400,#词表大小，这个是小词表
            rms_norm_eps: float = 1e-05,#一个极小量
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,#高效注意力机制的开关

            use_moe: bool = False,
            num_experts_per_tok: int = 2,#每个token选2个专家
            n_routed_experts: int = 4,#可供路由选择的专家总数是4个
            n_shared_experts: int = 1,#有一个共享专家，可能所有的token都会经过它
            scoring_func: str = 'softmax',#router使用softmax算专家分数

            aux_loss_alpha: float = 0.01,#辅助损失权重，用于负载均衡
            seq_aux: bool = True,#辅助损失是否按序列级别统计
            norm_topk_prob: bool = True,#选出top-k后，是否把它们的概率重新归一化
            **kwargs
    ):
        super().__init__(**kwargs)#将用户传入的参数传递给父类的构造函数，PretrainedConfig会处理一些通用的配置项
        #以下的赋值本质上是把传入的参数保存成这个配置对象的成员变量
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,#转32圈视为高频
            "beta_slow": 1,#转1圈视为低频
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn

        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 继承nn.Module类


class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
# norm
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
# forward
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)

#提前计算每个维度的频率，生成cos和sin矩阵
def precompute_freqs_cis(dim: int, end: int = 32 * 1024, rope_base=10000, rope_scaling: Optional[dict] = None):
    # 初始化 RoPE 的基础频率
    freqs, attn_factor = (
        1.0 / rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim),
        1.0
    )#torch.arang(0,dim,2)生成一个0到dim的偶数序列，公差为2,。如果dim本身是偶数，那么[:(dim//2)]这一步就会保留所有元素。最后除以dim进行归一化。
    #factor是一个缩放因子，默认是1，这里留一个接口。

    #如果要做长上下文扩展，就进入scaling逻辑；否则用原始的RoPE频率。
    if rope_scaling is not None:
        #从字典中提取参数
        #orig_max模型训练时的原始最大长度
        #factor扩长倍数
        #beta_fast/beta_slow控制从哪些频率开始过渡缩放
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"]
        )
        #只有推理长度超过训练长度才进行缩放
        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            low, high = (
                #low是开始缩放的索引
                #high是完全缩放的索引
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
                #floor和ceil分别是向上取整和向下取整
            )

            #ramp是一个缩放系数
            #当索引小于low时，ramp为0，不进行缩放；当索引大于high时，ramp为1，完全缩放；在low与high之间进行线性缩放。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )

            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device).float()
    #outer是外积操作
    #下面的代码创建出来一个维度cos和sin的查询表
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

#编写RoPE代码
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #rotate_half是一个辅助旋转函数
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed