from transformers import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional,Tuple
from transformers.activations import ACT2FN
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

#重复使用KV的工具函数
#x:输入张量
#n_rep每个KV头要重复多少次
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    #bs：batch size
    #slen：序列长度
    #num_key_value_heads：KV头数
    #head_dim：每个KV头的维度
    #x是一个四维的张量
    bs,slen,num_key_value_heads,head_dim=x.shape
    if n_rep==1:
        return x
    #先插入一个新的维度，再把这个新的张量reshape成我们需要的形状，这样每个KV头就被重复了n_rep次。
    return x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim)

class Attention(nn.Module):
    def __init__(self,args:YxMindConfig):
        super().__init__()

        self.num_key_value_heads=args.num_key_value_heads if args.num_key_value_heads is None else args.num_attention_heads

        #Attention头的数量必须是KV头的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0,"num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(
            self,
            x:torch.Tensor,
            position_embedding:Tuple[torch.Tensor,torch.Tensor],
            past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
            use_cache=False,
            attention_mask:Optional[torch.Tensor]=None,
            )->torch.Tensor:
        #投影，计算qkv
        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.K_proj(x),self.v_proj(x)
        #把qkv分成多个头
        q=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        k=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        v=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        #q和k使用RoPE编码
        cos,sin=position_embedding
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
        #对于k和v，使用repeat
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv=(xk,xv) if use_cache else None

        xq,xk,xv=(
            #[bsz,n_local_heads,seq_len,head_dim] pytorch中的计算习惯发生在最后两位
            xq.transpose(1,2),
            repeat_kv[xk,self.n_rep].transpose(1,2),
            repeat_kv[xv,self.n_rep].transpose(1,2)
        )
        #进行attention计算
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
                    output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):#这是一个可训练的网络模块
    def __init__(self, config: YxMindConfig):
        super().__init__()#这是调用父类nn.Module的初始化方法
        #传入一个config配置对象，包含以下这些部分
        #hidden_size隐藏层的维度
        #intermediate_size中间层的维度
        #dropout丢弃率
        #hidden_act激活函数类型

        #如果用户没有指定中间层的维度，那么就根据hidden_size计算一个默认的维度，通常是hidden_size的8/3倍
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)#这一步是取到一个大于等于intermediate_size的最小的的64的倍数，这样可以更好地利用GPU的并行计算能力
        #定义一个线性层，把hidden_size维度的输入映射到intermediate_size维度
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        #定义一个映射回去hidden_size维度的线性层
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        #先经过一个门控信号，再经过一个激活函数，接着进行升维再降维的过程，最后经过dropout层。
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    
class YxMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: YxMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value