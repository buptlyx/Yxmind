from transformers import PreTrainedModel,PretrainedConfig,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
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
        #一个极小量
        self.eps=eps
        #weight是一个可训练的参数，初始化为全1，维度是dim。
        #在RMSNorm中需要使用weight对输入进行缩放，所以它是一个可学习的参数。
        self.weight=nn.Parameter(torch.ones(dim))
# norm
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
# forward
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)

#提前计算每个维度的频率，生成cos和sin矩阵
def precompute_freqs_cis(dim: int, end: int = 32 * 1024, rope_base=10000, rope_scaling: Optional[dict] = None):#rope_scaling是一个可选的字典参数
    # 初始化 RoPE 的基础频率
    # 公式中的i指的是第几对二维对子
    freqs, attn_factor = (
        1.0 / rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim),
        1.0
    )#torch.arange(0,dim,2)生成一个0到dim的偶数序列，公差为2,。如果dim本身是偶数，那么[:(dim//2)]这一步就会保留所有元素。最后除以dim进行归一化。
    #factor是一个缩放因子，默认是1，这里留一个接口。

    #如果要做长上下文扩展，就进入scaling逻辑；否则用原始的RoPE频率。Optional类型保证了rope_scaling是一个字典或者是None。
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
        #以下的公式偏向工程化，通过某个频率边界推算出对应维度的索引
        if end > orig_max:
            #这个公式可以理解为，b是维度旋转的圈数，inv_dim是通过圈数反推的维度索引。
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
                #low是缩放开始的索引，dim//2是每一对二维对子的索引，所以说dim//2-low就是所有小于low的索引距离low的距离，high-low是一个区间长度，所以这个公式表达的意思是，在low和high之间进行线性缩放，超过high之后完全缩放。
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )

            freqs = freqs * (1 - ramp + ramp / factor)

    #生成所有的位置序列，也就是论文公式中的m和n
    t = torch.arange(end, device=freqs.device).float()
    #outer是外积操作
    #下面的代码创建出来一个维度cos和sin的查询表，其中每一行对应一个位置的token，每一列对应一个维度的频率。
    freqs = torch.outer(t, freqs).float()
    #为什么需要复制一份？因为token的QK的最后一维是dim而不是dim//2，所以为了适配维度，需要把频率表复制一份拼接起来。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

#编写RoPE代码
def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    #rotate_half是一个辅助旋转函数
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    #unsqueeze是因为cos和sin比q和k少一个维度heads，unsqueeze利用了广播机制，每一个batch中每一个头都有对应的cos和sin
    #这一步很巧妙地把前半段和后半段的维度配对起来了。
    #比如如果dim=8，则1和5配对，2和6配对。
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

#重复使用KV的工具函数
#x:输入张量
#n_rep每个KV头要重复多少次
#GQA中每一个KV头被多个Q头共享
def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    #bs：batch size
    #slen：序列长度
    #num_key_value_heads：KV头数，一般是Attention头数的1/4或者1/8
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

        #Wq的值，把输入的hidden_size维度映射成num_attention_heads * head_dim维度，也就是每个头的维度乘以头的数量
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        #Wk和Wv的值，把输入的hidden_size维度映射成num_key_value_heads * head_dim维度，也就是每个KV头的维度乘以KV头的数量
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #线性层，把多头注意力的输出映射回hidden_size维度
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
        bsz,seq_len,_=x.shape#x的形状是[bsz,seq_len,hidden_size]
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
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
            #计算相似度
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #对最后的seq_len个位置进行mask，保证每个位置只能看到自己和之前的位置，不能看到之后的位置。
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                #如果用户传入了attention_mask，那么就把它扩展成和scores一样的形状，然后把mask的位置设置为一个很大的负数，这样在softmax之后这些位置的权重就接近于0了。
                #1表示可以看，0表示不能看
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                #-1e9是一个很大的负数，保证被mask的位置在softmax后权重为0
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
            #scores的形状是[bsz,n_local_heads,seq_len,seq_len]，每一个位置对应着两个token之间的相似度分布
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            #注意力的dropout，在训练时随机屏蔽掉一些token之间的关系，防止过拟合。
            scores = self.attn_dropout(scores)
            #xv的形状是[bsz,n_local_kv_heads,seq_len,head_dim]，scores的形状是[bsz,n_local_heads,seq_len,seq_len]，通过矩阵乘法把它们结合起来，得到每个位置的加权平均值，形状是[bsz,n_local_heads,seq_len,head_dim]
            output = scores @ xv
        #把最后一个维度转换成hidden_size的形状
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        #resid_dropout是输出的dropout，在最后输出之前再随机屏蔽掉一些神经元，防止过拟合。
        #一般来说attention模块的输出会经过一个残差连接，是x+attention(x)，所以这个dropout的作用是让模型在训练时更加健壮，不会过度依赖某些特定的神经元。
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
        #依次经过门控、激活函数、线性变换和dropout。
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    
class YxMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: YxMindConfig):
        super().__init__()
        #注意力的头数
        self.num_attention_heads = config.num_attention_heads
        #隐藏层的维度
        self.hidden_size = config.hidden_size
        #每个头的维度
        self.head_dim = config.hidden_size // config.num_attention_heads
        #把Attention模块封装成一个成员变量
        self.self_attn = Attention(config)

        #记录当前是第几层
        self.layer_id = layer_id
        #LayerNorm采用RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #在attention之后的一个RMSNorm层
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        #hidden_states是输入的隐藏状态，也就是一个从上一层传递而来的张量，形状是[batch_size, seq_length, hidden_size]

        #存储原始输入，后面做残差连接时使用
        residual = hidden_states

        #进行attention计算，得到新的hidden_states和present_key_value
        #pre-norm结构，先做layernorm，再做attention。
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )

        #attention层的残差连接
        hidden_states += residual
        #FFN层的残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class YxMindModel(nn.Module):
    def __init__(self, config: YxMindConfig):
        super().__init__()
        #传入配置对象
        self.config = config
        #传入配置对象中的词表大小和隐藏层的层数
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        #定义一个词嵌入层，把输入的token_id映射成一个hidden_size维度的向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #定义dropout层
        self.dropout = nn.Dropout(config.dropout)
        #定义一个Modulelist，包含了num_hidden_layers个YxMindBlock，每个block都是一个Transformer层
        self.layers = nn.ModuleList([YxMindBlock(i, config) for i in range(self.num_hidden_layers)])
        #定义RMSNorm层，作为最后的输出层的归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #提前计算好RoPE的位置编码频率，并存储在模型的buffer中，这样在前向传播中就不需要重复计算了
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        #创建kv cache的缓存列表
        presents = []
        #依次通过每一层Transformer层进行计算，每一层返回新的hidden)states和past_key_value
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        #最后进行一次RMSNorm的归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents

#把隐藏层输出映射到词表大小的维度上。
class YxMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class=YxMindConfig

    def __init__(self,config:YxMindConfig):
        self.config=config
        super().__init__(config)

        self.model=YxMindModel(config)

        #把输出的隐藏层维度映射到词表大小中，得到每个token的预测分布
        self.lm_head=nn.Linear(
            self.config.hidden_size,self.config.vocab_size,bias=False
        )
        
        #输出层的权重和嵌入层的权重共享
        self.model.embed_tokens.weight=self.lm_head.weight

        self.OUT=CausalLMOutputWithPast()

    def forward(self,
                    input_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None,
                    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                    use_cache: bool = False,
                    logits_to_keep: Union[int, torch.Tensor] = 0,
                    **args):
            hidden_states, past_key_values= self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **args
            )
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)

            return output

