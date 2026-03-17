from transformers import PretrainedConfig#从huggingface中导入pretanedconfig的类


class MiniMindConfig(PretrainedConfig):#类的括号中是父类，说明minimindconfig继承了huggingface中的pretrainedconfig类
    model_type = "minimind"

    def __init__(#类中的__init__是构造函数，当创建类的时候，会自动调用这个函数
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
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
