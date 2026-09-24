import numpy as np
import torch

def assign(target_param, source_weight):
    if target_param.shape != source_weight.shape:
        raise ValueError("shapeがあってないぞ")

    with torch.no_grad():
        target_param.copy_(
            torch.tensor(
                source_weight,
                dtype=target_param.dtype,
                device=target_param.device
            )
        )

def load_weights_into_gpt(gpt, params):
    assign(gpt.token_emb_layer.weight, params["wte"])
    assign(gpt.position_emb_layer.weight, params["wpe"])

    blocks = params["blocks"]
    for i, source_block in enumerate(blocks):
        gpt_block = gpt.trf_blocks[i]

        w_q, w_k, w_v = np.split(source_block["attn"]["c_attn"]["w"], 3, axis=-1)
        assign(gpt_block.attention.w_query.weight, w_q.T)
        assign(gpt_block.attention.w_key.weight, w_k.T)
        assign(gpt_block.attention.w_value.weight, w_v.T)

        b_q, b_k, b_v = np.split(source_block["attn"]["c_attn"]["b"], 3, axis=-1)
        assign(gpt_block.attention.w_query.bias, b_q)
        assign(gpt_block.attention.w_key.bias, b_k)
        assign(gpt_block.attention.w_value.bias, b_v)

        assign(gpt_block.attention.out_proj.weight, source_block["attn"]["c_proj"]["w"].T)
        assign(gpt_block.attention.out_proj.bias, source_block["attn"]["c_proj"]["b"])

        assign(gpt_block.feedforward.layers[0].weight, source_block["mlp"]["c_fc"]["w"].T)
        assign(gpt_block.feedforward.layers[0].bias, source_block["mlp"]["c_fc"]["b"])
        assign(gpt_block.feedforward.layers[2].weight, source_block["mlp"]["c_proj"]["w"].T)
        assign(gpt_block.feedforward.layers[2].bias, source_block["mlp"]["c_proj"]["b"])

        assign(gpt_block.layer_norm_1.scale, source_block["ln_1"]["g"])
        assign(gpt_block.layer_norm_1.shift, source_block["ln_1"]["b"])
        assign(gpt_block.layer_norm_2.scale, source_block["ln_2"]["g"])
        assign(gpt_block.layer_norm_2.shift, source_block["ln_2"]["b"])

    assign(gpt.final_norm.scale, params["g"])
    assign(gpt.final_norm.shift, params["b"])
    assign(gpt.out_head.weight, params["wte"])
