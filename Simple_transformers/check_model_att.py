
from model import *



model = MyTransformer_TS(n_input=100, n_output=1, n_head=5, num_layers=2)
input = torch.rand((10, 100))
print(model(input))

model.eval()

save_output = SaveOutput()

for module in model.modules():
    if isinstance(module, nn.MultiheadAttention):
        patch_attention(module)
        module.register_forward_hook(save_output)


all_params = []
for name, param in model.named_parameters():
    if 'self_attn.out_proj.weight' in name:
        all_params.append(param.detach().numpy())



save_output


# save_output = SaveOutput()
# patch_attention(model.transformer_encoder.layers[-1].self_attn)
# hook_handle = model.transformer_encoder.layers[-1].self_attn.register_forward_hook(save_output)
#
# seq_len = 20

with torch.no_grad():
    out = model(input)

print(save_output.outputs[0][0])


# print(save_output.outputs)