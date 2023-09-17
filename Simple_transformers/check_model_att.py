
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


model_weights =[]
att_layers = []
model_children = list(model.children())
counter = 0
for i in range(len(model_children)):
    # if 'transformer' in type(model_children[i]):
    if type(model_children[i]) == nn.MultiheadAttention:
        counter += 1
        model_weights.append(model_children[i].weight)
        att_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.MultiheadAttention:
                    counter+=1
                    model_weights.append(child.weight)
                    att_layers.append(child)
print(f"Total attention layers: {counter}")
# print("conv_layers")




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