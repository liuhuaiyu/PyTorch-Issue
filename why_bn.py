import torch
import torch.nn as nn

class ActionHead(nn.Module):

    def __init__(self, inplanes=64, imsize=16, num_classes=14):
        super(ActionHead, self).__init__()
        outplanes = 4

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size = 1, stride = 1, bias = False)
        self.bn = nn.BatchNorm2d(outplanes, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_features=imsize * imsize * outplanes,
                            out_features=num_classes,
                            bias=True)

    def forward(self, x):


        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class TorchScript_ActionHead(torch.jit.ScriptModule):

    def __init__(self, inplanes=64, imsize=16, num_classes=14):
        super(TorchScript_ActionHead, self).__init__()
        outplanes = 4

        self.conv = torch.jit.trace(nn.Conv2d(inplanes, outplanes, kernel_size = 1, 
            stride = 1, bias = False),
            torch.randn(1, inplanes, imsize, imsize))

        # if self.disable_bn_in_resnet == 0:
        self.bn = torch.jit.trace(nn.BatchNorm2d(outplanes, 
            track_running_stats=True),
        torch.randn(1, outplanes, imsize, imsize))
        self.relu = torch.jit.trace(nn.ReLU(inplace=True), 
            torch.randn(1, outplanes, imsize, imsize))
        self.fc = torch.jit.trace(nn.Linear(in_features=imsize * imsize * outplanes,
            out_features=num_classes,
            bias=True),
            torch.randn(1, imsize * imsize * outplanes))

    @torch.jit.script_method
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

default_net = ActionHead()
torchscript_net = TorchScript_ActionHead()

checkpoint = torch.load('ActionHead_parameters.pt')

dict_for_params_match = {}
for param_tensor in checkpoint.keys():
    dict_for_params_match[param_tensor] = checkpoint[param_tensor]

default_net.load_state_dict(dict_for_params_match)
torchscript_net.load_state_dict(dict_for_params_match)

for p1, p2 in zip(default_net.parameters(), torchscript_net.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
        raise Exception('network parameters are not matching!')
print('2 network parameters are same')

# for param_tensor in default_net.state_dict():
#     print(default_net.state_dict()[param_tensor].size())
#     print(torchscript_net.state_dict()[param_tensor].size())

Input = torch.randn(1, 64, 16, 16)

default_net_output_train = default_net.forward(Input)
torchscript_net_output_train = torchscript_net.forward(Input)

default_net.eval()
torchscript_net.eval()

default_net_output_eval = default_net.forward(Input)
torchscript_net_output_eval = torchscript_net.forward(Input)

print('\nTesting MSE between origian network with construted TorchScript network:')

print('\ndefault train vs torch_script train:')
print(torch.sum(default_net_output_train - torchscript_net_output_train) ** 2)
print('\ndefault eval vs torch_script eval:')
print(torch.sum(default_net_output_eval - torchscript_net_output_eval) ** 2)
print('\ndefault train vs torch_script eval:')
print(torch.sum(default_net_output_train - torchscript_net_output_eval) ** 2)
print('\ndefault eval vs torch_script train:')
print(torch.sum(default_net_output_eval  - torchscript_net_output_train) ** 2)

