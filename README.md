# PyTorch-Issue


change input in line 78, input: (batch_size, 64, 16, 16) 

run <code>python why.py</code>

output is like:

<code>

2 network parameters are same

Testing MSE between origian network with construted TorchScript network:

default train vs torch_script train:
tensor(0., grad_fn=<PowBackward0>)

default eval vs torch_script eval:
tensor(56.4769, grad_fn=<PowBackward0>)

default train vs torch_script eval:
tensor(0., grad_fn=<PowBackward0>)

default eval vs torch_script train:
tensor(56.4769, grad_fn=<PowBackward0>)

</code>
