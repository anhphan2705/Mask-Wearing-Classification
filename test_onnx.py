# Export

import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Dummy input for ONNX
dummy_input = torch.randn(10, 3, 224, 224)

# Export with ONNX
torch.onnx.export(model, dummy_input, "./EfficientNet-Models/{model_name}.onnx", verbose=True)



# Test export 
import onnx

model = onnx.load("efficientnet-b1.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)

