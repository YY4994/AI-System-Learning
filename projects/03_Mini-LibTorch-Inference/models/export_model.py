import torch 
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()

# example_input = torch.randn(1, 3, 224, 224)
example_input = torch.randn(1, 3, 224, 224)

#导出TorchScriptcd 
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("resnet50.pt")

#保存输入输出
torch.save(example_input, "example_input.pt")
with torch.no_grad():
    example_output = model(example_input)
    torch.save(example_output, "example_output.pt")
    print("Example input and output saved successfully!")
print("Model exported successfully!")