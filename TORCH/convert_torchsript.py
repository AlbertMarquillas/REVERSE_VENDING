import torch

def convert(model, weights_path, image = None):

    print("Starting conversion to .ts:")
    model.eval()
    model.cpu()
    model.load_state_dict(torch.load(weights_path))

    if image is None:                           #if we pass an image it will trace instead of script
        model_scripted = torch.jit.script(model) # Export to TorchScript
   
    else:
        model_scripted = torch.jit.trace(model,image)

    model_scripted.save('model.ts') # Save
    print("Done")
