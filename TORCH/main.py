import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import imblearn

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

#importo funciones de otros scripts
from custom_dataset import *
from train import *
from convert_torchsript import convert

from PIL import Image

################ PARAMETROS ##########################

split_ratio = 0.80               #porcentaje del dataset que sera train
img_folder = "/home/deep/imgs"
learning_rate = 1e-2
scheduler_param = [4,8,12,16] #[4,6,8]  
batch_size = 16
num_epochs = 8
freeze = False
which_net = 0

train = True
convert_ts = True

############### MODELS EFFICIENTNET V2 ################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("DEVICE: ",device)

if which_net == 0:
    weights_model = EfficientNet_B0_Weights.IMAGENET1K_V1
    model_efficient = efficientnet_b0(weights=weights_model)
elif which_net == 1:
    weights_model = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model_efficient = mobilenet_v3_small(weights=weights_model)

############## DATASET ###############################

data_transforms = {
    'train': transforms.Compose([          
        #transforms.RandomRotation(360),
        #transforms.RandomPerspective(distortion_scale=0.1, p=0.25),
        #transforms.RandomAffine((30,70)),
        #transforms.RandomCrop((350,350)),
        RandomRotationTransform(angles=[0, 90, 180, 270, 360]),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        #transforms.RandomAutocontrast(),
        #transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.Grayscale(num_output_channels=3),
        #transforms.Resize((400,400)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Grayscale(num_output_channels=3), 
        #transforms.Resize((400,400)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset = get_data(img_folder)
n_classes = len(dataset.classes)
print("Numero de clases:",n_classes)
label_classes = dataset.classes
print("Labels dataset: ",label_classes)

# data split
train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset)*split_ratio), len(dataset)-int(len(dataset)*split_ratio)])
sampler_train = class_sampler(dataset, train_data)
sampler_val = class_sampler(dataset, val_data)
cnn = imblearn.over_sampling.SMOTE(random_state=42)   #random_state=42
#train_data = TransformsDataset_balance(train_data, data_transforms["train"], cnn)
train_data = TransformsDataset(train_data, data_transforms["train"])
val_data = TransformsDataset(val_data, data_transforms["val"])


train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler_train) #, shuffle=True) #
val_dataloader = DataLoader(val_data, batch_size=batch_size, sampler=sampler_val)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}


#freeze all params
if freeze:
    for params in model_efficient.parameters():
        params.requires_grad_ = False
    for params in model_efficient.classifier.parameters():
        params.requires_grad_ = True

if which_net == 0:
    num_features_classifier = model_efficient.classifier[1].in_features
    model_efficient.classifier[1] = nn.Sequential(
				nn.Dropout(p=0.3, inplace=True),
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))
elif which_net == 1:
    num_features_classifier = model_efficient.classifier[3].in_features
    model_efficient.classifier[3] = nn.Sequential(
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))

############# TRAINING PARAMETERS ###################

criterion = nn.CrossEntropyLoss()
optimizer_conv = torch.optim.SGD(model_efficient.parameters(), lr=learning_rate, momentum=0.9)
#optimizer_conv = torch.optim.Adam(model_efficient.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#optimizer_conv = torch.optim.RMSprop(model_efficient.parameters())
exp_lr_scheduler = scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=scheduler_param, gamma=0.1)
save_model = True

if train:
    model_efficient, acc, loss, acc_val, loss_val = train_model_logits(num_epochs=num_epochs, model=model_efficient, dataloader=dataloaders, classes_labels=label_classes,
    									criterion= criterion, optimizer=optimizer_conv,scheduler=exp_lr_scheduler, device=device,
                                                                        save=save_model, save_name="model_out.pth")

######## CONVERTIR A TORCHSCRIPT ######################

if convert_ts:
    transf_val = data_transforms["val"]
    im = torch.zeros((1,3,400,400))
    convert(model_efficient, "model_out.pth", im)

