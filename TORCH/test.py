import torch
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms, io
import torch.nn as nn

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from custom_dataset import get_data
import time
import tqdm

data_test = get_data("/home/deep/imgs")
n_classes = len(data_test.classes)
classes_labels = data_test.classes
which_net = 0

if which_net == 0:
    weights_model = EfficientNet_B0_Weights.IMAGENET1K_V1
    model_efficient = efficientnet_b0(weights=weights_model)
elif which_net == 1:
    weights_model = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model_efficient = mobilenet_v3_small(weights=weights_model)

if which_net == 0:
    num_features_classifier = model_efficient.classifier[1].in_features
    model_efficient.classifier[1] = nn.Sequential(
				nn.Dropout(p=0, inplace=True),
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))
elif which_net == 1:
    num_features_classifier = model_efficient.classifier[3].in_features
    model_efficient.classifier[3] = nn.Sequential(
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))

model_efficient.load_state_dict(torch.load("model_out.pth"))
model_efficient.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


normalize = transforms.Compose([
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    #transforms.Resize((400,400)),    #tamaño final imagenes
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #normalizacion imagenes con valores ImageNet dataset
                    ])

def inferencia(model, images:torch.Tensor , transforms, device):
    """
    Función que devuelve una lista con las inferencias de un batch de imágenes
    model: modelo en formato .ts
    images: imagen unica en formato np.array
    transforms: transformaciones necesarias para las imagenes. Deben estar en formato torch.Tensor
    device: sitio en el que se trabaja (CPU o GPU)
    """

    model.eval()
    model.to(device)

    starttime = time.time()
    
    images = transforms(images).unsqueeze(0).to(device)

    # Cargamos las imagenes en el device
    # Obtenemos inferencias
    output = model(images)
    # Aplicamos softmax para normalizar probabilidades a 1
    output = torch.nn.functional.softmax(output, dim=1)
    times.append(time.time() - starttime)

    return output.cpu().detach().numpy()


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


images = []
predictions = []
labels = []
times = []

errors = 0
for i,im in tqdm.tqdm(enumerate(data_test)):    
    results = inferencia(model_efficient, im[0], normalize, device)
    prediction = np.argmax(results)
    if im[1] != prediction:
        im[0].save('/home/deep/ERRORS/'+str(errors)+'_REAL_'+classes_labels[im[1]]+'_PRED_'+classes_labels[prediction]+'.jpg')
        errors += 1
    predictions.append(prediction)
    labels.append(im[1])
    
cm = confusion_matrix(labels, predictions)
report = classification_report(labels, predictions, target_names=classes_labels)
print("Temps d'inferencia:", np.mean(times[10:]),"\n")
print("Confusion matrix:\n")
print_cm(cm, classes_labels)
print()
print("Report:\n", report)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_labels)
disp.plot()
plt.show()


    





    

