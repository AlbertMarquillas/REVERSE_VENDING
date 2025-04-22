import os
import numpy as np
import cv2
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn


# Ruta a las imágenes
images_dir = "/home/deep/imgs"

# Cargar modelo preentrenado ResNet50
"""model = models.resnet50(pretrained=True)
model.eval()  # Modo evaluación

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])"""
n_classes = 250

which_net = 1

if which_net == 0:
    weights_model = EfficientNet_B0_Weights.IMAGENET1K_V1
    model_efficient = efficientnet_b0(weights=weights_model, pretrained=True)
elif which_net == 1:
    weights_model = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model_efficient = mobilenet_v3_small(weights=weights_model, pretrained=True)

if which_net == 0:
    num_features_classifier = model_efficient.classifier[1].in_features
    model_efficient.classifier[1] = nn.Sequential(
				nn.Dropout(p=0, inplace=True),
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))
elif which_net == 1:
    num_features_classifier = model_efficient.classifier[3].in_features
    model_efficient.classifier[3] = nn.Sequential(
				nn.Linear(in_features=num_features_classifier, out_features=n_classes, bias=True))

#model_efficient.load_state_dict(torch.load("model_out.pth"))
model_efficient.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


normalize = transforms.Compose([
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Resize((400,400)),    #tamaño final imagenes
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      #normalizacion imagenes con valores ImageNet dataset
                    ])

# Obtener embeddings de una imagen
def get_image_embedding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = normalize(img).unsqueeze(0)  # Añadir dimensión de batch
    img = Variable(img)

    with torch.no_grad():
        embedding = model_efficient(img)

    return embedding.squeeze().cpu().numpy()

def rotate_image(img, angle):
    """Rota la imagen en el ángulo especificado."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    return rotated

def get_image_embedding_rot(img_path):
    """Extrae embeddings para 4 rotaciones de la imagen (0°, 90°, 180°, 270°)."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    embeddings = []
    
    with torch.no_grad():
        for angle in [0, 90, 180, 270]:
            rotated_img = rotate_image(img, angle)
            rotated_img = normalize(rotated_img).unsqueeze(0)  # Normalizar y añadir batch
            rotated_img = Variable(rotated_img)

            embedding = model_efficient(rotated_img)
            embeddings.append(embedding.squeeze().cpu().numpy())

    return embeddings  # Devuelve una lista con 4 embeddings (uno por cada rotación)

# Obtener lista de imágenes
image_files = os.listdir(images_dir)

# Agrupar imágenes en pares
pairs = defaultdict(list)
for img_name in image_files:
    pair_id = img_name.split('_')[0]  # Extraer prefijo del par
    pairs[pair_id].append(img_name)

# Obtener embeddings
image_embeddings = {}
for img_name in tqdm(image_files, desc="Obteniendo embeddings", unit="imagen"):
    img_path = os.path.join(images_dir, img_name)
    image_embeddings[img_name] = get_image_embedding_rot(img_path)

# Calcular similitud dentro de los pares
same_pair_similarities = []
"""for pair_id, imgs in pairs.items():
    if len(imgs) == 2:  # Asegurar que sea un par exacto
        emb1 = image_embeddings[imgs[0]]
        emb2 = image_embeddings[imgs[1]]
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        same_pair_similarities.append(similarity)"""

for pair_id, imgs in pairs.items():
    if len(imgs) == 2:  # Asegurar que sea un par exacto
        emb1_list = image_embeddings[imgs[0]]
        emb2_list = image_embeddings[imgs[1]]
        max_similarity = max(cosine_similarity([emb1], [emb2])[0][0] 
                             for emb1 in emb1_list for emb2 in emb2_list)
        
        same_pair_similarities.append(max_similarity) 

# Calcular similitud entre pares distintos
diff_pair_similarities = []
pair_keys = list(pairs.keys())
for i in tqdm(range(len(pair_keys)), desc="Calculando diferencias entre pares"):
    for j in range(i + 1, len(pair_keys)):  # Evitar comparar consigo mismo
        imgs1 = pairs[pair_keys[i]]
        imgs2 = pairs[pair_keys[j]]

        """if len(imgs1) == 2 and len(imgs2) == 2:
            emb1 = image_embeddings[imgs1[0]]
            emb2 = image_embeddings[imgs2[0]]
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            diff_pair_similarities.append(similarity)"""

        if len(imgs1) == 2 and len(imgs2) == 2:
            emb1_list = image_embeddings[imgs1[0]]
            emb2_list = image_embeddings[imgs2[0]]
            max_similarity = max(cosine_similarity([emb1], [emb2])[0][0] 
                                 for emb1 in emb1_list for emb2 in emb2_list)

            diff_pair_similarities.append(max_similarity)

# Resultados
mean_same = np.mean(same_pair_similarities)
mean_diff = np.mean(diff_pair_similarities)
print(f"Similitud media entre imágenes del mismo par: {mean_same:.4f}")
print(f"Similitud media entre imágenes de pares diferentes: {mean_diff:.4f}")
print(f"Diferencia media (mayor es mejor): {mean_same - mean_diff:.4f}")



# Asociar cada imagen a su par
image_to_pair = {img: pair_id for pair_id, imgs in pairs.items() for img in imgs}

# Variables para métricas
correct_matches = 0
incorrect_matches = []
total_images = len(image_files)

min_correct_similarity = float("inf")
max_incorrect_similarity = float("-inf")

# Umbral de similitud
SIMILARITY_THRESHOLD = 0.85

# Calcular el porcentaje de acierto y métricas de similitud
for img_name in tqdm(image_files, desc="Calculando precisión", unit="imagen"):
    img_emb_list = image_embeddings[img_name]  # Lista con los embeddings rotados

    # Calcular similitud con todas las demás imágenes
    similarities = {}
    for other_img_name, other_emb_list in image_embeddings.items():
        if img_name != other_img_name:  # No compararse consigo misma
            max_sim = max(cosine_similarity([emb1], [emb2])[0][0] 
                          for emb1 in img_emb_list for emb2 in other_emb_list)
            similarities[other_img_name] = max_sim

    # Encontrar la imagen con la mayor similitud
    best_match = max(similarities, key=similarities.get)
    best_similarity = similarities[best_match]

    # Obtener la similitud con la imagen correcta (si existe otra en el mismo par)
    correct_similarities = [
        similarities[other_img] for other_img in pairs[image_to_pair[img_name]] if other_img in similarities
    ]
    real_similarity = max(correct_similarities) if correct_similarities else None

    # Registrar la mínima similitud correcta
    if real_similarity is not None:
        min_correct_similarity = min(min_correct_similarity, real_similarity)

    # 🔹 **Condiciones de error** 🔹
    error_flag = False
    
    # Si la imagen con mayor similitud no es la correcta → Error**
    if image_to_pair[img_name] != image_to_pair[best_match]:
        error_flag = True

    # Si cualquier otra imagen tiene similitud > 0.8 → Error**
    for other_img, sim in similarities.items():
        if image_to_pair[other_img] != image_to_pair[img_name] and sim > SIMILARITY_THRESHOLD:
            error_flag = True
            max_incorrect_similarity = max(max_incorrect_similarity, sim)

    # Si el valor de similitud más alto es < 0.8 → Error**
    if best_similarity < SIMILARITY_THRESHOLD:
        error_flag = True

    if error_flag:
        incorrect_matches.append((img_name, best_match, best_similarity, real_similarity))
    else:
        correct_matches += 1  # Solo se cuenta si no hay error

# Calcular porcentaje de acierto
accuracy = (correct_matches / total_images) * 100 if total_images > 0 else 0

print(f"Total imágenes analizadas: {total_images}")
print(f"Porcentaje de acierto: {accuracy:.2f}%")
print(f"Similitud mínima entre pares correctos: {min_correct_similarity:.4f}")
print(f"Similitud máxima entre pares incorrectos: {max_incorrect_similarity:.4f}")

# Imprimir los errores
print("\nErrores en la clasificación:")
for img_name, best_match, best_similarity, real_similarity in incorrect_matches:
    print(f"Imagen: {img_name} | Predicción: {best_match} | Similitud real: {real_similarity:.4f} | Similitud máxima: {best_similarity:.4f}")

"""# Crear un diccionario que asocia cada imagen con su par correspondiente
image_to_pair = {}
for pair_id, imgs in pairs.items():
    for img in imgs:
        image_to_pair[img] = pair_id  # Asociar cada imagen a su par

# Calcular el porcentaje de acierto y métricas de similitud
for img_name in tqdm(image_files, desc="Calculando precisión", unit="imagen"):
    img_emb = image_embeddings[img_name]
    
    # Calcular similitud con todas las demás imágenes
    similarities = {}
    for other_img_name, other_emb in image_embeddings.items():
        if img_name != other_img_name:  # No compararse consigo misma
            similarities[other_img_name] = cosine_similarity([img_emb], [other_emb])[0][0]

    # Encontrar la imagen con la mayor similitud
    best_match = max(similarities, key=similarities.get)
    best_similarity = similarities[best_match]

    # Obtener la similitud con la imagen correcta (si existe otra en el mismo par)
    correct_similarities = [
        similarities[other_img] for other_img in pairs[image_to_pair[img_name]] if other_img in similarities
    ]
    if correct_similarities:
        real_similarity = max(correct_similarities)  # Elegimos la máxima similitud con el par real
        min_correct_similarity = min(min_correct_similarity, real_similarity)
    else:
        real_similarity = None

    # Verificar si pertenece al mismo par
    if image_to_pair[img_name] == image_to_pair[best_match]:
        correct_matches += 1
    else:
        # Guardar el error
        incorrect_matches.append((img_name, best_match, best_similarity, real_similarity))
        max_incorrect_similarity = max(max_incorrect_similarity, best_similarity)

# Calcular porcentaje de acierto
accuracy = (correct_matches / total_images) * 100
print(f"Porcentaje de acierto: {accuracy:.2f}%")
print(f"Similitud mínima entre pares correctos: {min_correct_similarity:.4f}")
print(f"Similitud máxima entre pares incorrectos: {max_incorrect_similarity:.4f}")

# Imprimir los errores
print("\nErrores en la clasificación:")
for img_name, best_match, best_similarity, real_similarity in incorrect_matches:
    print(f"Imagen: {img_name} | Predicción: {best_match} | Similitud real: {real_similarity:.4f} | Similitud máxima: {best_similarity:.4f}")"""
