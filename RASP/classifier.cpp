#include "classifier.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

// Constructor: carga el modelo TorchScript desde el path proporcionado
Classifier::Classifier(const std::string& model_path) : model_path(model_path) {
    try {
        model = torch::jit::load(model_path);
        std::cout << "Modelo cargado correctamente." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error al cargar el modelo: " << e.what() << std::endl;
    }
}

// Función auxiliar para calcular la similitud coseno entre dos tensores
float cosine_similarity(const torch::Tensor& a, const torch::Tensor& b) {
    float dot = torch::dot(a, b).item<float>();               // Producto punto
    float norm_a = torch::norm(a).item<float>();              // Norma del primer tensor
    float norm_b = torch::norm(b).item<float>();              // Norma del segundo tensor
    return dot / (norm_a * norm_b + 1e-8);                     // Evita división por cero
}

// Carga un vector de embedding desde un archivo .txt
std::vector<float> load_embedding_txt(const std::string& path) {
    std::vector<float> vec;
    std::ifstream file(path);
    float val;
    while (file >> val) {
        vec.push_back(val);  // Lee cada valor del archivo como float
    }
    return vec;
}

// Función principal de predicción a partir de una imagen
int Classifier::predict(const std::string& image_path) {
    try {
        // Carga la imagen con OpenCV
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error al cargar la imagen." << std::endl;
            return -1;
        }

        // Convierte la imagen a un tensor Torch (NCHW, normalizado)
        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2});  // NCHW
        img_tensor = img_tensor.to(torch::kFloat) / 255.0;  // Normaliza entre 0 y 1

        // Pasa el tensor por el modelo
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(img_tensor);
        torch::Tensor output = model.forward(inputs).toTensor();  // Suponemos que retorna embedding

        // Verifica el formato del output
        torch::Tensor embedding;
        if (output.dim() == 2 && output.size(1) > 10) {
            embedding = output.squeeze();  // El modelo devuelve un solo embedding
        } else {
            std::cerr << "Formato de salida no válido del modelo." << std::endl;
            return -1;
        }

        // Thresholds individuales para cada clase (0, 1, 2). Clase 3 ignorada para similitud
        std::map<int, float> class_thresholds = {
            {0, 0.80}, // Latas
            {2, 0.80}, // Botella PET
            {3, 0.90}  // Botella vidrio
        };
        
        int matched_class = -1;
        float max_sim = 0.0;
        
        for (const auto& file : fs::directory_iterator("EMBEDINGS")) {
            std::vector<float> vec = load_embedding_txt(file.path().string());
            if (vec.size() != embedding.size(0)) continue;
        
            // Extrae la clase del nombre del archivo, ejemplo: clase0.txt -> 0
            std::string filename = file.path().filename().string();
            int class_id = -1;
            try {
                size_t pos1 = filename.find_first_of("0123456789");
                size_t pos2 = filename.find_first_not_of("0123456789", pos1);
                class_id = std::stoi(filename.substr(pos1, pos2 - pos1));
            } catch (...) {
                continue; // Ignora archivos con nombres no válidos
            }
        
            // Solo comparar si la clase tiene threshold definido
            if (class_thresholds.find(class_id) != class_thresholds.end()) {
                torch::Tensor saved_tensor = torch::from_blob(vec.data(), {(long)vec.size()}, torch::kFloat32).clone();
                float sim = cosine_similarity(embedding, saved_tensor);
        
                if (sim >= class_thresholds[class_id] && sim > max_sim) {
                    max_sim = sim;
                    matched_class = class_id;
                }
            }
        }

        // Si se encontró una coincidencia por similitud
        if (matched_class != -1) {
            std::cout << "Similitud alta con clase " << matched_class << ": " << max_sim << std::endl;
            return 4;  // Clase especial si se supera el threshold
        }

        // Si no es un embedding conocido, usa argmax del embedding como predicción directa de clase
        int predicted_class = embedding.argmax().item<int>();
        return predicted_class;

    } catch (const c10::Error& e) {
        std::cerr << "Error en la predicción: " << e.what() << std::endl;
        return -1;
    }
}
