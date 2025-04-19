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

        // Compara el embedding con todos los emb. conocidos en la carpeta EMBEDINGS/
        float max_sim = 0.0;
        const float threshold = 0.95;

        for (const auto& file : fs::directory_iterator("EMBEDINGS")) {
            std::vector<float> vec = load_embedding_txt(file.path().string());
            if (vec.size() != embedding.size(0)) continue;  // Skip si tamaño no coincide

            // Convierte el vector a tensor y calcula similitud
            torch::Tensor saved_tensor = torch::from_blob(vec.data(), {(long)vec.size()}, torch::kFloat32).clone();
            float sim = cosine_similarity(embedding, saved_tensor);
            if (sim > max_sim) max_sim = sim;
        }

        // Si la similitud supera el umbral, lo marcamos como clase especial "4"
        if (max_sim >= threshold) {
            std::cout << "Similitud alta con embedding conocido: " << max_sim << std::endl;
            return 4;
        }

        // Si no es un embedding conocido, usa argmax del embedding como predicción directa de clase
        int predicted_class = embedding.argmax().item<int>();
        return predicted_class;

    } catch (const c10::Error& e) {
        std::cerr << "Error en la predicción: " << e.what() << std::endl;
        return -1;
    }
}
