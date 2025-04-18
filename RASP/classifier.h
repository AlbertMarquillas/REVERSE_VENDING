#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>

class Classifier {
public:
    Classifier(const std::string& model_path);
    int predict(const std::string& image_path);
private:
    std::string model_path;
};

#endif
