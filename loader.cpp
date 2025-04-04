#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// Read big-endian integers from file
int readInt(std::ifstream &ifs) {
    uint8_t bytes[4];
    ifs.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Read MNIST images
std::vector<std::vector<uint8_t> > readImages(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    int magic = readInt(file);
    int num_images = readInt(file);
    int rows = readInt(file);
    int cols = readInt(file);
    std::vector<std::vector<uint8_t> > images(num_images, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < num_images; i++) {
        file.read((char*)images[i].data(), rows * cols);
    }
    return images;
}

// Read MNIST labels
std::vector<uint8_t> readLabels(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    int magic = readInt(file);
    int num_labels = readInt(file);
    std::vector<uint8_t> labels(num_labels);
    file.read((char*)labels.data(), num_labels);
    return labels;
}

#include <iostream>
#include <vector>
#include <string>

int main() {
    // File paths (Change these to actual file paths after decompression)
    std::string image_file = "train-images.idx3-ubyte"; 
    std::string label_file = "train-labels.idx1-ubyte"; 

    // Read images and labels
    std::vector<std::vector<uint8_t> > images = readImages(image_file);
    std::vector<uint8_t> labels = readLabels(label_file);

    // Print dataset information
    std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels.\n";

    // Display the first image and label
    if (!images.empty() && !labels.empty()) {
        std::cout << "Label: " << (int)labels[0] << std::endl;
        std::cout << "First image:\n";
        
        int rows = 28, cols = 28;  // MNIST image dimensions
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << (images[0][i * cols + j] > 128 ? '#' : '.'); // ASCII visualization
            }
            std::cout << std::endl;
        }
    } else {
        std::cerr << "Error: No images or labels loaded!" << std::endl;
    }

    return 0;
}
