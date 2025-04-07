#include "loader.h"

using namespace std;

// Read big-endian integers from file
int readInt(ifstream &ifs) {
    uint8_t bytes[4];
    ifs.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

vector<vector<float>> convertToFloat(const vector<vector<uint8_t>>& img, bool normalize) {
    int rows = img.size();
    int cols = img[0].size();
    vector<vector<float>> result(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i][j] = normalize ? img[i][j] / 255.0f : static_cast<float>(img[i][j]);
    return result;
}



// Read MNIST images
vector<vector<vector<uint8_t>>> readImages(const string &filename) {
    ifstream file(filename, ios::binary);
    int magic = readInt(file);
    int num_images = readInt(file);
    int rows = readInt(file);
    int cols = readInt(file);
	cout<<"Rows: "<<rows<<" cols: "<<cols<<endl;
    vector<vector<vector<uint8_t> >> images(num_images, vector<vector<uint8_t>>(rows, vector<uint8_t>(cols)));
    vector<uint8_t> buffer(rows * cols); // Temp buffer to read flat image
    for (int i = 0; i < num_images; ++i) {
        file.read((char *)buffer.data(), rows * cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                images[i][r][c] = buffer[r * cols + c];
            }
        }
    }
    return images;
}


// Read MNIST labels
vector<uint8_t> readLabels(const string &filename) {
    ifstream file(filename, ios::binary);
    int magic = readInt(file);
    int num_labels = readInt(file);
    vector<uint8_t> labels(num_labels);
    file.read((char*)labels.data(), num_labels);
    return labels;
}

// int main() {
//     // File paths (Change these to actual file paths after decompression)
//     string image_file = "train-images.idx3-ubyte"; 
//     string label_file = "train-labels.idx1-ubyte"; 

//     // Read images and labels
//     vector<vector<vector<uint8_t>>> images = readImages(image_file);
//     vector<uint8_t> labels = readLabels(label_file);

//     // Print dataset information
//     cout << "Loaded " << images.size() << " images and " << labels.size() << " labels.\n";

//     // Display the first image and label
//     if (!images.empty() && !labels.empty()) {
//         cout << "Label: " << (int)labels[0] << endl;
//         cout << "First image:\n";
        
//         int rows = 28, cols = 28;  // MNIST image dimensions
//         for (int i = 0; i < rows; i++) {
//             for (int j = 0; j < cols; j++) {
//                 cout << (images[0][i][j] > 128 ? '#' : '.'); // ASCII visualization
//             }
//             cout << endl;
//         }
//     } else {
//         cerr << "Error: No images or labels loaded!" << endl;
//     }

//     return 0;
// }
