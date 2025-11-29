// mnist_loader.h
#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// Function to swap endianness for reading the MNIST file format
int swap_endian(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to load MNIST image data
// Normalizes pixel values from [0, 255] to [0.0, 1.0]
std::vector<std::vector<float>> load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    file.read((char*)&num_images, sizeof(num_images));
    num_images = swap_endian(num_images);
    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = swap_endian(num_rows);
    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = swap_endian(num_cols);

    std::vector<std::vector<float>> images(num_images);
    int image_size = num_rows * num_cols;
    unsigned char* buffer = new unsigned char[image_size];

    for (int i = 0; i < num_images; ++i) {
        images[i].resize(image_size);
        file.read((char*)buffer, image_size);
        for (int j = 0; j < image_size; ++j) {
            images[i][j] = buffer[j] / 255.0f; // Normalize to 0.0-1.0
        }
    }

    delete[] buffer;
    return images;
}

// Function to load MNIST label data
std::vector<int> load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << path << std::endl;
        return {};
    }

    int magic_number = 0, num_labels = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = swap_endian(num_labels);

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }

    return labels;
}