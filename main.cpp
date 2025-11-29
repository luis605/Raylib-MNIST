
#include <raylib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <iomanip>
#include <algorithm>

#include "mnist_loader.h"

struct Neuron
{
    std::vector<float> weights;
    float bias;
    float output;
    float delta;
};

struct Layer
{
    std::vector<Neuron> neurons;
};

struct NeuralNetwork
{
    std::vector<Layer> layers;
    std::vector<int> layerSizes;
};

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x)
{
    return x * (1.0f - x);
}

void InitializeWeights(NeuralNetwork &network)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (size_t i = 1; i < network.layers.size(); ++i)
    {
        int previousLayerSize = network.layers[i - 1].neurons.size();
        for (Neuron &neuron : network.layers[i].neurons)
        {
            neuron.weights.resize(previousLayerSize);
            for (int w = 0; w < previousLayerSize; ++w)
            {
                neuron.weights[w] = dist(gen);
            }
            neuron.bias = dist(gen);
        }
    }
}

NeuralNetwork CreateNetwork(const std::vector<int> &layerSizes)
{
    NeuralNetwork network;
    network.layerSizes = layerSizes;

    for (int size : layerSizes)
    {
        Layer layer;
        layer.neurons.resize(size);
        network.layers.emplace_back(layer);
    }
    InitializeWeights(network);
    return network;
}

void FeedForward(NeuralNetwork &network, const std::vector<float> &inputs)
{
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        network.layers[0].neurons[i].output = inputs[i];
    }

    for (size_t i = 1; i < network.layers.size(); ++i)
    {
        const Layer &previousLayer = network.layers[i - 1];
        for (Neuron &neuron : network.layers[i].neurons)
        {
            float weightedSum = 0.0f;
            for (size_t j = 0; j < previousLayer.neurons.size(); ++j)
            {
                weightedSum += previousLayer.neurons[j].output * neuron.weights[j];
            }
            weightedSum += neuron.bias;
            neuron.output = sigmoid(weightedSum);
        }
    }
}

void Backpropagate(NeuralNetwork &network, const std::vector<float> &targets)
{

    Layer &outputLayer = network.layers.back();
    for (size_t i = 0; i < outputLayer.neurons.size(); ++i)
    {
        Neuron &neuron = outputLayer.neurons[i];
        float error = neuron.output - targets[i];
        neuron.delta = error * sigmoid_derivative(neuron.output);
    }

    for (size_t i = network.layers.size() - 2; i > 0; --i)
    {
        Layer &hiddenLayer = network.layers[i];
        const Layer &nextLayer = network.layers[i + 1];
        for (size_t j = 0; j < hiddenLayer.neurons.size(); ++j)
        {
            Neuron &neuron = hiddenLayer.neurons[j];
            float error = 0.0f;
            for (size_t k = 0; k < nextLayer.neurons.size(); ++k)
            {
                error += nextLayer.neurons[k].weights[j] * nextLayer.neurons[k].delta;
            }
            neuron.delta = error * sigmoid_derivative(neuron.output);
        }
    }
}

void UpdateWeights(NeuralNetwork &network, float learningRate)
{
    for (size_t i = 1; i < network.layers.size(); ++i)
    {
        const Layer &previousLayer = network.layers[i - 1];
        for (Neuron &neuron : network.layers[i].neurons)
        {
            for (size_t j = 0; j < previousLayer.neurons.size(); ++j)
            {
                neuron.weights[j] -= learningRate * neuron.delta * previousLayer.neurons[j].output;
            }
            neuron.bias -= learningRate * neuron.delta;
        }
    }
}

std::vector<float> to_one_hot(int label, int num_classes)
{
    std::vector<float> one_hot(num_classes, 0.0f);
    if (label < num_classes)
    {
        one_hot[label] = 1.0f;
    }
    return one_hot;
}

float calculate_mse(const Layer &outputLayer, const std::vector<float> &targets)
{
    float mse = 0.0f;
    for (size_t i = 0; i < targets.size(); ++i)
    {
        mse += pow(targets[i] - outputLayer.neurons[i].output, 2);
    }
    return mse / targets.size();
}

int get_prediction(const NeuralNetwork &network)
{
    const Layer &outputLayer = network.layers.back();
    float maxOutput = -1.0f;
    int prediction = -1;
    for (size_t i = 0; i < outputLayer.neurons.size(); ++i)
    {
        if (outputLayer.neurons[i].output > maxOutput)
        {
            maxOutput = outputLayer.neurons[i].output;
            prediction = i;
        }
    }
    return prediction;
}

void DrawNetworkArchitecture(const NeuralNetwork &network, int x, int y, int width, int height)
{
    const int max_neurons_to_draw = 16;
    float neuronRadius = 7.0f;

    std::vector<std::vector<Vector2>> neuronPositions(network.layers.size());

    float layerSpacing = (float)width / (network.layers.size() - 1);

    for (size_t i = 0; i < network.layers.size(); ++i)
    {
        int neuronsInLayer = network.layers[i].neurons.size();
        int neuronsToDraw = (neuronsInLayer > max_neurons_to_draw) ? max_neurons_to_draw : neuronsInLayer;
        float neuronSpacing = (float)height / (neuronsToDraw + 1);

        for (int j = 0; j < neuronsToDraw; ++j)
        {
            neuronPositions[i].push_back({x + i * layerSpacing, y + (j + 1) * neuronSpacing});
        }
    }

    for (size_t i = 0; i < network.layers.size() - 1; ++i)
    {
        int neuronsToDrawCurrent = neuronPositions[i].size();
        int neuronsToDrawNext = neuronPositions[i + 1].size();

        for (int j = 0; j < neuronsToDrawCurrent; ++j)
        {
            for (int k = 0; k < neuronsToDrawNext; ++k)
            {
                int realNeuronIndexPrev = (j * network.layers[i].neurons.size()) / neuronsToDrawCurrent;
                int realNeuronIndexNext = (k * network.layers[i + 1].neurons.size()) / neuronsToDrawNext;

                float weight = network.layers[i + 1].neurons[realNeuronIndexNext].weights[realNeuronIndexPrev];
                Color weightColor = (weight > 0) ? BLUE : RED;
                weightColor.a = (unsigned char)(fmin(fabs(weight) * 2.0f, 1.0f) * 255);

                DrawLineV(neuronPositions[i][j], neuronPositions[i + 1][k], weightColor);
            }
        }
    }

    for (size_t i = 0; i < network.layers.size(); ++i)
    {
        int neuronsInLayer = network.layers[i].neurons.size();
        for (size_t j = 0; j < neuronPositions[i].size(); ++j)
        {
            int realNeuronIndex = (j * neuronsInLayer) / neuronPositions[i].size();
            float activation = network.layers[i].neurons[realNeuronIndex].output;
            Color neuronColor = ColorLerp(GRAY, YELLOW, activation);
            DrawCircleV(neuronPositions[i][j], neuronRadius, neuronColor);
            DrawCircleLinesV(neuronPositions[i][j], neuronRadius, DARKGRAY);
        }
    }
}

void DrawOutputBars(const NeuralNetwork &network, int correctLabel, int x, int y, int width, int height)
{
    const Layer &outputLayer = network.layers.back();
    float barWidth = (float)width / outputLayer.neurons.size();
    for (size_t i = 0; i < outputLayer.neurons.size(); ++i)
    {
        float barHeight = outputLayer.neurons[i].output * height;
        Color barColor = (i == correctLabel) ? LIME : SKYBLUE;
        DrawRectangle(x + i * barWidth, y + height - barHeight, barWidth - 2, barHeight, barColor);
        DrawText(TextFormat("%d", i), x + i * barWidth + barWidth / 2 - 5, y + height + 5, 20, BLACK);
    }
}

void DrawUI(const NeuralNetwork &network, Texture2D digitTexture, int currentLabel,
            int epoch, int imageIndex, float loss, float accuracy, int screenWidth, int screenHeight)
{

    BeginDrawing();
    ClearBackground(RAYWHITE);

    DrawText("MNIST Neural Network Training", 20, 20, 40, DARKGRAY);
    DrawText(TextFormat("Epoch: %d / 5", epoch), 20, 70, 20, GRAY);
    DrawText(TextFormat("Image: %d / 60000", imageIndex), 20, 95, 20, GRAY);
    DrawText(TextFormat("Current Loss (MSE): %.6f", loss), 20, 120, 20, GRAY);
    DrawText(TextFormat("Epoch Accuracy: %.2f%%", accuracy * 100.0f), 20, 145, 20, DARKGREEN);

    int topRowY = 200;

    DrawText("Input Digit", 100, topRowY, 20, BLACK);
    DrawTextureEx(digitTexture, {50, (float)topRowY + 30}, 0, 8, WHITE);
    DrawRectangleLines(50, topRowY + 30, 28 * 8, 28 * 8, BLACK);

    DrawText("Network Prediction", 500, topRowY, 20, BLACK);
    DrawOutputBars(network, currentLabel, 400, topRowY + 30, 550, 224);
    DrawRectangleLines(400, topRowY + 30, 550, 224, BLACK);

    int bottomRowY = 500;
    DrawText("Network Architecture (Sampled)", 20, bottomRowY, 20, BLACK);
    DrawNetworkArchitecture(network, 50, bottomRowY + 30, screenWidth - 100, 250);

    EndDrawing();
}

int main()
{

    const int screenWidth = 1000;
    const int screenHeight = 800;
    InitWindow(screenWidth, screenHeight, "Raylib - MNIST Neural Network Training");
    SetTargetFPS(60);

    std::cout << "Loading MNIST dataset..." << std::endl;
    auto train_images = load_mnist_images("data/train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels("data/train-labels.idx1-ubyte");

    if (train_images.empty() || train_labels.empty())
    {
        std::cerr << "Failed to load dataset. Make sure the 'data' folder exists and contains the unzipped MNIST files." << std::endl;
        CloseWindow();
        return -1;
    }
    std::cout << "Dataset loaded successfully. " << train_images.size() << " images." << std::endl;

    const int input_size = 28 * 28;
    const int hidden_size = 100;
    const int output_size = 10;
    NeuralNetwork network = CreateNetwork({input_size, hidden_size, output_size});

    unsigned char *digitPixelData = new unsigned char[28 * 28 * 4];
    Image digitImage = {digitPixelData, 28, 28, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    Texture2D digitTexture = LoadTextureFromImage(digitImage);

    float learningRate = 0.1f;
    int epochs = 5;
    int currentEpoch = 1;
    int currentImageIndex = 0;
    float currentLoss = 0.0f;

    int correct_predictions_in_epoch = 0;
    float currentAccuracy = 0.0f;

    while (!WindowShouldClose())
    {
        if (currentEpoch <= epochs)
        {

            const auto &image = train_images[currentImageIndex];
            int label = train_labels[currentImageIndex];
            auto target = to_one_hot(label, output_size);

            FeedForward(network, image);
            Backpropagate(network, target);
            UpdateWeights(network, learningRate);

            currentLoss = calculate_mse(network.layers.back(), target);

            int predicted_label = get_prediction(network);
            if (predicted_label == label)
            {
                correct_predictions_in_epoch++;
            }
            currentAccuracy = (float)correct_predictions_in_epoch / (currentImageIndex % train_images.size() + 1);

            for (int i = 0; i < 28 * 28; i++)
            {
                unsigned char pixel_val = (unsigned char)(image[i] * 255.0f);
                digitPixelData[i * 4 + 0] = pixel_val;
                digitPixelData[i * 4 + 1] = pixel_val;
                digitPixelData[i * 4 + 2] = pixel_val;
                digitPixelData[i * 4 + 3] = 255;
            }
            UpdateTexture(digitTexture, digitPixelData);

            DrawUI(network, digitTexture, label, currentEpoch, currentImageIndex + 1, currentLoss, currentAccuracy, screenWidth, screenHeight);

            currentImageIndex++;
            if (currentImageIndex >= train_images.size())
            {
                std::cout << "Epoch " << currentEpoch << " finished. Accuracy: " << std::fixed << std::setprecision(2) << currentAccuracy * 100.0f << "%" << std::endl;
                currentImageIndex = 0;
                currentEpoch++;

                correct_predictions_in_epoch = 0;
                currentAccuracy = 0.0f;

                if (currentEpoch <= epochs)
                {
                    std::cout << "Starting Epoch " << currentEpoch << std::endl;
                }
            }
        }
        else
        {

            BeginDrawing();
            ClearBackground(RAYWHITE);
            DrawText("Training Complete!", screenWidth / 2 - 250, screenHeight / 2 - 40, 50, DARKGREEN);
            DrawText(TextFormat("Final Accuracy: %.2f%%", currentAccuracy * 100.0f), screenWidth / 2 - 200, screenHeight / 2 + 20, 30, DARKGRAY);
            EndDrawing();
        }
    }

    UnloadTexture(digitTexture);
    delete[] digitPixelData;
    CloseWindow();
    return 0;
}