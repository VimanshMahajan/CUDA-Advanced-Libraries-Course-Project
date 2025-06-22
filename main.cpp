#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tiffio.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>

using namespace std;
namespace fs = std::filesystem;

// Initialize cuDNN and display GPU specs
__host__ cudnnHandle_t initializeCuDNN()
{
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    cout << "Total GPUs available: " << numGPUs << endl;

    cudaSetDevice(0); // Use GPU 0 by default
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    cout << "Using GPU with Compute Capability: " << props.major << "." << props.minor << endl;

    cudnnHandle_t cudnnHandle;
    cudnnCreate(&cudnnHandle);
    cout << "cuDNN handle created successfully." << endl;
    return cudnnHandle;
}

// Load a TIFF image, convert to RGB float tensor, and define a cuDNN tensor descriptor
__host__ tuple<cudnnTensorDescriptor_t, float*, int, int, int> loadTIFFImage(const char* filePath)
{
    TIFF* tiff = TIFFOpen(filePath, "r");
    if (!tiff) {
        cerr << "Error: Cannot open TIFF file." << endl;
        exit(1);
    }

    uint32_t width, height;
    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);

    size_t totalPixels = width * height;
    uint32_t* raster = (uint32_t*)_TIFFmalloc(totalPixels * sizeof(uint32_t));

    if (!raster || !TIFFReadRGBAImage(tiff, width, height, raster, 0)) {
        cerr << "Error reading or allocating TIFF image." << endl;
        exit(1);
    }

    TIFFClose(tiff);

    // Create tensor descriptor
    cudnnTensorDescriptor_t tensorDesc;
    cudnnCreateTensorDescriptor(&tensorDesc);
    cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 3, height, width);

    // Allocate and normalize RGB values
    float* inputData;
    cudaMallocManaged(&inputData, totalPixels * 3 * sizeof(float));
    for (size_t i = 0; i < totalPixels; ++i) {
        inputData[i * 3 + 0] = TIFFGetR(raster[i]) / 255.0f;
        inputData[i * 3 + 1] = TIFFGetG(raster[i]) / 255.0f;
        inputData[i * 3 + 2] = TIFFGetB(raster[i]) / 255.0f;
    }

    _TIFFfree(raster);
    return {tensorDesc, inputData, 1, static_cast<int>(height), static_cast<int>(width)};
}

// Dummy model that generates random class scores
__host__ float* mockModelForward(cudnnHandle_t handle, cudnnTensorDescriptor_t inputDesc, float* inputData, int numClasses)
{
cudnnTensorDescriptor_t outputDesc;
cudnnCreateTensorDescriptor(&outputDesc);
cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, numClasses, 1, 1);

float* outputData;
cudaMallocManaged(&outputData, numClasses * sizeof(float));

// Fill with random values (mock behavior)
for (int i = 0; i < numClasses; ++i) {
outputData[i] = static_cast<float>(rand()) / RAND_MAX;
}

return outputData;
}

// Print predicted class and class probabilities to output file
__host__ void logPrediction(ofstream& outFile, const string& imagePath, float* classProbs, int numClasses)
{
    int predictedClass = max_element(classProbs, classProbs + numClasses) - classProbs;
    outFile << "File: " << imagePath << "\n";
    outFile << "Predicted class: " << predictedClass << "\n";
    outFile << "Class probabilities: ";
    for (int i = 0; i < numClasses; ++i)
        outFile << classProbs[i] << " ";
    outFile << "\n\n";
}

// Main execution logic
int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_directory> <output_file.txt>" << endl;
        return 1;
    }

    string inputDir = argv[1];
    string outputPath = argv[2];
    const int numClasses = 10;

    cudnnHandle_t cudnnHandle = initializeCuDNN();
    ofstream outFile(outputPath);

    // Process all .tif/.tiff files
    for (const auto& file : fs::directory_iterator(inputDir)) {
        if (file.path().extension() == ".tif" || file.path().extension() == ".tiff") {
            auto [tensorDesc, inputData, n, h, w] = loadTIFFImage(file.path().c_str());
            float* output = mockModelForward(cudnnHandle, tensorDesc, inputData, numClasses);
            logPrediction(outFile, file.path().string(), output, numClasses);

            cudaFree(inputData);
            cudaFree(output);
        }
    }

    cudnnDestroy(cudnnHandle);
    outFile.close();
    cout << "Classification complete. cuDNN handle destroyed." << endl;
    return 0;
}
