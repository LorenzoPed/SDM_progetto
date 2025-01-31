#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 16

// Kernel per calcolare la somma cumulativa per riga
__global__ void rowCumSum(float *image, float *rowSum, int width, int height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;
    
    float sum = 0;
    for (int x = 0; x < width; x++) {
        sum += image[INDEX(x, y, width)];
        rowSum[INDEX(x, y, width)] = sum;
    }
}

// Kernel per calcolare la somma cumulativa per colonna
__global__ void colCumSum(float *rowSum, float *sumTable, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;
    
    float sum = 0;
    for (int y = 0; y < height; y++) {
        sum += rowSum[INDEX(x, y, width)];
        sumTable[INDEX(x, y, width)] = sum;
    }
}

// Funzione device per calcolare la somma in una regione usando la sum table
__device__ float getRegionSum(float *sumTable, int width, int x, int y, int kx, int ky) {
    x--; y--;
    float S = sumTable[INDEX(x + kx, y + ky, width)]
             - (x >= 0 ? sumTable[INDEX(x, y + ky, width)] : 0)
             - (y >= 0 ? sumTable[INDEX(x + kx, y, width)] : 0)
             + (x >= 0 && y >= 0 ? sumTable[INDEX(x, y, width)] : 0);
    return S;
}

// Kernel CUDA ottimizzato per calcolare SSD
__global__ void computeSSD(
    float *imageSum,      // Immagine integrale della sorgente
    float *imageSqSum,    // Immagine integrale dei quadrati
    float templateSum,    // Somma dei pixel del template
    float templateSqSum,  // Somma dei quadrati dei pixel del template
    int width,
    int height,
    int kx,
    int ky,
    float *ssdResult
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width - kx + 1 || y >= height - ky + 1) return;
    
    // Calcolo delle somme usando le immagini integrali
    float S1 = getRegionSum(imageSum, width, x, y, kx, ky);      // Somma della regione immagine
    float S2 = getRegionSum(imageSqSum, width, x, y, kx, ky);    // Somma dei quadrati
    
    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * (S1 * templateSum) + templateSqSum;
    
    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}

// Funzione principale per il template matching
cudaError_t templateMatchingSSD(
    const cv::Mat& image,
    const cv::Mat& templ,
    cv::Point* bestLoc
) {
    // Conversione delle immagini in float
    cv::Mat imageFloat, templFloat;
    image.convertTo(imageFloat, CV_32F);
    templ.convertTo(templFloat, CV_32F);
    
    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;
    
    // Calcolo delle somme del template
    float templateSum = 0;
    float templateSqSum = 0;
    for (int i = 0; i < kx * ky; i++) {
        float val = templFloat.ptr<float>()[i];
        templateSum += val;
        templateSqSum += val * val;
    }
    
    // Allocazione memoria su device
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult;
    size_t imageSize = width * height * sizeof(float);
    size_t resultSize = (width - kx + 1) * (height - ky + 1) * sizeof(float);
    
    cudaError_t cudaStatus;
    
    // Allocazione memoria su device
    cudaStatus = cudaMalloc(&d_image, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    cudaStatus = cudaMalloc(&d_imageSum, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    cudaStatus = cudaMalloc(&d_imageSqSum, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    cudaStatus = cudaMalloc(&d_ssdResult, resultSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    // Copia immagine su device e creazione immagine dei quadrati
    cudaStatus = cudaMemcpy(d_image, imageFloat.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    // Creazione dell'immagine dei quadrati
    cv::Mat imageSq;
    cv::multiply(imageFloat, imageFloat, imageSq);
    cudaStatus = cudaMemcpy(d_imageSqSum, imageSq.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    // Calcolo delle immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width - kx + BLOCK_SIZE) / BLOCK_SIZE,
        (height - ky + BLOCK_SIZE) / BLOCK_SIZE
    );
    
    

    // Calcolo immagini integrali
    rowCumSum<<<(height+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_image, d_imageSum, width, height);
    colCumSum<<<(width+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_imageSum, d_imageSum, width, height);
    
    rowCumSum<<<(height+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_imageSqSum, d_imageSqSum, width, height);
    colCumSum<<<(width+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(d_imageSqSum, d_imageSqSum, width, height);
    
    // Calcolo SSD
    computeSSD<<<gridSize, blockSize>>>(
        d_imageSum,
        d_imageSqSum,
        templateSum,
        templateSqSum,
        width,
        height,
        kx,
        ky,
        d_ssdResult
    );
    
    // Copia risultati su host
    cv::Mat ssdResult(height - ky + 1, width - kx + 1, CV_32F);
    cudaStatus = cudaMemcpy(ssdResult.ptr<float>(), d_ssdResult, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) return cudaStatus;
    
    // Trova la posizione del minimo SSD
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(ssdResult, &minVal, &maxVal, &minLoc, &maxLoc);
    *bestLoc = minLoc;
    
    // Cleanup
    cudaFree(d_image);
    cudaFree(d_imageSum);
    cudaFree(d_imageSqSum);
    cudaFree(d_ssdResult);
    
    return cudaStatus;
}

int main() {
    // Carica le immagini
    cv::Mat image = cv::imread("source.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread("template.jpg", cv::IMREAD_GRAYSCALE);
    
    if (image.empty() || templ.empty()) {
        printf("Errore nel caricamento delle immagini\n");
        return -1;
    }
    
    cv::Point bestLoc;
    cudaError_t status = templateMatchingSSD(image, templ, &bestLoc);
    
    if (status != cudaSuccess) {
        printf("Errore CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }
    
    // Disegna un rettangolo intorno al match trovato
    cv::rectangle(
        image,
        bestLoc,
        cv::Point(bestLoc.x + templ.cols, bestLoc.y + templ.rows),
        cv::Scalar(0, 0, 255),
        2
    );
    
    // Mostra il risultato
    cv::imwrite("Result.jpg", image);
    cv::waitKey(0);
    
    return 0;
}