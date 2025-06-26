#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "tMatchSeq.h"

using namespace std::chrono;

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 32

// Kernel per calcolare la somma cumulativa per riga
__global__ void rowCumSum(float *image, float *rowSum, int width, int height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height)
        return;

    float sum = 0;
    for (int x = 0; x < width; x++)
    {
        sum += image[INDEX(x, y, width)];
        rowSum[INDEX(x, y, width)] = sum;
    }
}

// Kernel per calcolare la somma cumulativa per colonna
__global__ void colCumSum(float *rowSum, float *imageSum, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width)
        return;

    float sum = 0;
    for (int y = 0; y < height; y++)
    {
        sum += rowSum[INDEX(x, y, width)];
        imageSum[INDEX(x, y, width)] = sum;
    }
}

// Funzione device per calcolare la somma in una regione usando la sum table
__device__ int getRegionSum(const float *sumTable, int width, int height, int x, int y, int kx, int ky)
{
    int x1 = x - 1;
    int y1 = y - 1;
    int x2 = min(x + kx - 1, width - 1);
    int y2 = min(y + ky - 1, height - 1);

    float A = (x1 >= 0 && y1 >= 0) ? sumTable[INDEX(x1, y1, width)] : 0.0f;
    float B = (y1 >= 0) ? sumTable[INDEX(x2, y1, width)] : 0.0f;
    float C = (x1 >= 0) ? sumTable[INDEX(x1, y2, width)] : 0.0f;
    float D = sumTable[INDEX(x2, y2, width)];

    return D - B - C + A;
}

// Kernel CUDA calcolare SSD
__global__ void computeSSD(
    float *imageSqSum,    // Immagine integrale dei quadrati
    float *templateSqSum, // Somma dei quadrati dei pixel del template
    int width,            // x immagine
    int height,           // y immagine
    int kx,               // x template
    int ky,               // y template
    float *ssdResult,
    float *crossCorrelation,
    int paddedCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - kx + 1 || y >= height - ky + 1)
        return;

    // Calcolo delle somme usando le immagini integrali
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky); // Somma dei quadrati
    float SC = crossCorrelation[INDEX(x, y, paddedCols)];
    // Calcolo SSD 
    float ssd = S2 - 2 * SC + (*templateSqSum);

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}
__global__ void multiply(
    float *image, 
    int width,
    int height,
    float *d_imageSq)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width + 1 || y >= height + 1)
        return;

    d_imageSq[INDEX(x, y, width)] = image[INDEX(x, y, width)] * image[INDEX(x, y, width)];
}

// Kernel per padding a zero
__global__ void padToZero(const float *src, float *dst,
                          int srcRows, int srcCols,
                          int dstRows, int dstCols)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= dstRows || x >= dstCols)
        return;
    int dstIdx = y * dstCols + x;
    if (y < srcRows && x < srcCols)
    {
        dst[dstIdx] = src[y * srcCols + x];
    }
    else
    {
        dst[dstIdx] = 0.0f;
    }
}

// Kernel moltiplicazione complessa coniugata
__global__ void mulConjAndScale(cufftComplex *imageF,
                                const cufftComplex *kernelF,
                                int rows, int colsFreq)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows || x >= colsFreq)
        return;
    int idx = y * colsFreq + x;
    cufftComplex A = imageF[idx];
    cufftComplex B = kernelF[idx];
    // coniugato di B
    B.y = -B.y;
    // moltiplicazione

    cufftComplex C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.x * B.y + A.y * B.x;
    imageF[idx] = C;
}

// Kernel normalizzazione
__global__ void normalize(float *data, int totalSize, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
        data[idx] *= scale;
}

// Funzione principale per il template matching
cudaError_t templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLoc)
{
    // Conversione delle immagini in float con valori normalizzati tra 0 e 1
    cv::Mat imageN, templN;
    image.convertTo(imageN, CV_32F, 1.0 / 255.0);
    templ.convertTo(templN, CV_32F, 1.0 / 255.0);

    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    // dimensioni ottimali FFT
    int m = cv::getOptimalDFTSize(height + ky - 1);
    int n = cv::getOptimalDFTSize(width + kx - 1);
    int freqCols = n / 2 + 1;
    size_t realSize = sizeof(float) * m * n;
    size_t freqSize = sizeof(cufftComplex) * m * freqCols;

    float *d_imgPad, *d_tmpPad;
    cufftComplex *d_imgFreq, *d_tmpFreq;
    // Calcolo delle somme del template
    float templateSqSum = 0;

    float *d_image, *d_imageSqSum, *d_ssdResult, *d_imageSq, *d_rowSqSum, *d_templ, *d_templSq, *d_TrowSqSum, *d_templSqSum, *d_crossCorrelation, *d_templateSqSumLV;

    size_t templSize = kx * ky * sizeof(float);
    size_t imageSize = width * height * sizeof(float);
    size_t resultSize = (width - kx + 1) * (height - ky + 1) * sizeof(float);

    cudaError_t cudaStatus;

    // Allocazione memoria su device
    cudaStatus = cudaMalloc(&d_imgPad, realSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_tmpPad, realSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_imgFreq, freqSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_tmpFreq, freqSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_image, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_imageSqSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_imageSq, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_ssdResult, resultSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_crossCorrelation, resultSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_rowSqSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templ, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSq, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSqSum, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_TrowSqSum, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templateSqSumLV, sizeof(float));
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Copia immagine su device
    cudaStatus = cudaMemcpy(d_image, imageN.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMemcpy(d_templ, templN.ptr<float>(), templSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Calcolo delle immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calcolo immagini integrali
    // Immagine
    multiply<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_imageSq, d_rowSqSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSqSum, d_imageSqSum, width, height);

    dim3 gridSizeT(
        (kx + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (ky + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Template
    multiply<<<gridSizeT, blockSize>>>(d_templ, kx, ky, d_templSq);
    rowCumSum<<<(ky + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_templSq, d_TrowSqSum, kx, ky);
    colCumSum<<<(kx + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_TrowSqSum, d_templSqSum, kx, ky);

    // Copia del last value del template syl davice
    size_t offset = ((ky - 1) * kx + (kx - 1)) * sizeof(float);
    cudaMemcpy(d_templateSqSumLV,
               (char *)d_templSqSum + offset,
               sizeof(float),
               cudaMemcpyDeviceToDevice);

    // calcolo matrice cross correlation
    auto start_cross = high_resolution_clock::now();
    
    // Pad to zero
    dim3 b1(16, 16), g1((n + 15) / 16, (m + 15) / 16);
    padToZero<<<g1, b1>>>(d_image, d_imgPad, height, width, m, n);
    padToZero<<<g1, b1>>>(d_templ, d_tmpPad, ky, kx, m, n);

    // cuFFT plan
    cufftHandle planFwd, planInv;
    cufftPlan2d(&planFwd, m, n, CUFFT_R2C);
    cufftPlan2d(&planInv, m, n, CUFFT_C2R);

    // FFT 
    cufftExecR2C(planFwd, d_imgPad, d_imgFreq);
    cufftExecR2C(planFwd, d_tmpPad, d_tmpFreq);

    // Multiply with conjugate
    dim3 b2(16, 16), g2((freqCols + 15) / 16, (m + 15) / 16);
    mulConjAndScale<<<g2, b2>>>(d_imgFreq, d_tmpFreq, m, freqCols);

    //  FFT inversa
    cufftExecC2R(planInv, d_imgFreq, d_imgPad);

    // Normalize
    int totalReal = m * n;
    normalize<<<(totalReal + 255) / 256, 256>>>(d_imgPad, totalReal, 1.0f / (m * n));

    // auto start_cross = high_resolution_clock::now();
    // cv::Mat crossCorrelation;
    // crossCorrelation = crossCorrelationFFT(imageN, templN);
    auto end_cross = high_resolution_clock::now();
    auto duration_cross = duration_cast<milliseconds>(end_cross - start_cross);
    std::cout << "Tempo cross correlation (ms): " << duration_cross.count() << std::endl;

    // cudaStatus = cudaMemcpy(d_crossCorrelation, crossCorrelation.ptr<float>(), resultSize, cudaMemcpyHostToDevice);
    // if (cudaStatus != cudaSuccess)
    //     return cudaStatus;

    // Calcolo SSD
    dim3 gridSSDSize((width - kx + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height - ky + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeSSD<<<gridSSDSize, blockSize>>>(
        d_imageSqSum,
        d_templateSqSumLV,
        width,
        height,
        kx,
        ky,
        d_ssdResult,
        d_imgPad,
        n);

    // Copia risultati su host
    cv::Mat ssdResult(height - ky + 1, width - kx + 1, CV_32F);
    cudaStatus = cudaMemcpy(ssdResult.ptr<float>(), d_ssdResult, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Trova la posizione del minimo SSD
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(ssdResult, &minVal, &maxVal, &minLoc, &maxLoc);
    *bestLoc = minLoc;

    // Cleanup
    cufftDestroy(planFwd);
    cufftDestroy(planInv);
    cudaFree(d_imgPad);
    cudaFree(d_tmpPad);
    cudaFree(d_imgFreq);
    cudaFree(d_tmpFreq);
    cudaFree(d_image);
    cudaFree(d_imageSqSum);
    cudaFree(d_ssdResult);
    cudaFree(d_imageSq);
    cudaFree(d_rowSqSum);
    cudaFree(d_crossCorrelation);
    cudaFree(d_TrowSqSum);
    cudaFree(d_templateSqSumLV);

    return cudaStatus;
}

int main()
{
    // Carica le immagini
    cv::Mat image = cv::imread("immagini/sourceC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("immagini/sourceC.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("immagini/templateC.jpg", cv::IMREAD_GRAYSCALE);
    float scaleFactor = 1; // Fattore di ridimensionamento
    cv::Mat imageR, templateR;
    cv::resize(image, imageR, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    cv::resize(templ, templateR, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    if (image.empty() || templ.empty())
    {
        printf("Errore nel caricamento delle immagini\n");
        return -1;
    }

    cv::Point bestLoc;
    cv::Point bestLocSeq;
    cv::Point LocCheck;

    auto start_gpu = high_resolution_clock::now();
    cudaError_t status = templateMatchingSSD(imageR, templateR, &bestLoc);
    cudaDeviceSynchronize();
    auto end_gpu = high_resolution_clock::now();
    auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu);
    std::cout << "Tempo GPU (ms): " << duration_gpu.count() << std::endl;

    auto start_seq = high_resolution_clock::now();
    seq_templateMatchingSSD(imageR, templateR, &LocCheck);
    auto end_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(end_seq - start_seq);
    std::cout << "Tempo CPU (ms): " << duration_seq.count() << std::endl;

    if (status != cudaSuccess)
    {
        printf("Errore CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }

    // Disegna un rettangolo intorno al match trovato

    // risultato template matching BLUE
    LocCheck.x = static_cast<int>(LocCheck.x) / scaleFactor;
    LocCheck.y = static_cast<int>(LocCheck.y) / scaleFactor;
    cv::Rect matchRectTM(LocCheck, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRectTM, cv::Scalar(255, 0, 0), 3);

    // risultato sequenziale  ROSSO
    // bestLocSeq.x = static_cast<int>(bestLocSeq.x) / scaleFactor;
    // bestLocSeq.y = static_cast<int>(bestLocSeq.y) / scaleFactor;
    // cv::Rect matchRectSeq(bestLocSeq, cv::Size(templ.cols, templ.rows));
    // rectangle(imageColor, matchRectSeq, cv::Scalar(0, 0, 255), 3);

    // risultato cuda VERDE
    bestLoc.x = static_cast<int>(bestLoc.x) / scaleFactor;
    bestLoc.y = static_cast<int>(bestLoc.y) / scaleFactor;
    cv::Rect matchRect(bestLoc, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRect, cv::Scalar(0, 255, 0), 2);

    // Mostra il risultato
    cv::imwrite("Result.jpg", imageColor);
    // cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    // imshow("Immagine principale", imageColor);
    // cv::waitKey(0);

    return 0;
}