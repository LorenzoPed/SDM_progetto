/*
MAIN COMPLETO E PULITO
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include "tMatchSeq.h"

using namespace std::chrono;

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 32

// PARAMETRI PER PREFIX SUM

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define SHARED_MEM_SIZE 2048

// Parametri per transpose ottimizzato
#define TILE_DIM 32
#define BLOCK_ROWS 8

// ============================================================================
// FASE 1: PREFIX SUM INTRA-WARP (più efficiente)
// ============================================================================

__device__ __forceinline__ float warpPrefixSum(float val) {
    // Algoritmo di prefix sum dentro un warp (32 thread)
    // Sfrutta __shfl_up_sync per comunicazione veloce tra thread
    
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x >= offset) {
            val += temp;
        }
    }
    return val;
}

// ============================================================================
// FASE 2: PREFIX SUM INTRA-BLOCK (usa shared memory)
// ============================================================================

__device__ float blockPrefixSum(float val, float* sharedMem) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    
    // Fase 1: Prefix sum dentro ogni warp
    float warpSum = warpPrefixSum(val);
    
    // Il thread finale di ogni warp salva la somma totale del warp
    if (laneId == WARP_SIZE - 1) {
        sharedMem[warpId] = warpSum;
    }
    __syncthreads();
    
    // Fase 2: Prefix sum delle somme dei warp (solo il primo warp lavora)
    if (warpId == 0) {
        float warpTotal = (tid < blockDim.x / WARP_SIZE) ? sharedMem[tid] : 0.0f;
        warpTotal = warpPrefixSum(warpTotal);
        sharedMem[tid] = warpTotal;
    }
    __syncthreads();
    
    // Fase 3: Aggiungi l'offset dei warp precedenti
    float blockOffset = (warpId > 0) ? sharedMem[warpId - 1] : 0.0f;
    return warpSum + blockOffset;
}

// ============================================================================
// KERNEL OTTIMIZZATO: PREFIX SUM PER RIGHE (COALESCENTE)
// ============================================================================

__global__ void optimizedRowPrefixSum(float* input, float* output, 
                                     float* blockSums, int width, int height) {
    // Configurazione: ogni blocco gestisce una riga
    // Thread accedono in modo coalescente
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height) return;
    
    __shared__ float sharedData[MAX_BLOCK_SIZE];
    __shared__ float warpSums[MAX_BLOCK_SIZE / WARP_SIZE];
    
    // Carica dati in shared memory (accesso coalescente)
    float val = 0.0f;
    if (col < width) {
        val = input[row * width + col];  // Accesso coalescente!
    }
    
    // Prefix sum all'interno del blocco
    float prefixSum = blockPrefixSum(val, warpSums);
    
    // Scrivi risultato (accesso coalescente)
    if (col < width) {
        output[row * width + col] = prefixSum;
    }
    
    // Salva la somma finale del blocco per la fase successiva
    if (threadIdx.x == blockDim.x - 1 && col < width) {
        int blockId = row * gridDim.x + blockIdx.x;
        blockSums[blockId] = prefixSum;
    }
}

// ============================================================================
// KERNEL: APPLICA OFFSET FINALE
// ============================================================================

__global__ void addBlockOffsets(float* data, float* blockSums, 
                               int width, int height) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height || col >= width) return;
    
    // Calcola l'offset cumulativo dai blocchi precedenti nella stessa riga
    float offset = 0.0f;
    int baseBlockId = row * gridDim.x;
    
    for (int i = 0; i < blockIdx.x; i++) {
        offset += blockSums[baseBlockId + i];
    }
    
    // Applica l'offset
    data[row * width + col] += offset;
}

// ============================================================================
// KERNEL: PREFIX SUM PER COLONNE (TRANSPOSE + PREFIX SUM)
// ============================================================================

__global__ void optimizedTranspose(float* input, float* output, 
                                  int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 per evitare bank conflict
    
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    
    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;
    
    // Carica tile in shared memory
    if (x < width) {
        for (int j = 0; j < TILE_DIM; j += blockDim.y) {
            if (y + j < height) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
            }
        }
    }
    __syncthreads();
    
    // Calcola coordinate per output trasposto
    x = blockIdx_y * TILE_DIM + threadIdx.x;
    y = blockIdx_x * TILE_DIM + threadIdx.y;
    
    // Scrivi tile trasposto
    if (x < height) {
        for (int j = 0; j < TILE_DIM; j += blockDim.y) {
            if (y + j < width) {
                output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// ============================================================================
// FUNZIONE PRINCIPALE: CALCOLO IMMAGINE INTEGRALE OTTIMIZZATA
// ============================================================================

cudaError_t computeIntegralImageOptimized(float* d_input, float* d_output,
                                         int width, int height) {
    
    // Allocazioni temporanee
    float* d_temp = nullptr;
    float* d_blockSums = nullptr;
    float* d_transposed = nullptr;
    
    size_t imageSize = width * height * sizeof(float);
    
    // Calcola dimensioni grid ottimali
    int threadsPerBlock = 256;  // Multiplo di warp size
    int blocksPerRow = (width + threadsPerBlock - 1) / threadsPerBlock;
    int totalBlocks = height * blocksPerRow;
    
    // Allocazioni
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_blockSums, totalBlocks * sizeof(float));
    cudaMalloc(&d_transposed, imageSize);
    
    // === FASE 1: PREFIX SUM PER RIGHE ===
    
    dim3 blockSize1(threadsPerBlock);
    dim3 gridSize1(blocksPerRow, height);
    
    // Primo passo: prefix sum locale per blocchi
    optimizedRowPrefixSum<<<gridSize1, blockSize1>>>(
        d_input, d_temp, d_blockSums, width, height);
    
    // Secondo passo: calcola prefix sum degli offset dei blocchi
    // (Questo può essere fatto in modo ricorsivo per immagini molto larghe)
    
    // Terzo passo: applica gli offset
    addBlockOffsets<<<gridSize1, blockSize1>>>(
        d_temp, d_blockSums, width, height);
    
    // === FASE 2: TRANSPOSE + PREFIX SUM PER COLONNE ===
    
    // Trasponi l'immagine
    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid((width + TILE_DIM - 1) / TILE_DIM,
                      (height + TILE_DIM - 1) / TILE_DIM);
    
    optimizedTranspose<<<transposeGrid, transposeBlock>>>(
        d_temp, d_transposed, width, height);
    
    // Ricalcola parametri per l'immagine trasposta
    int blocksPerCol = (height + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridSize2(blocksPerCol, width);
    
    // Prefix sum sulle "righe" dell'immagine trasposta (= colonne originali)
    optimizedRowPrefixSum<<<gridSize2, blockSize1>>>(
        d_transposed, d_temp, d_blockSums, height, width);  // Nota: dimensioni scambiate
    
    addBlockOffsets<<<gridSize2, blockSize1>>>(
        d_temp, d_blockSums, height, width);
    
    // Trasponi di nuovo per ottenere il risultato finale
    dim3 transposeGrid2((height + TILE_DIM - 1) / TILE_DIM,
                       (width + TILE_DIM - 1) / TILE_DIM);
    
    optimizedTranspose<<<transposeGrid2, transposeBlock>>>(
        d_temp, d_output, height, width);  // Dimensioni scambiate
    
    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_blockSums);
    cudaFree(d_transposed);
    
    return cudaGetLastError();
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
    float *crossCorrelation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - kx + 1 || y >= height - ky + 1)
        return;

    // Calcolo delle somme usando le immagini integrali
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky); // Somma dei quadrati
    float SC = crossCorrelation[INDEX(x, y, width - kx + 1)];
    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * SC + (*templateSqSum);

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}
__global__ void multiply(
    float *image, // Immagine sorgente
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

    // Calcolo delle somme del template
    float templateSqSum = 0;

    float *d_image, *d_imageSqSum, *d_ssdResult, *d_imageSq, *d_rowSqSum, *d_templ, *d_templSq, *d_TrowSqSum, *d_templSqSum, *d_crossCorrelation, *d_templateSqSumLV;

    size_t templSize = kx * ky * sizeof(float);
    size_t imageSize = width * height * sizeof(float);
    size_t resultSize = (width - kx + 1) * (height - ky + 1) * sizeof(float);

    cudaError_t cudaStatus;

    // Allocazione memoria su device
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
    computeIntegralImageOptimized(d_imageSq, d_imageSqSum, width, height);

    dim3 gridSizeT(
        (kx + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (ky + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Template
    multiply<<<gridSizeT, blockSize>>>(d_templ, kx, ky, d_templSq);
    computeIntegralImageOptimized(d_templSq, d_templSqSum, width, height);

    // Copia del last value del template syl davice
    size_t offset = ((ky - 1) * kx + (kx - 1)) * sizeof(float);
    cudaMemcpy(d_templateSqSumLV,
               (char *)d_templSqSum + offset,
               sizeof(float),
               cudaMemcpyDeviceToDevice);

    // calcolo matrice cross correlation
    cv::Mat crossCorrelation;
    crossCorrelation = crossCorrelationFFT(imageN, templN);

    cudaStatus = cudaMemcpy(d_crossCorrelation, crossCorrelation.ptr<float>(), resultSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

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
        d_crossCorrelation);

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
    cv::Mat image = cv::imread("sourceC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("sourceC.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("templateC.jpg", cv::IMREAD_GRAYSCALE);
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