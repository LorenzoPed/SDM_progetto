#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "tMatchSeq.h"

using namespace std::chrono;

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 32
#define BLUE_BLOCK_SIZE 16

// PARAMETRI PER PREFIX SUM

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define SHARED_MEM_SIZE 2048

// Parametri per transpose ottimizzato
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Per blue pixel
#define OPTIMIZED_BLOCK_SIZE 256  // Multiplo di warp_size per migliore occupancy


// ============================================================================
// FASE 1: PREFIX SUM INTRA-WARP (più efficiente)
// ============================================================================

__device__ __forceinline__ float warpPrefixSum(float val) {
    // Algoritmo di prefix sum dentro un warp (32 thread)
    // Sfrutta __shfl_up_sync, permette ad un thread di accedere direttamente al valore
    // di un altro thread senza usare mem condivisa
    
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset); 
        /*  T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);
            mask: specifica i thread che partecipano alla comunicazione: 0xFFFFFFFF significa che tutti i thread partecipano
            var: il dato condiviso dal thread
            delta: num di posizioni verso il basso da cui prelevare
            width: opzionale, indica la dim del sotto warp */
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
    // Se sono al thread finale di un warp (es: T31) -> laneId = 31 % 32 = 31 (0 con resto 31), allo stesso modo per i successivi.
    if (laneId == WARP_SIZE - 1) {
        sharedMem[warpId] = warpSum;
    }
    __syncthreads();
    
    // Fase 2: Prefix sum delle somme dei warp (solo il primo warp lavora)
    // Creaiamo l'offset tra i warp 
    if (warpId == 0) {
        float warpTotal = (tid < blockDim.x / WARP_SIZE) ? sharedMem[tid] : 0.0f;
        warpTotal = warpPrefixSum(warpTotal);
        sharedMem[tid] = warpTotal;
    }
    __syncthreads();
    
    // Fase 3: Aggiungi l'offset dei warp precedenti ai thread successivi
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
    
    //__shared__ float sharedData[MAX_BLOCK_SIZE];
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

    // Alla fine di questa funzione ho le prefix sum di tutti i blocchi
    // La fase successiva è uguale alla fase 3 di blockPrefixSum
    // vado ad aggiungere ai thread di TUTTI i blocchi (tranne lo 0), l offset calcolato
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
    
    // Evita race conditions calcolando l'offset solo per i blocchi necessari
    if (blockIdx.x > 0) {
        for (int i = 0; i < blockIdx.x; i++) {
            offset += blockSums[baseBlockId + i];
        }
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
    float B = (y1 >= 0 && x2 < width && y1 < height) ? sumTable[INDEX(x2, y1, width)] : 0.0f;
    float C = (x1 >= 0 && x1 < width && y2 < height) ? sumTable[INDEX(x1, y2, width)] : 0.0f;
    float D = (x2 < width && y2 < height) ? sumTable[INDEX(x2, y2, width)] : 0.0f;

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
    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * SC + (*templateSqSum);

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}


__global__ void multiply_optimized(
    float *image, 
    int width,
    int height,
    float *d_imageSq)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = INDEX(x, y, width);
    
    // Usa cache L2 per il caricamento
    float val = __ldg(&image[idx]);
    d_imageSq[idx] = val * val;
}

// FUNZIONI PER FATTORE DI CROSSCORRELAZIONE

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
__global__ void mulConj(cufftComplex *imageF,
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
    // moltiplicazione complessa
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
    int m = cv::getOptimalDFTSize(height + ky - 1); // In pratica, si arrotonda alla potenza di 2 più vicina per efficienza nella FFT.
    int n = cv::getOptimalDFTSize(width + kx - 1);  // In particolare risulta più efficiente se i numeri altamente composti (sono fattorizzabili in 2,3 e 5). FFT -> tante DFT + semplici, più bassi sono i coefficienti (2,3,5) minori sono le operazioni aritmetiche
    int freqCols = n / 2 + 1;                       // La FFT di un segnale reale ha simmetria hermitiana, per risparmiare, quindi, la cuFFT memorizza solo metà dello spettro.                   
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
    multiply_optimized<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    computeIntegralImageOptimized(d_imageSq, d_imageSqSum, width, height);

    dim3 gridSizeT(
        (kx + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (ky + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Template
    multiply_optimized<<<gridSizeT, blockSize>>>(d_templ, kx, ky, d_templSq);
    computeIntegralImageOptimized(d_templSq, d_templSqSum, kx, ky);

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

    // FFT forward
    cufftExecR2C(planFwd, d_imgPad, d_imgFreq);
    cufftExecR2C(planFwd, d_tmpPad, d_tmpFreq);

    // Multiply with conjugate
    dim3 b2(16, 16), g2((freqCols + 15) / 16, (m + 15) / 16);
    mulConj<<<g2, b2>>>(d_imgFreq, d_tmpFreq, m, freqCols);

    // IFFT
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

// FUNZIONI PER COLOR FILTERING

// Funzione device per conversione BGR a HSV (Hue (tonalità), Saturation (saturazione), Value (luminosità)) corretta 
__device__ void bgrToHsv(unsigned char b, unsigned char g, unsigned char r, 
                        unsigned char& h, unsigned char& s, unsigned char& v) {
    float fb = b / 255.0f;
    float fg = g / 255.0f;
    float fr = r / 255.0f;
    
    float maxVal = fmaxf(fb, fmaxf(fg, fr));
    float minVal = fminf(fb, fminf(fg, fr));
    float delta = maxVal - minVal;
    
    // Value
    v = (unsigned char)(maxVal * 255);
    
    // Saturation
    if (maxVal == 0) {
        s = 0;
    } else {
        s = (unsigned char)((delta / maxVal) * 255);
    }
    
    // Hue
    float hue = 0;
    if (delta != 0) {
        if (maxVal == fr) {
            hue = 60.0f * fmodf((fg - fb) / delta, 6.0f);
        } else if (maxVal == fg) {
            hue = 60.0f * ((fb - fr) / delta + 2.0f);
        } else {
            hue = 60.0f * ((fr - fg) / delta + 4.0f);
        }
    }
    
    if (hue < 0) hue += 360.0f;
    
    // Converti in range OpenCV (0-179)
    h = (unsigned char)(hue / 2.0f);
}

// Kernel per contare i pixel blu in un'area specifica
__global__ void countBluePixelsKernel(uchar3* image, int width, int height, 
                                     int startX, int startY, int roiWidth, int roiHeight,
                                     int* blueCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Converti alle coordinate dell'immagine completa
    int imgX = startX + x;
    int imgY = startY + y;

    // Controlla se siamo dentro l'ROI e dentro i bounds dell'immagine
    if (x < roiWidth && y < roiHeight && imgX < width && imgY < height) {
        int idx = imgY * width + imgX;
        uchar3 pixel = image[idx]; // BGR format (OpenCV standard)

        // Converti BGR to HSV usando la funzione corretta
        unsigned char h, s, v;
        bgrToHsv(pixel.x, pixel.y, pixel.z, h, s, v);

        // Condizione per il blu in HSV
        bool isBlue = (h >= BLUE_H_MIN && h <= BLUE_H_MAX) &&
                      (s >= BLUE_S_MIN) &&
                      (v >= BLUE_V_MIN);

        if (isBlue) {
            atomicAdd(blueCount, 1);
        } 
    }
}

__global__ void convertBGRtoUchar3Kernel_OptimalSimple(const uchar* cvImage, uchar3* cudaImage,
                                                       int width, int height, size_t step) {
    // Thread ID lineare per coalescenza garantita
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total_pixels = width * height;
    
    // Grid-stride loop per alta compute intensity
    for (int pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += stride) {
        // Coordinate 2D
        int y = pixel_idx / width;
        int x = pixel_idx % width;
        
        // Accesso ai dati
        int src_idx = y * step + x * 3;
        
        // Lettura e conversione
        cudaImage[pixel_idx] = make_uchar3(
            cvImage[src_idx],      // B
            cvImage[src_idx + 1],  // G
            cvImage[src_idx + 2]   // R
        );
    }
}

// Funzione per contare i pixel blu in una ROI
void countBlueInROI(cv::Mat& image, const cv::Rect& roi, int& blueCount) {
    // Verifica che l'ROI sia valido
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > image.cols || 
        roi.y + roi.height > image.rows) {
        std::cerr << "ROI fuori dai bounds dell'immagine!" << std::endl;
        return;
    }

    printf("Analizzando ROI: x=%d, y=%d, width=%d, height=%d\n", roi.x, roi.y, roi.width, roi.height);

    // Alloca memoria device
    uchar* d_cvImage = nullptr;
    uchar3* d_image = nullptr;
    int* d_blueCount = nullptr;


    size_t cvImageSize = image.rows * image.step;  // Usa step per gestire padding
    size_t uchar3ImageSize = image.rows * image.cols * sizeof(uchar3);
    
    // Alloca memoria per entrambi i formati
    cudaMalloc(&d_cvImage, cvImageSize);
    cudaMalloc(&d_image, uchar3ImageSize);
    cudaMalloc(&d_blueCount, sizeof(int));

    // Copia l'immagine OpenCV direttamente su device
    cudaMemcpy(d_cvImage, image.data, cvImageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_blueCount, 0, sizeof(int));

    
    int totalPixels = image.rows * image.cols;
    int blockSizePerConversion = 256;
    int gridSizePerConversion = std::min(65535, (totalPixels + blockSizePerConversion - 1) / blockSizePerConversion);

    convertBGRtoUchar3Kernel_OptimalSimple<<<gridSizePerConversion, blockSizePerConversion>>>(
        d_cvImage, d_image, image.cols, image.rows, image.step
    );


    // Configura kernel
    dim3 blockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 gridSize((roi.width + blockSize.x - 1) / blockSize.x, 
                 (roi.height + blockSize.y - 1) / blockSize.y);

    // Esegui kernel
    countBluePixelsKernel<<<gridSize, blockSize>>>(d_image, image.cols, image.rows,
                                                 roi.x, roi.y, roi.width, roi.height,
                                                 d_blueCount);

    // Controlla errori CUDA
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Errore kernel CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    cudaDeviceSynchronize();

    // Copia risultati
    cudaMemcpy(&blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Libera memoria
    cudaFree(d_image);
    cudaFree(d_blueCount);
}

int main()
{
    // Carica le immagini
    cv::Mat image = cv::imread("immagini/sourceC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("immagini/sourceC.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("immagini/templateC.jpg", cv::IMREAD_GRAYSCALE);


    if (image.empty() || templ.empty())
    {
        printf("Errore nel caricamento delle immagini\n");
        return -1;
    }

    cv::Point bestLoc;
    cv::Point LocCheck;

    printf("\n### INIZIO FASE 1: TEMPLATE MATCHING ###\n");

    auto start_gpu = high_resolution_clock::now();
    cudaError_t status = templateMatchingSSD(image, templ, &bestLoc);
    cudaDeviceSynchronize();
    auto end_gpu = high_resolution_clock::now();
    auto duration_gpu = duration_cast<microseconds>(end_gpu - start_gpu);

    auto start_seq = high_resolution_clock::now();
    seq_templateMatchingSSD(image, templ, &LocCheck);
    auto end_seq = high_resolution_clock::now();
    auto duration_seq = duration_cast<microseconds>(end_seq - start_seq);

    if (status != cudaSuccess)
    {
        printf("Errore CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }

    printf("\n### FASE 2: CONTEGGIO PIXEL BLU ###\n");

    // Crea ROI basata sul risultato del template matching CUDA e seq
    cv::Rect roi(bestLoc.x, bestLoc.y, templ.cols, templ.rows);
    cv::Rect seq_roi(LocCheck.x, LocCheck.y, templ.cols, templ.rows);
    
    // Verifica e correggi ROI se necessario
    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    roi.width = std::min(roi.width, imageColor.cols - roi.x);
    roi.height = std::min(roi.height, imageColor.rows - roi.y);

    // uguale per sequenziale
    seq_roi.x = std::max(0, seq_roi.x);
    seq_roi.y = std::max(0, seq_roi.y);
    seq_roi.width = std::min(seq_roi.width, imageColor.cols - seq_roi.x);
    seq_roi.height = std::min(seq_roi.height, imageColor.rows - seq_roi.y);
    
    printf("ROI per analisi blu CUDA: x=%d, y=%d, width=%d, height=%d\n", 
           roi.x, roi.y, roi.width, roi.height);

    printf("ROI per analisi blu SEQ: x=%d, y=%d, width=%d, height=%d\n", 
            seq_roi.x, seq_roi.y, seq_roi.width, seq_roi.height);
        
    // Conta pixel blu nell'ROI con CUDA
    int blueCount = 0;

    auto start_colorGpu = high_resolution_clock::now();
    countBlueInROI(imageColor, roi, blueCount);
    auto end_colorGpu = high_resolution_clock::now();
    auto duration_colorGPU = duration_cast<microseconds>(end_colorGpu - start_colorGpu);

    // Conta dei pixel blu nella ROI sequenziale
    auto start_colorSeq = high_resolution_clock::now();
    int seq_blue = seq_countBluePixels(imageColor, seq_roi.x, seq_roi.y, seq_roi.width, seq_roi.height);
    auto end_colorSeq = high_resolution_clock::now();
    auto duration_colorSeq = duration_cast<microseconds>(end_colorSeq - start_colorSeq);
    
    // Visualizza risultati Color Figltering CUDA
    printf("Pixel blu trovati con CUDA: %d\n", blueCount);
    printf("Area analizzata: %dx%d (%d pixel totali)\n", 
           roi.width, roi.height, roi.width * roi.height);
    printf("Percentuale pixel blu: %.2f%%\n", 
           (100.0 * blueCount) / (roi.width * roi.height));

    // Visualizza risultati Color Figltering SEQ
    printf("Pixel blu trovati con SEQ: %d\n", seq_blue);
    printf("Area analizzata: %dx%d (%d pixel totali)\n", 
           seq_roi.width, seq_roi.height, seq_roi.width * seq_roi.height);
    printf("Percentuale pixel blu: %.2f%%\n", 
           (100.0 * seq_blue) / (seq_roi.width * seq_roi.height));


    // Disegna un rettangolo intorno al match trovato

    // risultato template matching sequenziale BLUE
    LocCheck.x = static_cast<int>(LocCheck.x);
    LocCheck.y = static_cast<int>(LocCheck.y);
    cv::Rect matchRectTM(LocCheck, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRectTM, cv::Scalar(255, 0, 0), 3);

    // risultato cuda VERDE
    bestLoc.x = static_cast<int>(bestLoc.x);
    bestLoc.y = static_cast<int>(bestLoc.y);
    cv::Rect matchRect(bestLoc, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRect, cv::Scalar(0, 255, 0), 2);

    // Aggiungi testo con informazioni
    std::string infoText = "Blu: " + std::to_string(blueCount) + " px (" + 
                          std::to_string((int)((100.0 * blueCount) / (roi.width * roi.height))) + "%)";
    cv::putText(imageColor, infoText, cv::Point(bestLoc.x, bestLoc.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    std::string seq_infoText = "Blu: " + std::to_string(seq_blue) + " px (" + 
                          std::to_string((int)((100.0 * blueCount) / (roi.width * roi.height))) + "%)";
    cv::putText(imageColor, seq_infoText, cv::Point(LocCheck.x + 250, LocCheck.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    
    // STAMPE FINALI DEI TEMPI
    printf("\nGPU:\n");
    printf("\tTemplate Matching: %ld us;\n", duration_gpu.count());
    printf("\tColor filtering: %ld us;\n", duration_colorGPU.count());

    printf("\nCPU:\n");
    printf("\tTemplate Matching: %ld us;\n", duration_seq.count());
    printf("\tColor filtering: %ld us;\n", duration_colorSeq.count());

    // Mostra il risultato
    cv::imwrite("Result.jpg", imageColor);

    return 0;
}

