#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
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
#define OPTIMIZED_BLOCK_SIZE 256 // Multiplo di warp_size per migliore occupancy

// Range HSV per il blu (compatibili con OpenCV)
const unsigned char BLUE_H_MIN = 90;
const unsigned char BLUE_H_MAX = 140;
const unsigned char BLUE_S_MIN = 30;
const unsigned char BLUE_V_MIN = 30;

// FASE 1: PREFIX SUM INTRA-WARP (più efficiente)
__device__ __forceinline__ float warpPrefixSum(float val)
{
    // Algoritmo di prefix sum dentro un warp (32 thread)
    // Sfrutta __shfl_up_sync, permette ad un thread di accedere direttamente al valore
    // di un altro thread senza usare mem condivisa

#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        /*
            T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);
            mask: specifica i thread che partecipano alla comunicazione: 0xFFFFFFFF significa che tutti i thread partecipano
            var: il dato condiviso dal thread
            delta: num di posizioni verso il basso da cui prelevare
            width: opzionale, indica la dim del sotto warp
        */
        if (threadIdx.x >= offset)
        {
            val += temp;
        }
    }
    return val;
}

// FASE 2: PREFIX SUM INTRA-BLOCK (usa shared memory)
__device__ float blockPrefixSum(float val, float *sharedMem)
{
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    // Fase 1: Prefix sum dentro ogni warp
    float warpSum = warpPrefixSum(val);

    // Il thread finale di ogni warp salva la somma totale del warp
    // Se sono al thread finale di un warp (es: T31) -> laneId = 31 % 32 = 31 (0 con resto 31), allo stesso modo per i successivi.
    if (laneId == WARP_SIZE - 1)
    {
        sharedMem[warpId] = warpSum;
    }
    __syncthreads();

    // Fase 2: Prefix sum delle somme dei warp (solo il primo warp lavora)
    // Creaiamo l'offset tra i warp
    if (warpId == 0)
    {
        float warpTotal = (tid < blockDim.x / WARP_SIZE) ? sharedMem[tid] : 0.0f;
        warpTotal = warpPrefixSum(warpTotal);
        sharedMem[tid] = warpTotal;
    }
    __syncthreads();

    // Fase 3: Aggiungi l'offset dei warp precedenti ai thread successivi
    float blockOffset = (warpId > 0) ? sharedMem[warpId - 1] : 0.0f;
    return warpSum + blockOffset;
}

// KERNEL OTTIMIZZATO: PREFIX SUM PER RIGHE (COALESCENTE)
__global__ void optimizedRowPrefixSum(float *input, float *output,
                                      float *blockSums, int width, int height)
{
    // Configurazione: ogni blocco gestisce una riga
    // Thread accedono in modo coalescente
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height)
        return;

    //__shared__ float sharedData[MAX_BLOCK_SIZE];
    __shared__ float warpSums[MAX_BLOCK_SIZE / WARP_SIZE];

    // Carica dati in shared memory (accesso coalescente)
    float val = 0.0f;
    if (col < width)
    {
        val = input[row * width + col]; // Accesso coalescente!
    }

    // Prefix sum all'interno del blocco
    float prefixSum = blockPrefixSum(val, warpSums);

    // Scrivi risultato (accesso coalescente)
    if (col < width)
    {
        output[row * width + col] = prefixSum;
    }

    // Salva la somma finale del blocco per la fase successiva
    if (threadIdx.x == blockDim.x - 1 && col < width)
    {
        int blockId = row * gridDim.x + blockIdx.x;
        blockSums[blockId] = prefixSum;
    }

    // Alla fine di questa funzione ho le prefix sum di tutti i blocchi
    // La fase successiva è uguale alla fase 3 di blockPrefixSum
    // vado ad aggiungere ai thread di TUTTI i blocchi (tranne lo 0), l offset calcolato
}

// KERNEL: APPLICA OFFSET FINALE
__global__ void addBlockOffsets(float *data, float *blockSums,
                                int width, int height)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width)
        return;

    // Calcola l'offset cumulativo dai blocchi precedenti nella stessa riga
    float offset = 0.0f;
    int baseBlockId = row * gridDim.x;

    // Evita race conditions calcolando l'offset solo per i blocchi necessari
    if (blockIdx.x > 0)
    {
        for (int i = 0; i < blockIdx.x; i++)
        {
            offset += blockSums[baseBlockId + i];
        }
    }

    // Applica l'offset
    data[row * width + col] += offset;
}

// KERNEL: PREFIX SUM PER COLONNE (TRANSPOSE)
__global__ void optimizedTranspose(float *input, float *output,
                                   int width, int height)
{

    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 per evitare bank conflict

    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;

    // Carica tile in shared memory
    if (x < width)
    {
        for (int j = 0; j < TILE_DIM; j += blockDim.y)
        {
            if (y + j < height)
            {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
            }
        }
    }
    __syncthreads();

    // Calcola coordinate per output trasposto
    x = blockIdx_y * TILE_DIM + threadIdx.x;
    y = blockIdx_x * TILE_DIM + threadIdx.y;

    // Scrivi tile trasposto
    if (x < height)
    {
        for (int j = 0; j < TILE_DIM; j += blockDim.y)
        {
            if (y + j < width)
            {
                output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// FUNZIONE PRINCIPALE: CALCOLO IMMAGINE INTEGRALE OTTIMIZZATA
cudaError_t computeIntegralImageOptimized(float *d_input, float *d_output,
                                          int width, int height)
{

    // Allocazioni temporanee
    float *d_temp = nullptr;
    float *d_blockSums = nullptr;
    float *d_blockSumsT = nullptr;
    float *d_transposed = nullptr;

    size_t imageSize = width * height * sizeof(float);

    // Calcola dimensioni grid ottimali
    int threadsPerBlock = 256; // Multiplo di warp size
    int blocksPerRow = (width + threadsPerBlock - 1) / threadsPerBlock;
    int totalBlocks = height * blocksPerRow;

    // Allocazioni
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_temp, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_blockSums, totalBlocks * sizeof(float));
    if (cudaStatus != cudaSuccess)
        return cudaStatus;
    cudaStatus = cudaMalloc(&d_transposed, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // FASE 1: PREFIX SUM PER RIGHE ===

    dim3 blockSize1(threadsPerBlock);
    dim3 gridSize1(blocksPerRow, height);

    // Primo passo: prefix sum locale per blocchi + offsets
    optimizedRowPrefixSum<<<gridSize1, blockSize1>>>(
        d_input, d_temp, d_blockSums, width, height);

    // Secondo passo: applica gli offset
    addBlockOffsets<<<gridSize1, blockSize1>>>(
        d_temp, d_blockSums, width, height);

    // FASE 2: TRANSPOSE + PREFIX SUM PER COLONNE ===

    // Trasponi l'immagine
    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid((width + TILE_DIM - 1) / TILE_DIM,
                       (height + TILE_DIM - 1) / TILE_DIM);

    optimizedTranspose<<<transposeGrid, transposeBlock>>>(
        d_temp, d_transposed, width, height);

    // Ricalcola parametri per l'immagine trasposta
    int blocksPerCol = (height + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridSize2(blocksPerCol, width);

    int totalBlocksT = width * blocksPerCol;

    cudaStatus = cudaMalloc(&d_blockSumsT, totalBlocksT * sizeof(float));
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Prefix sum sulle "righe" dell'immagine trasposta (= colonne originali)
    optimizedRowPrefixSum<<<gridSize2, blockSize1>>>(
        d_transposed, d_temp, d_blockSumsT, height, width); // Nota: dimensioni scambiate

    addBlockOffsets<<<gridSize2, blockSize1>>>(
        d_temp, d_blockSumsT, height, width);

    // Trasponi di nuovo per ottenere il risultato finale
    dim3 transposeGrid2((height + TILE_DIM - 1) / TILE_DIM,
                        (width + TILE_DIM - 1) / TILE_DIM);

    optimizedTranspose<<<transposeGrid2, transposeBlock>>>(
        d_temp, d_output, height, width); // Dimensioni scambiate

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_blockSums);
    cudaFree(d_blockSumsT);
    cudaFree(d_transposed);

    return cudaGetLastError();
}

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
    int padX = x + kx - 1;
    int padY = y + ky - 1;

    // Verifica bounds del risultato SSD
    int resultWidth = width - kx + 1;
    int resultHeight = height - ky + 1;

    if (x >= resultWidth || y >= resultHeight)
        return;

    // Calcolo delle somme usando le immagini integrali
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky); // Somma dei quadrati
    float SC = crossCorrelation[INDEX(padX, padY, paddedCols)];
    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * SC + (*templateSqSum);

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}

// Funzione per il quadrato delle immagini ottimizzata
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

// FUNZIONI PER COLOR FILTERING
// Funzione device per conversione BGR a HSV (Hue (tonalità), Saturation (saturazione), Value (luminosità)) corretta
__device__ void bgrToHsv(unsigned char b, unsigned char g, unsigned char r,
                         unsigned char &h, unsigned char &s, unsigned char &v)
{
    float fb = b / 255.0f;
    float fg = g / 255.0f;
    float fr = r / 255.0f;

    float maxVal = fmaxf(fb, fmaxf(fg, fr));
    float minVal = fminf(fb, fminf(fg, fr));
    float delta = maxVal - minVal;

    // Value
    v = (unsigned char)(maxVal * 255);

    // Saturation
    if (maxVal == 0)
    {
        s = 0;
    }
    else
    {
        s = (unsigned char)((delta / maxVal) * 255);
    }

    // Hue
    float hue = 0;
    if (delta != 0)
    {
        if (maxVal == fr)
        {
            hue = 60.0f * fmodf((fg - fb) / delta, 6.0f);
        }
        else if (maxVal == fg)
        {
            hue = 60.0f * ((fb - fr) / delta + 2.0f);
        }
        else
        {
            hue = 60.0f * ((fr - fg) / delta + 4.0f);
        }
    }

    if (hue < 0)
        hue += 360.0f;

    // Converti in range OpenCV (0-179)
    h = (unsigned char)(hue / 2.0f);
}

// Kernel per contare i pixel blu in un'area specifica
__global__ void countBluePixelsKernel(uchar3 *image, int width, int height,
                                      int startX, int startY, int roiWidth, int roiHeight,
                                      int *blueCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Converti alle coordinate dell'immagine completa
    int imgX = startX + x;
    int imgY = startY + y;

    // Controlla se siamo dentro l'ROI e dentro i bounds dell'immagine
    if (x < roiWidth && y < roiHeight && imgX < width && imgY < height)
    {
        int idx = imgY * width + imgX;
        uchar3 pixel = image[idx]; // BGR format (OpenCV standard)

        // Converti BGR to HSV usando la funzione corretta
        unsigned char h, s, v;
        bgrToHsv(pixel.x, pixel.y, pixel.z, h, s, v);

        // Condizione per il blu in HSV
        bool isBlue = (h >= BLUE_H_MIN && h <= BLUE_H_MAX) &&
                      (s >= BLUE_S_MIN) &&
                      (v >= BLUE_V_MIN);

        if (isBlue)
        {
            atomicAdd(blueCount, 1);
        }
    }
}

// Funzione per conversione BGR -> Uchar3
__global__ void convertBGRtoUchar3Kernel_OptimalSimple(const uchar *cvImage, uchar3 *cudaImage,
                                                       int width, int height, size_t step)
{
    // Thread ID lineare per coalescenza garantita
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total_pixels = width * height;

    // Loop per alta compute intensity -> con stride, un thread può elaborare + pixels
    for (int pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += stride)
    {
        // Coordinate 2D
        int y = pixel_idx / width;
        int x = pixel_idx % width;

        // Accesso ai dati
        int src_idx = y * step + x * 3;

        // Lettura e conversione
        cudaImage[pixel_idx] = make_uchar3(
            cvImage[src_idx],     // B
            cvImage[src_idx + 1], // G
            cvImage[src_idx + 2]  // R
        );
    }
}

// Funzione per contare i pixel blu in una ROI
void countBlueInROI(const cv::Mat &image, const cv::Rect &roi, int &blueCount)
{
    // Verifica che l'ROI sia valido
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > image.cols ||
        roi.y + roi.height > image.rows)
    {
        std::cerr << "ROI fuori dai bounds dell'immagine!" << std::endl;
        return;
    }

    printf("Analizzando ROI: x=%d, y=%d, width=%d, height=%d\n", roi.x, roi.y, roi.width, roi.height);

    // Alloca memoria device
    uchar *d_cvImage = nullptr;
    uchar3 *d_image = nullptr;
    int *d_blueCount = nullptr;

    size_t cvImageSize = image.rows * image.step; // Usa step per gestire padding
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
        d_cvImage, d_image, image.cols, image.rows, image.step);

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
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "Errore kernel CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaDeviceSynchronize();

    // Copia risultati
    cudaMemcpy(&blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Libera memoria
    cudaFree(d_image);
    cudaFree(d_blueCount);
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

// ============================================================================
// CLASSE PER GESTIONE MEMORIA PERSISTENTE GPU
// ============================================================================

class GPUMemoryManager
{
private:
    // Memoria per template matching
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult;
    float *d_rowSum, *d_imageSq, *d_rowSqSum, *d_crossCorrelation;
    float *d_templateSqSumLV;

    float *d_imgPad, *d_tmpPad;
    cufftComplex *d_imgFreq, *d_tmpFreq;

    // Memoria per conteggio pixel blu
    uchar3 *d_colorImage, *d_debugImage;
    int *d_blueCount;

    // Dimensioni correnti
    int currentWidth, currentHeight;
    int currentResultWidth, currentResultHeight;

    // Var per FFT
    int m, n, freqCols;
    size_t realSize, freqSize, resultSize;

    bool initialized;

public:
    GPUMemoryManager() : initialized(false), currentWidth(0), currentHeight(0) {}

    ~GPUMemoryManager()
    {
        cleanup();
    }

    bool initialize(int width, int height, int templateWidth, int templateHeight)
    {
        if (initialized && width == currentWidth && height == currentHeight)
        {
            return true; // Già inizializzato con le dimensioni corrette
        }

        cleanup(); // Libera memoria precedente se presente

        size_t imageSize = width * height * sizeof(float);
        size_t colorImageSize = width * height * sizeof(uchar3);
        size_t resultSize = (width - templateWidth + 1) * (height - templateHeight + 1) * sizeof(float);

        // dimensioni ottimali FFT
        int m = cv::getOptimalDFTSize(height + templateHeight - 1); // In pratica, si arrotonda alla potenza di 2 più vicina per efficienza nella FFT.
        int n = cv::getOptimalDFTSize(width + templateWidth - 1);   // In particolare risulta più efficiente se i numeri altamente composti (sono fattorizzabili in 2,3 e 5).
        int freqCols = n / 2 + 1;                                   // FFT -> tante DFT + semplici, più bassi sono i coefficienti (2,3,5) minori sono le operazioni aritmetiche
        size_t realSize = sizeof(float) * m * n;                    // La FFT di un segnale reale ha simmetria hermitiana, per risparmiare, quindi, la cuFFT memorizza solo metà dello spettro.
        size_t freqSize = sizeof(cufftComplex) * m * freqCols;

        // Alloca memoria per template matching
        if (cudaMalloc(&d_image, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSqSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSq, imageSize) != cudaSuccess ||
            cudaMalloc(&d_rowSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_rowSqSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_ssdResult, resultSize) != cudaSuccess ||
            cudaMalloc(&d_crossCorrelation, resultSize) != cudaSuccess ||
            cudaMalloc(&d_templateSqSumLV, sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_imgPad, realSize) != cudaSuccess ||
            cudaMalloc(&d_tmpPad, realSize) != cudaSuccess ||
            cudaMalloc(&d_imgFreq, freqSize) != cudaSuccess ||
            cudaMalloc(&d_tmpFreq, freqSize) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        // Alloca memoria per conteggio pixel blu
        if (cudaMalloc(&d_colorImage, colorImageSize) != cudaSuccess ||
            cudaMalloc(&d_debugImage, colorImageSize) != cudaSuccess ||
            cudaMalloc(&d_blueCount, sizeof(int)) != cudaSuccess)
        {
            cleanup();
            return false;
        }

        currentWidth = width;
        currentHeight = height;
        currentResultWidth = width - templateWidth + 1;
        currentResultHeight = height - templateHeight + 1;
        initialized = true;

        return true;
    }

    void cleanup()
    {
        if (initialized)
        {
            cudaFree(d_image);
            cudaFree(d_imageSum);
            cudaFree(d_imageSqSum);
            cudaFree(d_imageSq);
            cudaFree(d_rowSum);
            cudaFree(d_rowSqSum);
            cudaFree(d_ssdResult);
            cudaFree(d_crossCorrelation);
            cudaFree(d_colorImage);
            cudaFree(d_debugImage);
            cudaFree(d_blueCount);
            cudaFree(d_templateSqSumLV);
            cudaFree(d_imgPad);
            cudaFree(d_tmpPad);
            cudaFree(d_imgFreq);
            cudaFree(d_tmpFreq);
            initialized = false;
        }
    }

    // Getter per accesso ai puntatori memoria
    float *getImagePtr() { return d_image; }
    float *getImageSumPtr() { return d_imageSum; }
    float *getImageSqSumPtr() { return d_imageSqSum; }
    float *getImageSqPtr() { return d_imageSq; }
    float *getRowSumPtr() { return d_rowSum; }
    float *getRowSqSumPtr() { return d_rowSqSum; }
    float *getSSDResultPtr() { return d_ssdResult; }
    float *getTemplateSqSumLV() { return d_templateSqSumLV; }
    float *getCrossCorrelationPtr() { return d_crossCorrelation; }
    uchar3 *getColorImagePtr() { return d_colorImage; }
    uchar3 *getDebugImagePtr() { return d_debugImage; }
    int *getBlueCountPtr() { return d_blueCount; }
    float *getImagePadPtr() { return d_imgPad; }
    float *getTempPadPtr() { return d_tmpPad; }
    cufftComplex *getImageFreqPtr() { return d_imgFreq; }
    cufftComplex *getTmpFreqPtr() { return d_tmpFreq; }
    int getM() { return m; }
    int getN() { return n; }
    int getFreqCols() { return freqCols; }
    size_t getRealSize() { return realSize; }
    size_t getFreqSize() { return freqSize; }
    size_t getResultSize() { return resultSize; }

    int getResultWidth() { return currentResultWidth; }
    int getResultHeight() { return currentResultHeight; }
};

// ============================================================================
// FUNZIONI OTTIMIZZATE PER VIDEO
// ============================================================================

cudaError_t processVideoFrame(
    const cv::Mat &grayFrame,
    const cv::Mat &colorFrame,
    const cv::Mat &templ,
    GPUMemoryManager &memManager,
    cv::Point &bestLoc,
    int &blueCount,
    float *d_temp)
{
    // Conversione frame in float
    cv::Mat frameFloat;
    grayFrame.convertTo(frameFloat, CV_32F, 1.0 / 255.0);

    int width = grayFrame.cols;
    int height = grayFrame.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    // Inizializza memoria GPU se necessario
    if (!memManager.initialize(width, height, kx, ky))
    {
        return cudaErrorMemoryAllocation;
    }

    printf("Dim source: %dx%d; Dim Template: %dx%d\n\n", width, height, kx, ky);

    // Copia frame su GPU
    size_t imageSize = width * height * sizeof(float);
    size_t colorImageSize = width * height * sizeof(uchar3);

    cudaError_t status = cudaMemcpy(memManager.getImagePtr(), frameFloat.ptr<float>(),
                                    imageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
        return status;

    // Copia frame a colori
    std::vector<uchar3> hostColorImage(height * width);
    for (int y = 0; y < height; ++y)
    {
        const cv::Vec3b *row = colorFrame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x)
        {
            hostColorImage[y * width + x] = make_uchar3(row[x][0], row[x][1], row[x][2]);
        }
    }
    status = cudaMemcpy(memManager.getColorImagePtr(), hostColorImage.data(),
                        colorImageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
        return status;

    // Configurazione kernel
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeIntegralImageOptimized(memManager.getImagePtr(), memManager.getImageSumPtr(), width, height);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess)
    {
        printf("Errore computeIntegralImageOptimized: %s\n", cudaGetErrorString(err2));
        return err2;
    }

    // Calcolo immagine source integrali
    multiply_optimized<<<gridSize, blockSize>>>(memManager.getImagePtr(), width, height, memManager.getImageSqSumPtr());
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
        printf("Errore multiply_optimized: %s\n", cudaGetErrorString(err1));
        return err1;
    }

    computeIntegralImageOptimized(memManager.getImageSqPtr(), memManager.getImageSqSumPtr(), width, height);
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess)
    {
        printf("Errore computeIntegralImageOptimized: %s\n", cudaGetErrorString(err2));
        return err3;
    }

    cudaDeviceSynchronize();
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        printf("Errore kernel: %s\n", cudaGetErrorString(kernelError));
        return kernelError;
    }

    // Calcolo cross correlation
    // Pad to zero
    dim3 b1(16, 16);
    int gridX = min(65535, (memManager.getN() + 15) / 16);
    int gridY = min(65535, (memManager.getM() + 15) / 16);
    dim3 g1(gridX, gridY);

    padToZero<<<g1, b1>>>(memManager.getImagePtr(), memManager.getImagePadPtr(), height, width, memManager.getM(), memManager.getN());
    padToZero<<<g1, b1>>>(d_temp, memManager.getTempPadPtr(), ky, kx, memManager.getM(), memManager.getN());

    // cuFFT plan
    cufftHandle planFwd, planInv;
    cufftPlan2d(&planFwd, memManager.getM(), memManager.getN(), CUFFT_R2C);
    cufftPlan2d(&planInv, memManager.getM(), memManager.getN(), CUFFT_C2R);

    // FFT forward
    cufftExecR2C(planFwd, memManager.getImagePadPtr(), memManager.getImageFreqPtr());
    cufftExecR2C(planFwd, memManager.getTempPadPtr(), memManager.getTmpFreqPtr());

    // Multiply with conjugate
    dim3 b2(16, 16);
    int freqGridX = min(65535, (memManager.getFreqCols() + 15) / 16);
    int freqGridY = min(65535, (memManager.getM() + 15) / 16);
    dim3 g2(freqGridX, freqGridY);

    mulConjAndScale<<<g2, b2>>>(memManager.getImageFreqPtr(), memManager.getTmpFreqPtr(), memManager.getM(), memManager.getFreqCols());

    // IFFT
    cufftExecC2R(planInv, memManager.getImageFreqPtr(), memManager.getImagePadPtr());

    // Normalize
    int totalReal = memManager.getM() * memManager.getN();
    normalize<<<(totalReal + 255) / 256, 256>>>(memManager.getImagePadPtr(), totalReal, 1.0f / (memManager.getM() * memManager.getN()));

    cufftDestroy(planFwd);
    cufftDestroy(planInv);

    // Calcolo SSD
    int resultWidth = width - kx + 1;
    int resultHeight = height - ky + 1;
    dim3 gridSSDSize((resultWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (resultHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSSDSize(16, 16);

    printf("DEBUG: Result=%dx%d, Grid=%dx%d, MaxThread=(%d,%d)\n",
           resultWidth, resultHeight, gridSSDSize.x, gridSSDSize.y,
           gridSSDSize.x * BLOCK_SIZE - 1, gridSSDSize.y * BLOCK_SIZE - 1);

    printf("Launching SSD kernel with grid %dx%d, block %dx%d\n",
           gridSSDSize.x, gridSSDSize.y, blockSSDSize.x, blockSSDSize.y);

    computeSSD<<<gridSSDSize, blockSSDSize>>>(
        memManager.getImageSqSumPtr(),
        memManager.getTemplateSqSumLV(),
        width,
        height,
        kx,
        ky,
        memManager.getSSDResultPtr(),
        memManager.getImagePadPtr(),
        memManager.getN());

    cudaDeviceSynchronize();
    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        printf("Errore kernel computeSSD: %s\n", cudaGetErrorString(kernelError));
        return kernelError;
    }

    // Trova minimo SSD (questo richiede copia su CPU)
    cv::Mat ssdResult(memManager.getResultHeight(), memManager.getResultWidth(), CV_32F);
    status = cudaMemcpy(ssdResult.ptr<float>(), memManager.getSSDResultPtr(),
                        memManager.getResultSize(), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        return status;

    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(ssdResult, &minVal, nullptr, &minLoc, nullptr);
    bestLoc = minLoc;

    // Conteggio pixel blu nell'ROI trovata
    cv::Rect roi(bestLoc.x, bestLoc.y, kx, ky);
    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    roi.width = std::min(roi.width, width - roi.x);
    roi.height = std::min(roi.height, height - roi.y);

    // Reset contatore blu
    // status = cudaMemcpy(&blueCount, memManager.getBlueCountPtr(), sizeof(int), cudaMemcpyDeviceToHost);

    countBlueInROI(colorFrame, roi, blueCount);

    memManager.cleanup();

    return status;
}

// Funzione per precalcolare i parametri del template (chiamata una sola volta)
void precomputeTemplateParameters(float *d_templ, float *d_templSq, float *d_templSqSum, float *d_templateSqSumLV, int kx, int ky)
{

    size_t templSize = kx * ky * sizeof(float);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSizeT(
        (kx + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (ky + BLOCK_SIZE - 1) / BLOCK_SIZE);

    multiply_optimized<<<gridSizeT, blockSize>>>(d_templ, kx, ky, d_templSq);
    computeIntegralImageOptimized(d_templSq, d_templSqSum, kx, ky);

    // Copia del last value del template syl davice
    size_t offset = ((ky - 1) * kx + (kx - 1)) * sizeof(float);
    cudaMemcpy(d_templateSqSumLV,
               (char *)d_templSqSum + offset,
               sizeof(float),
               cudaMemcpyDeviceToDevice);
}

// ============================================================================
// MAIN FUNCTION PER VIDEO PROCESSING
// ============================================================================

int main()
{
    // Carica template
    cv::Mat templ = cv::imread("template.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat templFloat;

    cv::resize(templ, templ, cv::Size(templ.cols / 3, templ.rows / 3));
    templ.convertTo(templFloat, CV_32F, 1.0 / 255.0);

    int kx = templ.cols;
    int ky = templ.rows;
    printf("template resized dim: %dx%d", kx, ky);

    size_t templSize = kx * ky * sizeof(float);
    float *d_temp;

    if (cudaMalloc(&d_temp, templSize) != cudaSuccess)
    {
        printf("Errore\n");
        return -1;
    }

    if (templ.empty())
    {
        printf("Errore nel caricamento del template\n");
        return -1;
    }

    // Precalcola parametri template

    float *d_templSq, *d_templSqSum, *d_templateSqSumLV, *d_templ;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&d_templateSqSumLV, sizeof(float));
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSqSum, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSq, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templ, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMemcpy(d_templ, templFloat.ptr<float>(), templSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    precomputeTemplateParameters(d_templ, d_templSq, d_templSqSum, d_templateSqSumLV, kx, ky);

    // Apri video (webcam o file)
    cv::VideoCapture cap;

    // Prova prima webcam, poi file video
    cap.open(0); // Webcam
    if (!cap.isOpened())
    {
        cap.open("input_video.mp4"); // File video
        if (!cap.isOpened())
        {
            printf("Errore nell'apertura del video\n");
            return -1;
        }
    }

    // Configurazione video output (opzionale)
    cv::VideoWriter writer;
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0)
        fps = 30;

    // Manager memoria GPU
    GPUMemoryManager memManager;

    // Variabili per timing
    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    cv::Mat frame, grayFrame;

    printf("=== INIZIO ELABORAZIONE VIDEO ===\n");
    printf("Premi 'q' per uscire\n");

    while (true)
    {
        // Cattura frame
        if (!cap.read(frame))
        {
            printf("Fine video o errore nella cattura\n");
            break;
        }

        cv::resize(frame, frame, cv::Size(640, 360));

        // Converti in scala di grigi
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Processa frame
        cv::Point bestLoc;
        int blueCount = 0;

        cudaError_t status = processVideoFrame(grayFrame, frame, templFloat, memManager,
                                               bestLoc, blueCount, d_templ);

        if (status != cudaSuccess)
        {
            printf("Errore CUDA nel processamento frame: %s\n", cudaGetErrorString(status));
            continue;
        }

        // Visualizza risultati sul frame
        cv::Rect detectionRect(bestLoc, cv::Size(templ.cols, templ.rows));
        cv::rectangle(frame, detectionRect, cv::Scalar(0, 255, 0), 2);

        // Aggiungi informazioni testuali
        std::string infoText = "Blu: " + std::to_string(blueCount) + " px";
        cv::putText(frame, infoText, cv::Point(bestLoc.x, bestLoc.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        std::string posText = "Pos: (" + std::to_string(bestLoc.x) + "," + std::to_string(bestLoc.y) + ")";
        cv::putText(frame, posText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // Calcola e mostra FPS
        frameCount++;
        if (frameCount % 1 == 0)
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
            double fps = (frameCount * 1000.0) / duration.count();
            printf("FPS: %.2f, Frame: %d, Posizione: (%d,%d), Pixel blu: %d\n",
                   fps, frameCount, bestLoc.x, bestLoc.y, blueCount);
        }

        // Salva video (opzionale)
        if (!writer.isOpened() && frameCount == 1)
        {
            writer.open("output_video.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                        fps, frame.size(), true);
        }
        if (writer.isOpened())
        {
            writer.write(frame);
        }

        // Controllo uscita
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27)
        { // 'q' o ESC
            break;
        }
    }

    printf("\n=== ELABORAZIONE COMPLETATA ===\n");
    printf("Frame totali processati: %d\n", frameCount);

    // Cleanup
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}