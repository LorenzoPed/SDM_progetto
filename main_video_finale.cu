#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

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

#define SEARCH_HEIGHT 350

const unsigned char BLUE_H_MIN = 90;
const unsigned char BLUE_H_MAX = 140;
const unsigned char BLUE_S_MIN = 30;
const unsigned char BLUE_V_MIN = 30;

// Tracker
struct Tracker {
    cv::Point currentPosition;
    float currentScore;
    bool hasValidPosition;
    
    // Parametri per movimento graduale
    float maxValidScore;
    int minValidBluePixels;
    float movementSmoothingFactor;
    
    Tracker() {
        currentPosition = cv::Point(0, 0);
        currentScore = FLT_MAX;
        hasValidPosition = false;
        
        // Soglie di validazione
        maxValidScore = 0.5f;
        minValidBluePixels = 1000;
        movementSmoothingFactor = 0.3f; // Aumentta la fluidità
    }
    
    cv::Point updatePosition(cv::Point newMatch, float newScore, int bluePixels) {
        // Verifica se il match è valido
        bool isValidMatch = (newScore < maxValidScore) && (bluePixels >= minValidBluePixels);
        
        if (!isValidMatch) {
            // Match non valido, mantieni posizione precedente se esiste
            if (hasValidPosition) {
                return currentPosition;
            } else {
                // Prima volta e match non valido, usalo comunque
                currentPosition = newMatch;
                currentScore = newScore;
                hasValidPosition = true;
                return currentPosition;
            }
        }
        
        // Match valido
        if (!hasValidPosition) {
            // Prima posizione valida
            currentPosition = newMatch;
            currentScore = newScore;
            hasValidPosition = true;
        } else {
            // Movimento graduale verso la nuova posizione
            float deltaX = newMatch.x - currentPosition.x;
            float deltaY = newMatch.y - currentPosition.y;
            
            currentPosition.x += (int)(deltaX * movementSmoothingFactor);
            currentPosition.y += (int)(deltaY * movementSmoothingFactor);
            currentScore = newScore;
        }
        
        return currentPosition;
    }
    
    bool isValid() const {
        return hasValidPosition;
    }
};


// Funzioni kernel e device
// FASE 1: PREFIX SUM INTRA-WARP (più efficiente)
__device__ __forceinline__ float warpPrefixSum(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset); 
        if (threadIdx.x >= offset) {
            val += temp;
        }
    }
    return val;
}

// FASE 2: PREFIX SUM INTRA-BLOCK (usa shared memory)
__device__ float blockPrefixSum(float val, float* sharedMem) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    
    // Fase 1: Prefix sum dentro ogni warp
    float warpSum = warpPrefixSum(val);
    
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
    
    // Fase 3: Aggiungi l'offset dei warp precedenti ai thread successivi
    float blockOffset = (warpId > 0) ? sharedMem[warpId - 1] : 0.0f;
    return warpSum + blockOffset;
}

// KERNEL OTTIMIZZATO: PREFIX SUM PER RIGHE (COALESCENTE)
__global__ void optimizedRowPrefixSum(float* input, float* output, 
                                     float* blockSums, int width, int height) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height) return;
    
    __shared__ float warpSums[MAX_BLOCK_SIZE / WARP_SIZE];
    
    float val = 0.0f;
    if (col < width) {
        val = input[row * width + col];
    }
    
    float prefixSum = blockPrefixSum(val, warpSums);
    
    if (col < width) {
        output[row * width + col] = prefixSum;
    }
    
    if (threadIdx.x == blockDim.x - 1 && col < width) {
        int blockId = row * gridDim.x + blockIdx.x;
        blockSums[blockId] = prefixSum;
    }
}

// KERNEL: APPLICA OFFSET FINALE
__global__ void addBlockOffsets(float* data, float* blockSums, 
                               int width, int height) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= height || col >= width) return;
    
    float offset = 0.0f;
    int baseBlockId = row * gridDim.x;
    
    if (blockIdx.x > 0) {
        for (int i = 0; i < blockIdx.x; i++) {
            offset += blockSums[baseBlockId + i];
        }
    }
    
    data[row * width + col] += offset;
}

// KERNEL: PREFIX SUM PER COLONNE (TRANSPOSE + PREFIX SUM)
__global__ void optimizedTranspose(float* input, float* output, 
                                  int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    
    int x = blockIdx_x * TILE_DIM + threadIdx.x;
    int y = blockIdx_y * TILE_DIM + threadIdx.y;
    
    if (x < width) {
        for (int j = 0; j < TILE_DIM; j += blockDim.y) {
            if (y + j < height) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
            }
        }
    }
    __syncthreads();
    
    x = blockIdx_y * TILE_DIM + threadIdx.x;
    y = blockIdx_x * TILE_DIM + threadIdx.y;
    
    if (x < height) {
        for (int j = 0; j < TILE_DIM; j += blockDim.y) {
            if (y + j < width) {
                output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// FUNZIONE PRINCIPALE: CALCOLO IMMAGINE INTEGRALE OTTIMIZZATA
cudaError_t computeIntegralImageOptimized(float* d_input, float* d_output,
                                         int width, int height) {
    float* d_temp = nullptr;
    float* d_blockSums = nullptr;
    float* d_transposed = nullptr;
    
    size_t imageSize = width * height * sizeof(float);
    
    int threadsPerBlock = 256;
    int blocksPerRow = (width + threadsPerBlock - 1) / threadsPerBlock;
    int totalBlocks = height * blocksPerRow;
    
    cudaMalloc(&d_temp, imageSize);
    cudaMalloc(&d_blockSums, totalBlocks * sizeof(float));
    cudaMalloc(&d_transposed, imageSize);
    
    // === FASE 1: PREFIX SUM PER RIGHE ===
    dim3 blockSize1(threadsPerBlock);
    dim3 gridSize1(blocksPerRow, height);
    
    optimizedRowPrefixSum<<<gridSize1, blockSize1>>>(
        d_input, d_temp, d_blockSums, width, height);
    
    addBlockOffsets<<<gridSize1, blockSize1>>>(
        d_temp, d_blockSums, width, height);
    
    // === FASE 2: TRANSPOSE + PREFIX SUM PER COLONNE ===
    dim3 transposeBlock(TILE_DIM, BLOCK_ROWS);
    dim3 transposeGrid((width + TILE_DIM - 1) / TILE_DIM,
                      (height + TILE_DIM - 1) / TILE_DIM);
    
    optimizedTranspose<<<transposeGrid, transposeBlock>>>(
        d_temp, d_transposed, width, height);
    
    int blocksPerCol = (height + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridSize2(blocksPerCol, width);
    
    optimizedRowPrefixSum<<<gridSize2, blockSize1>>>(
        d_transposed, d_temp, d_blockSums, height, width);
    
    addBlockOffsets<<<gridSize2, blockSize1>>>(
        d_temp, d_blockSums, height, width);
    
    dim3 transposeGrid2((height + TILE_DIM - 1) / TILE_DIM,
                       (width + TILE_DIM - 1) / TILE_DIM);
    
    optimizedTranspose<<<transposeGrid2, transposeBlock>>>(
        d_temp, d_output, height, width);
    
    cudaFree(d_temp);
    cudaFree(d_blockSums);
    cudaFree(d_transposed);
    
    return cudaGetLastError();
}

// Funzione device per calcolare la somma in una regione usando la sum table
__device__ int getRegionSum(const float *sumTable, int width, int height, int x, int y, int kx, int ky) {
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
    float *imageSqSum,
    float *templateSqSum,
    int width,
    int height,
    int kx,
    int ky,
    float *ssdResult,
    float *crossCorrelation,
    int paddedCols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - kx + 1 || y >= height - ky + 1)
        return;

    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky);
    float SC = crossCorrelation[INDEX(x, y, paddedCols)];
    float ssd = S2 - 2 * SC + (*templateSqSum);

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}

__global__ void multiply_optimized(
    float *image, 
    int width,
    int height,
    float *d_imageSq) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = INDEX(x, y, width);
    float val = __ldg(&image[idx]);
    d_imageSq[idx] = val * val;
}

// FUNZIONI PER FATTORE DI CROSSCORRELAZIONE
__global__ void padToZero(const float *src, float *dst,
                          int srcRows, int srcCols,
                          int dstRows, int dstCols) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= dstRows || x >= dstCols)
        return;
    int dstIdx = y * dstCols + x;
    if (y < srcRows && x < srcCols) {
        dst[dstIdx] = src[y * srcCols + x];
    } else {
        dst[dstIdx] = 0.0f;
    }
}

__global__ void mulConjAndScale(cufftComplex *imageF,
                                const cufftComplex *kernelF,
                                int rows, int colsFreq) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows || x >= colsFreq)
        return;
    int idx = y * colsFreq + x;
    cufftComplex A = imageF[idx];
    cufftComplex B = kernelF[idx];
    B.y = -B.y;
    cufftComplex C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.x * B.y + A.y * B.x;
    imageF[idx] = C;
}

__global__ void normalize(float *data, int totalSize, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
        data[idx] *= scale;
}

// Template matching semplificato - cerca solo nella ROI
cudaError_t templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLoc,
    cv::Mat &ssdResult,
    const cv::Rect &searchROI) {
    
    // Estrai la ROI dall'immagine
    cv::Mat imageROI = image(searchROI);
    
    cv::Mat imageN, templN;
    imageROI.convertTo(imageN, CV_32F, 1.0 / 255.0);
    templ.convertTo(templN, CV_32F, 1.0 / 255.0);

    int width = imageROI.cols;
    int height = imageROI.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    int m = cv::getOptimalDFTSize(height + ky - 1);
    int n = cv::getOptimalDFTSize(width + kx - 1);
    int freqCols = n / 2 + 1;
    size_t realSize = sizeof(float) * m * n;
    size_t freqSize = sizeof(cufftComplex) * m * freqCols;

    float *d_imgPad, *d_tmpPad;
    cufftComplex *d_imgFreq, *d_tmpFreq;

    float *d_image, *d_imageSqSum, *d_ssdResult, *d_imageSq, *d_templ, *d_templSq, *d_templSqSum, *d_templateSqSumLV;

    size_t templSize = kx * ky * sizeof(float);
    size_t imageSize = width * height * sizeof(float);
    size_t resultSize = (width - kx + 1) * (height - ky + 1) * sizeof(float);

    // Allocazione memoria
    cudaMalloc(&d_imgPad, realSize);
    cudaMalloc(&d_tmpPad, realSize);
    cudaMalloc(&d_imgFreq, freqSize);
    cudaMalloc(&d_tmpFreq, freqSize);
    cudaMalloc(&d_image, imageSize);
    cudaMalloc(&d_imageSqSum, imageSize);
    cudaMalloc(&d_imageSq, imageSize);
    cudaMalloc(&d_ssdResult, resultSize);
    cudaMalloc(&d_templ, templSize);
    cudaMalloc(&d_templSq, templSize);
    cudaMalloc(&d_templSqSum, templSize);
    cudaMalloc(&d_templateSqSumLV, sizeof(float));

    // Copia dati
    cudaMemcpy(d_image, imageN.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_templ, templN.ptr<float>(), templSize, cudaMemcpyHostToDevice);

    // Calcolo immagini integrali
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    multiply_optimized<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    computeIntegralImageOptimized(d_imageSq, d_imageSqSum, width, height);

    dim3 gridSizeT((kx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ky + BLOCK_SIZE - 1) / BLOCK_SIZE);
    multiply_optimized<<<gridSizeT, blockSize>>>(d_templ, kx, ky, d_templSq);
    computeIntegralImageOptimized(d_templSq, d_templSqSum, kx, ky);

    size_t offset = ((ky - 1) * kx + (kx - 1)) * sizeof(float);
    cudaMemcpy(d_templateSqSumLV, (char *)d_templSqSum + offset, sizeof(float), cudaMemcpyDeviceToDevice);

    // Cross correlation via FFT
    dim3 b1(16, 16), g1((n + 15) / 16, (m + 15) / 16);
    padToZero<<<g1, b1>>>(d_image, d_imgPad, height, width, m, n);
    padToZero<<<g1, b1>>>(d_templ, d_tmpPad, ky, kx, m, n);

    cufftHandle planFwd, planInv;
    cufftPlan2d(&planFwd, m, n, CUFFT_R2C);
    cufftPlan2d(&planInv, m, n, CUFFT_C2R);

    cufftExecR2C(planFwd, d_imgPad, d_imgFreq);
    cufftExecR2C(planFwd, d_tmpPad, d_tmpFreq);

    dim3 b2(16, 16), g2((freqCols + 15) / 16, (m + 15) / 16);
    mulConjAndScale<<<g2, b2>>>(d_imgFreq, d_tmpFreq, m, freqCols);

    cufftExecC2R(planInv, d_imgFreq, d_imgPad);

    int totalReal = m * n;
    normalize<<<(totalReal + 255) / 256, 256>>>(d_imgPad, totalReal, 1.0f / (m * n));

    // Calcolo SSD
    dim3 gridSSDSize((width - kx + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height - ky + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeSSD<<<gridSSDSize, blockSize>>>(
        d_imageSqSum, d_templateSqSumLV, width, height, kx, ky, d_ssdResult, d_imgPad, n);

    // Copia risultati
    ssdResult = cv::Mat(height - ky + 1, width - kx + 1, CV_32F);
    cudaMemcpy(ssdResult.ptr<float>(), d_ssdResult, resultSize, cudaMemcpyDeviceToHost);

    // Trova minimo
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(ssdResult, &minVal, &maxVal, &minLoc, &maxLoc);
    
    // Converti coordinate relative alla ROI in coordinate assolute
    bestLoc->x = minLoc.x + searchROI.x;
    bestLoc->y = minLoc.y + searchROI.y;

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_imageSqSum);
    cudaFree(d_ssdResult);
    cudaFree(d_imageSq);
    cudaFree(d_imgPad);
    cudaFree(d_tmpPad);
    cudaFree(d_imgFreq);
    cudaFree(d_tmpFreq);
    cudaFree(d_templ);
    cudaFree(d_templSq);
    cudaFree(d_templSqSum);
    cudaFree(d_templateSqSumLV);

    cufftDestroy(planFwd);
    cufftDestroy(planInv);

    return cudaGetLastError();
}

// FUNZIONI PER COLOR FILTERING
__device__ void bgrToHsv(unsigned char b, unsigned char g, unsigned char r, 
                        unsigned char& h, unsigned char& s, unsigned char& v) {
    float fb = b / 255.0f;
    float fg = g / 255.0f;
    float fr = r / 255.0f;
    
    float maxVal = fmaxf(fb, fmaxf(fg, fr));
    float minVal = fminf(fb, fminf(fg, fr));
    float delta = maxVal - minVal;
    
    v = (unsigned char)(maxVal * 255);
    
    if (maxVal == 0) {
        s = 0;
    } else {
        s = (unsigned char)((delta / maxVal) * 255);
    }
    
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
    h = (unsigned char)(hue / 2.0f);
}

__global__ void countBluePixelsKernel(uchar3* image, int width, int height, 
                                     int startX, int startY, int roiWidth, int roiHeight,
                                     int* blueCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int imgX = startX + x;
    int imgY = startY + y;

    if (x < roiWidth && y < roiHeight && imgX < width && imgY < height) {
        int idx = imgY * width + imgX;
        uchar3 pixel = image[idx];

        unsigned char h, s, v;
        bgrToHsv(pixel.x, pixel.y, pixel.z, h, s, v);

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total_pixels = width * height;
    
    for (int pixel_idx = tid; pixel_idx < total_pixels; pixel_idx += stride) {
        int y = pixel_idx / width;
        int x = pixel_idx % width;
        
        int src_idx = y * step + x * 3;
        
        cudaImage[pixel_idx] = make_uchar3(
            cvImage[src_idx],      // B
            cvImage[src_idx + 1],  // G
            cvImage[src_idx + 2]   // R
        );
    }
}

void countBlueInROI(cv::Mat& image, const cv::Rect& roi, int& blueCount) {
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > image.cols || 
        roi.y + roi.height > image.rows) {
        return;
    }

    uchar* d_cvImage = nullptr;
    uchar3* d_image = nullptr;
    int* d_blueCount = nullptr;

    size_t cvImageSize = image.rows * image.step;
    size_t uchar3ImageSize = image.rows * image.cols * sizeof(uchar3);
    
    cudaMalloc(&d_cvImage, cvImageSize);
    cudaMalloc(&d_image, uchar3ImageSize);
    cudaMalloc(&d_blueCount, sizeof(int));

    cudaMemcpy(d_cvImage, image.data, cvImageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_blueCount, 0, sizeof(int));

    int totalPixels = image.rows * image.cols;
    int blockSizePerConversion = 256;
    int gridSizePerConversion = std::min(65535, (totalPixels + blockSizePerConversion - 1) / blockSizePerConversion);

    convertBGRtoUchar3Kernel_OptimalSimple<<<gridSizePerConversion, blockSizePerConversion>>>(
        d_cvImage, d_image, image.cols, image.rows, image.step
    );

    dim3 blockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 gridSize((roi.width + blockSize.x - 1) / blockSize.x, 
                 (roi.height + blockSize.y - 1) / blockSize.y);

    countBluePixelsKernel<<<gridSize, blockSize>>>(d_image, image.cols, image.rows,
                                                 roi.x, roi.y, roi.width, roi.height,
                                                 d_blueCount);

    cudaDeviceSynchronize();
    cudaMemcpy(&blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_cvImage);
    cudaFree(d_image);
    cudaFree(d_blueCount);
}

// FUNZIONE PRINCIPALE

int processVideo(const std::string& videoPath, const std::string& templatePath, 
                const std::string& outputPath) {
    
    // Carica il template
    cv::Mat templ = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);
    if (templ.empty()) {
        printf("Errore nel caricamento del template\n");
        return -1;
    }

    // Apri il video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        printf("Errore nell'apertura del video: %s\n", videoPath.c_str());
        return -1;
    }

    // Proprietà video
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    printf("Video: %dx%d, %.2f FPS, %d frames\n", frameWidth, frameHeight, fps, frameCount);
    printf("Template: %dx%d\n", templ.cols, templ.rows);

    // Video writer
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M','P','4','V'), 
                          fps, cv::Size(frameWidth, frameHeight));
    
    if (!writer.isOpened()) {
        printf("Errore nella creazione del video di output\n");
        return -1;
    }

    // Tracker
    Tracker tracker;
    
    cv::Mat frame, grayFrame;
    cv::Point rawBestMatch, trackedPosition;
    int frameNum = 0;
    
    // Statistiche timing
    auto totalStart = high_resolution_clock::now();
    double totalTemplateTime = 0.0;
    double totalBlueTime = 0.0;
    
    printf("\nElaborazione video...\n");
    
    // ROI di ricerca - fascia centrale
    int searchHeight = SEARCH_HEIGHT; // 350px
    int searchY = (frameHeight - searchHeight) / 2;
    cv::Rect searchROI(0, searchY, frameWidth, searchHeight);
    
    printf("ROI di ricerca: %dx%d alla posizione Y=%d\n", 
           searchROI.width, searchROI.height, searchROI.y);
    
    while (cap.read(frame)) {
        frameNum++;
        int bluePixelCount = 0;

        // Converti in scala di grigi
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        
        // Template Matching
        auto tmStart = high_resolution_clock::now();
        
        cv::Mat ssdResult;
        cudaError_t result = templateMatchingSSD(grayFrame, templ, &rawBestMatch, ssdResult, searchROI);
        if (result != cudaSuccess) {
            printf("Errore CUDA nel frame %d: %s\n", frameNum, cudaGetErrorString(result));
            continue;
        }
        
        auto tmEnd = high_resolution_clock::now();
        double tmTime = duration_cast<microseconds>(tmEnd - tmStart).count() / 1000.0;
        totalTemplateTime += tmTime;
        
        // Ottieni il punteggio SSD del match trovato
        float rawScore = ssdResult.at<float>(rawBestMatch.y - searchROI.y, rawBestMatch.x - searchROI.x);
        
        // Conteggio blue pixels
        auto blueStart = high_resolution_clock::now();
        
        // Usa la posizione raw per il conteggio blu (per validazione)
        cv::Rect templateROI(rawBestMatch.x, rawBestMatch.y, templ.cols, templ.rows);
        templateROI &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        countBlueInROI(frame, templateROI, bluePixelCount);
        
        auto blueEnd = high_resolution_clock::now();
        double blueTime = duration_cast<microseconds>(blueEnd - blueStart).count() / 1000.0;
        totalBlueTime += blueTime;
        
        // Tracking
        // Il tracker segue sempre il miglior SSD con movimento graduale
        trackedPosition = tracker.updatePosition(rawBestMatch, rawScore, bluePixelCount);
        
        // Stats a video
        // ROI di ricerca (blu)
        cv::rectangle(frame, searchROI, cv::Scalar(255, 0, 0), 2);
        
        // Posizione tracciata (verde se valida, rosso se no)
        cv::Rect trackedROI(trackedPosition.x, trackedPosition.y, templ.cols, templ.rows);
        trackedROI &= cv::Rect(0, 0, frame.cols, frame.rows);
        
        cv::Scalar trackColor = tracker.isValid() ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(frame, trackedROI, trackColor, 3);
        
        // Posizione raw (giallo, sottile)
        cv::rectangle(frame, templateROI, cv::Scalar(0, 255, 255), 1);
        
        // Output
        std::string trackedInfo = "Tracked: (" + std::to_string(trackedPosition.x) + "," + 
                                 std::to_string(trackedPosition.y) + ")";
        std::string rawInfo = "Raw: (" + std::to_string(rawBestMatch.x) + "," + 
                             std::to_string(rawBestMatch.y) + ")";
        std::string frameInfo = "Frame: " + std::to_string(frameNum) + "/" + std::to_string(frameCount);
        
        // Stato capsula
        std::string capsuleStatus = (bluePixelCount > 1000) ? "Capsula PIENA" : "Capsula vuota";
        cv::Scalar capsuleColor = (bluePixelCount > 1000) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        
        std::string blueInfo = "Blue pixels: " + std::to_string(bluePixelCount);
        
        // Calcola distanza tra raw e tracked
        float distance = sqrt(pow(rawBestMatch.x - trackedPosition.x, 2) + 
                             pow(rawBestMatch.y - trackedPosition.y, 2));
        std::string distanceInfo = "Distance: " + std::to_string((int)distance) + "px";
        
        // Disegna testo
        cv::putText(frame, trackedInfo, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, trackColor, 2);
        cv::putText(frame, rawInfo, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, frameInfo, cv::Point(10, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, blueInfo, cv::Point(10, 120), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, distanceInfo, cv::Point(10, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Stato
        cv::putText(frame, capsuleStatus, cv::Point(10, 190), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, capsuleColor, 3);
        
        // Scrivi frame nel video
        writer.write(frame);
        
        // Progress ogni 30 frame
        if (frameNum % 30 == 0) {
            double progress = (double)frameNum / frameCount * 100.0;
            printf("Progresso: %.1f%% (Frame %d/%d) - Blue: %d - %s\n", 
                   progress, frameNum, frameCount, bluePixelCount, capsuleStatus.c_str());
        }
    }
    
    auto totalEnd = high_resolution_clock::now();
    double totalTime = duration_cast<milliseconds>(totalEnd - totalStart).count();
    
    // Stas finali
    printf("\n=== STATISTICHE ELABORAZIONE SEMPLIFICATA ===\n");
    printf("Frame processati: %d\n", frameNum);
    printf("Tempo totale: %.2f secondi\n", totalTime / 1000.0);
    printf("FPS medio: %.2f\n", frameNum / (totalTime / 1000.0));
    printf("ROI di ricerca: %dx%d (%.1f%% del frame)\n", 
           searchROI.width, searchROI.height, 
           (float)(searchROI.width * searchROI.height) / (frameWidth * frameHeight) * 100.0f);
    printf("\nTempo medio per frame:\n");
    printf("  Template matching: %.2f ms\n", totalTemplateTime / frameNum);
    printf("  Conteggio blu: %.2f ms\n", totalBlueTime / frameNum);
    printf("  Totale elaborazione: %.2f ms\n", 
           (totalTemplateTime + totalBlueTime) / frameNum);
    printf("\nSpeedup rispetto a tempo reale: %.2fx\n", 
           (frameNum / fps) / (totalTime / 1000.0));
    
    // Cleanup
    cap.release();
    writer.release();
    
    printf("\nVideo semplificato salvato in: %s\n", outputPath.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <video_input> <template_image> <video_output>\n", argv[0]);
        printf("Esempio: %s input.mp4 template.png output.mp4\n", argv[0]);
        return -1;
    }
    
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed! Controllare che sia presente una GPU CUDA.\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Memoria: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    std::string videoPath = argv[1];
    std::string templatePath = argv[2];
    std::string outputPath = argv[3];
    
    printf("\nParametri:\n");
    printf("  Video input: %s\n", videoPath.c_str());
    printf("  Template: %s\n", templatePath.c_str());
    printf("  Video output: %s\n", outputPath.c_str());
    
    int result = processVideo(videoPath, templatePath, outputPath);
    
    cudaDeviceReset();
    return result;
}