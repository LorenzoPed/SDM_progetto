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
#define BLUE_BLOCK_SIZE 16

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
    float SC = crossCorrelation[INDEX(x, y, paddedCols)]; // INDEX(x, y, width) ((y) * (width) + (x)) 
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

// KERNEL per FFT

// Kernel per padding a zero
__global__ void padToZero(const float *src, float *dst,
                          int srcRows, int srcCols,      
                          int dstRows, int dstCols)   // Aggiunge una riga sotto e una colonna a destra, tutti 0.
{                                                     // e copia la matrice tale e quale
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
    // Pad to zero: evita l'aliasing, previene effetti di warparound nella convoluzione circolare
    dim3 b1(16, 16), g1((n + 15) / 16, (m + 15) / 16);
    padToZero<<<g1, b1>>>(d_image, d_imgPad, height, width, m, n); // 
    padToZero<<<g1, b1>>>(d_templ, d_tmpPad, ky, kx, m, n);

    // cuFFT plan: crea l'ambiente corretto per la creazione dellla FFT e IFFT
    cufftHandle planFwd, planInv; 
    cufftPlan2d(&planFwd, m, n, CUFFT_R2C); // crea un piano per la trasformata diretta
    cufftPlan2d(&planInv, m, n, CUFFT_C2R); // crea un piano per la trasformata inversa

    // FFT forward: passa dal dominio spaziale a quello delle frequenze 
    cufftExecR2C(planFwd, d_imgPad, d_imgFreq); 
    cufftExecR2C(planFwd, d_tmpPad, d_tmpFreq);

    // Multiply with conjugate: fa la matrice complessa coniugata
    dim3 b2(16, 16), g2((freqCols + 15) / 16, (m + 15) / 16);
    mulConjAndScale<<<g2, b2>>>(d_imgFreq, d_tmpFreq, m, freqCols); 

    // IFFT: ritorno nel dominio spaziale 
    cufftExecC2R(planInv, d_imgFreq, d_imgPad);

    // Normalize: bisogna mantenere il bilancio energetico tra dominio spaziale e frequenziale
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

// Kernel per COLOR FILTERING

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

// Kernel per la conversione da BGR a Uchar3
__global__ void convertBGRtoUchar3Kernel(uchar* cvImage, uchar3* cudaImage, int width, int height, size_t step) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int src_idx = y * step + x * 3;  // OpenCV step may include padding
        int dst_idx = y * width + x;
        
        // OpenCV stores as BGR
        cudaImage[dst_idx] = make_uchar3(cvImage[src_idx],     // B
                                     cvImage[src_idx + 1], // G
                                     cvImage[src_idx + 2]); // R
    }
}


// Funzione ottimizzata per contare i pixel blu in una ROI
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

    // PRIMO KERNEL: Converti da OpenCV BGR a uchar3 usando il tuo kernel esistente
    dim3 convertBlockSize(16, 16);
    dim3 convertGridSize((image.cols + convertBlockSize.x - 1) / convertBlockSize.x, 
                        (image.rows + convertBlockSize.y - 1) / convertBlockSize.y);
    
    convertBGRtoUchar3Kernel<<<convertGridSize, convertBlockSize>>>(
        d_cvImage, d_image, image.cols, image.rows, image.step
    );

    // Controlla errori del primo kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Errore kernel conversione: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // SECONDO KERNEL: Conta i pixel blu usando il kernel esistente
    dim3 countBlockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 countGridSize((roi.width + countBlockSize.x - 1) / countBlockSize.x, 
                      (roi.height + countBlockSize.y - 1) / countBlockSize.y);

    countBluePixelsKernel<<<countGridSize, countBlockSize>>>(
        d_image, image.cols, image.rows,
        roi.x, roi.y, roi.width, roi.height,
        d_blueCount
    );

    // Controlla errori del secondo kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Errore kernel conteggio: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    cudaDeviceSynchronize();

    // Copia risultati
    cudaMemcpy(&blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Libera memoria
    if (d_cvImage) cudaFree(d_cvImage);
    if (d_image) cudaFree(d_image);
    if (d_blueCount) cudaFree(d_blueCount);
}

// Funzione per testare la detection con OpenCV come riferimento
void testBlueDetectionOpenCV(cv::Mat& image, const cv::Rect& roi) {
    cv::Mat roiImg = image(roi);
    cv::Mat hsvRoi;
    cv::cvtColor(roiImg, hsvRoi, cv::COLOR_BGR2HSV);
    
    cv::Mat blueMask;
    cv::inRange(hsvRoi, cv::Scalar(BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN), 
                       cv::Scalar(BLUE_H_MAX, 255, 255), blueMask);
    
    int openCVBlueCount = cv::countNonZero(blueMask);
    std::cout << "Pixel blu rilevati da OpenCV (riferimento): " << openCVBlueCount << std::endl;
    
    // Salva la maschera per debug
    cv::imwrite("opencv_blue_mask.jpg", blueMask);
}

int main()
{
    // Carica le immagini
    cv::Mat image = cv::imread("immagini/sourceC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("immagini/sourceC.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("immagini/templateC.jpg", cv::IMREAD_GRAYSCALE);
    float scaleFactor = 1; // Fattore di ridimensionamento
    //cv::Mat image, template;
    //cv::resize(image, image, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    //cv::resize(templ, template, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

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
    
    // Testa prima con OpenCV per confronto
    testBlueDetectionOpenCV(imageColor, roi);
    
    // Conta pixel blu nell'ROI con CUDA
    int blueCount = 0;

    auto start_colorGpu = high_resolution_clock::now();
    countBlueInROI(imageColor, roi, blueCount);
    auto end_colorGpu = high_resolution_clock::now();
    auto duration_colorGPU = duration_cast<microseconds>(end_colorGpu - start_colorGpu);

    // Conta dei pixel blu nella ROI sequenziale
    auto start_colorSeq = high_resolution_clock::now();
    int seq_blue = seq_countBluePixels(imageColor, roi.x, roi.y, roi.width, roi.height);
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

    // Aggiungi testo con informazioni
    std::string infoText = "Blu: " + std::to_string(blueCount) + " px (" + 
                          std::to_string((int)((100.0 * blueCount) / (roi.width * roi.height))) + "%)";
    cv::putText(imageColor, infoText, cv::Point(bestLoc.x, bestLoc.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    std::string seq_infoText = "Blu: " + std::to_string(seq_blue) + " px (" + 
                          std::to_string((int)((100.0 * blueCount) / (roi.width * roi.height))) + "%)";
    cv::putText(imageColor, seq_infoText, cv::Point(LocCheck.x, LocCheck.y + 10), 
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
    // cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    // imshow("Immagine principale", imageColor);
    // cv::waitKey(0);

    return 0;
}