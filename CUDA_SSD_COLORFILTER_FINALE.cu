#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "tMatchSeq.h"

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 32
#define BLUE_BLOCK_SIZE 16

// Range HSV per il blu (compatibili con OpenCV)
const unsigned char BLUE_H_MIN = 90;
const unsigned char BLUE_H_MAX = 140;
const unsigned char BLUE_S_MIN = 30;
const unsigned char BLUE_V_MIN = 30;

// ============================================================================
// KERNELS PER TEMPLATE MATCHING SSD
// ============================================================================

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

// Kernel CUDA ottimizzato per calcolare SSD
__global__ void computeSSD(
    float *imageSum,     // Immagine integrale della sorgente
    float *imageSqSum,   // Immagine integrale dei quadrati
    float templateSum,   // Somma dei pixel del template
    float templateSqSum, // Somma dei quadrati dei pixel del template
    int width,
    int height,
    int kx,
    int ky,
    float *ssdResult,
    float *crossCorrelation)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - kx + 1 || y >= height - ky + 1)
        return;

    // Calcolo delle somme usando le immagini integrali
    float S1 = getRegionSum(imageSum, width, height, x, y, kx, ky);   // Somma della regione immagine
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky); // Somma dei quadrati
    float SC = crossCorrelation[INDEX(x, y, width - kx + 1)];
    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * SC + templateSqSum;

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}

__global__ void multiply(
    float *imageSum, // Immagine integrale della sorgente
    int width,
    int height,
    float *d_imageSqSum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width + 1 || y >= height + 1)
        return;

    d_imageSqSum[INDEX(x, y, width)] = imageSum[INDEX(x, y, width)] * imageSum[INDEX(x, y, width)];
}

// ============================================================================
// KERNELS PER CONTEGGIO PIXEL BLU
// ============================================================================

// Funzione device per conversione BGR a HSV corretta
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
                                     int* blueCount, uchar3* debugImage) {
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
            if (debugImage) {
                // Marca i pixel blu in rosso nell'immagine di debug
                debugImage[idx] = make_uchar3(0, 0, 255); // BGR: rosso
            }
        } else if (debugImage) {
            debugImage[idx] = pixel; // Mantieni colore originale
        }
    }
}

// ============================================================================
// FUNZIONI PRINCIPALI
// ============================================================================

// Funzione principale per il template matching
cudaError_t templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLoc,
    cv::Point *bestLocSeq)
{
    // Conversione delle immagini in float
    cv::Mat imageN, templN;
    image.convertTo(imageN, CV_32F, 1.0 / 255.0);
    templ.convertTo(templN, CV_32F, 1.0 / 255.0);

    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    // Allocazione memoria su device
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult, *d_rowSum, *d_imageSq, *d_rowSqSum, *d_crossCorrelation;

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

    cudaStatus = cudaMalloc(&d_imageSq, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMalloc(&d_ssdResult, resultSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMalloc(&d_crossCorrelation, resultSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMalloc(&d_rowSum, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    cudaStatus = cudaMalloc(&d_rowSqSum, imageSize);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    // Copia immagine su device
    cudaStatus = cudaMemcpy(d_image, imageN.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    // Calcolo delle immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calcolo immagini integrali
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_image, d_rowSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSum, d_imageSum, width, height);

    multiply<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_imageSq, d_rowSqSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSqSum, d_imageSqSum, width, height);

    // Calcolo somme template
    cv::Mat templeSq;
    cv::multiply(templN, templN, templeSq);

    cv::Mat temp_integral_img, temp_integral_Sq_img;
    cv::integral(templN, temp_integral_img, CV_32F);
    cv::integral(templeSq, temp_integral_Sq_img, CV_32F);
    
    temp_integral_img = temp_integral_img(cv::Rect(1, 1, templ.cols, templ.rows));
    temp_integral_Sq_img = temp_integral_Sq_img(cv::Rect(1, 1, templ.cols, templ.rows));

    float templateSum = temp_integral_img.at<float>(ky - 1, kx - 1);
    float templateSqSum = temp_integral_Sq_img.at<float>(ky - 1, kx - 1);

    // Calcolo cross correlation
    cv::Mat crossCorrelation;
    crossCorrelation = crossCorrelationFFT(imageN, templN);
    cudaStatus = cudaMemcpy(d_crossCorrelation, crossCorrelation.ptr<float>(), resultSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    // Calcolo SSD
    dim3 gridSSDSize((width - kx + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height - ky + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeSSD<<<gridSSDSize, blockSize>>>(
        d_imageSum,
        d_imageSqSum,
        templateSum,
        templateSqSum,
        width,
        height,
        kx,
        ky,
        d_ssdResult,
        d_crossCorrelation);

    // Copia risultati su host
    cv::Mat ssdResult(height - ky + 1, width - kx + 1, CV_32F);
    cudaStatus = cudaMemcpy(ssdResult.ptr<float>(), d_ssdResult, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) return cudaStatus;

    // Trova la posizione del minimo SSD
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(ssdResult, &minVal, &maxVal, &minLoc, &maxLoc);
    *bestLoc = minLoc;

    // Calcolo sequenziale per confronto
    cv::Mat integralSeq_img, integralSeq_Sq_img;
    computeIntegralImagesSequential(imageN, integralSeq_Sq_img);
    
    cv::Mat seqSSDResult;
    computeSSDSequentialWithIntegrals(integralSeq_Sq_img, templateSqSum, width, height, kx, ky, seqSSDResult, crossCorrelation);
    
    double minValSeq, maxValSeq;
    cv::Point minLocSeq, maxLocSeq;
    cv::minMaxLoc(seqSSDResult, &minValSeq, &maxValSeq, &minLocSeq, &maxLocSeq);
    *bestLocSeq = minLocSeq;

    printf("Template matching completato - CUDA: (%d,%d), Sequenziale: (%d,%d)\n", 
           bestLoc->x, bestLoc->y, bestLocSeq->x, bestLocSeq->y);

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_imageSum);
    cudaFree(d_imageSqSum);
    cudaFree(d_ssdResult);
    cudaFree(d_rowSum);
    cudaFree(d_imageSq);
    cudaFree(d_rowSqSum);
    cudaFree(d_crossCorrelation);

    return cudaStatus;
}

// Funzione per contare i pixel blu in una ROI
void countBlueInROI(cv::Mat& image, const cv::Rect& roi, int& blueCount, cv::Mat& debugImage) {
    // Verifica che l'ROI sia valido
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > image.cols || 
        roi.y + roi.height > image.rows) {
        std::cerr << "ROI fuori dai bounds dell'immagine!" << std::endl;
        return;
    }

    printf("Analizzando ROI: x=%d, y=%d, width=%d, height=%d\n", roi.x, roi.y, roi.width, roi.height);

    // Alloca memoria device
    uchar3* d_image = nullptr;
    uchar3* d_debugImage = nullptr;
    int* d_blueCount = nullptr;

    size_t imageSize = image.rows * image.cols * sizeof(uchar3);
    
    // Converti l'immagine OpenCV in formato adatto per CUDA
    std::vector<uchar3> hostImage(image.rows * image.cols);
    for (int y = 0; y < image.rows; ++y) {
        cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < image.cols; ++x) {
            hostImage[y * image.cols + x] = make_uchar3(row[x][0], row[x][1], row[x][2]);
        }
    }

    // Alloca e copia dati su device
    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, hostImage.data(), imageSize, cudaMemcpyHostToDevice);

    // Alloca memoria per debug image (opzionale)
    if (!debugImage.empty()) {
        cudaMalloc(&d_debugImage, imageSize);
    }
    
    // Alloca e inizializza contatore
    cudaMalloc(&d_blueCount, sizeof(int));
    cudaMemset(d_blueCount, 0, sizeof(int));

    // Configura kernel
    dim3 blockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 gridSize((roi.width + blockSize.x - 1) / blockSize.x, 
                 (roi.height + blockSize.y - 1) / blockSize.y);

    // Esegui kernel
    countBluePixelsKernel<<<gridSize, blockSize>>>(d_image, image.cols, image.rows,
                                                 roi.x, roi.y, roi.width, roi.height,
                                                 d_blueCount, d_debugImage);

    // Controlla errori CUDA
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Errore kernel CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    
    cudaDeviceSynchronize();

    // Copia risultati
    cudaMemcpy(&blueCount, d_blueCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Se richiesto, copia l'immagine di debug
    if (!debugImage.empty() && d_debugImage) {
        std::vector<uchar3> debugHost(image.rows * image.cols);
        cudaMemcpy(debugHost.data(), d_debugImage, imageSize, cudaMemcpyDeviceToHost);
        
        for (int y = 0; y < image.rows; ++y) {
            cv::Vec3b* row = debugImage.ptr<cv::Vec3b>(y);
            for (int x = 0; x < image.cols; ++x) {
                uchar3 pixel = debugHost[y * image.cols + x];
                row[x] = cv::Vec3b(pixel.x, pixel.y, pixel.z);
            }
        }
    }

    // Libera memoria
    cudaFree(d_image);
    if (d_debugImage) cudaFree(d_debugImage);
    cudaFree(d_blueCount);
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

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main()
{
    // Carica le immagini
    cv::Mat image = cv::imread("sourceC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("sourceC.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("templateC.jpg", cv::IMREAD_GRAYSCALE);
    
    float scaleFactor = 1; // Fattore di ridimensionamento
    cv::Mat imageR, templateR, imageColorR;
    cv::resize(image, imageR, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    cv::resize(templ, templateR, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    cv::resize(imageColor, imageColorR, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    if (image.empty() || templ.empty() || imageColor.empty())
    {
        printf("Errore nel caricamento delle immagini\n");
        return -1;
    }

    printf("=== FASE 1: TEMPLATE MATCHING ===\n");
    
    // Esegui template matching
    cv::Point bestLoc, bestLocSeq, LocCheck;
    cudaError_t status = templateMatchingSSD(imageR, templateR, &bestLoc, &bestLocSeq);
    seq_templateMatchingSSD(imageR, templateR, &LocCheck);

    if (status != cudaSuccess)
    {
        printf("Errore CUDA nel template matching: %s\n", cudaGetErrorString(status));
        return -1;
    }

    printf("Template matching completato con successo!\n");
    printf("Posizione trovata (CUDA): (%d, %d)\n", bestLoc.x, bestLoc.y);
    printf("Posizione trovata (Sequenziale): (%d, %d)\n", bestLocSeq.x, bestLocSeq.y);

    printf("\n=== FASE 2: CONTEGGIO PIXEL BLU ===\n");
    
    // Crea ROI basata sul risultato del template matching
    cv::Rect roi(bestLoc.x, bestLoc.y, templateR.cols, templateR.rows);
    
    // Verifica e correggi ROI se necessario
    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    roi.width = std::min(roi.width, imageColorR.cols - roi.x);
    roi.height = std::min(roi.height, imageColorR.rows - roi.y);
    
    printf("ROI per analisi blu: x=%d, y=%d, width=%d, height=%d\n", 
           roi.x, roi.y, roi.width, roi.height);
    
    // Testa prima con OpenCV per confronto
    testBlueDetectionOpenCV(imageColorR, roi);
    
    // Conta pixel blu nell'ROI con CUDA
    int blueCount = 0;
    cv::Mat debugImage = cv::Mat::zeros(imageColorR.rows, imageColorR.cols, CV_8UC3);
    countBlueInROI(imageColorR, roi, blueCount, debugImage);
    
    // Visualizza risultati
    printf("Pixel blu trovati con CUDA: %d\n", blueCount);
    printf("Area analizzata: %dx%d (%d pixel totali)\n", 
           roi.width, roi.height, roi.width * roi.height);
    printf("Percentuale pixel blu: %.2f%%\n", 
           (100.0 * blueCount) / (roi.width * roi.height));

    printf("\n=== FASE 3: SALVATAGGIO RISULTATI ===\n");
    
    // Disegna rettangolo per template matching (verde)
    bestLoc.x = static_cast<int>(bestLoc.x / scaleFactor);
    bestLoc.y = static_cast<int>(bestLoc.y / scaleFactor);
    cv::Rect displayRect(bestLoc, cv::Size(templ.cols, templ.rows));
    cv::rectangle(imageColor, displayRect, cv::Scalar(0, 255, 0), 2);
    
    // Aggiungi testo con informazioni
    std::string infoText = "Blu: " + std::to_string(blueCount) + " px (" + 
                          std::to_string((int)((100.0 * blueCount) / (roi.width * roi.height))) + "%)";
    cv::putText(imageColor, infoText, cv::Point(bestLoc.x, bestLoc.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    // Salva risultati
    cv::imwrite("unified_result.jpg", imageColor);
    
    if (!debugImage.empty()) {
        cv::imwrite("debug_blue_pixels.jpg", debugImage);
    }
    
    printf("Risultati salvati in 'unified_result.jpg' e 'debug_blue_pixels.jpg'\n");
    printf("Analisi completata con successo!\n");

    return 0;
}