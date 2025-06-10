#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
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
// KERNELS PER TEMPLATE MATCHING SSD (invariati dal codice originale)
// ============================================================================

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

__device__ int getRegionSum(const float *sumTable, int width, int height, int x, int y, int kx, int ky)
{
    int x1 = x - 1;
    int y1 = y - 1;
    int x2 = min(x + kx - 1, width - 1);
    int y2 = min(y + ky - 1, height - ky - 1);

    float A = (x1 >= 0 && y1 >= 0) ? sumTable[INDEX(x1, y1, width)] : 0.0f;
    float B = (y1 >= 0) ? sumTable[INDEX(x2, y1, width)] : 0.0f;
    float C = (x1 >= 0) ? sumTable[INDEX(x1, y2, width)] : 0.0f;
    float D = sumTable[INDEX(x2, y2, width)];

    return D - B - C + A;
}

__global__ void computeSSD(
    float *imageSum,
    float *imageSqSum,
    float templateSum,
    float templateSqSum,
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

    float S1 = getRegionSum(imageSum, width, height, x, y, kx, ky);
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky);
    float SC = crossCorrelation[INDEX(x, y, width - kx + 1)];
    float ssd = S2 - 2 * SC + templateSqSum;

    ssdResult[INDEX(x, y, width - kx + 1)] = ssd;
}

__global__ void multiply(float *imageSum, int width, int height, float *d_imageSqSum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width + 1 || y >= height + 1)
        return;

    d_imageSqSum[INDEX(x, y, width)] = imageSum[INDEX(x, y, width)] * imageSum[INDEX(x, y, width)];
}

// ============================================================================
// KERNELS PER CONTEGGIO PIXEL BLU (invariati dal codice originale)
// ============================================================================

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
                                     int* blueCount, uchar3* debugImage) {
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
            if (debugImage) {
                debugImage[idx] = make_uchar3(0, 0, 255);
            }
        } else if (debugImage) {
            debugImage[idx] = pixel;
        }
    }
}

// ============================================================================
// CLASSE PER GESTIONE MEMORIA PERSISTENTE GPU
// ============================================================================

class GPUMemoryManager {
private:
    // Memoria per template matching
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult;
    float *d_rowSum, *d_imageSq, *d_rowSqSum, *d_crossCorrelation;
    
    // Memoria per conteggio pixel blu
    uchar3 *d_colorImage, *d_debugImage;
    int *d_blueCount;
    
    // Dimensioni correnti
    int currentWidth, currentHeight;
    int currentResultWidth, currentResultHeight;
    
    bool initialized;

public:
    GPUMemoryManager() : initialized(false), currentWidth(0), currentHeight(0) {}
    
    ~GPUMemoryManager() {
        cleanup();
    }
    
    bool initialize(int width, int height, int templateWidth, int templateHeight) {
        if (initialized && width == currentWidth && height == currentHeight) {
            return true; // Già inizializzato con le dimensioni corrette
        }
        
        cleanup(); // Libera memoria precedente se presente
        
        size_t imageSize = width * height * sizeof(float);
        size_t colorImageSize = width * height * sizeof(uchar3);
        size_t resultSize = (width - templateWidth + 1) * (height - templateHeight + 1) * sizeof(float);
        
        // Alloca memoria per template matching
        if (cudaMalloc(&d_image, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSqSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_imageSq, imageSize) != cudaSuccess ||
            cudaMalloc(&d_rowSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_rowSqSum, imageSize) != cudaSuccess ||
            cudaMalloc(&d_ssdResult, resultSize) != cudaSuccess ||
            cudaMalloc(&d_crossCorrelation, resultSize) != cudaSuccess) {
            cleanup();
            return false;
        }
        
        // Alloca memoria per conteggio pixel blu
        if (cudaMalloc(&d_colorImage, colorImageSize) != cudaSuccess ||
            cudaMalloc(&d_debugImage, colorImageSize) != cudaSuccess ||
            cudaMalloc(&d_blueCount, sizeof(int)) != cudaSuccess) {
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
    
    void cleanup() {
        if (initialized) {
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
            initialized = false;
        }
    }
    
    // Getter per accesso ai puntatori memoria
    float* getImagePtr() { return d_image; }
    float* getImageSumPtr() { return d_imageSum; }
    float* getImageSqSumPtr() { return d_imageSqSum; }
    float* getImageSqPtr() { return d_imageSq; }
    float* getRowSumPtr() { return d_rowSum; }
    float* getRowSqSumPtr() { return d_rowSqSum; }
    float* getSSDResultPtr() { return d_ssdResult; }
    float* getCrossCorrelationPtr() { return d_crossCorrelation; }
    uchar3* getColorImagePtr() { return d_colorImage; }
    uchar3* getDebugImagePtr() { return d_debugImage; }
    int* getBlueCountPtr() { return d_blueCount; }
    
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
    float templateSum,
    float templateSqSum)
{
    // Conversione frame in float
    cv::Mat frameFloat;
    grayFrame.convertTo(frameFloat, CV_32F, 1.0 / 255.0);
    
    int width = grayFrame.cols;
    int height = grayFrame.rows;
    int kx = templ.cols;
    int ky = templ.rows;
    
    // Inizializza memoria GPU se necessario
    if (!memManager.initialize(width, height, kx, ky)) {
        return cudaErrorMemoryAllocation;
    }
    
    // Copia frame su GPU
    size_t imageSize = width * height * sizeof(float);
    size_t colorImageSize = width * height * sizeof(uchar3);
    
    cudaError_t status = cudaMemcpy(memManager.getImagePtr(), frameFloat.ptr<float>(), 
                                   imageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Copia frame a colori
    std::vector<uchar3> hostColorImage(height * width);
    for (int y = 0; y < height; ++y) {
        cv::Vec3b* row = colorFrame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            hostColorImage[y * width + x] = make_uchar3(row[x][0], row[x][1], row[x][2]);
        }
    }
    status = cudaMemcpy(memManager.getColorImagePtr(), hostColorImage.data(), 
                       colorImageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Configurazione kernel
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Calcolo immagini integrali
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getImagePtr(), memManager.getRowSumPtr(), width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getRowSumPtr(), memManager.getImageSumPtr(), width, height);
    
    multiply<<<gridSize, blockSize>>>(memManager.getImagePtr(), width, height, memManager.getImageSqPtr());
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getImageSqPtr(), memManager.getRowSqSumPtr(), width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getRowSqSumPtr(), memManager.getImageSqSumPtr(), width, height);
    
    // Calcolo cross correlation (questo richiede ancora CPU per FFT)
    cv::Mat crossCorrelation = crossCorrelationFFT(frameFloat, templ);
    size_t resultSize = memManager.getResultWidth() * memManager.getResultHeight() * sizeof(float);
    status = cudaMemcpy(memManager.getCrossCorrelationPtr(), crossCorrelation.ptr<float>(), 
                       resultSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Calcolo SSD
    dim3 gridSSDSize((memManager.getResultWidth() + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                     (memManager.getResultHeight() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeSSD<<<gridSSDSize, blockSize>>>(
        memManager.getImageSumPtr(),
        memManager.getImageSqSumPtr(),
        templateSum,
        templateSqSum,
        width, height, kx, ky,
        memManager.getSSDResultPtr(),
        memManager.getCrossCorrelationPtr());
    
    // Trova minimo SSD (questo richiede copia su CPU)
    cv::Mat ssdResult(memManager.getResultHeight(), memManager.getResultWidth(), CV_32F);
    status = cudaMemcpy(ssdResult.ptr<float>(), memManager.getSSDResultPtr(), 
                       resultSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    
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
    cudaMemset(memManager.getBlueCountPtr(), 0, sizeof(int));
    
    // Configura kernel per conteggio blu
    dim3 blueBlockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 blueGridSize((roi.width + blueBlockSize.x - 1) / blueBlockSize.x, 
                     (roi.height + blueBlockSize.y - 1) / blueBlockSize.y);
    
    countBluePixelsKernel<<<blueGridSize, blueBlockSize>>>(
        memManager.getColorImagePtr(), width, height,
        roi.x, roi.y, roi.width, roi.height,
        memManager.getBlueCountPtr(), nullptr);
    
    // Copia risultato conteggio blu
    status = cudaMemcpy(&blueCount, memManager.getBlueCountPtr(), sizeof(int), cudaMemcpyDeviceToHost);
    
    return status;
}

// Funzione per precalcolare i parametri del template (chiamata una sola volta)
void precomputeTemplateParameters(const cv::Mat &templ, float &templateSum, float &templateSqSum) {
    cv::Mat templFloat;
    templ.convertTo(templFloat, CV_32F, 1.0 / 255.0);
    
    cv::Mat templSq;
    cv::multiply(templFloat, templFloat, templSq);
    
    cv::Mat temp_integral_img, temp_integral_Sq_img;
    cv::integral(templFloat, temp_integral_img, CV_32F);
    cv::integral(templSq, temp_integral_Sq_img, CV_32F);
    
    temp_integral_img = temp_integral_img(cv::Rect(1, 1, templ.cols, templ.rows));
    temp_integral_Sq_img = temp_integral_Sq_img(cv::Rect(1, 1, templ.cols, templ.rows));
    
    templateSum = temp_integral_img.at<float>(templ.rows - 1, templ.cols - 1);
    templateSqSum = temp_integral_Sq_img.at<float>(templ.rows - 1, templ.cols - 1);
}

// ============================================================================
// MAIN FUNCTION PER VIDEO PROCESSING
// ============================================================================

int main()
{
    // Carica template
    cv::Mat templ = cv::imread("templateC.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat templFloat;
    templ.convertTo(templFloat, CV_32F, 1.0 / 255.0);
    
    if (templ.empty()) {
        printf("Errore nel caricamento del template\n");
        return -1;
    }
    
    // Precalcola parametri template
    float templateSum, templateSqSum;
    precomputeTemplateParameters(templ, templateSum, templateSqSum);
    
    // Apri video (webcam o file)
    cv::VideoCapture cap;
    
    // Prova prima webcam, poi file video
    cap.open(0); // Webcam
    if (!cap.isOpened()) {
        cap.open("input_video.mp4"); // File video
        if (!cap.isOpened()) {
            printf("Errore nell'apertura del video\n");
            return -1;
        }
    }
    
    // Configurazione video output (opzionale)
    cv::VideoWriter writer;
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    if (fps <= 0) fps = 30;
    
    // Manager memoria GPU
    GPUMemoryManager memManager;
    
    // Variabili per timing
    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    
    cv::Mat frame, grayFrame;
    
    printf("=== INIZIO ELABORAZIONE VIDEO ===\n");
    printf("Premi 'q' per uscire\n");
    
    while (true) {
        // Cattura frame
        if (!cap.read(frame)) {
            printf("Fine video o errore nella cattura\n");
            break;
        }
        
        // Converti in scala di grigi
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        
        // Processa frame
        cv::Point bestLoc;
        int blueCount = 0;
        
        cudaError_t status = processVideoFrame(grayFrame, frame, templFloat, memManager, 
                                             bestLoc, blueCount, templateSum, templateSqSum);
        
        if (status != cudaSuccess) {
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
        if (frameCount % 30 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
            double fps = (frameCount * 1000.0) / duration.count();
            printf("FPS: %.2f, Frame: %d, Posizione: (%d,%d), Pixel blu: %d\n", 
                   fps, frameCount, bestLoc.x, bestLoc.y, blueCount);
        }
        
        // Mostra frame
        cv::imshow("Template Matching + Blue Detection", frame);
        
        // Salva video (opzionale)
        if (!writer.isOpened() && frameCount == 1) {
            writer.open("output_video.mp4", cv::VideoWriter::fourcc('X','V','I','D'), 
                       fps, frame.size(), true);
        }
        if (writer.isOpened()) {
            writer.write(frame);
        }
        
        // Controllo uscita
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' o ESC
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