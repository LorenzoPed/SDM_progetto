#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <limits>
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
// STRUTTURE DATI
// ============================================================================

struct SearchConfig {
    int roiX, roiY;           // Angolo superiore sinistro della ROI
    int roiWidth, roiHeight;  // Dimensioni della ROI
    int gridStepX, gridStepY; // Passo della griglia (multipli delle dimensioni template)
    bool useGrid;             // Se true usa ricerca a griglia, se false ricerca normale
    float threshold;          // Soglia per considerare un match valido
};

struct TemplateMatch {
    cv::Point position;
    float score;
    int bluePixels;
    bool isValid;
};

struct TrackingState {
    cv::Rect searchROI;      // Regione di ricerca corrente
    cv::Rect lastMatch;      // Ultima posizione trovata
    int lostCount = 0;       // Contatore frame persi
    float lastScore = 0;     // Ultimo score del match
    bool isTracking = false; // Se sta tracciando un oggetto
};

// ============================================================================
// KERNELS CUDA (rimangono invariati)
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

__device__ float getRegionSum(const float *sumTable, int width, int height, int x, int y, int kx, int ky)
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

__global__ void multiply(float *imageSum, int width, int height, float *d_imageSqSum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width + 1 || y >= height + 1)
        return;

    d_imageSqSum[INDEX(x, y, width)] = imageSum[INDEX(x, y, width)] * imageSum[INDEX(x, y, width)];
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

__global__ void computeSSDGrid(
    float *imageSum,
    float *imageSqSum,
    float templateSum,
    float templateSqSum,
    int width,
    int height,
    int kx,
    int ky,
    float *ssdResult,
    float *crossCorrelation,
    int roiX, int roiY,
    int roiWidth, int roiHeight,
    int gridStepX, int gridStepY,
    bool useGrid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int x, y;
    int resultWidth = roiWidth - kx + 1;
    int resultHeight = roiHeight - ky + 1;
    
    if (useGrid) {
        // Calcola posizioni sulla griglia
        int gridCols = (roiWidth - kx) / gridStepX + 1;
        int gridRows = (roiHeight - ky) / gridStepY + 1;
        int totalGridPoints = gridCols * gridRows;
        
        if (idx >= totalGridPoints) return;
        
        int gridX = idx % gridCols;
        int gridY = idx / gridCols;
        
        x = roiX + gridX * gridStepX;
        y = roiY + gridY * gridStepY;
        
        // Assicurati che sia dentro i limiti
        if (x + kx > roiX + roiWidth || y + ky > roiY + roiHeight) return;
        
    } else {
        // Ricerca normale all'interno della ROI
        if (idx >= resultWidth * resultHeight) return;
        
        int localX = idx % resultWidth;
        int localY = idx / resultWidth;
        
        x = roiX + localX;
        y = roiY + localY;
    }
    
    // Verifica limiti generali
    if (x >= width - kx + 1 || y >= height - ky + 1) return;
    
    float S1 = getRegionSum(imageSum, width, height, x, y, kx, ky);
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky);
    
    // Calcola indice per cross correlation
    int ccIdx = (y - roiY) * resultWidth + (x - roiX);
    float SC = crossCorrelation[ccIdx];
    float ssd = S2 - 2 * SC + templateSqSum;
    
    // Salva risultato
    if (useGrid) {
        ssdResult[idx] = ssd;
    } else {
        ssdResult[ccIdx] = ssd;
    }
}

// ============================================================================
// CLASSE GPUMemoryManager (rimane invariata)
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
// FUNZIONI HELPER
// ============================================================================

void updateSearchROI(TrackingState& state, const cv::Size& frameSize, const cv::Size& templSize) {
    const int margin = 50; // Margine attorno all'oggetto
    
    if (state.isTracking) {
        // Calcola ROI centrata sull'ultima posizione con margine
        int width = state.lastMatch.width + 2 * margin;
        int height = state.lastMatch.height + 2 * margin;
        
        // Assicurati che la ROI sia dentro il frame
        state.searchROI.x = std::max(0, state.lastMatch.x - margin);
        state.searchROI.y = std::max(0, state.lastMatch.y - margin);
        state.searchROI.width = std::min(width, frameSize.width - state.searchROI.x);
        state.searchROI.height = std::min(height, frameSize.height - state.searchROI.y);
    } else {
        // Se non sta tracciando, cerca in tutto il frame
        state.searchROI = cv::Rect(0, 0, frameSize.width, frameSize.height);
    }
}

cudaError_t countBluePixelsInRect(GPUMemoryManager &memManager, const cv::Rect &rect, 
                                 int imageWidth, int imageHeight, int &blueCount) {
    // Reset contatore
    cudaMemset(memManager.getBlueCountPtr(), 0, sizeof(int));
    
    // Configura kernel
    dim3 blueBlockSize(BLUE_BLOCK_SIZE, BLUE_BLOCK_SIZE);
    dim3 blueGridSize((rect.width + blueBlockSize.x - 1) / blueBlockSize.x, 
                     (rect.height + blueBlockSize.y - 1) / blueBlockSize.y);
    
    countBluePixelsKernel<<<blueGridSize, blueBlockSize>>>(
        memManager.getColorImagePtr(), imageWidth, imageHeight,
        rect.x, rect.y, rect.width, rect.height,
        memManager.getBlueCountPtr(), nullptr);
    
    cudaDeviceSynchronize();
    
    // Copia risultato
    return cudaMemcpy(&blueCount, memManager.getBlueCountPtr(), sizeof(int), cudaMemcpyDeviceToHost);
}

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
// FUNZIONE PRINCIPALE PER ELABORAZIONE FRAME
// ============================================================================

cudaError_t processVideoFrameWithROI(
    const cv::Mat &grayFrame,
    const cv::Mat &colorFrame,
    const cv::Mat &templ,
    GPUMemoryManager &memManager,
    const SearchConfig &config,
    std::vector<TemplateMatch> &matches,
    float templateSum,
    float templateSqSum)
{
    cv::Mat frameFloat;
    grayFrame.convertTo(frameFloat, CV_32F, 1.0 / 255.0);
    
    int width = grayFrame.cols;
    int height = grayFrame.rows;
    int kx = templ.cols;
    int ky = templ.rows;
    
    // Verifica che la ROI sia valida
    SearchConfig validConfig = config;
    validConfig.roiX = std::max(0, std::min(validConfig.roiX, width - kx));
    validConfig.roiY = std::max(0, std::min(validConfig.roiY, height - ky));
    validConfig.roiWidth = std::min(validConfig.roiWidth, width - validConfig.roiX);
    validConfig.roiHeight = std::min(validConfig.roiHeight, height - validConfig.roiY);
    
    // Inizializza memoria GPU
    if (!memManager.initialize(width, height, kx, ky)) {
        return cudaErrorMemoryAllocation;
    }
    
    // Copia dati su GPU
    size_t imageSize = width * height * sizeof(float);
    size_t colorImageSize = width * height * sizeof(uchar3);
    
    cudaError_t status = cudaMemcpy(memManager.getImagePtr(), frameFloat.ptr<float>(), 
                                   imageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Copia frame a colori
    std::vector<uchar3> hostColorImage(height * width);
    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = colorFrame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            hostColorImage[y * width + x] = make_uchar3(row[x][0], row[x][1], row[x][2]);
        }
    }
    status = cudaMemcpy(memManager.getColorImagePtr(), hostColorImage.data(), 
                       colorImageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Calcolo immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getImagePtr(), memManager.getRowSumPtr(), width, height);
    cudaDeviceSynchronize();
    
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getRowSumPtr(), memManager.getImageSumPtr(), width, height);
    cudaDeviceSynchronize();
    
    multiply<<<gridSize, blockSize>>>(memManager.getImagePtr(), width, height, memManager.getImageSqPtr());
    cudaDeviceSynchronize();
    
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getImageSqPtr(), memManager.getRowSqSumPtr(), width, height);
    cudaDeviceSynchronize();
    
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        memManager.getRowSqSumPtr(), memManager.getImageSqSumPtr(), width, height);
    cudaDeviceSynchronize();
    
    // Calcolo cross correlation solo per la ROI
    cv::Rect roi(validConfig.roiX, validConfig.roiY, validConfig.roiWidth, validConfig.roiHeight);
    cv::Mat roiFrame = frameFloat(roi);
    cv::Mat crossCorrelation = crossCorrelationFFT(roiFrame, templ);
    
    size_t ccSize = crossCorrelation.rows * crossCorrelation.cols * sizeof(float);
    status = cudaMemcpy(memManager.getCrossCorrelationPtr(), crossCorrelation.ptr<float>(), 
                       ccSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;
    
    // Calcola numero di punti da processare
    int numPoints;
    if (validConfig.useGrid) {
        int gridCols = (validConfig.roiWidth - kx) / validConfig.gridStepX + 1;
        int gridRows = (validConfig.roiHeight - ky) / validConfig.gridStepY + 1;
        numPoints = gridCols * gridRows;
    } else {
        numPoints = (validConfig.roiWidth - kx + 1) * (validConfig.roiHeight - ky + 1);
    }
    
    // Launch kernel modificato
    int blocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    computeSSDGrid<<<blocks, threadsPerBlock>>>(
        memManager.getImageSumPtr(),
        memManager.getImageSqSumPtr(),
        templateSum,
        templateSqSum,
        width, height, kx, ky,
        memManager.getSSDResultPtr(),
        memManager.getCrossCorrelationPtr(),
        validConfig.roiX, validConfig.roiY,
        validConfig.roiWidth, validConfig.roiHeight,
        validConfig.gridStepX, validConfig.gridStepY,
        validConfig.useGrid);
    
    cudaDeviceSynchronize();
    
    // Copia risultati e trova tutti i match sotto la soglia
    size_t resultSize = numPoints * sizeof(float);
    std::vector<float> ssdResults(numPoints);
    status = cudaMemcpy(ssdResults.data(), memManager.getSSDResultPtr(), 
                       resultSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    
    // Trova tutti i match validi
    matches.clear();
    
    if (validConfig.useGrid) {
        int gridCols = (validConfig.roiWidth - kx) / validConfig.gridStepX + 1;
        
        for (int i = 0; i < numPoints; i++) {
            if (ssdResults[i] < validConfig.threshold) {
                int gridX = i % gridCols;
                int gridY = i / gridCols;
                
                cv::Point pos(validConfig.roiX + gridX * validConfig.gridStepX,
                             validConfig.roiY + gridY * validConfig.gridStepY);
                
                // Conta pixel blu per questo match
                int blueCount = 0;
                cv::Rect matchRect(pos, cv::Size(kx, ky));
                countBluePixelsInRect(memManager, matchRect, width, height, blueCount);
                
                TemplateMatch match;
                match.position = pos;
                match.score = ssdResults[i];
                match.bluePixels = blueCount;
                match.isValid = true;
                
                matches.push_back(match);
            }
        }
    } else {
        // Ricerca normale - trova il migliore
        int minIdx = std::min_element(ssdResults.begin(), ssdResults.end()) - ssdResults.begin();
        
        if (ssdResults[minIdx] < validConfig.threshold) {
            int resultWidth = validConfig.roiWidth - kx + 1;
            int localX = minIdx % resultWidth;
            int localY = minIdx / resultWidth;
            
            cv::Point pos(validConfig.roiX + localX, validConfig.roiY + localY);
            
            int blueCount = 0;
            cv::Rect matchRect(pos, cv::Size(kx, ky));
            countBluePixelsInRect(memManager, matchRect, width, height, blueCount);
            
            TemplateMatch match;
            match.position = pos;
            match.score = ssdResults[minIdx];
            match.bluePixels = blueCount;
            match.isValid = true;
            
            matches.push_back(match);
        }
    }
    
    return cudaSuccess;
}


// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Carica template
    cv::Mat templ = cv::imread("template.jpg", cv::IMREAD_GRAYSCALE);
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
    cap.open(0); // Prova prima webcam
    if (!cap.isOpened()) {
        cap.open("input_video.mp4"); // Poi file video
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
    
    // Stato del tracker
    TrackingState tracker;
    const cv::Size templSize = templ.size();
    const float trackingThreshold = 5000.0f; // Soglia per considerare un buon match
    const int maxLostFrames = 5; // Massimo frame persi prima di reset
    
    // Variabili per timing
    auto startTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    
    cv::Mat frame, grayFrame;
    std::vector<TemplateMatch> matches;
    
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
        
        // Aggiorna la ROI di ricerca
        updateSearchROI(tracker, frame.size(), templSize);
        
        // Configura la ricerca per la ROI corrente
        SearchConfig config;
        config.roiX = tracker.searchROI.x;
        config.roiY = tracker.searchROI.y;
        config.roiWidth = tracker.searchROI.width;
        config.roiHeight = tracker.searchROI.height;
        config.useGrid = false; // Ricerca densa nella ROI
        config.threshold = std::numeric_limits<float>::max(); // Prendi sempre il migliore
        
        // Processa frame
        cudaError_t status = processVideoFrameWithROI(grayFrame, frame, templFloat, 
                                                    memManager, config, matches, 
                                                    templateSum, templateSqSum);
        
        if (status != cudaSuccess) {
            printf("Errore CUDA nel processamento frame: %s\n", cudaGetErrorString(status));
            continue;
        }
        
        // Elabora i risultati
        if (!matches.empty()) {
            auto& bestMatch = matches[0]; // Prendi il migliore
            
            // Aggiorna lo stato del tracker
            tracker.lastMatch = cv::Rect(bestMatch.position, templSize);
            tracker.lastScore = bestMatch.score;
            
            if (bestMatch.score < trackingThreshold) {
                tracker.isTracking = true;
                tracker.lostCount = 0;
                
                // Conta pixel blu nell'area trovata
                int blueCount = 0;
                cv::Rect matchRect(bestMatch.position, templSize);
                countBluePixelsInRect(memManager, matchRect, frame.cols, frame.rows, blueCount);
                
                // Visualizza il risultato
                cv::rectangle(frame, tracker.lastMatch, cv::Scalar(0, 255, 0), 2);
                cv::rectangle(frame, tracker.searchROI, cv::Scalar(255, 0, 0), 1);
                
                std::string info = "Score: " + std::to_string((int)bestMatch.score) + 
                                 " | Blu: " + std::to_string(blueCount) +
                                 " | Tracking: ON";
                cv::putText(frame, info, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            } else {
                tracker.lostCount++;
                if (tracker.lostCount > maxLostFrames) {
                    tracker.isTracking = false;
                }
                
                std::string info = "Score: " + std::to_string((int)bestMatch.score) + 
                                 " | Tracking: LOW CONFIDENCE";
                cv::putText(frame, info, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
            }
        } else {
            tracker.lostCount++;
            if (tracker.lostCount > maxLostFrames) {
                tracker.isTracking = false;
            }
            
            std::string info = "Nessun match trovato | Tracking: " + 
                             std::string(tracker.isTracking ? "RECOVERING" : "OFF");
            cv::putText(frame, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
        // Calcola e mostra FPS
        frameCount++;
        if (frameCount % 10 == 0) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
            double fps = (10 * 1000.0) / duration.count();
            printf("FPS: %.2f, Frame: %d, Stato: %s, Score: %.1f\n", 
                   fps, frameCount, 
                   tracker.isTracking ? "Tracking" : "Ricerca", 
                   tracker.lastScore);
            startTime = currentTime;
        }
        
        // Mostra frame
        //cv::imshow("Dynamic ROI Template Matching", frame);
        
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
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    return 0;
}