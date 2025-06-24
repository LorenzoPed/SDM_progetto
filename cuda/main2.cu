#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "tMatchSeq.h"

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
    //    float S1 = getRegionSum(imageSum, width, height, x, y, kx, ky);   // Somma della regione immagine
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

// Funzione principale per il template matching
cudaError_t templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLoc,
    cv::Point *bestLocSeq)
{
    // Conversione delle immagini in float
    // image.convertTo(imagefloat, CV_32F, 1.0 / 255.0); // Normalizza i valori tra 0 e 1
    cv::Mat imageN, templN;
    image.convertTo(imageN, CV_32F, 1.0 / 255.0);
    templ.convertTo(templN, CV_32F, 1.0 / 255.0);

    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    // Calcolo delle somme del template
    float templateSum = 0;
    float templateSqSum = 0;
    // for (int i = 0; i < kx * ky; i++)
    //{
    //     float val = templN.ptr<float>()[i];
    //     templateSum += val;
    //     templateSqSum += val * val;
    // }
    //  Allocazione memoria su device
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult, *d_rowSum, *d_imageSq, *d_rowSqSum, *d_templSum, *d_templSqSum, *d_crossCorrelation;

    size_t templSize = kx * ky * sizeof(float);
    size_t imageSize = width * height * sizeof(float);
    size_t resultSize = (width - kx + 1) * (height - ky + 1) * sizeof(float);

    cudaError_t cudaStatus;

    // Allocazione memoria su device
    cudaStatus = cudaMalloc(&d_image, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_imageSum, imageSize);
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

    cudaStatus = cudaMalloc(&d_rowSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_rowSqSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSum, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_templSqSum, templSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Copia immagine su device e creazione immagine dei quadrati
    cudaStatus = cudaMemcpy(d_image, imageN.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Creazione dell'immagine dei quadrati
    // cv::Mat imageSq;
    // cv::multiply(imageN, imageN, imageSq);
    // cudaStatus = cudaMemcpy(d_imageSq, imageSq.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    // if (cudaStatus != cudaSuccess)
    //     return cudaStatus;

    // Calcolo delle immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calcolo immagini integrali
    //   rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_image, d_rowSum, width, height);
    //   colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSum, d_imageSum, width, height);

    multiply<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_imageSq, d_rowSqSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSqSum, d_imageSqSum, width, height);

    cv::Mat integralSeq_img;
    cv::Mat integralSeq_Sq_img;
    computeIntegralImagesSequential(imageN, integralSeq_img, integralSeq_Sq_img);

    // template
    cv::Mat temp_integral_img;
    cv::integral(templN, temp_integral_img, CV_32F);
    temp_integral_img = temp_integral_img(cv::Rect(1, 1, templ.cols, templ.rows));

    // templatesq
    cv::Mat templeSq;
    cv::multiply(templN, templN, templeSq);

    cv::Mat temp_integral_Sq_img;
    cv::integral(templeSq, temp_integral_Sq_img, CV_32F);
    temp_integral_Sq_img = temp_integral_Sq_img(cv::Rect(1, 1, templ.cols, templ.rows));

    //     int last_value = mat.at<int>(mat.rows - 1, mat.cols - 1);
    templateSum = temp_integral_img.at<float>(ky - 1, kx - 1);
    templateSqSum = temp_integral_Sq_img.at<float>(ky - 1, kx - 1);

    // Copia delle immagini integrali calcolate da CUDA su host
    cv::Mat cudaIntegralImage(height, width, CV_32F);
    cv::Mat cudaIntegralSqImage(height, width, CV_32F);

    cudaStatus = cudaMemcpy(cudaIntegralImage.ptr<float>(), d_imageSum, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMemcpy(cudaIntegralSqImage.ptr<float>(), d_imageSqSum, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Confronto delle immagini integrali
    bool integralMatch = compareImages(cudaIntegralImage, integralSeq_img);
    bool integralSqMatch = compareImages(cudaIntegralSqImage, integralSeq_Sq_img);

    if (!integralMatch || !integralSqMatch)
    {
        printf("Errore: Le immagini integrali calcolate da CUDA non corrispondono a quelle sequenziali.\n");
        // return cudaErrorUnknown;
    }
    else
    {
        printf("Ottimo: Le immagini integrali calcolate da CUDA corrispondono a quelle sequenziali.\n");
    }

    // calcolo matrice cross correlation
    cv::Mat crossCorrelation;
   // cv::matchTemplate(imageN, templN, crossCorrelation, cv::TM_CCORR);
    crossCorrelation = crossCorrelationFFT(imageN, templN);
    cudaStatus = cudaMemcpy(d_crossCorrelation, crossCorrelation.ptr<float>(), resultSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

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
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Calcolo SSD sequenziale con immagini integrali
    cv::Mat seqSSDResult;
    computeSSDSequentialWithIntegrals(integralSeq_img, integralSeq_Sq_img, templateSum, templateSqSum, width, height, kx, ky, seqSSDResult, crossCorrelation);

    // Confronto dei risultati SSD
    bool ssdMatch = compareImages(ssdResult, seqSSDResult);
    if (!ssdMatch)
    {
        printf("Errore: I risultati SSD calcolati da CUDA non corrispondono a quelli sequenziali.\n");
    }
    else
    {
        printf("Ottimo: I risultati SSD calcolati da CUDA corrispondono a quelli sequenziali.\n");
    }

    // Trova la posizione del minimo SSD
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(ssdResult, &minVal, &maxVal, &minLoc, &maxLoc);
    *bestLoc = minLoc;

    double minValSeq, maxValSeq;
    cv::Point minLocSeq, maxLocSeq;
    cv::minMaxLoc(seqSSDResult, &minValSeq, &maxValSeq, &minLocSeq, &maxLocSeq);
    *bestLocSeq = minLocSeq;

    std::cout << "Matrice immagine:" << std::endl;
    stampMta(imageN, imageN.rows, imageN.cols);
    std::cout << "Matrice immagine integrale:" << std::endl;
    stampMta(cudaIntegralImage, cudaIntegralImage.rows, cudaIntegralImage.cols);
    std::cout << "Matrice immagine integrale al quadrato:" << std::endl;
    stampMta(cudaIntegralSqImage, cudaIntegralSqImage.rows, cudaIntegralSqImage.cols);
    std::cout << "Matrice immagine template:" << std::endl;
    stampMta(templN, templN.rows, templN.cols);
    std::cout << "Matrice sum template:" << std::endl;
    stampMta(temp_integral_img, temp_integral_img.rows, temp_integral_img.cols);
    std::cout << "Matrice sum sq template:" << std::endl;
    stampMta(temp_integral_Sq_img, temp_integral_Sq_img.rows, temp_integral_Sq_img.cols);
    std::cout << "template:" << std::endl;
    std::cout << templateSum << "\n";
    std::cout << templateSqSum << "\n";
    std::cout << "Matrice SSD result:" << std::endl;
    stampMta(ssdResult, ssdResult.rows, ssdResult.cols);
    std::cout << "Matrice SSD result seq:" << std::endl;
    stampMta(seqSSDResult, seqSSDResult.rows, seqSSDResult.cols);
    std::cout << "Cross correlation" << std::endl;
    stampMta(crossCorrelation, crossCorrelation.rows, crossCorrelation.cols);

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
    cudaError_t status = templateMatchingSSD(imageR, templateR, &bestLoc, &bestLocSeq);
    seq_templateMatchingSSD(imageR, templateR, &LocCheck);

    if (status != cudaSuccess)
    {
        printf("Errore CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }

    // Disegna un rettangolo intorno al match trovato

    // risultato template matching BLUE
    //  LocCheck.x = static_cast<int>(LocCheck.x) / scaleFactor;
    //  LocCheck.y = static_cast<int>(LocCheck.y) / scaleFactor;
    //  cv::Rect matchRectTM(LocCheck, cv::Size(templ.cols, templ.rows));
    //  rectangle(imageColor, matchRectTM, cv::Scalar(255, 0, 0), 3);

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
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    imshow("Immagine principale", imageColor);
    cv::waitKey(0);

    return 0;
}