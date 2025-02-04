#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>

#define INDEX(x, y, width) ((y) * (width) + (x))
#define BLOCK_SIZE 32

// immagine integrele in manaore sequanziale
void computeIntegralImagesSequential(const cv::Mat &image, cv::Mat &integralImage, cv::Mat &integralSqImage)
{
    int width = image.cols;
    int height = image.rows;

    integralImage.create(height, width, CV_32F);
    integralSqImage.create(height, width, CV_32F);

    // Calcolo dell'immagine integrale
    for (int y = 0; y < height; y++)
    {
        float rowSum = 0;
        float rowSqSum = 0;
        for (int x = 0; x < width; x++)
        {
            float pixelValue = image.at<float>(y, x);
            rowSum += pixelValue;
            rowSqSum += pixelValue * pixelValue;

            if (y == 0)
            {
                integralImage.at<float>(y, x) = rowSum;
                integralSqImage.at<float>(y, x) = rowSqSum;
            }
            else
            {
                integralImage.at<float>(y, x) = integralImage.at<float>(y - 1, x) + rowSum;
                integralSqImage.at<float>(y, x) = integralSqImage.at<float>(y - 1, x) + rowSqSum;
            }
        }
    }
}

// Funzione sequenziale per calcolare la somma in una regione usando l'immagine integrale
int getRegionSumSequential(const cv::Mat &sumTable, int width, int x, int y, int kx, int ky)
{
    int x1 = x - 1;
    int y1 = y - 1;
    int x2 = x + kx - 1;
    int y2 = y + ky - 1;

    float A = (x1 >= 0 && y1 >= 0) ? sumTable.at<float>(y1, x1) : 0.0f;
    float B = (y1 >= 0) ? sumTable.at<float>(y1, x2) : 0.0f;
    float C = (x1 >= 0) ? sumTable.at<float>(y2, x1) : 0.0f;
    float D = sumTable.at<float>(y2, x2);

    return D - B - C + A;
}

void computeSSDSequentialWithIntegrals(
    const cv::Mat &integralImage,   // Immagine integrale della sorgente
    const cv::Mat &integralSqImage, // Immagine integrale dei quadrati
    float templateSum,              // Somma dei pixel del template
    float templateSqSum,            // Somma dei quadrati dei pixel del template
    int width, int height,          // Dimensioni dell'immagine
    int kx, int ky,                 // Dimensioni del template
    cv::Mat &ssdResult)             // Matrice di output per i risultati SSD
{
    ssdResult.create(height - ky + 1, width - kx + 1, CV_32F);

    for (int y = 0; y < height - ky + 1; y++)
    {
        for (int x = 0; x < width - kx + 1; x++)
        {
            // Calcola le somme usando le immagini integrali
            float S1 = getRegionSumSequential(integralImage, width, x, y, kx, ky);   // Somma della regione immagine
            float S2 = getRegionSumSequential(integralSqImage, width, x, y, kx, ky); // Somma dei quadrati

            // Calcola l'SSD
            float ssd = S2 - 2 * (S1 * templateSum) + templateSqSum;
            ssdResult.at<float>(y, x) = ssd;
        }
    }
}

// comparare le immagini cuda e sequanziali
bool compareImages(const cv::Mat &image1, const cv::Mat &image2, float tolerance = 1e-5)
{
    if (image1.size() != image2.size() || image1.type() != image2.type())
    {
        return false;
    }

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);

    double maxDiff;
    cv::minMaxLoc(diff, nullptr, &maxDiff);

    return maxDiff <= tolerance;
}
void stampMta(const cv::Mat &image, int height, int width)
{
    for (int y = 0; y < 10; ++y)
    {
        for (int x = 0; x < 10; ++x)
        {
            std::cout << image.at<float>(x, y) << "\t";
        }
        std::cout << std::endl;
    }
}
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
    float *ssdResult)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - kx + 1 || y >= height - ky + 1)
        return;

    // Calcolo delle somme usando le immagini integrali
    float S1 = getRegionSum(imageSum, width, height, x, y, kx, ky);   // Somma della regione immagine
    float S2 = getRegionSum(imageSqSum, width, height, x, y, kx, ky); // Somma dei quadrati

    // Calcolo SSD diretto usando le somme
    float ssd = S2 - 2 * (S1 * templateSum) + templateSqSum;

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
    // image.convertTo(imageFloat, CV_32F, 1.0 / 255.0); // Normalizza i valori tra 0 e 1
    cv::Mat imageint, templint;
    image.convertTo(imageint, CV_32F, 1.0 / 255.0);
    templ.convertTo(templint, CV_32F, 1.0 / 255.0);

    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    // Calcolo delle somme del template
    float templateSum = 0;
    float templateSqSum = 0;
    for (int i = 0; i < kx * ky; i++)
    {
        float val = templint.ptr<float>()[i];
        templateSum += val;
        templateSqSum += val * val;
    }
    // Allocazione memoria su device
    float *d_image, *d_imageSum, *d_imageSqSum, *d_ssdResult, *d_rowSum, *d_imageSq, *d_rowSqSum;
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

    cudaStatus = cudaMalloc(&d_rowSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMalloc(&d_rowSqSum, imageSize);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Copia immagine su device e creazione immagine dei quadrati
    cudaStatus = cudaMemcpy(d_image, imageint.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Creazione dell'immagine dei quadrati
    cv::Mat imageSq;
    cv::multiply(imageint, imageint, imageSq);
    cudaStatus = cudaMemcpy(d_imageSq, imageSq.ptr<float>(), imageSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Calcolo delle immagini integrali
    int threadsPerBlock = BLOCK_SIZE;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calcolo immagini integrali
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_image, d_rowSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSum, d_imageSum, width, height);

    // multiply<<<gridSize, blockSize>>>(d_image, width, height, d_imageSq);
    rowCumSum<<<(height + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_imageSq, d_rowSqSum, width, height);
    colCumSum<<<(width + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(d_rowSqSum, d_imageSqSum, width, height);

    // Copia delle immagini integrali calcolate da CUDA su host
    cv::Mat cudaIntegralImage(height, width, CV_32F);
    cv::Mat cudaIntegralSqImage(height, width, CV_32F);
    cudaStatus = cudaMemcpy(cudaIntegralImage.ptr<float>(), d_imageSum, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    cudaStatus = cudaMemcpy(cudaIntegralSqImage.ptr<float>(), d_imageSqSum, imageSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Calcolo delle immagini integrali sequenziali
    cv::Mat seqIntegralImage, seqIntegralSqImage;
    computeIntegralImagesSequential(imageint, seqIntegralImage, seqIntegralSqImage);

    // Confronto delle immagini integrali
    bool integralMatch = compareImages(cudaIntegralImage, seqIntegralImage);
    bool integralSqMatch = compareImages(cudaIntegralSqImage, seqIntegralSqImage);

    if (!integralMatch || !integralSqMatch)
    {
        printf("Errore: Le immagini integrali calcolate da CUDA non corrispondono a quelle sequenziali.\n");
        // return cudaErrorUnknown;
    }
    else
    {
        printf("Ottimo: Le immagini integrali calcolate da CUDA corrispondono a quelle sequenziali.\n");
    }

    // Calcolo SSD
    dim3 gridSSDSize((width - kx + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height - ky + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computeSSD<<<gridSSDSize, blockSize>>>(
        d_imageSum,
        d_imageSqSum,
        //  d_rowSum,
        //  d_rowSqSum,
        templateSum,
        templateSqSum,
        width,
        height,
        kx,
        ky,
        d_ssdResult);

    // Copia risultati su host
    cv::Mat ssdResult(height - ky + 1, width - kx + 1, CV_32F);
    cudaStatus = cudaMemcpy(ssdResult.ptr<float>(), d_ssdResult, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    // Calcolo SSD sequenziale con immagini integrali
    cv::Mat seqSSDResult;
    computeSSDSequentialWithIntegrals(seqIntegralImage, seqIntegralSqImage, templateSum, templateSqSum, width, height, kx, ky, seqSSDResult);

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
    stampMta(imageint, imageint.rows, imageint.cols);
    std::cout << "Matrice immagine integrale:" << std::endl;
    stampMta(cudaIntegralImage, cudaIntegralImage.rows, cudaIntegralImage.cols);
    std::cout << "Matrice immagine integrale al quadrato:" << std::endl;
    stampMta(cudaIntegralSqImage, cudaIntegralSqImage.rows, cudaIntegralSqImage.cols);
    std::cout << "Matrice immagine template:" << std::endl;
    stampMta(templint, templint.rows, templint.cols);
    std::cout << "template:" << std::endl;
    std::cout << templateSum << "\n";
    std::cout << templateSqSum << "\n";
    std::cout << "Matrice SSD result:" << std::endl;
    stampMta(ssdResult, ssdResult.rows, ssdResult.cols);
    std::cout << "Matrice SSD result seq:" << std::endl;
    stampMta(seqSSDResult, seqSSDResult.rows, seqSSDResult.cols);

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_imageSum);
    cudaFree(d_imageSqSum);
    cudaFree(d_ssdResult);
    cudaFree(d_rowSum);
    cudaFree(d_imageSq);
    cudaFree(d_rowSqSum);

    return cudaStatus;
}

int main()
{
    // Carica le immagini
    cv::Mat image = cv::imread("immagini/source.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat imageColor = cv::imread("immagini/source.jpg", cv::IMREAD_COLOR);
    cv::Mat templ = cv::imread("immagini/template.jpg", cv::IMREAD_GRAYSCALE);
    double scaleFactor = 0.10; // Fattore di ridimensionamento
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
    cudaError_t status = templateMatchingSSD(imageR, templateR, &bestLoc, &bestLocSeq);

    if (status != cudaSuccess)
    {
        printf("Errore CUDA: %s\n", cudaGetErrorString(status));
        return -1;
    }

    // Disegna un rettangolo intorno al match trovato
    //  cv::rectangle(
    //      imageColor,
    //      bestLoc,
    //      cv::Point(bestLoc.x/scaleFactor + templ.cols, bestLoc.y/scaleFactor + templ.rows),
    //      cv::Scalar(0, 0, 255),
    //      3);
    cv::rectangle(
        imageColor,
        bestLocSeq,
        cv::Point(bestLocSeq.x /scaleFactor+ templ.cols, bestLocSeq.y/scaleFactor + templ.rows),
        cv::Scalar(0, 255, 0),
        2);
    bestLoc.x = static_cast<int>(bestLoc.x)/scaleFactor;
    bestLoc.y = static_cast<int>(bestLoc.y)/scaleFactor;
    cv::Rect matchRect(bestLoc, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRect, cv::Scalar(255, 0, 0), 2);
    // Mostra il risultato
    cv::imwrite("Result.jpg", imageColor);
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    imshow("Immagine principale", imageColor);
    cv::waitKey(0);

    return 0;
}