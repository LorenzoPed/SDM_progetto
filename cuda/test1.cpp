#include <opencv2/opencv.hpp>
#include <stdio.h>

cv::Mat crossCorrelationFFT(const cv::Mat &image, const cv::Mat &kernel)
{
    // 1. Prepara le immagini e il kernel per la FFT

    // Ottieni le dimensioni ottimali per la FFT (potenze di 2 per efficienza)
    int m = cv::getOptimalDFTSize(image.rows + kernel.rows - 1);
    int n = cv::getOptimalDFTSize(image.cols + kernel.cols - 1);

    // Padding delle immagini e del kernel per le dimensioni ottimali
    cv::Mat paddedImage, paddedKernel;
    cv::copyMakeBorder(image, paddedImage, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(kernel, paddedKernel, 0, m - kernel.rows, 0, n - kernel.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 2. Calcola le FFT

    cv::Mat imageComplex, kernelComplex, kernelFFT, imageFFT;

    // Converti le immagini in virgola mobile e crea matrici complesse per la FFT
    paddedImage.convertTo(paddedImage, CV_32F);
    paddedKernel.convertTo(paddedKernel, CV_32F);

    cv::Mat zeroPaddingImage = cv::Mat::zeros(paddedImage.size(), CV_32F);
    cv::Mat zeroPaddingKernel = cv::Mat::zeros(paddedKernel.size(), CV_32F);

    cv::Mat imagePlanes[] = {paddedImage, zeroPaddingImage};
    cv::Mat kernelPlanes[] = {paddedKernel, zeroPaddingKernel};

    cv::merge(imagePlanes, 2, imageComplex);
    cv::merge(kernelPlanes, 2, kernelComplex);

    cv::dft(imageComplex, imageFFT, 0, image.rows);
    cv::dft(kernelComplex, kernelFFT, 0, kernel.rows);

    // 3. Moltiplicazione nel dominio della frequenza (e coniugazione per cross-correlazione)

    cv::Mat kernelFFTConjugated;
    kernelFFTConjugated = kernelFFT.clone(); // Clona kernelFFT
    // Inverti la parte immaginaria per ottenere il complesso coniugato
    for (int i = 0; i < kernelFFTConjugated.rows; ++i)
    {
        for (int j = 0; j < kernelFFTConjugated.cols; ++j)
        {
            kernelFFTConjugated.at<cv::Vec2f>(i, j)[1] *= -1; // Moltiplica la parte immaginaria per -1
        }
    }

    cv::Mat resultFFT;
    cv::mulSpectrums(imageFFT, kernelFFTConjugated, resultFFT, 0); // Moltiplicazione punto a punto

    // 4. Trasformata di Fourier Inversa

    cv::Mat resultComplex;
    cv::dft(resultFFT, resultComplex, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, image.rows); // FFT Inversa

    // 5. Ritaglia la parte valida del risultato (rimuovi il padding)

    cv::Mat crossCorrelationResult;
    cv::Rect roi(0, 0, image.cols - kernel.cols + 1, image.rows - kernel.rows + 1); // Calcola la ROI valida
    crossCorrelationResult = resultComplex(roi);

    return crossCorrelationResult;
}
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
    const cv::Mat &integralImage,    // Immagine integrale della sorgente
    const cv::Mat &integralSqImage,  // Immagine integrale dei quadrati
    float templateSum,               // Somma dei pixel del template
    float templateSqSum,             // Somma dei quadrati dei pixel del template
    int width, int height,           // Dimensioni dell'immagine
    int kx, int ky,                  // Dimensioni del template
    cv::Mat &cvSSD,                  // Matrice di output per i risultati SSD
    const cv::Mat &crossCorrelation) // Matrice con i valori di cross correletion
{
    cvSSD.create(height - ky + 1, width - kx + 1, CV_32F);

    for (int y = 0; y < height - ky + 1; y++)
    {
        for (int x = 0; x < width - kx + 1; x++)
        {
            // Calcola le somme usando le immagini integrali
            float S1 = getRegionSumSequential(integralImage, width, x, y, kx, ky);   // Somma della regione immagine
            float S2 = getRegionSumSequential(integralSqImage, width, x, y, kx, ky); // Somma dei quadrati
            float SC = crossCorrelation.at<float>(y, x);
            // Calcola l'SSD
            // float ssd = S2 - 2 * (S1 * templateSum) + templateSqSum;
            float ssd = S2 - 2 * SC + templateSqSum;
            cvSSD.at<float>(y, x) = ssd;
        }
    }
}

// comparare le immagini cuda e sequanziali
bool compareImages(const cv::Mat &image1, const cv::Mat &image2, float tolerance = 0.1) // 1e-5
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
            std::cout << image.at<float>(height - 10 + y, width - 10 + x) << "\t";
        }
        std::cout << std::endl;
    }
}
void seq_templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLoc,
    cv::Point *bestLocSeq)
{
    // Conversione delle immagini in float
    // image.convertTo(imagefloat, CV_32F, 1.0 / 255.0); // Normalizza i valori tra 0 e 1
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

    // Creazione dell'immagine dei quadrati
    cv::Mat imageSq;
    cv::multiply(imageint, imageint, imageSq);

    // Creazione dell'immagine integrale
    cv::Mat integral_imgCV;
    cv::integral(imageint, integral_imgCV, CV_32F);
    integral_imgCV = integral_imgCV(cv::Rect(1, 1, image.cols, image.rows));

    // Creazione dell'immagine integrale dei quadrati
    cv::Mat integral_Sq_imgCV;
    cv::integral(imageSq, integral_Sq_imgCV, CV_32F);
    integral_Sq_imgCV = integral_Sq_imgCV(cv::Rect(1, 1, image.cols, image.rows));

    // Creazione dell'immagine dei quadrati del template
    cv::Mat templeSq;
    cv::multiply(templint, templint, templeSq);

    // Creazione dell'immagine integrale del template
    cv::Mat temp_integral_imgCV;
    cv::integral(templint, temp_integral_imgCV, CV_32F);
    temp_integral_imgCV = temp_integral_imgCV(cv::Rect(1, 1, templ.cols, templ.rows));

    // Creazione dell'immagine integrale dei quadrati del template
    cv::Mat temp_integral_Sq_imgCV;
    cv::integral(templeSq, temp_integral_Sq_imgCV, CV_32F);
    temp_integral_Sq_imgCV = temp_integral_Sq_imgCV(cv::Rect(1, 1, templ.cols, templ.rows));

    // Estrazione dei valori totali del template
    templateSum = temp_integral_imgCV.at<float>(ky - 1, kx - 1);
    templateSqSum = temp_integral_Sq_imgCV.at<float>(ky - 1, kx - 1);

    // Calcolo delle immagini integrali sequenziali
    cv::Mat seqIntegralImage, seqIntegralSqImage;
    computeIntegralImagesSequential(imageint, seqIntegralImage, seqIntegralSqImage);

    // Confronto delle immagini integrali
    bool integralMatch = compareImages(integral_imgCV, seqIntegralImage);
    bool integralSqMatch = compareImages(integral_Sq_imgCV, seqIntegralSqImage);

    if (!integralMatch || !integralSqMatch)
    {
        printf("Errore: Le immagini integrali calcolate da OpenCV non corrispondono a quelle sequenziali.\n");
        // return cudaErrorUnknown;
    }
    else
    {
        printf("Ottimo: Le immagini integrali calcolate da OpenCV corrispondono a quelle sequenziali.\n");
    }

    cv::Mat crossCorrelation;
    crossCorrelation = crossCorrelationFFT(imageint, templint);
    cv::Mat crossCorrelationCV;
    cv::matchTemplate(imageint, templint, crossCorrelationCV, cv::TM_CCORR);

    cv::Mat cvSSD;
    computeSSDSequentialWithIntegrals(integral_imgCV, integral_Sq_imgCV, templateSum, templateSqSum, width, height, kx, ky, cvSSD, crossCorrelation);
    cv::Mat seqSSD;
    computeSSDSequentialWithIntegrals(seqIntegralImage, seqIntegralSqImage, templateSum, templateSqSum, width, height, kx, ky, seqSSD, crossCorrelation);

    // Confronto dei risultati della cross correlation
    bool crosscorrelationMatch = compareImages(crossCorrelationCV, crossCorrelation);
    if (!crosscorrelationMatch)
    {
        printf("Errore: I risultati Cross correlation calcolati con OpenCV non corrispondono a quelli sequenziali.\n");
    }
    else
    {
        printf("Ottimo: I risultati Cross correlation calcolati con OpenCV corrispondono a quelli sequenziali.\n");
    }
    // Confronto dei risultati SSD
    bool ssdMatch = compareImages(cvSSD, seqSSD);
    if (!ssdMatch)
    {
        printf("Errore: I risultati SSD calcolati con matrici int OpenCV non corrispondono a quelli sequenziali.\n");
    }
    else
    {
        printf("Ottimo: I risultati SSD calcolati con matrici OpenCV corrispondono a quelli sequenziali.\n");
    }

    // Trova la posizione del minimo SSD
    // double minVal, maxVal;
    // cv::Point minLoc, maxLoc;
    // cv::minMaxLoc(crossCorrelation, &minVal, &maxVal, &minLoc, &maxLoc);
    // *bestLoc = minLoc;

    double minValSeq, maxValSeq;
    cv::Point minLocSeq, maxLocSeq;
    cv::minMaxLoc(seqSSD, &minValSeq, &maxValSeq, &minLocSeq, &maxLocSeq);
    *bestLocSeq = minLocSeq;

    std::cout << "Matrice immagine:" << std::endl;
    stampMta(imageint, imageint.rows, imageint.cols);
    std::cout << std::endl;
    std::cout << "Matrice immagine integrale OpenCV:" << std::endl;
    stampMta(integral_imgCV, integral_imgCV.rows, integral_imgCV.cols);
    std::cout << std::endl;
    std::cout << "Matrice immagine integrale calcolata:" << std::endl;
    stampMta(seqIntegralImage, seqIntegralImage.rows, seqIntegralImage.cols);
    std::cout << std::endl;
    std::cout << "Matrice immagine integrale quadratica OpenCV:" << std::endl;
    stampMta(integral_Sq_imgCV, integral_Sq_imgCV.rows, integral_Sq_imgCV.cols);
    std::cout << std::endl;
    std::cout << "Matrice immagine integrale quadratica calcolata:" << std::endl;
    stampMta(seqIntegralSqImage, seqIntegralSqImage.rows, seqIntegralSqImage.cols);
    std::cout << std::endl;
    std::cout << "Matrice sum template:" << std::endl;
    stampMta(temp_integral_imgCV, temp_integral_imgCV.rows, temp_integral_imgCV.cols);
    std::cout << std::endl;
    std::cout << "Matrice sum sq template:" << std::endl;
    stampMta(temp_integral_Sq_imgCV, temp_integral_Sq_imgCV.rows, temp_integral_Sq_imgCV.cols);
    std::cout << std::endl;
    std::cout << "template:" << std::endl;
    std::cout << templateSum << "\n";
    std::cout << templateSqSum << "\n";
    std::cout << std::endl;
    std::cout << "Matrice SSD result OpevCV:" << std::endl;
    stampMta(cvSSD, cvSSD.rows, cvSSD.cols);
    std::cout << std::endl;
    std::cout << "Matrice SSD result calcolata:" << std::endl;
    stampMta(seqSSD, seqSSD.rows, seqSSD.cols);
    std::cout << std::endl;
    std::cout << "Matrice cross correlation:" << std::endl;
    stampMta(crossCorrelation, crossCorrelation.rows, crossCorrelation.cols);
    std::cout << crossCorrelation.rows << "\n";
    std::cout << crossCorrelation.cols << "\n";
    std::cout << std::endl;
    std::cout << "Matrice cross correlation openCV:" << std::endl;
    stampMta(crossCorrelationCV, crossCorrelationCV.rows, crossCorrelationCV.cols);
    std::cout << crossCorrelationCV.rows << "\n";
    std::cout << crossCorrelationCV.cols << "\n";
    std::cout << std::endl;
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
    std::cout << "Immagine dimensioni:" << "\t";
    std::cout << imageR.rows << " x " << imageR.cols << std::endl;
    std::cout << "Template dimensioni:" << "\t";
    std::cout << templateR.rows << " x " << templateR.cols << std::endl;

    if (image.empty() || templ.empty())
    {
        printf("Errore nel caricamento delle immagini\n");
        return -1;
    }
    cv::Point bestLoc;
    cv::Point bestLocSeq;
    seq_templateMatchingSSD(imageR, templateR, &bestLoc, &bestLocSeq);

    bestLocSeq.x = static_cast<int>(bestLocSeq.x) / scaleFactor;
    bestLocSeq.y = static_cast<int>(bestLocSeq.y) / scaleFactor;
    cv::Rect matchRectSeq(bestLocSeq, cv::Size(templ.cols, templ.rows));
    rectangle(imageColor, matchRectSeq, cv::Scalar(0, 255, 0), 2);

    // bestLoc.x = static_cast<int>(bestLoc.x) / scaleFactor;
    // bestLoc.y = static_cast<int>(bestLoc.y) / scaleFactor;
    // cv::Rect matchRect(bestLoc, cv::Size(templ.cols, templ.rows));
    // rectangle(imageColor, matchRect, cv::Scalar(255, 0, 0), 2);

    // Mostra il risultato
    cv::imwrite("Result.jpg", imageColor);
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    imshow("Immagine principale", imageColor);
    cv::waitKey(0);

    return 0;
}