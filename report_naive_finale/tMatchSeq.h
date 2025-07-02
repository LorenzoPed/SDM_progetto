#include <opencv2/opencv.hpp>
#include <stdio.h>

const unsigned char BLUE_H_MIN = 90;
const unsigned char BLUE_H_MAX = 140;
const unsigned char BLUE_S_MIN = 30;
const unsigned char BLUE_V_MIN = 30;

// immagine integrale in maniere sequenziale
void computeIntegralImagesSequential(const cv::Mat &image, cv::Mat &integralSqImage)
{
    int width = image.cols;
    int height = image.rows;

    integralSqImage.create(height, width, CV_32F);

    // Calcolo dell'immagine integrale
    for (int y = 0; y < height; y++)
    {
        float rowSqSum = 0;
        for (int x = 0; x < width; x++)
        {
            float pixelValue = image.at<float>(y, x);
            rowSqSum += pixelValue * pixelValue;

            if (y == 0)
            {
                integralSqImage.at<float>(y, x) = rowSqSum;
            }
            else
            {
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
    const cv::Mat &integralSqImage,  // Immagine integrale dei valori al quadrato
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
            float S2 = getRegionSumSequential(integralSqImage, width, x, y, kx, ky); // Somma dei quadrati
            float SC = crossCorrelation.at<float>(y, x);
            // Calcola l'SSD
            float ssd = S2 - 2 * SC + templateSqSum;
            cvSSD.at<float>(y, x) = ssd;
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
            std::cout << image.at<float>(height - 10 + y, width - 10 + x) << "\t";
        }
        std::cout << std::endl;
    }
}

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
    cv::Mat crossCorrelationResultF;
    cv::Rect roi(0, 0, image.cols - kernel.cols + 1, image.rows - kernel.rows + 1); // Calcola la ROI valida
    crossCorrelationResult = resultComplex(roi);
    crossCorrelationResult.convertTo(crossCorrelationResultF, CV_32F);
    return crossCorrelationResultF;
}

void seq_templateMatchingSSD(
    const cv::Mat &image,
    const cv::Mat &templ,
    cv::Point *bestLocSeq)
{
    cv::Mat imageN, templN;
    image.convertTo(imageN, CV_32F, 1.0 / 255.0);
    templ.convertTo(templN, CV_32F, 1.0 / 255.0);

    int width = image.cols;
    int height = image.rows;
    int kx = templ.cols;
    int ky = templ.rows;

    float templateSqSum = 0;

    // Creazione dell'immagine integrale dei quadrati del template
    cv::Mat temp_integral_sq;
    computeIntegralImagesSequential(templN, temp_integral_sq);
    // Estraggo l'ultimo valore della matrice (soma totale di tutto il template)
    templateSqSum = temp_integral_sq.at<float>(ky - 1, kx - 1);

    // Calcolo della immagine integrale dei valori al quadrato
    cv::Mat seqIntegralSqImage;
    computeIntegralImagesSequential(imageN, seqIntegralSqImage);

    // Calcolo matrice di cross correlazione
    cv::Mat crossCorrelation;
    crossCorrelation = crossCorrelationFFT(imageN, templN);

    // Calcolo della matrice solzione SSD
    cv::Mat seqSSD;
    computeSSDSequentialWithIntegrals(seqIntegralSqImage, templateSqSum, width, height, kx, ky, seqSSD, crossCorrelation);

    // Trova il valore minore nella SSD
    double minValSeq, maxValSeq;
    cv::Point minLocSeq, maxLocSeq;
    cv::minMaxLoc(seqSSD, &minValSeq, &maxValSeq, &minLocSeq, &maxLocSeq);
    *bestLocSeq = minLocSeq;
}

void seq_bgrToHsv(unsigned char b, unsigned char g, unsigned char r, unsigned char &h, unsigned char &s, unsigned char &v)
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

bool isBlue(unsigned char b, unsigned char g, unsigned char r) {
    unsigned char h, s, v;
    
    // Conversione da BGR a HSV
    seq_bgrToHsv(b, g, r, h, s, v);
    
    
    // Controlla se il pixel rientra nell'intervallo del blu
    if (h >= BLUE_H_MIN && h <= BLUE_H_MAX && 
        s >= BLUE_S_MIN && 
        v >= BLUE_V_MIN) {
        return true;
    }
    
    return false;
}

int seq_countBluePixels(cv::Mat& image, int startX, int startY, int roiWidth, int roiHeight) 
{
    long result = 0;

    // Controlla che l'immagine sia a 3 canali (BGR)
    if(image.type() != CV_8UC3){
        printf("ERRORE: Immagine nel formato errato");
        return -1;
    }
    
    for(int y = startY; y < startY + roiHeight; y++) {
        // Puntatore alla riga y
        cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        
        for(int x = startX; x < startX + roiWidth; x++) {
            // Accesso diretto al pixel
            cv::Vec3b& pixel = row[x];
            
            if(isBlue(pixel[0], pixel[1], pixel[2])) {
                
                result++;
            }
        }
    }
    return result;
}