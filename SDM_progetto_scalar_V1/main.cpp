#include <opencv2/opencv.hpp>
#include <iostream>
#include <x86intrin.h> // Per __rdtsc()

using namespace cv;

// Funzione per rimuovere riflessi dall'immagine
cv::Mat removeReflections(const cv::Mat& imgGray) {
    // Applica un filtro Gaussiano per sfocare l'immagine
    cv::Mat blurred;
    cv::GaussianBlur(imgGray, blurred, cv::Size(15, 15), 0);

    imshow("Blurred", blurred);
    waitKey(0);
    // Sottrai il filtro Gaussiano dall'immagine originale per ottenere i dettagli ad alta frequenza
    cv::Mat highFreq;
    cv::subtract(imgGray, blurred, highFreq);

    // Normalizza per migliorare il contrasto
    cv::Mat normalized;
    cv::normalize(highFreq, normalized, 0, 255, cv::NORM_MINMAX);

    return normalized;
}

void templateMatching(const Mat& image,
                      const Mat& templateImg,
                      Mat& matchScores) {
    int imgRows = image.rows;
    int imgCols = image.cols;
    int tmplRows = templateImg.rows;
    int tmplCols = templateImg.cols;

    // Crea la matrice dei punteggi
    matchScores.create(imgRows - tmplRows + 1, imgCols - tmplCols + 1, CV_32F);

    // Scorri l'immagine
    for (int i = 0; i <= imgRows - tmplRows; ++i) {
        // Ottieni un puntatore alla riga corrente dell'immagine
        const uint8_t* imgPtr = image.ptr<uint8_t>(i);

        // Ottieni un puntatore alla riga corrente della matrice dei punteggi
        float* scoresPtr = matchScores.ptr<float>(i);

        for (int j = 0; j <= imgCols - tmplCols; ++j) {
            float score = 0.0f;

            // Confronta il template con l'immagine
            for (int x = 0; x < tmplRows; ++x) {
                // Ottieni un puntatore alla riga corrente del template
                const uint8_t* tmplPtr = templateImg.ptr<uint8_t>(x);

                // Ottieni un puntatore alla riga corrente dell'immagine (offset per la finestra corrente)
                const uint8_t* imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                for (int y = 0; y < tmplCols; ++y) {
                    // Calcola la differenza tra i pixel
                    float diff = static_cast<float>(imgWindowPtr[y] - tmplPtr[y]);
                    score += diff * diff; // Somma dei quadrati delle differenze
                }
            }

            // Salva il punteggio
            scoresPtr[j] = score;
        }
    }
}

cv::Mat increaseSaturation(const cv::Mat& img, double factor) {
    // Controlla che l'immagine non sia vuota
    if (img.empty()) {
        std::cerr << "Errore: immagine vuota fornita!" << std::endl;
        return cv::Mat();
    }

    // Converte l'immagine BGR in HSV
    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    // Splitta i canali HSV
    std::vector<cv::Mat> channels;
    cv::split(imgHSV, channels); // channels[0] = Hue, channels[1] = Saturation, channels[2] = Value

    // Aumenta la saturazione
    channels[1] *= factor; // Moltiplica ogni pixel per il fattore
    cv::threshold(channels[1], channels[1], 255, 255, cv::THRESH_TRUNC); // Limita i valori a 255

    // Ricombina i canali HSV
    cv::merge(channels, imgHSV);

    // Converte l'immagine HSV di nuovo in BGR
    cv::Mat imgBGR;
    cv::cvtColor(imgHSV, imgBGR, cv::COLOR_HSV2BGR);

    return imgBGR;
}

cv::uint64_t colorMatching(const Point& minLoc, const Mat& image, const Mat& templateImage){
    if(image.empty()){
        std::cerr<<"Errore: immagine inesistente!"<<std::endl;
        return -1;
    }

    uint64_t bluePixelCount = 0;

    int lowerBlueB = 100;
    int lowerBlueG = 0;
    int lowerBlueR = 0;
    int upperBlueB = 255;
    int upperBlueG = 100;
    int upperBlueR = 255;

    //minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    //minLoc.y = static_cast<int>(minLoc.y / scaleFactor);

    cv::Mat imageHighSat = increaseSaturation(image, 100.0);
    imshow("source high saturation", imageHighSat);

    for(int row = minLoc.x; row < templateImage.rows; ++row){
        for(int col = minLoc.y; col < templateImage.cols; ++col){

            cv::Vec3b pixel = imageHighSat.at<cv::Vec3b>(row, col);
            int B = pixel[0];
            int G = pixel[1];
            int R = pixel[2];

            if(B >= lowerBlueB && B <= upperBlueB &&
               G >= lowerBlueG && G <= upperBlueG &&
               R >= lowerBlueR && R <= upperBlueR){
                bluePixelCount++;
            }

        }
    }

    return bluePixelCount;
}


int main() {
    // Carica l'immagine principale e il template
    Mat image = imread("blueSource.jpg", IMREAD_COLOR);       // Immagine principale
    Mat templateImg = imread("blueTemplate_bg_white.jpg", IMREAD_COLOR); // Template

    if (image.empty() || templateImg.empty()) {
        std::cerr << "Errore: impossibile caricare le immagini." << std::endl;
        return -1;
    }

    // Converti le immagini in scala di grigi
    Mat grayImage, grayTemplate;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cvtColor(templateImg, grayTemplate, COLOR_BGR2GRAY);

    std::cout << "Dimensioni immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Dimensioni template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    // Ridimensiona le immagini 
    double scaleFactor = 1; // Fattore di ridimensionamento
    resize(grayImage, grayImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    resize(grayTemplate, grayTemplate, Size(), scaleFactor, scaleFactor, INTER_LINEAR);

    std::cout << "Resize immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Resize template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;
    
    // Inizializza il contatore dei clock di rimozione dei riflessi
    uint64_t start1 = __rdtsc();   

    // Rimozione dei riflessi
    grayImage = removeReflections(grayImage);
    grayTemplate = removeReflections(grayTemplate);

    uint64_t end1 = __rdtsc(); // Ferma il timer

    // Matrice dei punteggi
    Mat matchScores;
    // Inizializza il contatore dei clock di template matching
    uint64_t start2 = __rdtsc(); 

    // Esegui il template matching
    templateMatching(grayImage, grayTemplate, matchScores);

    uint64_t end2 = __rdtsc(); // Ferma il timer2
    uint64_t elapsed1 = end1 - start1; // Totale1 clock
    uint64_t elapsed2 = end2 - start2; // Totale2 clock
    uint64_t total = elapsed1 + elapsed2;
    std::cout << "Cicli di clock (rimozione riflessi): " << elapsed1 << std::endl; 
    std::cout << "Cicli di clock (template matching): " << elapsed2 << std::endl;
    std::cout << "Cicli di clock TOTALI: " << total << std::endl;         

    // Trova la posizione con il punteggio migliore (minimo)
    double minScore;
    Point minLoc;
    minMaxLoc(matchScores, &minScore, nullptr, &minLoc, nullptr);

    // Ridimensiona le coordinate della posizione trovata alle dimensioni originali
    minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    minLoc.y = static_cast<int>(minLoc.y / scaleFactor);

    // Color matching
    uint64_t bluePixel = colorMatching(minLoc, image, templateImg);

    if(bluePixel > 0){
        std::cout<<"Esito positivo: trovati"<<bluePixel<<"pixel BLUE!"<<std::endl;
    } else {
        std::cout<<"Esito negativo: trovati"<<bluePixel<<"pixel BLUE!"<<std::endl;
    }

    // Disegna un rettangolo attorno alla posizione trovata
    Rect matchRect(minLoc, Size(templateImg.cols, templateImg.rows));
    rectangle(image, matchRect, Scalar(0, 255, 0), 2); // Rettangolo verde

    // Mostra i risultati
    cv::namedWindow("Template", cv::WINDOW_NORMAL); // Finestra ridimensionabile
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
    imshow("Gray image", grayImage);
    imshow("Gray template", grayTemplate);
    imshow("Immagine principale", image);
    imshow("Template", templateImg);
    waitKey(0);

    return 0;
}