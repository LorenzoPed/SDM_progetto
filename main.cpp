#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

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
        for (int j = 0; j <= imgCols - tmplCols; ++j) {
            float score = 0.0f;

            // Confronta il template con l'immagine
            for (int x = 0; x < tmplRows; ++x) {
                for (int y = 0; y < tmplCols; ++y) {
                    // Calcola la differenza tra i pixel
                    float diff = static_cast<float>(
                        image.at<uint8_t>(i + x, j + y) - templateImg.at<uint8_t>(x, y)
                    );
                    score += diff * diff; // Somma dei quadrati delle differenze
                }
            }

            // Salva il punteggio
            matchScores.at<float>(i, j) = score;
        }
    }
}

int main() {
    // Carica l'immagine principale e il template
    Mat image = imread("cartaCoppe.jpg", IMREAD_COLOR);       // Immagine principale
    Mat templateImg = imread("templete.jpg", IMREAD_COLOR); // Template

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
    double scaleFactor = 0.25; // Fattore di ridimensionamento
    resize(grayImage, grayImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    resize(grayTemplate, grayTemplate, Size(), scaleFactor, scaleFactor, INTER_LINEAR);

    std::cout << "Resize immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Resize template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    // Matrice dei punteggi
    Mat matchScores;

    // Esegui il template matching
    templateMatching(grayImage, grayTemplate, matchScores);

    // Trova la posizione con il punteggio migliore (minimo)
    double minScore;
    Point minLoc;
    minMaxLoc(matchScores, &minScore, nullptr, &minLoc, nullptr);

    // Ridimensiona le coordinate della posizione trovata alle dimensioni originali
    minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    minLoc.y = static_cast<int>(minLoc.y / scaleFactor);

    // Disegna un rettangolo attorno alla posizione trovata
    Rect matchRect(minLoc, Size(templateImg.cols, templateImg.rows));
    rectangle(image, matchRect, Scalar(0, 255, 0), 2); // Rettangolo verde

    // Mostra i risultati
    cv::namedWindow("Template", cv::WINDOW_NORMAL); // Finestra ridimensionabile
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);
 
    imshow("Immagine principale", image);
    imshow("Template", templateImg);
    waitKey(0);

    return 0;
}