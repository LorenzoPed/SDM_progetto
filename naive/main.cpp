#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>


using namespace cv;
using namespace std::chrono;

void templateMatching(const Mat &image,
                      const Mat &templateImg,
                      Mat &matchScores)
{
    int imgRows = image.rows;
    int imgCols = image.cols;
    int tmplRows = templateImg.rows;
    int tmplCols = templateImg.cols;

    // Crea la matrice dei punteggi
    matchScores.create(imgRows - tmplRows + 1, imgCols - tmplCols + 1, CV_32F);

    // Scorri l'immagine
    for (int i = 0; i <= imgRows - tmplRows; ++i)
    {
        // Ottieni un puntatore alla riga corrente dell'immagine
        const uint8_t *imgPtr = image.ptr<uint8_t>(i);

        // Ottieni un puntatore alla riga corrente della matrice dei punteggi
        float *scoresPtr = matchScores.ptr<float>(i);

        for (int j = 0; j <= imgCols - tmplCols; ++j)
        {
            float score = 0.0f;

            // Confronta il template con l'immagine
            for (int x = 0; x < tmplRows; ++x)
            {
                // Ottieni un puntatore alla riga corrente del template
                const uint8_t *tmplPtr = templateImg.ptr<uint8_t>(x);

                // Ottieni un puntatore alla riga corrente dell'immagine (offset per la finestra corrente)
                const uint8_t *imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                for (int y = 0; y < tmplCols; ++y)
                {
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

int main()
{
    // Carica l'immagine principale e il template
    Mat image = imread("../../immagini/source.jpg", IMREAD_COLOR);     // Immagine principale
    Mat templateImg = imread("../../immagini/template100.jpg", IMREAD_COLOR); // Template

    if (image.empty() || templateImg.empty())
    {
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

    // Matrice dei punteggi
    Mat matchScores;

    cv::Mat seqIntegralSqImage;

     auto start_templateM = high_resolution_clock::now();
    templateMatching(grayImage, grayTemplate, matchScores);

    auto end_temaplateM = high_resolution_clock::now();
    auto duration_templateM = duration_cast<milliseconds>(end_temaplateM - start_templateM);
    std::cout << "Tempo calcolo immagine integrale (ms): " << duration_templateM.count() << std::endl;

    
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

    cv::imwrite("Result.jpg", image);
    //imshow("Immagine principale", image);
    //imshow("Template", templateImg);
    //waitKey(0);

    return 0;
}