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

    // crea la matrice dei punteggi
    matchScores.create(imgRows - tmplRows + 1, imgCols - tmplCols + 1, CV_32F);

    for (int i = 0; i <= imgRows - tmplRows; ++i)
    {
        // riga corrente dell'immagine
        const uint8_t *imgPtr = image.ptr<uint8_t>(i);

        
        float *scoresPtr = matchScores.ptr<float>(i);

        for (int j = 0; j <= imgCols - tmplCols; ++j)
        {   // colonna immagine 
            float score = 0.0f;

            // SSD dei pixel immagine e dei pixel template per la finestra
            for (int x = 0; x < tmplRows; ++x)
            {
                const uint8_t *tmplPtr = templateImg.ptr<uint8_t>(x);

                const uint8_t *imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                for (int y = 0; y < tmplCols; ++y)
                {
                    float diff = static_cast<float>(imgWindowPtr[y] - tmplPtr[y]);
                    score += diff * diff; // SSD
                }
            }

            scoresPtr[j] = score;
        }
    }
}

int main()
{
    
    Mat image = imread("../../immagini/source.jpg", IMREAD_COLOR);     // Immagine
    Mat templateImg = imread("../../immagini/template100.jpg", IMREAD_COLOR); // Template

    if (image.empty() || templateImg.empty())
    {
        std::cerr << "Errore: impossibile caricare le immagini." << std::endl;
        return -1;
    }

    // conversione in scala di grigi
    Mat grayImage, grayTemplate;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cvtColor(templateImg, grayTemplate, COLOR_BGR2GRAY);

    std::cout << "Dimensioni immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Dimensioni template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    
    double scaleFactor = 1; // fattore di scale (1 dim orignali)
    resize(grayImage, grayImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    resize(grayTemplate, grayTemplate, Size(), scaleFactor, scaleFactor, INTER_LINEAR);

    std::cout << "Resize immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Resize template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    // Matrice per raccogliere i valori ssd
    Mat matchScores;


     auto start_templateM = high_resolution_clock::now();

    templateMatching(grayImage, grayTemplate, matchScores);

    auto end_temaplateM = high_resolution_clock::now();
    auto duration_templateM = duration_cast<milliseconds>(end_temaplateM - start_templateM);
    std::cout << "Tempo calcolo template matching (ms): " << duration_templateM.count() << std::endl;

    
    // Trova la posizione del valore minimo 
    double minScore;
    Point minLoc;
    minMaxLoc(matchScores, &minScore, nullptr, &minLoc, nullptr);

    // Ridimensionamento in caso di scale factor
    minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    minLoc.y = static_cast<int>(minLoc.y / scaleFactor);

    Rect matchRect(minLoc, Size(templateImg.cols, templateImg.rows));
    rectangle(image, matchRect, Scalar(0, 255, 0), 2); // Rettangolo verde

    cv::namedWindow("Template", cv::WINDOW_NORMAL); 
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);

    cv::imwrite("Result.jpg", image);
   
    return 0;
}