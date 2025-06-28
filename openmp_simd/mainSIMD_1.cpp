#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>       // Per OpenMP
#include <chrono>

using namespace cv;
using namespace std::chrono;

void templateMatchingMixedSIMD(const Mat &image,
                               const Mat &templateImg,
                               Mat &matchScores)
{
    int imgRows = image.rows;
    int imgCols = image.cols;
    int tmplRows = templateImg.rows;
    int tmplCols = templateImg.cols;

    // Crea la matrice dei punteggi
    matchScores.create(imgRows - tmplRows + 1, imgCols - tmplCols + 1, CV_32F);

// Parallelizzazione per righe
//#pragma omp parallel for
    for (int i = 0; i <= imgRows - tmplRows; ++i)
    {
        float *scoresPtr = matchScores.ptr<float>(i);

        for (int j = 0; j <= imgCols - tmplCols; ++j)
        {
            __m128i sumInt = _mm_setzero_si128(); // Accumulatore SIMD per valori interi

            for (int x = 0; x < tmplRows; ++x)
            {
                const uint8_t *tmplPtr = templateImg.ptr<uint8_t>(x);
                const uint8_t *imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                int y = 0;
                // Elabora 16 pixel alla volta con SIMD
                for (; y <= tmplCols - 16; y += 16)
                {
                    // Carica 16 pixel dall'immagine e dal template
                    __m128i imgPixels = _mm_loadu_si128((__m128i *)&imgWindowPtr[y]);
                    __m128i tmplPixels = _mm_loadu_si128((__m128i *)&tmplPtr[y]);

                    // Calcola la differenza assoluta
                    // __m128i diff = _mm_abs_epi8(_mm_sub_epi8(imgPixels, tmplPixels));
                    __m128i diff = _mm_sub_epi8(imgPixels, tmplPixels); // Differenza
                    __m128i diffSquared = _mm_mullo_epi16(diff, diff);  // Quadrato della differenza
                    // Accumula le differenze
                    sumInt = _mm_add_epi32(sumInt, _mm_sad_epu8(diffSquared, _mm_setzero_si128()));
                    // sumInt = _mm_add_epi32(sumInt, _mm_sad_epu8(diff, _mm_setzero_si128()));
                }

                // Elabora i pixel rimanenti
                for (; y < tmplCols; ++y)
                {
                    int diff = static_cast<int>(imgWindowPtr[y]) - static_cast<int>(tmplPtr[y]);
                    sumInt = _mm_add_epi32(sumInt, _mm_set1_epi32(abs(diff)));
                }
            }

            // Converti il risultato intero in float
            int sumArray[4];
            _mm_storeu_si128((__m128i *)sumArray, sumInt);
            float score = static_cast<float>(sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3]);

            // Salva il punteggio
            scoresPtr[j] = score;
        }
    }
}

int main()
{
    // Carica l'immagine principale e il template
    Mat image = imread("../../immagini/sourceC.jpg", IMREAD_COLOR);         // Immagine principale
    Mat templateImg = imread("../../immagini/templateC2.jpg", IMREAD_COLOR); // Template

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
    double scaleFactor = 0.25; // Fattore di ridimensionamento
    resize(grayImage, grayImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    resize(grayTemplate, grayTemplate, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    grayImage.type() == CV_8UC1 && grayTemplate.type() == CV_8UC1;


    std::cout << "Resize immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Resize template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    // Matrice dei punteggi
    Mat matchScores;

    auto start_templateM = high_resolution_clock::now();

    // Esegui il template matching
    templateMatchingMixedSIMD(grayImage, grayTemplate, matchScores);

    auto end_temaplateM = high_resolution_clock::now();
    auto duration_templateM = duration_cast<milliseconds>(end_temaplateM - start_templateM);
    std::cout << "Tempo calcolo immagine integrale (ms): " << duration_templateM.count() << std::endl;
 
    // Trova la posizione con il punteggio migliore (minimo)
    double minScore;
    Point minLoc;
    minMaxLoc(matchScores, &minScore, nullptr, &minLoc, nullptr);

    // Disegna un rettangolo attorno alla posizione trovata
    minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    minLoc.y = static_cast<int>(minLoc.y / scaleFactor);
    Rect matchRect(minLoc, Size(templateImg.cols, templateImg.rows));
    rectangle(image, matchRect, Scalar(0, 255, 0), 2);

    // Mostra i risultati
    cv::namedWindow("Template", cv::WINDOW_NORMAL); // Finestra ridimensionabile
    cv::namedWindow("Immagine principale", cv::WINDOW_NORMAL);

    imshow("Immagine principale", image);
    imshow("Template", templateImg);
    waitKey(0);

    return 0;
}