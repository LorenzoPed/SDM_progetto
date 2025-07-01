#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h> // Per OpenMP
#include <immintrin.h>

using namespace cv;
using namespace std::chrono;

void templateMatchingSIMD(const Mat &image, const Mat &templateImg, Mat &matchScores)
{
    int imgRows = image.rows;
    int imgCols = image.cols;
    int tmplRows = templateImg.rows;
    int tmplCols = templateImg.cols;

    matchScores.create(imgRows - tmplRows + 1, imgCols - tmplCols + 1, CV_32F);

    #pragma omp parallel for
    for (int i = 0; i <= imgRows - tmplRows; ++i)
    {
        float *scoresPtr = matchScores.ptr<float>(i);

        for (int j = 0; j <= imgCols - tmplCols; ++j)
        {
            __m128i sumInt = _mm_setzero_si128();

            for (int x = 0; x < tmplRows; ++x)
            {
                const uint8_t *tmplPtr = templateImg.ptr<uint8_t>(x);
                const uint8_t *imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                int y = 0;
                for (; y <= tmplCols - 16; y += 16)
                {
                    __m128i imgPixels = _mm_loadu_si128((__m128i *)&imgWindowPtr[y]);
                    __m128i tmplPixels = _mm_loadu_si128((__m128i *)&tmplPtr[y]);

                    __m128i imgLo = _mm_unpacklo_epi8(imgPixels, _mm_setzero_si128());
                    __m128i imgHi = _mm_unpackhi_epi8(imgPixels, _mm_setzero_si128());
                    __m128i tmplLo = _mm_unpacklo_epi8(tmplPixels, _mm_setzero_si128());
                    __m128i tmplHi = _mm_unpackhi_epi8(tmplPixels, _mm_setzero_si128());

                    __m128i diffLo = _mm_sub_epi16(imgLo, tmplLo);
                    __m128i diffHi = _mm_sub_epi16(imgHi, tmplHi);

                    __m128i sqLo = _mm_madd_epi16(diffLo, diffLo);
                    __m128i sqHi = _mm_madd_epi16(diffHi, diffHi);

                    sumInt = _mm_add_epi32(sumInt, sqLo);
                    sumInt = _mm_add_epi32(sumInt, sqHi);
                }

                for (; y < tmplCols; ++y)
                {
                    int diff = static_cast<int>(imgWindowPtr[y]) - static_cast<int>(tmplPtr[y]);
                    sumInt = _mm_add_epi32(sumInt, _mm_set1_epi32(diff * diff));
                }
            }

            int sumArray[4];
            _mm_storeu_si128((__m128i *)sumArray, sumInt);
            float score = static_cast<float>(sumArray[0] + sumArray[1] + sumArray[2] + sumArray[3]);
            scoresPtr[j] = score;
        }
    }
}

int main()
{
    Mat image = imread("../../immagini/source.jpg", IMREAD_COLOR);
    Mat templateImg = imread("../../immagini/template100.jpg", IMREAD_COLOR);

    if (image.empty() || templateImg.empty())
    {
        std::cerr << "Errore: impossibile caricare le immagini." << std::endl;
        return -1;
    }

    Mat grayImage, grayTemplate;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    cvtColor(templateImg, grayTemplate, COLOR_BGR2GRAY);

    std::cout << "Dimensioni immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Dimensioni template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    double scaleFactor = 1;
    resize(grayImage, grayImage, Size(), scaleFactor, scaleFactor, INTER_LINEAR);
    resize(grayTemplate, grayTemplate, Size(), scaleFactor, scaleFactor, INTER_LINEAR);

    std::cout << "Resize immagine: " << grayImage.cols << "x" << grayImage.rows << std::endl;
    std::cout << "Resize template: " << grayTemplate.cols << "x" << grayTemplate.rows << std::endl;

    Mat matchScores;

    auto start_templateM = high_resolution_clock::now();

    templateMatchingSIMD(grayImage, grayTemplate, matchScores);
    auto end_temaplateM = high_resolution_clock::now();
    auto duration_templateM = duration_cast<milliseconds>(end_temaplateM - start_templateM);
    std::cout << "Tempo calcolo immagine integrale (ms): " << duration_templateM.count() << std::endl;

    double minScore;
    Point minLoc;
    minMaxLoc(matchScores, &minScore, nullptr, &minLoc, nullptr);

    minLoc.x = static_cast<int>(minLoc.x / scaleFactor);
    minLoc.y = static_cast<int>(minLoc.y / scaleFactor);
    Rect matchRect(minLoc, Size(templateImg.cols, templateImg.rows));
    rectangle(image, matchRect, Scalar(0, 255, 0), 2);

    namedWindow("Template", WINDOW_NORMAL);
    namedWindow("Immagine principale", WINDOW_NORMAL);

    imshow("Immagine principale", image);
    imshow("Template", templateImg);
    cv::imwrite("Result.jpg", image);
    waitKey(0);

    return 0;
}