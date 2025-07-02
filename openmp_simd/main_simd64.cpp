#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h> 

using namespace cv;
using namespace std::chrono;

#include <immintrin.h>

void templateMatchingSIMD_64bit(const cv::Mat &image, const cv::Mat &templateImg, cv::Mat &matchScores)
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
            __m128i sumInt64 = _mm_setzero_si128(); 

            for (int x = 0; x < tmplRows; ++x)
            {
                const uint8_t *tmplPtr = templateImg.ptr<uint8_t>(x);
                const uint8_t *imgWindowPtr = image.ptr<uint8_t>(i + x) + j;

                // 16 pixel a iterazione
                int y = 0;
                for (; y <= tmplCols - 16; y += 16)
                {
                    __m128i imgPixels = _mm_loadu_si128((__m128i *)(imgWindowPtr + y));
                    __m128i tmplPixels = _mm_loadu_si128((__m128i *)(tmplPtr + y));

                    // Da 8x16 a 16x8
                    __m128i imgLo = _mm_unpacklo_epi8(imgPixels, _mm_setzero_si128());
                    __m128i imgHi = _mm_unpackhi_epi8(imgPixels, _mm_setzero_si128());
                    __m128i tmplLo = _mm_unpacklo_epi8(tmplPixels, _mm_setzero_si128());
                    __m128i tmplHi = _mm_unpackhi_epi8(tmplPixels, _mm_setzero_si128());

                    
                    __m128i diffLo = _mm_sub_epi16(imgLo, tmplLo);
                    __m128i diffHi = _mm_sub_epi16(imgHi, tmplHi);

                    // 4x32
                    __m128i sqLo = _mm_madd_epi16(diffLo, diffLo);
                    __m128i sqHi = _mm_madd_epi16(diffHi, diffHi);
                  
                    // 2x64 (evita overflow per template > 150x150 circa)
                    __m128i sqLo_low = _mm_cvtepu32_epi64(sqLo);
                    __m128i sqLo_high = _mm_cvtepu32_epi64(_mm_bsrli_si128(sqLo, 8));
                    __m128i sqHi_low = _mm_cvtepu32_epi64(sqHi);
                    __m128i sqHi_high = _mm_cvtepu32_epi64(_mm_bsrli_si128(sqHi, 8));

                    sumInt64 = _mm_add_epi64(sumInt64, sqLo_low);
                    sumInt64 = _mm_add_epi64(sumInt64, sqLo_high);
                    sumInt64 = _mm_add_epi64(sumInt64, sqHi_low);
                    sumInt64 = _mm_add_epi64(sumInt64, sqHi_high);
                }

                // Per gli eventuali pixel restanti 
                for (; y < tmplCols; ++y)
                {
                    int64_t diff = static_cast<int64_t>(imgWindowPtr[y]) - static_cast<int64_t>(tmplPtr[y]);
                    sumInt64 = _mm_add_epi64(sumInt64, _mm_set1_epi64x(diff * diff));
                }
            }

            // Allineamento 128bit
            alignas(16) int64_t sumArray[2];
            _mm_store_si128((__m128i *)sumArray, sumInt64);
            scoresPtr[j] = static_cast<float>(sumArray[0] + sumArray[1]);
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

    templateMatchingSIMD_64bit(grayImage, grayTemplate, matchScores);
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
    cv::imwrite("Result.jpg", image);
    waitKey(0);

    return 0;
}