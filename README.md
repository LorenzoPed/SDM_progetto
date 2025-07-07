# Implementazione del Template Matching con Algoritmo SSD

Questo progetto confronta diverse implementazioni dell'algoritmo **Sum of Squared Differences (SSD)** per il **template matching**. Include versioni in **C++ sequenziale**, **SIMD**, e **CUDA** (naive e ottimizzata). 
In particolare le due versioni CUDA vengono eseguite insieme alla versione C++ ottimizzata che trova le sue funzioni nella libreria tMatchSeq.h. Abbiamo anche messo i report di NsightCompute per le versioni CUDA
naive e ottimizzata. Infine trovate la relazione sul progetto

## 📂 Struttura del progetto
- [Relazione](https://github.com/LorenzoPed/SDM_progetto/blob/master/SD_project_final_V2.pdf)
- [Presentazione PowerPoint (zip file)](https://github.com/LorenzoPed/SDM_progetto/blob/master/TEMPLATE-MATCHING-CON-ALGORITMO-SSD.zip)
- [`cpp-sequenziale/`](https://github.com/LorenzoPed/SDM_progetto/blob/master/naive/main.cpp
) — Implementazione base in C++ non ottimizzata
- [`simd32/`](https://github.com/LorenzoPed/SDM_progetto/blob/master/openmp_simd/main_simd32.cpp
) — Versione SIMD 32 bit
- [`simd64/`](https://github.com/LorenzoPed/SDM_progetto/blob/master/openmp_simd/main_simd64.cpp)
   — Versione SIMD 64 bit
- [`cuda-naive/`](https://github.com/LorenzoPed/SDM_progetto/tree/master/report_naive_finale
) — Versione parallela CUDA naive con report e libreria per seq. ottimizato
- [`cuda-optimized/`](https://github.com/LorenzoPed/SDM_progetto/tree/master/report_optimized
) — Versione CUDA ottimizzata con report e libreria per seq. ottimizato
