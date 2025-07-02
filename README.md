# Implementazione del Template Matching con Algoritmo SSD

Questo progetto confronta diverse implementazioni dell'algoritmo **Sum of Squared Differences (SSD)** per il **template matching**. Include versioni in **C++ sequenziale**, **SIMD**, e **CUDA** (naive e ottimizzata). 
In particolare le due versioni CUDA vengono eseguite insieme alla versione C++ ottimizzata che trova le sue funzioni nella libreria tMatchSeq.h. Abbiamo anche messo i report di NsightCompute per le versioni CUDA
naive e ottimizzata. Infine trovate la relazione sul progetto

## 📂 Struttura del progetto
- []
- [`cpp-sequenziale/`](./cpp-sequenziale/) — Implementazione base in C++ non ottimizzata
- [`simd32/`](./simd/) — Versione SIMD 32 bit
- [`simd64/`](./simd/) — Versione SIMD 64 bit
- [`cuda-naive/`](./cuda-naive/) — Versione parallela CUDA naive con report e libreria per seq. ottimizato
- [`cuda-optimized/`](./cuda-optimized/) — Versione CUDA ottimizzata con report e libreria per seq. ottimizato
