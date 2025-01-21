cmake_minimum_required(VERSION 3.10)  # Versione minima di CMake
project(cpu)             # Nome del progetto

# Imposta lo standard C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trova le librerie richieste (OpenCV)
find_package(OpenCV REQUIRED)

# Aggiungi il file sorgente
set(SOURCES main.cpp)

# Crea l'eseguibile
add_executable(main ${SOURCES})

# Collega le librerie trovate
target_link_libraries(main ${OpenCV_LIBS})
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(main PRIVATE "-msse4.1")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(main PRIVATE "/arch:SSE2")
endif()

# Abilita OpenMP (se necessario)
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    target_link_libraries(main OpenMP::OpenMP_CXX)
endif()