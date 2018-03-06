cmake_minimum_required(VERSION 3.9)
project(diku_OptionsPricing)

set(CMAKE_CXX_STANDARD 11)

add_executable(diku_OptionsPricing
        common/Domain.hpp
        common/FutharkArrays.hpp
        common/Option.hpp
        common/Real.hpp
        cuda/CudaErrors.cuh
        cuda/CudaMain.cu
        cuda/FormattedOptions.hpp
        cuda/Makefile
        cuda/ScanKernels.cuh
        data/book.in
        data/compare.fut
        futhark-cosmin/header32.fut
        futhark-cosmin/header64.fut
        futhark-cosmin/trinom-basic.fut
        futhark-cosmin/trinom-flat.fut
        seq/Makefile
        seq/SeqMain.cpp)
