#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=5
device=0
sorts="h H w W"
block_sizes="-1 1024"
versions="3"

# data
data_path="../data"
files=("book")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/100000"
# out_path="../data/100000/out"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")
yield="yield"

# executables
exe="../build/CudaMulti"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")

compile() {
    echo "Compiling float version..."
    make -B compile REAL=32
    mv $exe $exefloat
    echo "Compiling float version with 32 registers..."
    make -B compile REAL=32 REG=32
    mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B compile REAL=64
    mv $exe $exedouble
    echo "Compiling double version with 32 registers..."
    make -B compile REAL=64 REG=32
    mv $exe $exedoublereg
}

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time"
    for file in ${files[*]}
    do
        for index in ${!exes[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in -y $data_path/$yield.in -s $sorts -v $versions -b $block_sizes -r $rep -d $device | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    test
fi
