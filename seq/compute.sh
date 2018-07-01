#!/bin/bash

# Usage:
# $ sh compute.sh compile - to compile the executable
# $ sh compute.sh - to compute options in the specified files

# compile options
real=64

# data
data_path="../data"
out_path="../data/out"
files=("book")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/1000"
# out_path="../data/1000/out"
# files=("rand_hw_1000" "unif_book_hw_1000")
# data_path="../data/100000"
# out_path="../data/100000/out"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")
yield="yield"

# executable
exe="../build/Seq"

compile() {
    echo "Compiling sequential version..."
    make -B compile REAL=$real
}

compute() {
    for file in ${files[*]}
    do
        echo "Computing" $file
        ./$exe -o $data_path/$file.in -y $data_path/$yield.in > $out_path/$file.out
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    compute
fi
