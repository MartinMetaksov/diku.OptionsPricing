#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all executables once
# $ sh test.sh - to compare outputs with the sepcified parameters

# compile options
real=64
reg=""

# program options
device=0
sort="H"
version="4"
block_size="256"

# data
data_path="../data"
out_path="../data/out"
# files=("book")
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/100000"
# out_path="../data/100000/out"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")
yield="yield"

# executables
exe="../build/CudaOption"
compare="../build/Compare"
test_dir="../test"

compile() {
    echo "Compiling cuda option..."
    make -B compile REAL=$real REG=$REG
    echo "Compiling compare..."
    make --no-print-directory -C $test_dir -B compile-compare REAL=$real
}

compare() {
    for file in ${files[*]}
    do
        echo "Comparing" $file
        {
            ./$exe -o $data_path/$file.in -y $data_path/$yield.in -s $sort -v $version -b $block_size -d $device
            cat $out_path/$file.out
        } | ./$compare
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    compare
fi
