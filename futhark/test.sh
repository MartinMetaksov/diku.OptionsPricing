#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 2 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=5

# compiled options (change in futhark code first)
real=32

# data
data_path="../data"
files=("book")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/100000"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")
yield="yield"

# executables
exe_basic="../build/FutharkBasic"
exe_flat="../build/FutharkFlat"

compile() {
    echo "Compiling basic version..."
    make -B compile-basic
    echo "Compiling flat version..."
    make -B compile-flat
}

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time"
    for file in ${files[*]}
    do
        cat $data_path/$file.in $data_path/$yield.in | ./$1 -r 5 -t tmp > /dev/null
        total_time=$(sort -n tmp | head -1)
        echo "$file,$real,-,-,-,-,-,$total_time"
    done
}

if [ "$1" = "compile" ]; then
    compile
elif [ "$1" = "basic" ]; then
    test $exe_basic
elif [ "$1" = "flat" ]; then
    test $exe_flat
fi
