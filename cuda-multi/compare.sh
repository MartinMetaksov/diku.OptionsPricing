#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all executables once
# $ sh test.sh - to compare outputs with the sepcified parameters

# compile options
real=64
reg=32

# program options
device=0
sorts=("-" "h" "H" "w" "W")
versions=("1" "2" "3")
block_sizes=("-1" "1024")

# data
# data_path="../data"
# files=("book")
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/1000"
# files=("rand_hw_1000" "unif_book_hw_1000")
# data_path="../data/100000"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")
data_path="../data/new_sets"
out_path="$data_path/out"
yield="yield"

# executables
exe="../build/CudaMulti"
compare="../build/Compare"
test_dir="../test"

compile() {
    echo "Compiling cuda multi..."
    make -B compile REAL=$real REG=$reg
    echo "Compiling compare..."
    make --no-print-directory -C $test_dir -B compile-compare REAL=$real
}

compare() {
    for file in ${files[*]}
    do
        for version in ${versions[*]}
        do
            for block_size in ${block_sizes[*]}
            do
                for sort in ${sorts[*]}
                do
                    echo "Comparing $file (version $version, block size $block_size, sort $sort)"
                    {
                        ./$exe -o $data_path/$file.in -y $data_path/$yield.in -s $sort -v $version -b $block_size -d $device
                        cat $out_path/$file.out
                    } | ./$compare
                done
            done
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    compare
fi
