#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all executables once
# $ sh test.sh - to compare outputs with the sepcified parameters

# compile options
real=64
reg=32

# program options
device=0
sorts="- w W h H"
block_sizes="1024"
versions="1 2 3"

# data
data_path="../data"
# files=("book" "options-1000" "options-60000")
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/unifdist_100000"
# files=( "rand_h_unif_w_unifdist_100000" "rand_hw_unifdist_100000" "rand_hw_w_256_unifdist_100000" "rand_w_unif_h_unifdist_100000"
#         "skew_h_1_rand_w_unifdist_100000" "skew_h_10_rand_w_unifdist_100000" "skew_hw_1_unifdist_100000" "skew_hw_1_w_256_unifdist_100000"
#         "skew_hw_10_unifdist_100000" "skew_hw_10_w_256_unifdist_100000" "skew_w_1_rand_h_unifdist_100000" "skew_w_10_rand_h_unifdist_100000"
#         "unif_book_hw_100000" "unif_hw_100000")
# data_path="../data/normdist_100000"
# files=( "rand_h_unif_w_normdist_100000" "rand_hw_normdist_100000" "rand_hw_w_256_normdist_100000" "rand_w_unif_h_normdist_100000"
#         "skew_h_1_rand_w_normdist_100000" "skew_h_10_rand_w_normdist_100000" "skew_hw_1_normdist_100000" "skew_hw_1_w_256_normdist_100000"
#         "skew_hw_10_normdist_100000" "skew_hw_10_w_256_normdist_100000" "skew_w_1_rand_h_normdist_100000" "skew_w_10_rand_h_normdist_100000"
#         "unif_book_hw_100000" "unif_hw_100000")
yield="yield"
out_path="$data_path/out"
# out_path="$data_path/out32"

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
