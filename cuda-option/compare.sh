#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all executables once
# $ sh test.sh - to compare outputs with the sepcified parameters

# compile options
real=64
reg=""

# program options
device=0
sorts="- w W h H"
block_sizes="32 64 128 256 512"
versions="1 2 3 4"

# data
data_path="../data"
# files=("book" "options-1000" "options-60000")
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
yield="yield"
out_path="$data_path/out"
# out_path="$data_path/out32"

# executables
exe="../build/CudaOption"
compare="../build/Compare"
test_dir="../test"

compile() {
    echo "Compiling cuda option..."
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
