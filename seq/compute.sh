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
