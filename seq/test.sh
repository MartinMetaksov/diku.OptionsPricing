#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=1

# data
data_path="../data"
# files=("book" "options-1000" "options-60000")
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
yield="yield"

# executables
exe="../build/Seq"
exefloat=$exe"-float"
exedouble=$exe"-double"
exes=($exefloat $exedouble)
exes_names=("float,-" "double,-")
exes_to_run=(0 1)

compile() {
    echo "Compiling float version..."
    make -B compile REAL=32
    mv $exe $exefloat
    echo "Compiling double version..."
    make -B compile REAL=64
    mv $exe $exedouble
}

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time,memory"
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in -y $data_path/$yield.in -r $rep | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    test
fi
