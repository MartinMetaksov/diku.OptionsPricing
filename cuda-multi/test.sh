#!/bin/bash

rep=5
sorts=("-" "h" "H" "w" "W")
block_sizes=(1024)
versions=(1 2 3)
data="../data"
files=("options-60000" "options-1000")
yield="yield"
exe="../build/CudaMulti"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")

compile() {
    echo "Compiling float version..."
    make -B REAL=32
    mv $exe $exefloat
    echo "Compiling float version with 32 registers..."
    make -B REAL=32 REG=32
    mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B REAL=64
    mv $exe $exedouble
    echo "Compiling double version with 32 registers..."
    make -B REAL=64 REG=32
    mv $exe $exedoublereg
}

test_files() {
    for file in ${files[*]}
    do
        for version in ${versions[*]}
        do
            for block in ${block_sizes[*]}
            do
                for sort in ${sorts[*]}
                do
                    echo -e "$2,$file,$version,$block,$sort,"`./$1 -o $data/$file.in -y $data/$yield.in -s $sort -v $version -b $block -r $rep | tail -n 1`
                done
            done
        done
    done
}

test() {
    echo "precision,registers,file,version,block,sort,kernel time,total time"
    for index in ${!exes[*]}; do 
        test_files ${exes[$index]} ${exes_names[$index]}
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    test
fi
