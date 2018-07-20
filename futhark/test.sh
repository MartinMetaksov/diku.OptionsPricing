#!/bin/bash

# Usage:
# $ sh test.sh create - to create specified inputs as datasets joined with yield

# data
data_path="../data"
files=("options-60000" "0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
yield="yield"
data_fut_path="$data_path/fut"

create() {
    for file in ${files[*]}
    do
        echo "Joining $file with $yield"
        cat $data_path/$file.in $data_path/$yield.in > $data_fut_path/$file.in
    done
}

if [ "$1" = "create" ]; then
    create
fi
