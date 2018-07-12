#!/bin/bash

# Usage:
# $ sh test.sh create - to create specified inputs as datasets joined with yield

# data
data_path="../data"
files=("options-60000" "0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# data_path="../data/100000"
# files=("rand_h_unif_w_100000" "rand_hw_100000" "rand_w_unif_h_100000" "skew_h_1_rand_w_100000" "skew_h_10_rand_w_100000"
#        "skew_hw_1_100000" "skew_hw_10_100000" "skew_w_1_rand_h_100000" "skew_w_10_rand_h_100000" "unif_book_hw_100000" "unif_hw_100000")

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
