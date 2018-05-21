#!/bin/bash

version=1
while [ $version -le 4 ]
do
    echo Testing version $version
    block=64
    while [ $block -le 256 ]
    do
    echo -e Block size $block '\t' `make repeat FILE=options-60000 YIELD=yield VERSION=$version BLOCK=$block | tail -n 1`
    block=$(( block*2 ))
    done
    ((version++))
done
