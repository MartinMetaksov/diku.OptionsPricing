#!/bin/bash

version=1
while [ $version -le 1 ]
do
    echo Testing version $version
    block=128
    while [ $block -le 512 ]
    do
    echo -e Block size $block '\t' `make repeat FILE=options-60000 YIELD=yield VERSION=$version BLOCK=$block | tail -n 1`
    block=$(( block*2 ))
    done
    ((version++))
done
