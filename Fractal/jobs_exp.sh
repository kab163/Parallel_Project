#!/bin/bash

echo "*********single************"
./single 20000 256

echo "********20000, 2 groups**************"

./fast 20000 256 2
./fast 20000 256 2
./fast 20000 256 2

echo "********20000, 5 groups**************"

./fast 20000 256 5
./fast 20000 256 5
./fast 20000 256 5

echo "********20000, 10 groups**************"

./fast 20000 256 10
./fast 20000 256 10
./fast 20000 256 10

echo "********20000, 25 groups**************"

./fast 20000 256 25
./fast 20000 256 25
./fast 20000 256 25

echo "*********single************"
./single 30000 256

echo "********30000, 2 groups**************"

./fast 30000 256 2
./fast 30000 256 2
./fast 30000 256 2

echo "********30000, 5 groups**************"

./fast 30000 256 5
./fast 30000 256 5
./fast 30000 256 5

echo "********30000, 10 groups**************"

./fast 30000 256 10
./fast 30000 256 10
./fast 30000 256 10

echo "********30000, 25 groups**************"

./fast 30000 256 25
./fast 30000 256 25
./fast 30000 256 25

echo "*********single************"
./single 50000 256

echo "********50000, 2 groups**************"

./fast 50000 256 2
./fast 50000 256 2
./fast 50000 256 2

echo "********50000, 5 groups**************"

./fast 50000 256 5
./fast 50000 256 5
./fast 50000 256 5

echo "********50000, 10 groups**************"

./fast 50000 256 10
./fast 50000 256 10
./fast 50000 256 10

echo "********50000, 25 groups**************"

./fast 50000 256 25
./fast 50000 256 25
./fast 50000 256 25

