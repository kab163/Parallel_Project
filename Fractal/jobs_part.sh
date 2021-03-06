#!/bin/bash

echo "*********single************"
./single 1000 256

echo "********1000, 2 groups**************"

./fast 1000 256 2
./fast 1000 256 2
./fast 1000 256 2

echo "********1000, 5 groups**************"

./fast 1000 256 5
./fast 1000 256 5
./fast 1000 256 5

echo "********1000, 10 groups**************"

./fast 1000 256 10
./fast 1000 256 10
./fast 1000 256 10

echo "********1000, 25 groups**************"

./fast 1000 256 25
./fast 1000 256 25
./fast 1000 256 25

echo "*********single************"
./single 2000 256

echo "********2000, 2 groups**************"

./fast 2000 256 2
./fast 2000 256 2
./fast 2000 256 2

echo "********2000, 5 groups**************"

./fast 2000 256 5
./fast 2000 256 5
./fast 2000 256 5

echo "********2000, 10 groups**************"

./fast 2000 256 10
./fast 2000 256 10
./fast 2000 256 10

echo "********2000, 25 groups**************"

./fast 2000 256 25
./fast 2000 256 25
./fast 2000 256 25

echo "*********single************"
./single 5000 256

echo "********5000, 2 groups**************"

./fast 5000 256 2
./fast 5000 256 2
./fast 5000 256 2

echo "********5000, 5 groups**************"

./fast 5000 256 5
./fast 5000 256 5
./fast 5000 256 5

echo "********5000, 10 groups**************"

./fast 5000 256 10
./fast 5000 256 10
./fast 5000 256 10

echo "********5000, 25 groups**************"

./fast 5000 256 25
./fast 5000 256 25
./fast 5000 256 25

echo "*********single************"
./single 10000 256

echo "********10000, 2 groups**************"

./fast 10000 256 2
./fast 10000 256 2
./fast 10000 256 2

echo "********10000, 5 groups**************"

./fast 10000 256 5
./fast 10000 256 5
./fast 10000 256 5

echo "********10000, 10 groups**************"

./fast 10000 256 10
./fast 10000 256 10
./fast 10000 256 10

echo "********10000, 25 groups**************"

./fast 10000 256 25
./fast 10000 256 25
./fast 10000 256 25

echo "********timesteps************"
echo "*********single************"
./single 5000 128

echo "********5000, 2 groups**************"

./fast 5000 128 2
./fast 5000 128 2
./fast 5000 128 2

echo "********5000, 5 groups**************"

./fast 5000 128 5
./fast 5000 128 5
./fast 5000 128 5

echo "********5000, 10 groups**************"

./fast 5000 128 10
./fast 5000 128 10
./fast 5000 128 10

echo "********5000, 25 groups**************"

./fast 5000 128 25
./fast 5000 128 25
./fast 5000 128 25

echo "*********single************"
./single 5000 512

echo "********5000, 2 groups**************"

./fast 5000 512 2
./fast 5000 512 2
./fast 5000 512 2

echo "********5000, 5 groups**************"

./fast 5000 512 5
./fast 5000 512 5
./fast 5000 512 5

echo "********5000, 10 groups**************"

./fast 5000 512 10
./fast 5000 512 10
./fast 5000 512 10

echo "********5000, 25 groups**************"

./fast 5000 512 25
./fast 5000 512 25
./fast 5000 512 25































































































































