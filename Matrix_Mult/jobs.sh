#!/bin/bash

./single 100

echo "**********size 100, 2 groups**********"
./fast 100 2
./fast 100 2
./fast 100 2

echo "**********size 100, 5 groups**********"
./fast 100 5
./fast 100 5
./fast 100 5

echo "**********size 100, 10 groups**********"
./fast 100 10
./fast 100 10
./fast 100 10

echo "**********size 100, 25 groups*********"
./fast 100 25
./fast 100 25
./fast 100 25

echo "*************************************"

./single 500

echo "**********size 500, 2 groups**********"
./fast 500 2
./fast 500 2
./fast 500 2

echo "**********size 500, 5 groups**********"
./fast 500 5
./fast 500 5
./fast 500 5

echo "**********size 500, 10 groups**********"
./fast 500 10
./fast 500 10
./fast 500 10

echo "**********size 500, 25 groups*********"
./fast 500 25
./fast 500 25
./fast 500 25

echo "**********************************"

./single 1000

echo "**********size 1000, 2 groups**********"
./fast 1000 2
./fast 1000 2
./fast 1000 2

echo "**********size 1000, 5 groups**********"
./fast 1000 5
./fast 1000 5
./fast 1000 5

echo "**********size 1000, 10 groups**********"
./fast 1000 10
./fast 1000 10
./fast 1000 10

echo "**********size 1000, 25 groups*********"
./fast 1000 25
./fast 1000 25
./fast 1000 25

echo "**********************************"

./single 5000

echo "**********size 5000, 2 groups**********"
./fast 5000 2
./fast 5000 2
./fast 5000 2

echo "**********size 5000, 5 groups**********"
./fast 5000 5
./fast 5000 5
./fast 5000 5

echo "**********size 5000, 10 groups**********"
./fast 5000 10
./fast 5000 10
./fast 5000 10

echo "**********size 5000, 25 groups*********"
./fast 5000 25
./fast 5000 25
./fast 5000 25

echo "**********************************"


./single 10000


echo "**********size 10000, 2 groups*********"
./fast 10000 2
./fast 10000 2
./fast 10000 2

echo "**********size 10000, 5 groups*********"
./fast 10000 5
./fast 10000 5
./fast 10000 5

echo "**********size 10000, 10 groups********"
./fast 10000 10
./fast 10000 10
./fast 10000 10

echo "**********size 10000, 25 groups*********"
./fast 10000 25
./fast 10000 25
./fast 10000 25

echo "*************************************"

./single 20000

echo "**********size 20000, 2 groups**********"
./fast 20000 2
./fast 20000 2
./fast 20000 2

echo "**********size 20000, 5 groups**********"
./fast 20000 5
./fast 20000 5
./fast 20000 5

echo "**********size 20000, 10 groups**********"
./fast 20000 10
./fast 20000 10
./fast 20000 10

echo "**********size 20000, 25 groups*********"
./fast 20000 25
./fast 20000 25
./fast 20000 25

echo "*************************************"

./single 30000

echo "**********size 30000, 2 groups**********"
./fast 30000 2
./fast 30000 2
./fast 30000 2

echo "**********size 30000, 5 groups**********"
./fast 30000 5
./fast 30000 5
./fast 30000 5

echo "**********size 30000, 10 groups**********"
./fast 30000 10
./fast 30000 10
./fast 30000 10

echo "**********size 30000, 25 groups*********"
./fast 30000 25
./fast 30000 25
./fast 30000 25

echo "*************************************"

./single 100000

echo "**********size 100000, 2 groups"
./fast 100000 2
./fast 100000 2
./fast 100000 2

echo "**********size 100000, 5 groups"
./fast 100000 5
./fast 100000 5
./fast 100000 5

echo "**********size 100000, 10 groups"
./fast 100000 10
./fast 100000 10
./fast 100000 10

echo "**********size 100000, 25 groups"
./fast 100000 25
./fast 100000 25
./fast 100000 25

echo "***********out of place 50k below********************"

./single 50000

echo "**********size 50000, 2 groups**********"
./fast 50000 2
./fast 50000 2
./fast 50000 2

echo "**********size 50000, 5 groups**********"
./fast 50000 5
./fast 50000 5
./fast 50000 5

echo "**********size 50000, 10 groups**********"
./fast 50000 10
./fast 50000 10
./fast 50000 10

echo "**********size 50000, 25 groups*********"
./fast 50000 25
./fast 50000 25
./fast 50000 25

echo "**********************************"

