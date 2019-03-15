#!/bin/bash

./single 1000000 10 0

echo "**********1000000 nbodies, 10 timesteps, 2 groups**********"
./fast 1000000 10 2
./fast 1000000 10 2
./fast 1000000 10 2

echo "**********1000000 nbodies, 10 timesteps, 5 groups**********"
./fast 1000000 10 5
./fast 1000000 10 5
./fast 1000000 10 5

echo "**********1000000 nbodies, 10 timesteps, 10 groups**********"
./fast 1000000 10 10
./fast 1000000 10 10
./fast 1000000 10 10

echo "***********************************************"

./single 5000000 10 0

echo "**********5000000 nbodies, 10 timesteps, 2 groups**********"
./fast 5000000 10 2
./fast 5000000 10 2
./fast 5000000 10 2

echo "**********5000000 nbodies, 10 timesteps, 5 groups**********"
./fast 5000000 10 5
./fast 5000000 10 5
./fast 5000000 10 5

echo "**********5000000 nbodies, 10 timesteps, 10 groups**********"
./fast 5000000 10 10
./fast 5000000 10 10
./fast 5000000 10 10

echo "***********************************************"

./single 10000000 10 0

echo "**********10000000 nbodies, 10 timesteps, 2 groups**********"
./fast 10000000 10 2
./fast 10000000 10 2
./fast 10000000 10 2

echo "**********10000000 nbodies, 10 timesteps, 5 groups**********"
./fast 10000000 10 5
./fast 10000000 10 5
./fast 10000000 10 5

echo "**********10000000 nbodies, 10 timesteps, 10 groups**********"
./fast 10000000 10 10
./fast 10000000 10 10
./fast 10000000 10 10

echo "***********************************************"




















































































