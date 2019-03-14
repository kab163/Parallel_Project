SRC=nbpart.cu
SRC2=nbSingle.cu


default: single fast

single: $(SRC2)
	nvcc $(SRC2) -o single

fast: $(SRC)
	nvcc -O3 -arch=sm_35 $(SRC) -o fast


clean:
	rm single fast
