SRC=nbpart.cu
SRC2=nbSingle.cu


default: single part

single: $(SRC2)
	nvcc $(SRC2) -o single

part: $(SRC)
	nvcc -O3 -arch=sm_35 $(SRC) -o multi


clean:
	rm single part
