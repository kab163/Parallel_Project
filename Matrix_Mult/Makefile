SRC=gpuTest.cu
SRC2=partGPUtest.cu
SRC3=optGPUtest.cu

default: single multi fast

single: $(SRC)
	nvcc $(SRC) -o single

multi: $(SRC2)
	nvcc -O3 -arch=sm_35 $(SRC2) -o multi

fast: $(SRC3)
	nvcc -O3 -arch=sm_35 $(SRC3) -o fast

clean:
	rm single multi fast
