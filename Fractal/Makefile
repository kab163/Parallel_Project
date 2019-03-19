SRC=fractal.cu
SRC2=optFractal.cu


default: single fast

single: $(SRC)
	nvcc $(SRC) -o single

fast: $(SRC2)
	nvcc -O3 -arch=sm_35 $(SRC2) -o fast

clean:
	rm single fast
