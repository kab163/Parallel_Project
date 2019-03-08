SRC=gpublur.cu
CIMG_I=-I/opt/X11/include
CIMG_L=-L/opt/X11/lib
#SRC2=partGPUtest.cu
#SRC3=optGPUtest.cu

default: single

single: $(SRC)
	nvcc $(CIMG_I) -w -o single $(SRC) $(CIMG_L) -lm -lpthread -lX11 -ljpeg

#multi: $(SRC2)
#	nvcc -O3 -arch=sm_35 $(SRC2) -o multi

#fast: $(SRC3)
#	nvcc -O3 -arch=sm_35 $(SRC3) -o fast

clean:
	rm single
