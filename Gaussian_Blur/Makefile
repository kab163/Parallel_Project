SRC=gpublur.cu
CIMG_I=-I/opt/X11/include
CIMG_L=-L/opt/X11/lib
SRC2=PartGPUBlur.cu
#SRC3=optGPUtest.cu

default: single fast

single: $(SRC)
	nvcc $(CIMG_I) -w -o single $(SRC) $(CIMG_L) -lm -lpthread -lX11 -ljpeg

fast: $(SRC2)
	nvcc $(CIMG_I) -w -o fast $(SRC2) $(CIMG_L) -lm -lpthread -lX11 -ljpeg

clean:
	rm single fast
