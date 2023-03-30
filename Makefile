NVCCFLAGS	:= -lineinfo -arch=sm_86 --ptxas-options=-v --use_fast_math

all:	reduction

reduction:	reduction.cu Makefile
	nvcc reduction.cu -o reduction $(NVCCFLAGS)