all:
	/usr/local/cuda/bin/nvcc  $(nf).cu -o $(nf) -lineinfo --ptxas-options=-v  -arch sm_61  -L /usr/local/cuda/samples/common/lib -I /usr/local/cuda/samples/common/inc
