NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2

EXAMPLES := \
	tutorials/01_cuda_basics/main \
	tutorials/02_compilation_and_error_checking/main \
	tutorials/03_memory_management/main \
	tutorials/04_thread_indexing/main \
	tutorials/05_custom_kernel_workflow/main \
	tutorials/06_shared_memory_and_sync/main \
	tutorials/07_streams_and_async_execution/main \
	tutorials/08_inference_performance_analysis/main

.PHONY: all clean run-01 run-02 run-03 run-04 run-05 run-06 run-07 run-08

all: $(EXAMPLES)

tutorials/%/main: tutorials/%/main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-01: tutorials/01_cuda_basics/main
	./tutorials/01_cuda_basics/main

run-02: tutorials/02_compilation_and_error_checking/main
	./tutorials/02_compilation_and_error_checking/main

run-03: tutorials/03_memory_management/main
	./tutorials/03_memory_management/main

run-04: tutorials/04_thread_indexing/main
	./tutorials/04_thread_indexing/main

run-05: tutorials/05_custom_kernel_workflow/main
	./tutorials/05_custom_kernel_workflow/main

run-06: tutorials/06_shared_memory_and_sync/main
	./tutorials/06_shared_memory_and_sync/main

run-07: tutorials/07_streams_and_async_execution/main
	./tutorials/07_streams_and_async_execution/main

run-08: tutorials/08_inference_performance_analysis/main
	./tutorials/08_inference_performance_analysis/main

clean:
	rm -f $(EXAMPLES)
