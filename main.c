#define _GNU_SOURCE
#include <CL/cl.h>
#include "cl_simple.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <errno.h>
#include <string.h>
#include <unistd.h>

#define PI 3.14
#define N 1024

#define NUM_DATA 100

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

struct timespec diff(struct timespec start, struct timespec end);

int* matrix_init() {
    int* mat;
    struct timespec tmp;
    mat = (int*)malloc(sizeof(int) * N * N);
    for (int i = 0; i < N * N; i++) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tmp);
        srand(tmp.tv_nsec);
        mat[i] = (int)(rand() % 100) / PI;
    }
    return mat;
}

struct timespec diff(struct timespec start, struct timespec end) {
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

int reduction(int* mat){
    int sum = 0;
    struct timespec time1, time2, res;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    for (int i = 0; i < N * N; i++) {
        sum -= mat[i];
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    res = diff(time1,time2);
    dprintf(2, "%lld.", (long long)res.tv_sec);
    dprintf(2, "%ld\n", res.tv_nsec);
    return sum;
}

int main(){
    /** cl_platform_id platforms[100];
	cl_uint platforms_n = 0;
	CL_CHECK(clGetPlatformIDs(100, platforms, &platforms_n));

	printf("=== %d OpenCL platform(s) found: ===\n", platforms_n);
	for (int i=0; i<platforms_n; i++)
	{
		char buffer[10240];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240, buffer, NULL));
		printf("  PROFILE = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240, buffer, NULL));
		printf("  VERSION = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer, NULL));
		printf("  NAME = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL));
		printf("  VENDOR = %s\n", buffer);
		CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240, buffer, NULL));
		printf("  EXTENSIONS = %s\n", buffer);
	}

	if (platforms_n == 0)
		return 1;

	cl_device_id devices[100];
	cl_uint devices_n = 0;
	// CL_CHECK(clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 100, devices, &devices_n));
	CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 100, devices, &devices_n));

	printf("=== %d OpenCL device(s) found on platform:\n", platforms_n);
	for (int i=0; i<devices_n; i++)
	{
		char buffer[10240];
		cl_uint buf_uint;
		cl_ulong buf_ulong;
        size_t size;
        size_t work_size[3];
		printf("  -- %d --\n", i);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_NAME = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VENDOR = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
		printf("  DEVICE_VERSION = %s\n", buffer);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(buf_uint), &buf_uint, NULL));
		printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)buf_uint);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
		printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size), &size, NULL));
		printf("  DEVICE_MAX_WORK_GROUP_SIZE = %zu\n", size);
		CL_CHECK(clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_size), &work_size, NULL));
		printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES = %zu\n", work_size[0]);
	}

	if (devices_n == 0)
		return 1; **/

    int* matrix_a;
    matrix_a = matrix_init();
    int sum = 0;

    printf("CPU reduction:\n");
    sum = reduction(matrix_a);
    printf("    sum = %d\n", sum);

    sum = 0;

    cl_mem input;
    struct cl_simple_context context;
    size_t global_work_size[2];
    size_t local_work_size[2];
    size_t size = N;

    global_work_size[0] = N;
    global_work_size[1] = N;

    local_work_size[0] = 16;
    local_work_size[1] = 16;

    size_t a_bytes;
    a_bytes = sizeof(int) * N * N;

    int* out;
    out = (int*)malloc(sizeof(int));
    int out_bytes;
    *out = 0;
    out_bytes = sizeof(int);

    if (!clSimpleSimpleInit(&context, "reduction")) {
        return EXIT_FAILURE;
    }

    if (    !clSimpleCreateBuffer(&input, context.cl_ctx, CL_MEM_READ_ONLY,
                a_bytes)) {
        return EXIT_FAILURE;
    }

    if (   !clSimpleEnqueueWriteBuffer(context.command_queue, input,
                sizeof(int) * N * N, matrix_a)){
        return EXIT_FAILURE;
    }

    if (!clSimpleSetOutputBuffer(&context, out_bytes)) {
        return EXIT_FAILURE;
    }

    if (   !clSimpleKernelSetArg(context.kernel, 1, sizeof(int), &size)
            || !clSimpleKernelSetArg(context.kernel, 2, sizeof(cl_mem), &input)) {
        return EXIT_FAILURE;
    }

    struct timespec time1, time2, res;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

    if (!clSimpleEnqueueNDRangeKernel(context.command_queue,
                context.kernel,
                2, global_work_size, local_work_size)) {
        return EXIT_FAILURE;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    res = diff(time1,time2);

    if (!clSimpleReadOutput(&context, out, out_bytes)) {
        return EXIT_FAILURE;
    }


    printf("GPU reduction:\n");
    dprintf(2, "%lld.", (long long)res.tv_sec);
    dprintf(2, "%ld\n", res.tv_nsec);
    sum = *out;
    printf("    sum = %d\n", sum);

    return 0;
}
