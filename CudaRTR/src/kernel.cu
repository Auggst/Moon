/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include <cuda.h>
#include <book.h>
#include <cpu_bitmap.h>
#include <include/helper_cuda.h>
#include <thrust/extrema.h>
#include <include/my_utility.h>
#include <include/hittable_list.h>
#include <include/sphere.h>
#include <include/camera.h>

#define DIM 512
 
#define rnd( x ) (x * rand() / RAND_MAX)
#define SAMPLES 100
#define MAX_DEPTH 20

//打印设备信息
void get_device_info() {
    auto device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0)
    {
        printf("没有支持CUDA的设备!\n");
        return;
    }
    for (auto dev = 0; dev < device_count; dev++)
    {
        cudaSetDevice(dev);
        cudaDeviceProp device_prop{};
        cudaGetDeviceProperties(&device_prop, dev);
        printf("设备 %d: \"%s\"\n", dev, device_prop.name);
        char msg[256];
        sprintf_s(msg, sizeof(msg),
            "global memory大小:        %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
            static_cast<unsigned long long>(device_prop.totalGlobalMem));
        printf("%s", msg);
        printf("SM数:                    %2d \n每SM CUDA核心数:           %3d \n总CUDA核心数:             %d \n",
            device_prop.multiProcessorCount,
            _ConvertSMVer2Cores(device_prop.major, device_prop.minor),
            _ConvertSMVer2Cores(device_prop.major, device_prop.minor) *
            device_prop.multiProcessorCount);
        printf("静态内存大小:             %zu bytes\n",
            device_prop.totalConstMem);
        printf("每block共享内存大小:      %zu bytes\n",
            device_prop.sharedMemPerBlock);
        printf("每block寄存器数:          %d\n",
            device_prop.regsPerBlock);
        printf("线程束大小:               %d\n",
            device_prop.warpSize);
        printf("每处理器最大线程数:       %d\n",
            device_prop.maxThreadsPerMultiProcessor);
        printf("每block最大线程数:        %d\n",
            device_prop.maxThreadsPerBlock);
        printf("线程块最大维度大小        (%d, %d, %d)\n",
            device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
            device_prop.maxThreadsDim[2]);
        printf("网格最大维度大小          (%d, %d, %d)\n",
            device_prop.maxGridSize[0], device_prop.maxGridSize[1],
            device_prop.maxGridSize[2]);
        printf("\n");
    }
    printf("************设备信息打印完毕************\n\n");
}

//场景初始化
__device__ double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0)   return -1.0;
    else return (-half_b - sqrt(discriminant)) /  a;
}

//计算光线颜色
__device__ color ray_color(ray& r, hittable_list*const dev_world, curandStateXORWOW_t* state) {
    hit_record rec;
    size_t depth = MAX_DEPTH;
    color result = color(1.0, 1.0, 1.0);
    ray temp;
    color attenuation;

    while (true) {
        if (depth <= 0) return color(0.0, 0.0, 0.0);
        if (!dev_world->hit(r, 0.001, infinity, rec)) {
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5 * (unit_direction.y() + 1.0);
            return result * ((1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));
        }
        depth--;
        if (!rec.mat_ptr->scatter(r, rec, attenuation, temp, state)) return color(0.0, 0.0, 0.0);
        r = temp;
        result = result * attenuation;
    }
    return result;
    //while ((depth > 0) && dev_world->hit(r, 0.001, infinity, rec)) {
    //    depth--;
    //    if (rec.mat_ptr->scatter(r, rec, attenuation, temp, state)) {
    //        r = temp;
    //        result = cross(result, attenuation);
    //    }
    //    else {
    //        return color(0.0, 0.0, 0.0);
    //    }
    //}
    //if (depth == 0) return color(0.0, 0.0, 0.0);
    //vec3 unit_direction = unit_vector(r.direction());
    //auto t = 0.5 * (unit_direction.y() + 1.0);
    //return cross(result, (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));
}

//GPU计算像素
__global__ void kernel(unsigned char* ptr, Moon::camera** cam, hittable_list** world) {
    
    // map from threadIdx/BlockIdx to pixel position
     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     int offset = x + y * blockDim.x * gridDim.x;
     if (x >= 512 || y >= 512 || offset >= 262144) return;
     int seed = offset;
     curandStateXORWOW_t rand_state;
     curand_init(seed, 0, 0, &rand_state);
     
     color pixel_color(0.0, 0.0, 0.0);
     for (size_t s = 0; s < SAMPLES; ++s) {
         double u = double(x + Moon::random_double(&rand_state)) / (double)(DIM - 1);
         double v = double(y + Moon::random_double(&rand_state)) / (double)(DIM - 1);
         ray r = (*cam)->get_ray(u, v, &rand_state);

         pixel_color += ray_color(r, *world, &rand_state);
     }

     double r = pixel_color.x();
     double g = pixel_color.y();
     double b = pixel_color.z();
     double scale = 1.0 / (double)SAMPLES;
     r = sqrt(scale * r);
     g = sqrt(scale * g);
     b = sqrt(scale * b);
     ptr[offset * 4 + 0] = (int)(Moon::clamp(r, 0.0, 0.999) * 255);
     ptr[offset * 4 + 1] = (int)(Moon::clamp(g, 0.0, 0.999) * 255);
     ptr[offset * 4 + 2] = (int)(Moon::clamp(b, 0.0, 0.999) * 255);
     ptr[offset * 4 + 3] = 255;
 }

 // globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;
};
// 如果HANDLE_ERROR有问题的请参考下文"mydef.h"
int main(void) {

    get_device_info();


    DataBlock   data;   //gpu数据块
    // 计时器
    cudaEvent_t     start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    CPUBitmap bitmap(DIM, DIM, &data);  //cpu图像
    unsigned char* dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
        bitmap.image_size())); 

    //相机加载
    Moon::camera** device_cam = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&device_cam, sizeof(Moon::camera**)));
    Moon::make_camera_device << <1, 1 >> > (device_cam);
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    //场景加载
    hittable_list** device_world = nullptr;
    HANDLE_ERROR(cudaMalloc((void**)&device_world, sizeof(hittable_list**)));
    make_scene_device << <1, 1 >> > (device_world, 2);
    cudaDeviceSynchronize();
    cudaGetLastError();

    // generate a bitmap from our sphere data
    dim3    grids(DIM / 16, DIM / 16);
    dim3    threads(16, 16);
    kernel << <grids, threads >> > (dev_bitmap, device_cam, device_world);
    cudaDeviceSynchronize();

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time to generate:  %3.1f ms, %3.1f min\n", elapsedTime, elapsedTime / (1000 * 60));
    //printf("FPS: %3.1f\n", 1000.0 / elapsedTime);

    HANDLE_ERROR(cudaFree(device_world));
    destroy_scene_device << <1, 1 >> > (device_world);
    HANDLE_ERROR(cudaFree(device_cam));
    destroy_camera_device << <1, 1 >> > (device_cam);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaFree(dev_bitmap));

    // display
    bitmap.display_and_exit();
}


//#define DIM 1024
//
//#define rnd( x ) (x * rand() / RAND_MAX)
//#define INF     2e10f
//
//struct Sphere {
//    float   r, b, g;
//    float   radius;
//    float   x, y, z;
//    __device__ float hit(float ox, float oy, float* n) {
//        float dx = ox - x;
//        float dy = oy - y;
//        if (dx * dx + dy * dy < radius * radius) {
//            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
//            *n = dz / sqrtf(radius * radius);
//            return dz + z;
//        }
//        return -INF;
//    }
//};
//#define SPHERES 20
//
//__constant__ Sphere s[SPHERES];
//
//__global__ void kernel(unsigned char* ptr) {
//    // map from threadIdx/BlockIdx to pixel position
//    int x = threadIdx.x + blockIdx.x * blockDim.x;
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//    int offset = x + y * blockDim.x * gridDim.x;
//    float   ox = (x - DIM / 2);// move to center
//    float   oy = (y - DIM / 2);
//
//    float   r = 0, g = 0, b = 0;
//    float   maxz = -INF;
//    for (int i = 0; i < SPHERES; i++) {
//        float   n;
//        float   t = s[i].hit(ox, oy, &n);
//        if (t > maxz) {
//            float fscale = n;
//            r = s[i].r * fscale;
//            g = s[i].g * fscale;
//            b = s[i].b * fscale;
//            maxz = t;
//        }
//    }
//
//    ptr[offset * 4 + 0] = (int)(r * 255);
//    ptr[offset * 4 + 1] = (int)(g * 255);
//    ptr[offset * 4 + 2] = (int)(b * 255);
//    ptr[offset * 4 + 3] = 255;
//}
//
//// globals needed by the update routine
//struct DataBlock {
//    unsigned char* dev_bitmap;
//};
//// 如果HANDLE_ERROR有问题的请参考下文"mydef.h"
//int main(void) {
//    DataBlock   data;
//    // capture the start time
//    cudaEvent_t     start, stop;
//    HANDLE_ERROR(cudaEventCreate(&start));
//    HANDLE_ERROR(cudaEventCreate(&stop));
//    HANDLE_ERROR(cudaEventRecord(start, 0));
//
//    CPUBitmap bitmap(DIM, DIM, &data);
//    unsigned char* dev_bitmap;
//
//    // allocate memory on the GPU for the output bitmap
//    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
//        bitmap.image_size()));
//
//    // allocate temp memory, initialize it, copy to constant
//    // memory on the GPU, then free our temp memory
//    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
//    for (int i = 0; i < SPHERES; i++) {
//        temp_s[i].r = rnd(1.0f);
//        temp_s[i].g = rnd(1.0f);
//        temp_s[i].b = rnd(1.0f);
//        temp_s[i].x = rnd(1000.0f) - 500;
//        temp_s[i].y = rnd(1000.0f) - 500;
//        temp_s[i].z = rnd(1000.0f) - 500;
//        temp_s[i].radius = rnd(100.0f) + 20;
//    }
//    HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s,
//        sizeof(Sphere) * SPHERES));
//    free(temp_s);
//
//    // generate a bitmap from our sphere data
//    dim3    grids(DIM / 16, DIM / 16);
//    dim3    threads(16, 16);
//    kernel << <grids, threads >> > (dev_bitmap);
//
//    // copy our bitmap back from the GPU for display
//    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
//        bitmap.image_size(),
//        cudaMemcpyDeviceToHost));
//
//    // get stop time, and display the timing results
//    HANDLE_ERROR(cudaEventRecord(stop, 0));
//    HANDLE_ERROR(cudaEventSynchronize(stop));
//    float   elapsedTime;
//    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
//        start, stop));
//    printf("Time to generate:  %3.1f ms\n", elapsedTime);
//
//    HANDLE_ERROR(cudaEventDestroy(start));
//    HANDLE_ERROR(cudaEventDestroy(stop));
//
//    HANDLE_ERROR(cudaFree(dev_bitmap));
//
//    // display
//    bitmap.display_and_exit();
//}