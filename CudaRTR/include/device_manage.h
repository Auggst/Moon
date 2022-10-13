#pragma once
#ifndef MOON_DEVICE_MANAGE_H_
#define MOON_DEVICE_MANAGE_H_

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"

#include <include/helper_cuda.h>

#include <include/hittable_list.h>
#include <include/camera.h>
#include <include/vec3.h>

struct CPUDATA {
	Camera* cam_ptr;
	Hittable_list* scene_ptr;
	unsigned char* img_ptr;
	int width;
	int height;
	size_t img_size;
};  
using GPUDATA = CPUDATA;

class DeviceManager {
 public:
   DeviceManager();
   ~DeviceManager();

   void ToDevice();
   void ToHost();
   void PrintDeviceInfo();
   void BindOpenGL();
   unsigned char* get_img();


 public:
   CPUDATA host_data;
   GPUDATA dev_data;
   bool isInit;
   unsigned int bufferObj;
   cudaGraphicsResource *cudaObj;
};

#endif // !MOON_DEVICE_MANAGE_H_

