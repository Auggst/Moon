#pragma once
#ifndef MOON_RENDERER_H_
#define MOON_RENDERER_H_

#include <cuda.h>
#include "cuda_runtime.h"
#include <include/helper_cuda.h>

#include <include/camera.h>
#include <include/hittable_list.h> 

class RayTraceRenderer {
 public:
   RayTraceRenderer();
   ~RayTraceRenderer();

   void Render(unsigned char* img_ptr, RayTraceRenderer& render, Camera* dev_cam, Hittable_list* dev_scene);

 public:
   size_t max_depth;
   size_t samples;
   int width;
   int height;
};

#endif // !MOON_RENDERER_H_
