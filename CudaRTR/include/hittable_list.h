#pragma once
#ifndef MOON_HITTABLE_LIST_H_
#define MOON_HITTABLE_LIST_H_

#include <include/my_utility.h>
#include <include/hittable.h>
#include <include/myobject.h>
#include <include/material.h>
#include <include/texture.h>

#define SPHERES (512)

class Hittable_list {
 public:
   __device__ __host__ Hittable_list(const size_t size, Moon::ImageTexture* tex_img);
   __device__ __host__ explicit Hittable_list(const size_t size);
   //__device__ __host__ Hittable_list();

   __device__ __host__ ~Hittable_list();
	
   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec) ;
 
public:
   HitObject** objects;
   Material** mat_list;
   color background;
   int cur = 0;
};

__global__ void make_scene_device(Hittable_list** scene, const size_t size, unsigned char* img_ptr);

__global__ void destroy_scene_device(Hittable_list** scene);


#endif // !MOON_HITTABLE_LIST_H_