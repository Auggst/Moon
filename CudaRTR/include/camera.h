#pragma once
#ifndef MOON_CAMERA_H_
#define MOON_CAMERA_H_

#include <include/vec3.h>
#include <include/ray.h>

class Camera {
 public:
   __device__ __host__ Camera();
   __device__ __host__ Camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect_ratio, double aperture, double focus_dist, double _time0, double _time1);

   __device__ ray get_ray(double s, double t, curandStateXORWOW_t* state) const;

 public:
   point3 origin;
   point3 lower_left_corner;
   vec3 horizontal;
   vec3 vertical;
   vec3 u, v, w;
   double lens_radius;
   double time0, time1;
};

__global__ void make_camera_device(Camera** cam);

__global__ void destroy_camera_device(Camera** cam);

#endif // !MOON_CAMERA_H_