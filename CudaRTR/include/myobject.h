#pragma once
#ifndef MOON_MYOBJECT_H_
#define MOON_MYOBJECT_H_

#include <include/my_utility.h>
#include <include/material.h>

enum class ObjectType {SPHERE, MOVING_SPHERE, XY_RECT, XZ_RECT, YZ_RECT, BOX};

class Sphere {
 public:
   __device__  __host__ Sphere() { center = point3(0.0, 0.0, 0.0); radius = 1.0; mat_ptr = nullptr; }
   __device__  __host__ Sphere(point3 cen, double r, Material* _mat_ptr) : center(cen), radius(r), mat_ptr(_mat_ptr) {}
    
   __device__ void get_uv(const point3& p, double& u, double& v);
   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   point3 center;
   double radius;
   Material* mat_ptr;
};

class MovingSphere {
 public:
   __device__ __host__ MovingSphere();                        
   __device__ __host__ MovingSphere(point3 cen0, point3 cen1, double r, double _time0, double _time1, Material* _mat_ptr): 
                            center0(cen0), center1(cen1), radius(r), time0(_time0), time1(_time1), mat_ptr(_mat_ptr){}
   __device__ point3 Center(double time) const { return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0); }
   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   point3 center0, center1;
   double radius;
   double time0;
   double time1;
   Material* mat_ptr;
};

class XY_Rect {
 public:
   __device__ __host__ XY_Rect() { mat_ptr = nullptr; x0 = 0; x1 = 0; y0 = 0; y1 = 0; k = 0; }
   __device__ __host__ XY_Rect(double _x0, double _x1, double _y0, double _y1, double _k, Material* mat)
                              : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {}
   
   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   Material* mat_ptr;
   double x0, x1, y0, y1, k;
};

class XZ_Rect {
 public:
   __device__ __host__ XZ_Rect() { mat_ptr = nullptr; x0 = 0; x1 = 0; z0 = 0; z1 = 0; k = 0; }
   __device__ __host__ XZ_Rect(double _x0, double _x1, double _z0, double _z1, double _k, Material* mat)
                                : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {}

   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);

 public:
   Material* mat_ptr;
   double x0, x1, z0, z1, k;
};

class YZ_Rect {
 public:
   __device__ __host__ YZ_Rect() { mat_ptr = nullptr; y0 = 0;  y1 = 0; z0 = 0; z1 = 0; k = 0; }
   __device__ __host__ YZ_Rect(double _y0, double _y1, double _z0, double _z1, double _k, Material* mat)
                                : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {}

   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   Material* mat_ptr;
   double y0, y1, z0, z1, k;
};

class Box {
 public:
   __device__ __host__ Box(const point3& p0, const point3& p1, Material* ptr);
   __device__ __host__ void  Update();
   __device__
   bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   point3 box_min;
   point3 box_max;
   XY_Rect xy[2];
   XZ_Rect xz[2];
   YZ_Rect yz[2];
};

class HitObject {
public:
   __device__ __host__ HitObject(Sphere* _sphere) { hit_sphere = _sphere; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_box = nullptr;  hit_type = ObjectType::SPHERE;}
   __device__ __host__ HitObject(MovingSphere* _moving_sphere) { hit_moving_sphere = _moving_sphere; hit_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_box = nullptr;  hit_type = ObjectType::MOVING_SPHERE; }
   __device__ __host__ HitObject(XY_Rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = _rect; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_box = nullptr; hit_type = ObjectType::XY_RECT; }
   __device__ __host__ HitObject(XZ_Rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = _rect; hit_yz_rect = nullptr; hit_box = nullptr; hit_type = ObjectType::XZ_RECT; }
   __device__ __host__ HitObject(YZ_Rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = _rect; hit_box = nullptr; hit_type = ObjectType::YZ_RECT; }
   __device__ __host__ HitObject(Box* _box) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_box = _box; hit_type = ObjectType::BOX; }
  
   __device__ __host__ void Translate(const vec3& displacement);
   __device__ bool Hit(const ray& r, double t_min, double t_max, hit_record& rec);
 public:
   Sphere* hit_sphere;
   MovingSphere* hit_moving_sphere;
   XY_Rect* hit_xy_rect;
   XZ_Rect* hit_xz_rect;
   YZ_Rect* hit_yz_rect;
   Box* hit_box;
   ObjectType hit_type;
};


#endif // !MOON_MYOBJECT_H_