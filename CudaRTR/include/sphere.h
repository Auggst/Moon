#pragma once

#include "hittable.h"
#include "vec3.h"
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

class metal;
class lambertian;

enum Material_type;

class sphere {
public:
	point3 center;
	double radius;
    material* mat_ptr;
public:
	__device__  
		sphere() {}
	__device__  
        sphere(point3 cen, double r, material* _mat_ptr) : center(cen), radius(r), mat_ptr(_mat_ptr) {}
	__device__ 
		bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

//判断是否击中
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    //判断光线与圆相交的参数
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    //相交求交点的t值
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    return true;
}