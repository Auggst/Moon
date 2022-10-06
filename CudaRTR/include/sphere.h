#pragma once

#include "hittable.h"
#include "vec3.h"
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

class metal;
class lambertian;

enum class Material_type;

enum class Object_type {Sphere, Moving_sphere};

class sphere {
public:
	point3 center;
	double radius;
    material* mat_ptr;
public:
	__device__  
        sphere() { center = point3(0.0, 0.0, 0.0); radius = 1.0; mat_ptr = nullptr; }
	__device__  
        sphere(point3 cen, double r, material* _mat_ptr) : center(cen), radius(r), mat_ptr(_mat_ptr) {}
    __device__
        void get_uv(const point3& p, double& u, double& v);
	__device__ 
		bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

//求球面的uv坐标
void sphere::get_uv(const point3& p, double& u, double& v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2 * pi);
    v = theta / pi;
}

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
    get_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;
    return true;
}

class moving_sphere {
public:
    point3 center0, center1;
    double radius;
    double time0;
    double time1;
    material* mat_ptr;
public:
    __device__
        moving_sphere() { center0 = point3(0.0, 0.0, 0.0); center1 = point3(0.0, 1.0, 0.0); radius = 1.0; time0 = 0.0; time1 = 0.0; mat_ptr = nullptr; }
    __device__
    moving_sphere(point3 cen0, point3 cen1, double r, double _time0, double _time1, material* _mat_ptr): 
        center0(cen0), center1(cen1), radius(r), time0(_time0), time1(_time1), mat_ptr(_mat_ptr){}
    __device__
        point3 center(double time) const { return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0); }
    __device__
        bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

bool moving_sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    //光线与圆求交的参数
    vec3 oc = r.origin() - center(r.time());
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    //计算光线与圆交点
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    //更新hit_record
    rec.t = root;
    rec.p = r.at(rec.t);
    auto outward_normal = (rec.p - center(r.time())) / radius; //法线就是圆心到交点的向量
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

class hit_object {
public:
    sphere* hit_sphere;
    moving_sphere* hit_moving_sphere;
    Object_type hit_type;
public:
    __device__
        hit_object(sphere* _sphere) { hit_sphere = _sphere; hit_moving_sphere = nullptr; hit_type = Object_type::Sphere; }
    __device__
        hit_object(moving_sphere* _moving_sphere) { hit_moving_sphere = _moving_sphere; hit_sphere = nullptr; hit_type = Object_type::Moving_sphere; }
    __device__
        bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

bool hit_object::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    if (this->hit_type == Object_type::Sphere) {
        return hit_sphere->hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == Object_type::Moving_sphere) {
        return hit_moving_sphere->hit(r, t_min, t_max, rec);
    }
    else {
        return false;
    }
}