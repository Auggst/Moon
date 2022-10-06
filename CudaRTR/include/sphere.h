#pragma once

#include "hittable.h"
#include "vec3.h"
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

class metal;
class lambertian;

enum class Material_type;

enum class Object_type {Sphere, Moving_sphere, XY_rect, XZ_rect, YZ_rect};

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
    auto phi = atan2(-p.z(), p.x()) + PI;

    u = phi / (2 * PI);
    v = theta / PI;
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

class xy_rect {
public:
    material* mat_ptr;
    double x0, x1, y0, y1, k;
public:
    __device__
    xy_rect() {}
    __device__
        xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, material* mat)
            :   x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat) {}
    __device__
        bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

bool xy_rect::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max) return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)   return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    auto outward_normal = vec3(0.0, 0.0, 1.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

class xz_rect {
public:
    material* mat_ptr;
    double x0, x1, z0, z1, k;
public:
    __device__
        xz_rect() {}
    __device__
        xz_rect(double _x0, double _x1, double _z0, double _z1, double _k, material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {}
    __device__
        bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

bool xz_rect::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max) return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)   return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = vec3(0.0, 1.0, 0.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

class yz_rect {
public:
    material* mat_ptr;
    double y0, y1, z0, z1, k;
public:
    __device__
        yz_rect() {}
    __device__
        yz_rect(double _y0, double _y1, double _z0, double _z1, double _k, material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat) {}
    __device__
        bool hit(const ray& r, double t_min, double t_max, hit_record& rec);
};

bool yz_rect::hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max) return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)   return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = vec3(1.0, 0.0, 0.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

class hit_object {
public:
    sphere* hit_sphere;
    moving_sphere* hit_moving_sphere;
    xy_rect* hit_xy_rect;
    xz_rect* hit_xz_rect;
    yz_rect* hit_yz_rect;
    Object_type hit_type;
public:
    __device__
        hit_object(sphere* _sphere) { hit_sphere = _sphere; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_type = Object_type::Sphere; }
    __device__
        hit_object(moving_sphere* _moving_sphere) { hit_moving_sphere = _moving_sphere; hit_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_type = Object_type::Moving_sphere; }
    __device__
        hit_object(xy_rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = _rect; hit_xz_rect = nullptr; hit_yz_rect = nullptr; hit_type = Object_type::XY_rect; }
    __device__
        hit_object(xz_rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = _rect; hit_yz_rect = nullptr; hit_type = Object_type::XZ_rect; }
    __device__
        hit_object(yz_rect* _rect) { hit_sphere = nullptr; hit_moving_sphere = nullptr; hit_xy_rect = nullptr; hit_xz_rect = nullptr; hit_yz_rect = _rect; hit_type = Object_type::YZ_rect; }

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
    else if (this->hit_type == Object_type::XY_rect) {
        return hit_xy_rect->hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == Object_type::XZ_rect) {
        return hit_xz_rect->hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == Object_type::YZ_rect) {
        return hit_yz_rect->hit(r, t_min, t_max, rec);
    }
    else {
        return false;
    }
}