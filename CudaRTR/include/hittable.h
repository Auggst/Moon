#pragma once

#include "ray.h"

enum Material_type;
class material;
class lambertian;


struct hit_record {
	point3 p;
	vec3 normal;
	material* mat_ptr;
	double t;
	bool front_face;

	__forceinline__ __device__ __host__
	inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	__device__ __host__
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};