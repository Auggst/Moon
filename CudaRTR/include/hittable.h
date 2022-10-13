#pragma once
#ifndef MOON_HITTABLE_H_
#define MOON_HITTABLE_H_

#include <include/ray.h>

class Material;

struct hit_record {
	point3 p;
	vec3 normal;
	Material* mat_ptr;
	double t;
	double u, v;
	bool front_face;

	__forceinline__ __device__ __host__
	inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

#endif // !MOON_HITTABLE_H_