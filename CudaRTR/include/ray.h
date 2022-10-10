#pragma once

#include "vec3.h"

class ray {
public:
	point3 orig;
	vec3 dir;
	double tm;
public:
	//����������
	__device__ __host__
	ray() {}
	__device__ __host__
	ray(const point3& origin, const vec3& direction, double time = 0.0) : orig(origin), dir(direction), tm(time) {}

	__device__ __host__
	point3 origin() const { return orig; }
	__device__ __host__
	void origin_set(point3 _orig) { orig = _orig; }
	__device__ __host__
	vec3 direction() const { return dir; }
	__device__ __host__
	void direction_set(vec3 _dir) { dir = _dir; }
	__device__ __host__
	double time() const { return tm; }
	__device__ __host__
	point3 at(double t) const { return orig + t * dir; }

};