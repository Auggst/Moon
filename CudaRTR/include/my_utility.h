#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>

#define PI (3.1415926535897932385)
#define INF (2e10f)
#define infinity (2e8f)

using std::sqrt;

namespace Moon {
	__forceinline__ __device__ __host__
	inline double degrees_to_radians(double degrees) {
		return degrees * PI / 180.0;
	}

	__device__
	double random_double(double min, double max, curandStateXORWOW_t* state) {
		return min + (max - min) * curand_uniform_double(state);
	}
	__device__
	double random_double(curandStateXORWOW_t* state) {
		return random_double(0.0, 1.0, state);
	}


	__device__ __host__
	double clamp(double x, double min, double max) {
		if (x < min) return min;
		else if (x > max) return max;
		else return x;
	}
}

#include "ray.h"
#include "vec3.h"

__forceinline__ __device__
vec3 random_in_unit_sphere(curandStateXORWOW_t* state) {
	while (true) {
		vec3 p = vec3::random(-1.0, 1.0, state);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

__forceinline__ __device__
vec3 random_unit_vector(curandStateXORWOW_t* state){
	return unit_vector(random_in_unit_sphere(state));
}

__forceinline__ __device__
vec3 random_in_hemisphere(const vec3& normal, curandStateXORWOW_t* state) {
	while (true) {
		vec3 in_unit_sphere = random_in_unit_sphere(state);
		if (dot(in_unit_sphere, normal) > 0.0)
			return in_unit_sphere;
		else
			return -in_unit_sphere;
	}
}

__forceinline__ __device__
vec3 random_in_unit_disk(curandStateXORWOW_t* state) {
	while (true) {
		auto p = vec3(Moon::random_double(-1.0, 1.0, state), Moon::random_double(-1.0, 1.0, state), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}