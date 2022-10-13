#pragma once
#ifndef MOON_VEC3_H_
#define MOON_VEC3_H_

#include <cmath>
#include <cstdio>

#include <include/my_utility.h>

using std::sqrt;

class vec3 {
public:
		double val[3];
public:
	__device__ __host__ vec3() :val{ 0.0, 0.0, 0.0 } {}
	__device__ __host__ vec3(double _x, double _y, double _z) : val{ _x, _y,_z } {}

	__device__ __host__
	double x() const { return val[0]; }
	__device__ __host__
	double y() const { return val[1]; }
	__device__ __host__
	double z() const { return val[2]; }

	__device__ __host__
	vec3 operator-() const { return vec3(-val[0], -val[1], -val[2]); }
	__device__ __host__
	double operator[](int i) const { return val[i]; }
	__device__ __host__
	double& operator[](int i) { return val[i]; }

	__device__ __host__
	vec3& operator+=(const vec3& v) {
		val[0] += v.val[0];
		val[1] += v.val[1];
		val[2] += v.val[2];
		return *this;
	}

	__device__ __host__
	vec3& operator=(const vec3& v) {
		val[0] = v.val[0];
		val[1] = v.val[1];
		val[2] = v.val[2];
		return *this;
	}
	__device__ __host__
	vec3& operator*=(const double t) {
		val[0] *= t;
		val[1] *= t;
		val[2] *= t;
		return *this;
	}
	__device__ __host__
	vec3& operator/=(const double t) {
		return *this *= 1 / t;
	}

	__device__ __host__
	double length() const {
		return sqrt(length_squared());
	}
	__device__ __host__
	double length_squared() const {
		return val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
	}

	__forceinline__ __device__ 
	inline static vec3 random(curandStateXORWOW_t* state) {
		return vec3(Moon::random_double(state), Moon::random_double(state), Moon::random_double(state));
	}

	__forceinline__ __device__ 
	inline static vec3 random(double min, double max, curandStateXORWOW_t* state) {
		return vec3(Moon::random_double(min, max, state), Moon::random_double(min, max, state), Moon::random_double(min, max, state));
	}

	__device__
	bool near_zero() const {
		const auto s = 1e-8;
		return (fabs(val[0]) < s) && (fabs(val[1]) < s) && (fabs(val[2]) < s);
	}
};

using point3 = vec3;
using color = vec3;

// vec3 Utility Functions
__forceinline__ __device__ __host__
inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.val[0] + v.val[0], u.val[1] + v.val[1], u.val[2] + v.val[2]);
}

__forceinline__ __device__ __host__
inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.val[0] - v.val[0], u.val[1] - v.val[1], u.val[2] - v.val[2]);
}
__forceinline__ __device__ __host__
inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.val[0] * v.val[0], u.val[1] * v.val[1], u.val[2] * v.val[2]);
}

__forceinline__ __device__ __host__
inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.val[0], t * v.val[1], t * v.val[2]);
}

__forceinline__ __device__ __host__
inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

__forceinline__ __device__ __host__
inline vec3 operator/(vec3 v, double t) {
	return (1 / t) * v;
}

__forceinline__ __device__ __host__
inline double dot(const vec3& u, const vec3& v) {
	return u.val[0] * v.val[0]
		+ u.val[1] * v.val[1]
		+ u.val[2] * v.val[2];
}

__forceinline__ __device__ __host__
inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.val[1] * v.val[2] - u.val[2] * v.val[1],
		u.val[2] * v.val[0] - u.val[0] * v.val[2],
		u.val[0] * v.val[1] - u.val[1] * v.val[0]);
}

__forceinline__ __device__ __host__
inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

__forceinline__ __device__ __host__
inline vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

__forceinline__ __device__ __host__
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
	auto cos_theta = fmin(dot(-uv, n), 1.0);
	vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
	vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_perp + r_out_parallel;
}

__forceinline__ __device__ inline vec3 random_in_unit_sphere(curandStateXORWOW_t* state) {
	while (true) {
		vec3 p = vec3::random(-1.0, 1.0, state);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

__forceinline__ __device__ inline vec3 random_unit_vector(curandStateXORWOW_t* state){
	return unit_vector(random_in_unit_sphere(state));
}

__forceinline__ __device__ inline vec3 random_in_hemisphere(const vec3& normal, curandStateXORWOW_t* state) {
	while (true) {
		vec3 in_unit_sphere = random_in_unit_sphere(state);
		if (dot(in_unit_sphere, normal) > 0.0)
			return in_unit_sphere;
		else
			return -in_unit_sphere;
	}
}

__forceinline__ __device__ inline vec3 random_in_unit_disk(curandStateXORWOW_t* state) {
	while (true) {
		auto p = vec3(Moon::random_double(-1.0, 1.0, state), Moon::random_double(-1.0, 1.0, state), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

#endif // !MOON_VEC3_H_