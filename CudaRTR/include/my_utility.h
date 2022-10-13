#pragma once
#ifndef MOON_MY_UTILITY_H_
#define MOON_MY_UTILITY_H_

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
__forceinline__ __device__ __host__ inline double degrees_to_radians(double degrees) {
  return degrees * PI / 180.0;
}

__forceinline__ __device__ inline double random_double(double min, double max, curandStateXORWOW_t* state) {
  return min + (max - min) * curand_uniform_double(state);
}
__forceinline__ __device__ inline double random_double(curandStateXORWOW_t* state) {
  return random_double(0.0, 1.0, state);
}


__forceinline__ __device__ __host__ inline double clamp(double x, double min, double max) {
  if (x < min) return min;
  else if (x > max) return max;
  else return x;
}
}

#endif // !MOON_MY_UTILITY_H_