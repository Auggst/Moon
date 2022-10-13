#pragma once
#ifndef MOON_MATERIAL_H_
#define MOON_MATERIAL_H_

#include <include/my_utility.h>
#include <include/texture.h>
#include <include/hittable.h>
#include <include/ray.h>

enum class MaterialType { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

class Material {
 public:
   __device__ __host__ Material(const color& a, double f, const MaterialType& _mat_type) 
						: albedo(a), fuzz(f < 1 ? f : 1), mat_type(_mat_type) 
						{ tex = new Moon::Texture(new Moon::SolidTexture(a)); }	//TODO::ÐÞ¸Ä£¬ÄÚ´æÐ¹Â©
   __device__ __host__ Material(const color& a, double f, const MaterialType& _mat_type, Moon::Texture* _tex) 
						: albedo(a), fuzz(f < 1 ? f : 1), mat_type(_mat_type), tex(_tex) 
						{}
   __device__ __host__ ~Material() { delete tex; tex = nullptr; }
   
   __device__ bool Scatter(
						const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* state 
					);
	__device__ __host__ color Emitted(double u, double v, const point3& p);
 public:
  color albedo;
  double fuzz;
  MaterialType mat_type;
  Moon::Texture* tex;
private:
   __device__ static double Reflectance(double cosine, double ref_idx);
};

#endif // !MOON_MATERIAL_H_