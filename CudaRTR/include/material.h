#pragma once

#include "my_utility.h"
#include "texture.h"

struct hit_record;
enum class Material_type { Lambertian, Metal, Dielectric, Diffuse_light };

class material {
public:
	color albedo;
	double fuzz;
	Material_type mat_type;
	Moon::texture* tex;
public:
	__device__
		material(const color& a, double f, const Material_type& _mat_type) : albedo(a), fuzz(f < 1 ? f : 1), mat_type(_mat_type) { tex = new Moon::texture(); }
	__device__
		material(const color& a, double f, const Material_type& _mat_type, Moon::texture* _tex) : albedo(a), fuzz(f < 1 ? f : 1), mat_type(_mat_type), tex(_tex) {}
	__device__
		~material() { delete tex; tex = nullptr; }
	__device__
		bool scatter(
			const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* state 
		) const;
	__device__
		color emitted(double u, double v, const point3& p);
private:
	__device__
	static double reflectance(double cosine, double ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};


__device__
bool material::scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* state) const {
	if (this->mat_type == Material_type::Lambertian) {
		vec3 scatter_direction = rec.normal + fuzz * random_unit_vector(state);

		//避免法线反向
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;
		scatter_direction = unit_vector(scatter_direction);
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = tex->value(rec.u, rec.v, rec.p);
		return true;
	}
	else if (this->mat_type == Material_type::Metal) {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state), r_in.time());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
	else if (this->mat_type == Material_type::Dielectric) {
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / fuzz) : fuzz;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		bool schlick = this->reflectance(cos_theta, refraction_ratio) > Moon::random_double(state);
		vec3 direction = (cannot_refract || schlick) ? reflect(unit_direction, rec.normal) : refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}
	else if (this->mat_type == Material_type::Diffuse_light) {
		return false;
	}
	else {
		return false;
	}
}

__device__
color material::emitted(double u, double v, const point3& p) {
	if (this->mat_type == Material_type::Lambertian) {
		return tex->value(u, v, p);
	}
	else if (this->mat_type == Material_type::Metal) {
		return color(0.0, 0.0, 0.0);
	}
	else if (this->mat_type == Material_type::Dielectric) {
		return color(0.0, 0.0, 0.0);
	}
	else if (this->mat_type == Material_type::Diffuse_light) {
		return tex->value(u, v, p);
	}
	else {
		return color(0.0, 0.0, 0.0);
	}
}