#include <include/material.h>

bool Material::Scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandStateXORWOW_t* state) {
	if (this->mat_type == MaterialType::LAMBERTIAN) {
		vec3 scatter_direction = rec.normal + fuzz * random_unit_vector(state);

		//避免法线反向
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;
		scatter_direction = unit_vector(scatter_direction);
		scattered = ray(rec.p, scatter_direction, r_in.time());
		attenuation = this->tex->Value(rec.u, rec.v, rec.p);
		return true;
	}
	else if (this->mat_type == MaterialType::METAL) {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state), r_in.time());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}
	else if (this->mat_type == MaterialType::DIELECTRIC) {
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / fuzz) : fuzz;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		bool schlick = this->Reflectance(cos_theta, refraction_ratio) > Moon::random_double(state);
		vec3 direction = (cannot_refract || schlick) ? reflect(unit_direction, rec.normal) : refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction, r_in.time());
		return true;
	}
	else if (this->mat_type == MaterialType::DIFFUSE_LIGHT) {
		return false;
	}
	else {
		return false;
	}
}

color Material::Emitted(double u, double v, const point3& p) {
	if (this->mat_type == MaterialType::LAMBERTIAN) {
		return color(0.0, 0.0, 0.0);
	}
	else if (this->mat_type == MaterialType::METAL) {
		return color(0.0, 0.0, 0.0);
	}
	else if (this->mat_type == MaterialType::DIELECTRIC) {
		return color(0.0, 0.0, 0.0);
	}
	else if (this->mat_type == MaterialType::DIFFUSE_LIGHT) {
		return this->tex->Value(u, v, p);
	}
	else {
		return color(0.0, 0.0, 0.0);
	}
}

double Material::Reflectance(double cosine, double ref_idx) {
	auto r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}
