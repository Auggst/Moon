#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include "hittable.h"
#include "sphere.h"
#include "material.h"

#define SPHERES (256)

class hittable_list {
public:
	sphere** objects;
	material** mat_list;
	int cur = 0;
public:
	__device__
	hittable_list(const size_t size);

	__device__
	~hittable_list() {
		for (size_t i = 0; i < SPHERES; i++) {
			delete objects[i];
			delete mat_list[i];
		}
		delete[] objects;
		objects = nullptr;
		delete[] mat_list;
		mat_list = nullptr;
	}
	
	__device__
	bool hit(const ray& r, double t_min, double t_max, hit_record& rec) ;
};

__device__
hittable_list::hittable_list(const size_t size) {
	cur = 0;
	mat_list = new material* [SPHERES];
	objects = new sphere* [SPHERES];
	mat_list[cur] = new material(color(0.8, 0.8, 0.0), 1.0, Material_type::Lambertian);
	objects[cur++] = new sphere(point3(0.0, -100.5, -1.0), 100.0, mat_list[0]);

	mat_list[cur] = new material(color(0.7, 0.3, 0.3), 0.3, Material_type::Lambertian);
	objects[cur++] = new sphere(point3(0.0, 0.0, -1.0), 0.5, mat_list[1]);

	mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	objects[cur++] = new sphere(point3(-1.0, 0.0, -1.0), 0.5, mat_list[2]);
	
	mat_list[cur] = new material(color(0.8, 0.6, 0.2), 1.0, Material_type::Metal);
	objects[cur++] = new sphere(point3(1.0, 0.0, -1.0), 0.5, mat_list[3]);

	mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	objects[cur++] = new sphere(point3(-1.0, 0.0, -1.0), -0.4, mat_list[4]);
}

__device__
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec){
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (size_t i = 0; i < cur; i++) {
		if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

__global__
void make_scene_device(hittable_list** scene, const size_t size) {
	printf("Loading secne...\n");
	*scene = new hittable_list(size);
	printf("Scene loading is done!\n");
}

__global__
void destroy_scene_device(hittable_list** scene) {
	delete *scene;
	*scene = nullptr;
}