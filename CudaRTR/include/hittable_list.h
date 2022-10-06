#pragma once

#include "my_utility.h"
#include "hittable.h"
#include "sphere.h"
#include "material.h"

#define SPHERES (512)

class hittable_list {
public:
	hit_object** objects;
	material** mat_list;
	int cur = 0;
public:
	__device__
	hittable_list(const size_t size);
	//__device__
	//hittable_list();

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
	objects = new hit_object* [SPHERES];
	Moon::texture* check_tex_ptr = new Moon::texture(new Moon::check_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9)));
	mat_list[cur] = new material(color(0.8, 0.8, 0.0), 1.0, Material_type::Lambertian, check_tex_ptr);
	objects[cur++] = new hit_object(new sphere(point3(0.0, -100.5, -1.0), 100.0, mat_list[0]));

	mat_list[cur] = new material(color(0.7, 0.3, 0.3), 0.3, Material_type::Lambertian);
	objects[cur++] = new hit_object(new moving_sphere(point3(0.0, 0.0, -1.0), point3(0.0, 1.0, -1.0), 0.5, 0.0, 1.0, mat_list[1]));

	mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	objects[cur++] = new hit_object(new sphere(point3(-1.0, 0.0, -1.0), 0.5, mat_list[2]));
	
	mat_list[cur] = new material(color(0.8, 0.6, 0.2), 1.0, Material_type::Metal);
	objects[cur++] = new hit_object(new sphere(point3(1.0, 0.0, -1.0), 0.5, mat_list[3]));

	mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	objects[cur++] = new hit_object(new sphere(point3(-1.0, 0.0, -1.0), -0.4, mat_list[4]));
}

//__device__
//hittable_list::hittable_list() {
//	curandStateXORWOW_t rand_state;
//	curand_init(1, 0, 0, &rand_state);
//
//	cur = 0;
//	mat_list = new material * [SPHERES];
//	objects = new hit_object * [SPHERES];
//
//	//�ذ�
//	mat_list[cur] = new material(color(0.5, 0.5, 0.0), 1.0, Material_type::Lambertian);
//	objects[cur++] = new hit_object(new sphere(point3(0.0, -1000.0, 0.0), 1000.0, mat_list[0]));
//
//	//�������
//	for (int a = -11; a < 11; a++) {
//		for (int b = -11; b < 11; b++) {
//			auto choose_mat = Moon::random_double(&rand_state);
//			point3 center(a + 0.9 * Moon::random_double(&rand_state), 0.2, b + 0.9 * Moon::random_double(&rand_state));
//			
//			if ((center - point3(4.0, 0.2, 0.0)).length() > 0.9) {
//				if (choose_mat < 0.8) {
//					//diffuse
//					auto albedo = color::random(&rand_state);
//					mat_list[cur] = new material(albedo, 1.0, Material_type::Lambertian);
//					objects[cur] = new hit_object(new moving_sphere(center, center + vec3(0, Moon::random_double(0.0, 0.5, &rand_state), 0), 0.2, 0.0, 1.0, mat_list[cur]));
//					cur++;
//				}
//				else if (choose_mat < 0.95) {
//					//metal
//					auto albedo = color::random(0.5, 1.0, &rand_state);
//					auto fuzz = Moon::random_double(0.0, 0.5, &rand_state);
//					mat_list[cur] = new material(albedo, fuzz, Material_type::Metal);
//					objects[cur] = new hit_object(new sphere(center, 0.2, mat_list[cur]));
//					cur++;
//				}
//				else {
//					//glass
//					auto albedo = color::random(0.5, 1.0, &rand_state);
//					auto fuzz = Moon::random_double(1.0, 2.0, &rand_state);
//					mat_list[cur] = new material(albedo, fuzz, Material_type::Dielectric);
//					objects[cur] = new hit_object(new sphere(center, 0.2, mat_list[cur]));
//					cur++;
//				}
//			}
//		}
//	}
//
//	mat_list[cur] = new material(color(0.5, 0.5, 0.0), 1.5, Material_type::Dielectric);
//	objects[cur] = new hit_object(new sphere(point3(0.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//
//	mat_list[cur] = new material(color(0.4, 0.2, 0.1), 1.0, Material_type::Lambertian);
//	objects[cur] = new hit_object(new sphere(point3(-4.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//
//	mat_list[cur] = new material(color(0.7, 0.6, 0.5), 0.0, Material_type::Metal);
//	objects[cur] = new hit_object(new sphere(point3(4.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//}

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
	//*scene = new hittable_list();
	printf("Scene loading is done!\n");
}

__global__
void destroy_scene_device(hittable_list** scene) {
	delete *scene;
	*scene = nullptr;
}