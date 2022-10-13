#include <include/hittable_list.h>

Hittable_list::Hittable_list(const size_t size, Moon::ImageTexture* tex_img) {
	this->background = color(0.70, 0.80, 1.00);
	cur = 0;
	mat_list = new Material* [SPHERES];
	objects = new HitObject* [SPHERES];
	//Moon::texture* img_tex_ptr = new Moon::texture(tex_img);
	//mat_list[cur] = new material(color(0.8, 0.8, 0.0), 1.0, Material_type::Lambertian, img_tex_ptr);
	//objects[cur++] = new hit_object(new sphere(point3(0.0, -100.5, -1.0), 100.0, mat_list[0]));

	//mat_list[cur] = new material(color(0.7, 0.3, 0.3), 0.3, Material_type::Lambertian);
	//objects[cur++] = new hit_object(new moving_sphere(point3(0.0, 0.0, -1.0), point3(0.0, 1.0, -1.0), 0.5, 0.0, 1.0, mat_list[1]));

	//mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	//objects[cur++] = new hit_object(new sphere(point3(-1.0, 0.0, -1.0), 0.5, mat_list[2]));

	//mat_list[cur] = new material(color(0.8, 0.6, 0.2), 1.0, Material_type::Metal);
	//objects[cur++] = new hit_object(new sphere(point3(1.0, 0.0, -1.0), 0.5, mat_list[3]));

	//mat_list[cur] = new material(color(0.8, 0.8, 0.8), 1.5, Material_type::Dielectric);
	//objects[cur++] = new hit_object(new sphere(point3(-1.0, 0.0, -1.0), -0.4, mat_list[4]));
}

Hittable_list::Hittable_list(const size_t size) {
	this->background = color(0.0, 0.0, 0.0);
	cur = 0;
	mat_list = new Material * [size];
	objects = new HitObject * [size];

	mat_list[0] = new Material(color(0.65, 0.05, 0.05), 1.0, MaterialType::LAMBERTIAN);	//红
	mat_list[1] = new Material(color(0.73, 0.73, 0.73), 1.0, MaterialType::LAMBERTIAN);	//白
	mat_list[2] = new Material(color(0.12, 0.45, 0.15), 1.0, MaterialType::LAMBERTIAN);	//绿
	mat_list[3] = new Material(color(15, 15, 15), 1.0, MaterialType::DIFFUSE_LIGHT);		//光

	objects[cur++] = new HitObject(new YZ_Rect(0.0, 555.0, 0.0, 555.0, 555.0, mat_list[2]));	//左
	objects[cur++] = new HitObject(new YZ_Rect(0.0, 555.0, 0.0, 555.0, 0.0, mat_list[0]));		//右
	objects[cur++] = new HitObject(new XZ_Rect(213.0, 343.0, 227.0, 332.0, 554.0, mat_list[3]));	//灯光
	objects[cur++] = new HitObject(new XZ_Rect(0.0, 555.0, 0.0, 555.0, 0.0, mat_list[1]));		//下
	objects[cur++] = new HitObject(new XZ_Rect(0.0, 555.0, 0.0, 555.0, 555.0, mat_list[1]));	//上
	objects[cur++] = new HitObject(new XY_Rect(0.0, 555.0, 0.0, 555.0, 555.0, mat_list[1]));	//后
	objects[cur++] = new HitObject(new Box(point3(0.0, 0.0, 0.0), point3(165.0, 330.0, 165.0), mat_list[1]));  //前
	//objects[cur - 1]->rotate(15.0, 1);
	objects[cur - 1]->Translate(vec3(265.0, 0.0, 295.0));
	objects[cur++] = new HitObject(new Box(point3(0.0, 0.0, 0.0), point3(165.0, 165.0, 165.0), mat_list[1])); //后
	//objects[cur - 1]->rotate(-18.0,1);
	objects[cur - 1]->Translate(vec3(130.0, 0.0, 65.0));
}


Hittable_list::~Hittable_list() {
	//for (size_t i = 0; i < SPHERES; i++) {
	//	delete objects[i];
	//	delete mat_list[i];
	//}
	delete[] objects;
	objects = nullptr;
	delete[] mat_list;
	mat_list = nullptr;
}

//Hittable_list::Hittable_list() {
//	curandStateXORWOW_t rand_state;
//	curand_init(1, 0, 0, &rand_state);
//
//	cur = 0;
//	mat_list = new Material * [SPHERES];
//	objects = new HitObject * [SPHERES];
//
//	//地板
//	mat_list[cur] = new Material(color(0.5, 0.5, 0.0), 1.0, MaterialType::LAMBERTIAN);
//	objects[cur++] = new HitObject(new Sphere(point3(0.0, -1000.0, 0.0), 1000.0, mat_list[0]));
//
//	//随机场景
//	for (int a = -11; a < 11; a++) {
//		for (int b = -11; b < 11; b++) {
//			auto choose_mat = Moon::random_double(&rand_state);
//			point3 center(a + 0.9 * Moon::random_double(&rand_state), 0.2, b + 0.9 * Moon::random_double(&rand_state));
//			
//			if ((center - point3(4.0, 0.2, 0.0)).length() > 0.9) {
//				if (choose_mat < 0.8) {
//					//diffuse
//					auto albedo = color::random(&rand_state);
//					mat_list[cur] = new Material(albedo, 1.0, MaterialType::LAMBERTIAN);
//					objects[cur] = new HitObject(new MovingSphere(center, center + vec3(0, Moon::random_double(0.0, 0.5, &rand_state), 0), 0.2, 0.0, 1.0, mat_list[cur]));
//					cur++;
//				}
//				else if (choose_mat < 0.95) {
//					//metal
//					auto albedo = color::random(0.5, 1.0, &rand_state);
//					auto fuzz = Moon::random_double(0.0, 0.5, &rand_state);
//					mat_list[cur] = new Material(albedo, fuzz, MaterialType::METAL);
//					objects[cur] = new HitObject(new Sphere(center, 0.2, mat_list[cur]));
//					cur++;
//				}
//				else {
//					//glass
//					auto albedo = color::random(0.5, 1.0, &rand_state);
//					auto fuzz = Moon::random_double(1.0, 2.0, &rand_state);
//					mat_list[cur] = new Material(albedo, fuzz, MaterialType::DIELECTRIC);
//					objects[cur] = new HitObject(new Sphere(center, 0.2, mat_list[cur]));
//					cur++;
//				}
//			}
//		}
//	}
//
//	mat_list[cur] = new Material(color(0.5, 0.5, 0.0), 1.5, MaterialType::DIELECTRIC);
//	objects[cur] = new HitObject(new Sphere(point3(0.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//
//	mat_list[cur] = new Material(color(0.4, 0.2, 0.1), 1.0, MaterialType::LAMBERTIAN);
//	objects[cur] = new HitObject(new Sphere(point3(-4.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//
//	mat_list[cur] = new Material(color(0.7, 0.6, 0.5), 0.0, MaterialType::METAL);
//	objects[cur] = new HitObject(new Sphere(point3(4.0, 1.0, 0.0), 1.0, mat_list[cur]));
//	cur++;
//}

bool Hittable_list::Hit(const ray& r, double t_min, double t_max, hit_record& rec){
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (size_t i = 0; i < cur; i++) {
		if (objects[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

__global__ void make_scene_device(Hittable_list** scene, const size_t size, unsigned char* img_ptr) {
	printf("Loading secne...\n");
	//Moon::ImageTexture* tex_img = new Moon::ImageTexture(img_ptr, 1024, 512);
	//*scene = new Hittable_list(size, tex_img);
	*scene = new Hittable_list(size);
	//*scene = new Hittable_list();
	printf("Scene loading is done!\n");
}

__global__ void destroy_scene_device(Hittable_list** scene) {
	delete *scene;
	*scene = nullptr;
}