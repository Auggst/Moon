#pragma once
#include "my_utility.h"

namespace Moon {
	class camera {
	public:
		point3 origin;
		point3 lower_left_corner;
		vec3 horizontal;
		vec3 vertical;
	public:
		__forceinline__ __device__ __host__
			camera() {
			auto aspect_ratio = 1.0;
			auto viewport_height = 2.0;
			auto viewport_width = aspect_ratio * viewport_height;
			auto focal_length = 1.0;

			origin = point3(0.0, 0.0, 0.0);
			horizontal = vec3(viewport_width, 0.0, 0.0);
			vertical = vec3(0.0, viewport_height, 0.0);
			lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0.0, 0.0, focal_length);
		}
		__forceinline__ __device__
			camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect_ratio) {
			auto theta = degrees_to_radians(vfov);
			auto h = tan(theta / 2);
			auto viewport_height = 2.0 * h;
			auto viewport_width = aspect_ratio * viewport_height;

			auto w = unit_vector(lookfrom - lookat);
			auto u = unit_vector(cross(vup, w));
			auto v = cross(w, u);

			origin = lookfrom;
			horizontal = viewport_width * u;
			vertical = viewport_height * v;
			lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
		}

		__forceinline__ __device__
			ray get_ray(double s, double t) const {
			return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
		}
	};

	__global__
	void make_camera_device(camera** cam) {
		printf("Loading camera...\n");
		*cam = new camera();
		//*cam = new camera(point3(-2.0, 2.0, 1.0), point3(0.0, 0.0, -1.0), vec3(0.0, 1.0, 0.0), 90.0, 1.0);
		printf("Camera loading is done!\n");
	}

	__global__
	void destroy_camera_device(camera** cam) {
		delete* cam;
		*cam = nullptr;
	}
}
