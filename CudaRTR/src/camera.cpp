#include <include/camera.h>

Camera::Camera() {
  auto aspect_ratio = 1.0;
  auto viewport_height = 2.0;
  auto viewport_width = aspect_ratio * viewport_height;
  auto focal_length = 1.0;
  
  origin = point3(0.0, 0.0, 0.0);
  horizontal = vec3(viewport_width, 0.0, 0.0);
  vertical = vec3(0.0, viewport_height, 0.0);
  lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0.0, 0.0, focal_length);
  time0 = 0.0;
  time1 = 0.0;
}

Camera::Camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect_ratio, double aperture, double focus_dist, double _time0, double _time1) {
  auto theta = Moon::degrees_to_radians(vfov);
  auto h = tan(theta / 2);
  auto viewport_height = 2.0 * h;
  auto viewport_width = aspect_ratio * viewport_height;

  w = unit_vector(lookfrom - lookat);
  u = unit_vector(cross(vup, w));
  v = cross(w, u);

  origin = lookfrom;
  horizontal = focus_dist * viewport_width * u;
  vertical = focus_dist * viewport_height * v;
  lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

  lens_radius = aperture / 2;
  time0 = _time0;
  time1 = _time1;
}

ray Camera::get_ray(double s, double t, curandStateXORWOW_t* state) const {
	vec3 rd = lens_radius * random_in_unit_disk(state);
	vec3 offset = u * rd.x() + v * rd.y();
	return ray(origin + offset, 
		lower_left_corner + s * horizontal + t * vertical - origin - offset, 
		Moon::random_double(time0, time1, state)
	);
}

void make_camera_device(Camera** cam) {
  printf("Loading camera...\n");
  //*cam = new camera();
  point3 lookfrom(278.0, 278.0, -800.0);
  point3 lookat(278.0, 278.0, 0.0);
  vec3 vup(0.0, 1.0, 0.0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;
  *cam = new Camera(lookfrom, lookat, vup, 40.0, 1.0, aperture, dist_to_focus, 0.0, 1.0);
  printf("Camera loading is done!\n");
}

void destroy_camera_device(Camera** cam) {
  delete* cam;
  *cam = nullptr;
}