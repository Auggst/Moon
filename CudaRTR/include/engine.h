#pragma once
#ifndef MOON_ENGINE_H_
#define MOON_ENGINE_H_

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>

#include <include/hittable_list.h>
#include <include/device_manage.h>
#include <include/renderer.h>

extern class Camera;
class Engine {
 public:
  Engine();
  Engine(Camera* _cam, Hittable_list* _scene);
  Engine(const Engine& other) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine();

  static Engine& get_instance() { static Engine value; return value;}

  void Init(int window_width, int window_height, size_t depth_max, size_t samples);
  void Update();
  //void On_mouse_move(int x, int y);
  //void On_mouse_scroll(int a);
 public:
  Camera* cam;
  Hittable_list* scene;
  GLFWwindow* window;
  RayTraceRenderer* render;
  DeviceManager dev_manager;
  
};

#endif //MOON_ENGINE_H_