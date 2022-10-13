#pragma once
#ifndef MOON_ENGINE_H_
#define MOON_ENGINE_H_

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>

#include <include/hittable_list.h>

extern class Camera;
class Engine {
 public:
  Engine();
  Engine(Camera* _cam, Hittable_list* _scene);
  Engine(const Engine& other) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine();

  static Engine& get_instance() { static Engine value; return value;}

  void Init();
  void Update();
  //void On_mouse_move(int x, int y);
  //void On_mouse_scroll(int a);
 public:
  Camera* cam;
  Hittable_list* scene;
  GLFWwindow* window;
};

#endif //MOON_ENGINE_H_