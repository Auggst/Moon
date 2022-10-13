#include <include/engine.h>

#include<iostream>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <include/camera.h>
Engine::Engine() {
  this->cam = nullptr;
  this->window = nullptr;
  this->scene = nullptr;
}
Engine::Engine(Camera* _cam, Hittable_list* _scene) {
  this->cam = _cam;
  this->scene = _scene;
}
Engine::~Engine() {
  //单例模式存储在静态存储区，因此不需要释放
  delete this->cam;
  this->cam = nullptr;
  printf("Engine is done! \n");
}

void Engine::Init() {
  if (this->cam == nullptr) this->cam = new Camera();
  if (this->scene == nullptr) this->scene = new Hittable_list(0);

  glfwInit();//初始化
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);//配置GLFW
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);//配置GLFW
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);//
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  this->window =  glfwCreateWindow(512, 512, "Moon", nullptr, nullptr);

  if (this->window==nullptr)
  {
      std::cout << "Failed to create GLFW window" << std::endl;
      glfwTerminate();
      return;
  }
  glfwMakeContextCurrent(this->window);

  //设置Imgui
  ImGui::CreateContext();     // Setup Dear ImGui context
  ImGui::StyleColorsDark();       // Setup Dear ImGui style
  ImGui_ImplGlfw_InitForOpenGL(this->window, true);     // Setup Platform/Renderer backends
  ImGui_ImplOpenGL3_Init("#version 330");
}

void Engine::Update() {
  bool show_demo_window = true;
  bool show_another_window = false;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  while (!glfwWindowShouldClose(this->window)) {
	  /* Render here */

	  /* Swap front and back buffers */
	  ImGui_ImplOpenGL3_NewFrame();
	  ImGui_ImplGlfw_NewFrame();
	  ImGui::NewFrame();

	  // 1. Show the big demo window// (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
	  if (show_demo_window)
		  ImGui::ShowDemoWindow(&show_demo_window);

	  // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
	  {
		  static float f = 0.0f;
		  static int counter = 0;

		  ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

		  ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
		  ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
		  ImGui::Checkbox("Another Window", &show_another_window);

		  ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
		  ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

		  if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			  counter++;
		  ImGui::SameLine();
		  ImGui::Text("counter = %d", counter);

		  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		  ImGui::End();
	  }

	  // 3. Show another simple window.
	  if (show_another_window)
	  {
		  ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
		  ImGui::Text("Hello from another window!");
		  if (ImGui::Button("Close Me"))
			  show_another_window = false;
		  ImGui::End();
	  }

	  glClear(GL_COLOR_BUFFER_BIT);

	  // Rendering
	  ImGui::Render();
	  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	  glfwPollEvents();
	  glfwSwapBuffers(this->window);
   }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwTerminate();
}