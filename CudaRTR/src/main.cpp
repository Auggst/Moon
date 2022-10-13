#include<iostream>
#define GLEW_STATIC
#include <include/engine.h>

using namespace std;

int main(int argc, char** argv[])
{
    Engine& moon = Engine::get_instance();
    moon.Init();
    moon.Update();
    return 0;
}