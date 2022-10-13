#include <include/renderer.h>
#include <include/hittable.h>
#include <include/vec3.h>

//计算光线颜色
__device__ color ray_color(ray& r, Hittable_list*const dev_world, size_t depth_max, curandStateXORWOW_t* state) {
    hit_record rec;
    size_t depth = depth_max;
    color result = color(1.0, 1.0, 1.0);
    ray temp;
    color attenuation;

    while (true) {
        if (depth <= 0) return color(0.0, 0.0, 0.0);
        if (!dev_world->Hit(r, 0.001, infinity, rec)) {
            return result * dev_world->background;
        }
        depth--;
        color emitted = rec.mat_ptr->Emitted(rec.u, rec.v, rec.p);
        if (!rec.mat_ptr->Scatter(r, rec, attenuation, temp, state)) {
            result = result * emitted;
            break;
        }
        r = temp;
        result = result * attenuation + emitted;
    }
    return result;
}

//GPU计算像素
__global__ void RenderCUDA(unsigned char* ptr,RayTraceRenderer& render, Camera** cam, Hittable_list** world) {
    // map from threadIdx/BlockIdx to pixel position
    size_t depth = render.max_depth; //TODO::改为CPU设置
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    if (x >= 512 || y >= 512 || offset >= 262144) return;
    int seed = offset;
    curandStateXORWOW_t rand_state;
    curand_init(seed, 0, 0, &rand_state);

    color pixel_color(0.0, 0.0, 0.0);
    for (size_t s = 0; s < render.samples; ++s) {
        double u = double(x + Moon::random_double(&rand_state)) / (double)(render.width - 1);
        double v = double(y + Moon::random_double(&rand_state)) / (double)(render.height - 1);
        ray r = (*cam)->get_ray(u, v, &rand_state);

        pixel_color += ray_color(r, *world, depth, &rand_state);
    }

    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();
    double scale = 1.0 / (double)render.samples;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);
    ptr[offset * 4 + 0] = (int)(Moon::clamp(r, 0.0, 0.999) * 255);
    ptr[offset * 4 + 1] = (int)(Moon::clamp(g, 0.0, 0.999) * 255);
    ptr[offset * 4 + 2] = (int)(Moon::clamp(b, 0.0, 0.999) * 255);
    ptr[offset * 4 + 3] = 255;
}

RayTraceRenderer::RayTraceRenderer()
{
    this->max_depth = 20;
    this->samples = 100;
    this->width = 512;
    this->height = 512;
}

RayTraceRenderer::~RayTraceRenderer()
{
}

void RayTraceRenderer::Render(unsigned char* img_ptr, RayTraceRenderer& render, Camera* dev_cam, Hittable_list* dev_scene) {
    dim3 grids(this->width / 16, this->height / 16);
    dim3 threads(16, 16);
    RenderCUDA<<<grids, threads>>> (img_ptr, render, &dev_cam, &dev_scene);
    cudaDeviceSynchronize();
}
