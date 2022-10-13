#pragma once
#ifndef MOON_TEXTURE_H_
#define MOON_TEXTURE_H_

#include <include/vec3.h>

enum class TextureType {SOLID, CHECK, IMAGE};

namespace Moon {
class SolidTexture {
 public:
   __device__ __host__ SolidTexture() { color_value = color(1.0, 1.0, 1.0); }
   __device__ __host__ SolidTexture(color _col) { color_value = _col; }
   __device__ __host__ SolidTexture(double _r, double _g, double _b) : color_value(color(_r, _g, _b)) {}
		
   __device__ color Value(double u, double v, const vec3& p) const;
 public:
   color color_value;
};

class CheckTexture {
 public:
   __device__ __host__ CheckTexture() { odd = SolidTexture(color(0.0, 0.0, 0.0)); even = SolidTexture(); }
   __device__ __host__ CheckTexture(color col0, color col1) { odd = col0; even = col1; }

   __device__ color Value(double u, double v, const point3& p);
 public:
   SolidTexture odd;
   SolidTexture even;
};

class ImageTexture {
 public:
   __device__ __host__ ImageTexture() : _data(nullptr), _width(0), _height(0), _bytes_per_scanline(0) {}
   __device__ __host__ ImageTexture(unsigned char* file, int width, int height);

   __device__ color Value(double u, double v, const vec3& p) const;
 public:
   unsigned char* _data;
   int _width, _height;
   int _bytes_per_scanline;
   const static int bytes_per_pixel = 3;
};

class Texture {
 public:
   __device__ __host__ Texture();
   __device__ __host__ Texture(SolidTexture* _solid_ptr);
   __device__ __host__ Texture(CheckTexture* _check_ptr);
   __device__ __host__ Texture(ImageTexture* _img_ptr);

   __device__ color Value(double u, double v, const vec3& p) const;
 public:
   SolidTexture* solid_ptr;
   CheckTexture* check_ptr;
   ImageTexture* img_ptr;
   TextureType tex_type;
};
}

#endif // !MOON_TEXTURE_H_