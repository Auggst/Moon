#pragma once

#include "my_utility.h"

enum class Texture_type {Solid, Check, Image};

namespace Moon {
	class solid_texture {
	public:
		color color_value;
	public:
		__device__
			solid_texture() { color_value = color(1.0, 1.0, 1.0); }
		__device__
			solid_texture(color _col) { color_value = _col; }
		__device__
			solid_texture(double _r, double _g, double _b) : color_value(color(_r, _g, _b)) {}
		__device__
			color value(double u, double v, const vec3& p) const;
	};

	color solid_texture::value(double u, double v, const vec3& p) const {
		return color_value;
	}

	class check_texture {
	public:
		solid_texture odd;
		solid_texture even;
	public:
		__device__
			check_texture() { odd = solid_texture(color(0.0, 0.0, 0.0)); even = solid_texture(); }
		__device__
			check_texture(color col0, color col1) { odd = col0; even = col1; }

		__device__
			color value(double u, double v, const point3& p);
	};
	color check_texture::value(double u, double v, const point3& p) {
		auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)	return odd.value(u, v, p);
		else return even.value(u, v, p);
	}

	class image_texture {
	public:
		unsigned char* _data;
		int _width, _height;
		int _bytes_per_scanline;
		const static int bytes_per_pixel = 3;
	public:
		__device__
		image_texture() : _data(nullptr), _width(0), _height(0), _bytes_per_scanline(0) {}
		__device__
		image_texture(unsigned char* file, int width, int height) {
			this->_data = file;
			this->_height = height;
			this->_width = width;
			this->_bytes_per_scanline = bytes_per_pixel * _width;
		}

		__device__
			color value(double u, double v, const vec3& p) const;
	};

	color image_texture::value(double u, double v, const vec3& p) const {
		//如果没有纹理数据，就返回固定颜色
		if (_data == nullptr) {
			return color(0, 1.0, 0.0);
		}
		//输入坐标变换到[0, 1] × [1, 0]
		u = clamp(u, 0.0, 1.0);
		v = 1.0 - clamp(v, 0.0, 1.0);   //对于图片坐标要反转v

		int i = (int)(u * _width);
		int j = (int)(v * _height);

		//剪切int映射，因为确切坐标应当小于1.0
		if (i >= _width) i = _width - 1;
		if (j >= _height) j = _height - 1;

		const double color_scale = 1.0 / 255.0;
		auto pixel = _data + j * _bytes_per_scanline + i * bytes_per_pixel;

		return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
	}

	class texture {
	public:
		solid_texture* solid_ptr;
		check_texture* check_ptr;
		image_texture* img_ptr;
		Texture_type tex_type;
	public:
		__device__
			texture()
		{
			solid_ptr = new solid_texture(); check_ptr = nullptr; img_ptr = nullptr; tex_type = Texture_type::Solid;
		}
		__device__
			texture(solid_texture* _solid_ptr)
		{
			solid_ptr = _solid_ptr; check_ptr = nullptr; img_ptr = nullptr; tex_type = Texture_type::Solid;
		}
		__device__
			texture(check_texture* _check_ptr)
		{
			check_ptr = _check_ptr; img_ptr = nullptr; solid_ptr = nullptr; tex_type = Texture_type::Check;
		}

		__device__
			texture(image_texture* _img_ptr)
		{
			img_ptr = _img_ptr; check_ptr = nullptr; solid_ptr = nullptr; tex_type = Texture_type::Image;
		}

		__device__
			color value(double u, double v, const vec3& p) const;
	};

	color texture::value(double u, double v, const vec3& p) const {
		if (this->tex_type == Texture_type::Solid) return solid_ptr->value(u, v, p);
		else if (this->tex_type == Texture_type::Check) return check_ptr->value(u, v, p);
		else if (this->tex_type == Texture_type::Image) return img_ptr->value(u, v, p);
		else return color(0.0, 0.0, 0.0);
	}
}