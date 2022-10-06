#pragma once

#include "my_utility.h"

enum class Texture_type {Solid, Check};

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

	class texture {
	public:
		solid_texture* solid_ptr;
		check_texture* check_ptr;
		Texture_type tex_type;
	public:
		__device__
			texture()
		{
			solid_ptr = new solid_texture(); check_ptr = nullptr; tex_type = Texture_type::Solid;
		}
		__device__
			texture(solid_texture* _solid_ptr)
		{
			solid_ptr = _solid_ptr; check_ptr = nullptr; tex_type = Texture_type::Solid;
		}
		__device__
			texture(check_texture* _check_ptr)
		{
			check_ptr = _check_ptr; solid_ptr = nullptr; tex_type = Texture_type::Check;
		}
		__device__
			color value(double u, double v, const vec3& p) const;
	};

	color texture::value(double u, double v, const vec3& p) const {
		if (this->tex_type == Texture_type::Solid) return solid_ptr->value(u, v, p);
		else if (this->tex_type == Texture_type::Check) return check_ptr->value(u, v, p);
		else return color(0.0, 0.0, 0.0);
	}
}