#include <include/texture.h>

namespace Moon{
color SolidTexture::Value(double u, double v, const vec3& p) const {
  return  this->color_value;
}

color CheckTexture::Value(double u, double v, const point3& p) {
	auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
	if (sines < 0)	return this->odd.Value(u, v, p);
	else return this->even.Value(u, v, p);
}

ImageTexture::ImageTexture(unsigned char* file, int width, int height) {
	this->_data = file;
	this->_height = height;
	this->_width = width;
	this->_bytes_per_scanline = bytes_per_pixel * _width;
}

color ImageTexture::Value(double u, double v, const vec3& p) const {
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

Texture::Texture() {
	this->solid_ptr = new SolidTexture(); 
	this->check_ptr = nullptr; 
	this->img_ptr = nullptr; 
	this->tex_type = TextureType::SOLID;
}

Texture::Texture(SolidTexture* _solid_ptr) {
	this->solid_ptr = _solid_ptr; 
	this->check_ptr = nullptr; 
	this->img_ptr = nullptr; 
	this->tex_type = TextureType::SOLID;
}

Texture::Texture(CheckTexture* _check_ptr) {
	this->check_ptr = _check_ptr; 
	this->img_ptr = nullptr; 
	this->solid_ptr = nullptr; 
	this->tex_type = TextureType::CHECK;
}

Texture::Texture(ImageTexture* _img_ptr) {
	this->img_ptr = _img_ptr; 
	this->check_ptr = nullptr; 
	this->solid_ptr = nullptr; 
	this->tex_type = TextureType::IMAGE;
}

color Texture::Value(double u, double v, const vec3& p) const {
	if (this->tex_type == TextureType::SOLID) return this->solid_ptr->Value(u, v, p);
	else if (this->tex_type == TextureType::CHECK) return this->check_ptr->Value(u, v, p);
	else if (this->tex_type == TextureType::IMAGE) return this->img_ptr->Value(u, v, p);
	else return color(0.0, 0.0, 0.0);
}

}