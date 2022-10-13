#include <include/myobject.h>

//求球面的uv坐标
void Sphere::get_uv(const point3& p, double& u, double& v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + PI;

    u = phi / (2 * PI);
    v = theta / PI;
}

//判断是否击中
bool Sphere::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    //判断光线与圆相交的参数
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    //相交求交点的t值
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;
    return true;
}

MovingSphere::MovingSphere() { 
    center0 = point3(0.0, 0.0, 0.0); 
    center1 = point3(0.0, 1.0, 0.0); 
    radius = 1.0; 
    time0 = 0.0; 
    time1 = 0.0; 
    mat_ptr = nullptr; 
}

bool MovingSphere::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    //光线与圆求交的参数
    vec3 oc = r.origin() - Center(r.time());
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    //计算光线与圆交点
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    //更新hit_record
    rec.t = root;
    rec.p = r.at(rec.t);
    auto outward_normal = (rec.p - Center(r.time())) / radius; //法线就是圆心到交点的向量
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool XY_Rect::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max) return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)   return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    auto outward_normal = vec3(0.0, 0.0, 1.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

bool XZ_Rect::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max) return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)   return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = vec3(0.0, 1.0, 0.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

bool YZ_Rect::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max) return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)   return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = vec3(1.0, 0.0, 0.0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
}

Box::Box(const point3& p0, const point3& p1, Material* ptr) {
    box_min = p0;
    box_max = p1;

    xy[0] = XY_Rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    xy[1] = XY_Rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

    xz[0] = XZ_Rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    xz[1] = XZ_Rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

    yz[0] = YZ_Rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    yz[1] = YZ_Rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);
}

void Box::Update() {
    xy[0].x0 = box_min.x();
    xy[0].x1 = box_max.x();
    xy[0].y0 = box_min.y();
    xy[0].y1 = box_max.y();
    xy[0].k = box_max.z();

    xy[1].x0 = box_min.x();
    xy[1].x1 = box_max.x();
    xy[1].y0 = box_min.y();
    xy[1].y1 = box_max.y();
    xy[1].k = box_min.z();

    xz[0].x0 = box_min.x();
    xz[0].x1 = box_max.x();
    xz[0].z0 = box_min.z();
    xz[0].z1 = box_max.z();
    xz[0].k = box_max.y();

    xz[1].x0 = box_min.x();
    xz[1].x1 = box_max.x();
    xz[1].z0 = box_min.z();
    xz[1].z1 = box_max.z();
    xz[1].k = box_min.y();

    yz[0].y0 = box_min.y();
    yz[0].y1 = box_max.y();
    yz[0].z0 = box_min.z();
    yz[0].z1 = box_max.z();
    yz[0].k = box_max.x();

    yz[1].y0 = box_min.y();
    yz[1].y1 = box_max.y();
    yz[1].z0 = box_min.z();
    yz[1].z1 = box_max.z();
    yz[1].k = box_min.x();
}

bool Box::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;
    for (int i = 0; i < 2; i++) {
        if (xy[i].Hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    for (int i = 0; i < 2; i++) {
        if (xz[i].Hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    for (int i = 0; i < 2; i++) {
        if (yz[i].Hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

void HitObject::Translate(const vec3& displacement) {
    if (this->hit_type == ObjectType::SPHERE) {
        hit_sphere->center = hit_sphere->center + displacement;
    }
    else if (this->hit_type == ObjectType::MOVING_SPHERE) {
        hit_moving_sphere->center0 = hit_moving_sphere->center0 + displacement;
        hit_moving_sphere->center1 = hit_moving_sphere->center1 + displacement;
    }
    else if (this->hit_type == ObjectType::XY_RECT) {
        this->hit_xy_rect->x0 = this->hit_xy_rect->x0 + displacement.x();
        this->hit_xy_rect->x1 = this->hit_xy_rect->x1 + displacement.x();
        this->hit_xy_rect->y0 = this->hit_xy_rect->y0 + displacement.y();
        this->hit_xy_rect->y1 = this->hit_xy_rect->y1 + displacement.y();
        this->hit_xy_rect->k = this->hit_xy_rect->k + displacement.z();
    }
    else if (this->hit_type == ObjectType::XZ_RECT) {
        this->hit_xz_rect->x0 = this->hit_xz_rect->x0 + displacement.x();
        this->hit_xz_rect->x1 = this->hit_xz_rect->x1 + displacement.x();
        this->hit_xz_rect->z0 = this->hit_xz_rect->z0 + displacement.z();
        this->hit_xz_rect->z1 = this->hit_xz_rect->z1 + displacement.z();
        this->hit_xz_rect->k = this->hit_xz_rect->k + displacement.y();
    }
    else if (this->hit_type == ObjectType::YZ_RECT) {
        this->hit_yz_rect->y0 = this->hit_yz_rect->y0 + displacement.y();
        this->hit_yz_rect->y1 = this->hit_yz_rect->y1 + displacement.y();
        this->hit_yz_rect->z0 = this->hit_yz_rect->z0 + displacement.z();
        this->hit_yz_rect->z1 = this->hit_yz_rect->z1 + displacement.z();
        this->hit_yz_rect->k = this->hit_xy_rect->k + displacement.x();
    }
    else if (this->hit_type == ObjectType::BOX) {
        this->hit_box->box_min = this->hit_box->box_min + displacement;
        this->hit_box->box_max = this->hit_box->box_max + displacement;
        this->hit_box->Update();
    }
    else {
        return;
    }
}

bool HitObject::Hit(const ray& r, double t_min, double t_max, hit_record& rec) {
    if (this->hit_type == ObjectType::SPHERE) {
        return hit_sphere->Hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == ObjectType::MOVING_SPHERE) {
        return hit_moving_sphere->Hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == ObjectType::XY_RECT) {
        return hit_xy_rect->Hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == ObjectType::XZ_RECT) {
        return hit_xz_rect->Hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == ObjectType::YZ_RECT) {
        return hit_yz_rect->Hit(r, t_min, t_max, rec);
    }
    else if (this->hit_type == ObjectType::BOX) {
        return hit_box->Hit(r, t_min, t_max, rec);
    }
    else {
        return false;
    }
}

