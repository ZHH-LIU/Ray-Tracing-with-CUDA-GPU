#ifndef OBJECT_H
#define OBJECT_H

#include "hittable.h"
#include "float3.h"

enum objType
{
    Sphere,AARect
};

enum faceType
{
    faceX, faceY, faceZ
};

class object{
public:
    __host__ __device__ object() {}
    __host__ __device__ object(objType _type, point3 cen, float r, Material* _mat_ptr) : obj_type(_type), center(cen), radius(r), mat_ptr(_mat_ptr) {};
    __host__ __device__ object(objType _type, float _x0, float _x1, float _y0, float _y1, float _offset, faceType _k, Material* _mat_ptr)
        : obj_type(_type), x0(_x0), x1(_x1), y0(_y0), y1(_y1), offset(_offset), face(_k), mat_ptr(_mat_ptr){};
    __host__ __device__ bool hit_sphere(const ray& r, double t_min, double t_max, hit_record& rec) const;
    __host__ __device__ object* ToDevice();
    __host__ __device__ aabb* bounding_box_sphere() const;
    __host__ __device__ aabb* bounding_box_rect() const;
    __host__ __device__ bool hit_rect(const ray& r, double t_min, double t_max, hit_record& rec) const;
    __host__ __device__ bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
    __host__ __device__ aabb* bounding_box() const;
public:
    //sphere
    point3 center;
    float radius;
    //aarect
    float x0, x1, y0, y1, offset;
    faceType face;
    //normal
    objType obj_type;
    Material* mat_ptr;
};

object* object::ToDevice()
{
    object* device;
    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(object));
    cudaMemcpy(device, this, sizeof(object), cudaMemcpyHostToDevice);

    return device;
}

aabb* object::bounding_box() const
{
    switch (obj_type)
    {
    case Sphere:
        return bounding_box_sphere();
        break;
    case AARect:
        return bounding_box_rect();
        break;
    }
}

bool object::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    switch (obj_type)
    {
    case Sphere:
        return hit_sphere(r, t_min, t_max, rec);
        break;
    case AARect:
        return hit_rect(r, t_min, t_max, rec);
        break;
    }
}

bool object::hit_sphere(const ray& r, double t_min, double t_max, hit_record& rec) const {
    float3 oc = r.origin() - center;
    auto a = SquaredLength(r.direction());
    auto half_b = Dot(oc, r.direction());
    auto c = SquaredLength(oc)- radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    float3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

aabb* object::bounding_box_sphere() const {
    aabb* out = new aabb;
    out->maximum = center + make_float3(radius, radius, radius);
    out->minimum = center - make_float3(radius, radius, radius);
    return out;
}

aabb* object::bounding_box_rect() const {
    // The bounding box must have non-zero width in each dimension, so pad the Z
    // dimension a small amount.
    aabb* out = new aabb;
    switch (face)
    {
    case faceX:
        out->maximum = point3{offset - 0.0001f, x0, y0 };
        out->minimum = point3{offset + 0.0001f, x1, y1 };
        break;
    case faceY:
        out->maximum = point3{ y0,offset - 0.0001f, x0 };
        out->minimum = point3{ y1,offset + 0.0001f, x1 };
        break;
    case faceZ:
        out->maximum = point3{ x0 , y0,offset - 0.0001f };
        out->minimum = point3{ x1 , y1,offset + 0.0001f };
        break;
    }
    return out;
}

bool object::hit_rect(const ray& r, double t_min, double t_max, hit_record& rec) const {
    int xid, yid, zid;
    float3 outward_normal;
    switch (this->face)
    {
    case faceX:
        xid = 1; yid = 2; zid = 0; outward_normal = float3{ 1, 0, 0 }; break;
    case faceY:
        xid = 2; yid = 0; zid = 1; outward_normal = float3{ 0, 1, 0 }; break;
    case faceZ:
        xid = 0; yid = 1; zid = 2; outward_normal = float3{ 0, 0, 1 }; break;
    }
    auto t = (offset - Get(r.origin(),zid)) / Get(r.direction(),zid);
    if (t < t_min || t > t_max)
        return false;
    auto x = Get(r.origin(),xid) + t * Get(r.direction(),xid);
    auto y = Get(r.origin(),yid) + t * Get(r.direction(),yid);
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;

    rec.t = t;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);

    //printf("%d\n", face);
    return true;
}


#endif