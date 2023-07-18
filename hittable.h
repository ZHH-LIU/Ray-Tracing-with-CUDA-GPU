#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "aabb.h"
class Material;
enum material
{
    Diffuse,Specular, Dielectrics, Light
};

struct hit_record {
    point3 p;
    float3 normal;
    double t;
    bool front_face;
    Material* mat_ptr;

    __host__ __device__ inline void set_face_normal(const ray& r, const float3& outward_normal) {
        front_face = Dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
public:
    //__host__ __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
    __host__ __device__ hittable() = default;
    __host__ __device__ ~hittable() = default;
};

class Material
{
public:
    __host__ __device__ Material(material _mat_type, const color& a) : mat_type(_mat_type), albedo(a) {}
    __host__ __device__ Material(material _mat_type, const color& a, double _fuzz, double _ir) : mat_type(_mat_type), albedo(a), fuzz(_fuzz), ir(_ir) {}
    __host__ __device__ bool scatter_specular(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, const float3& unitVector
    ) const {
        float3 randSphere = (unitVector - make_float3(0.5, 0.5, 0.5)) * 2.0;
        randSphere = UnitVector(randSphere);
        float3 reflected = reflect(UnitVector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * randSphere);
        attenuation = albedo;
        return (Dot(scattered.direction(), rec.normal) > 0);
    }

    __host__ __device__ bool scatter_diffuse(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, const float3& unitVector
    ) const {
        float3 randSphere = (unitVector - make_float3(0.5, 0.5, 0.5)) * 2.0;
        randSphere = UnitVector(randSphere);
        auto scatter_direction = rec.normal + randSphere;

        // Catch degenerate scatter direction
        if (nearZero(scatter_direction))
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    __host__ __device__ bool scatter_dielectric(
        const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
    ) const {
        attenuation = albedo;
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        float3 unit_direction = UnitVector(r_in.direction());
        float3 refracted = refract(unit_direction, rec.normal, refraction_ratio);

        float cos_theta = FloatMin(Dot(-unit_direction, rec.normal), 1.0);
        float sin_theta_square = 1.0 - cos_theta * cos_theta;

        bool cannot_refract = refraction_ratio * refraction_ratio * sin_theta_square > 1.0;
        float3 direction;

        if (cannot_refract)
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

    __host__ __device__ bool scatter_light(color& lightcolor
    ) const {
        lightcolor = albedo;
        return true;
    }

    __host__ __device__ Material* ToDevice();

public:
    color albedo;
    double fuzz;
    double ir;
    material mat_type;
};

Material* Material::ToDevice()
{
    Material* device;

    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(Material));
    cudaMemcpy(device, this, sizeof(Material), cudaMemcpyHostToDevice);

    return device;
}


#endif