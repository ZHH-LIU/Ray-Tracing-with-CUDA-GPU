#ifndef CAMERA_H
#define CAMERA_H

#include "used.h"

class camera {
public:
    __host__ __device__ camera(
        point3 lookfrom,
        point3 lookat,
        float3   vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        auto w = UnitVector(lookfrom - lookat);
        auto u = UnitVector(Cross(vup, w));
        auto v = Cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
    }

    __host__ __device__ ray get_ray(double u, double v) const {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
    __host__ __device__ camera* ToDevice();
private:
    point3 origin;
    point3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
};

camera* camera::ToDevice()
{
    camera* device;

    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(camera));
    cudaMemcpy(device, this, sizeof(camera), cudaMemcpyHostToDevice);

    return device;
}
#endif
