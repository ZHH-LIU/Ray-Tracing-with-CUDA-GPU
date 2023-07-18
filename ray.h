#ifndef RAY_H
#define RAY_H

#include "float3.h"

class ray {
public:
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const float3& direction)
        : orig(origin), dir(direction)
    {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ float3 direction() const { return dir; }

    __host__ __device__ point3 at(double t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    float3 dir;
};

#endif
