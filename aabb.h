#ifndef AABB_H
#define AABB_H

#include "used.h"

class aabb {
public:
    __host__ __device__ aabb() {}
    __host__ __device__ aabb(const point3& a, const point3& b) { minimum = a; maximum = b; }

    __host__ __device__ point3 min() const { return minimum; }
    __host__ __device__ point3 max() const { return maximum; }

    __host__ __device__ bool hit(const ray& r, double t_min, double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto t0 = FloatMin((Get(minimum,a) - Get(r.origin(),a)) / Get(r.direction(),a),
                (Get(maximum, a) - Get(r.origin(), a)) / Get(r.direction(), a));
            auto t1 = FloatMax((Get(minimum, a) - Get(r.origin(), a)) / Get(r.direction(), a),
                (Get(maximum, a) - Get(r.origin(), a)) / Get(r.direction(), a));
            t_min = FloatMax(t0, t_min);
            t_max = FloatMin(t1, t_max);
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
    __host__ __device__ aabb* ToDevice();
    point3 minimum;
    point3 maximum;
};

__host__ __device__ aabb* aabb::ToDevice()
{
    aabb* device;
    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(aabb));
    cudaMemcpy(device, this, sizeof(aabb), cudaMemcpyHostToDevice);

    return device;
}

__host__ __device__ aabb* surrounding_box(const aabb* box0, const aabb* box1) {
    point3 small{ FloatMin(box0->min().x, box1->min().x),
        FloatMin(box0->min().y, box1->min().y),
        FloatMin(box0->min().z, box1->min().z) };

    point3 big{ FloatMax(box0->max().x, box1->max().x),
        FloatMax(box0->max().y, box1->max().y),
        FloatMax(box0->max().z, box1->max().z) };
    aabb* temp = (aabb*) new aabb;
    temp->minimum = small;
    temp->maximum = big;
    return temp;
}

#endif
