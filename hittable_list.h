#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>
#include "object.h"
#include "scene.h"

class hittable_list{
public:
    __host__ __device__ hittable_list(std::vector<object*>_objects) {
        for (int i = 0; i != HITTABLE_COUNT; i++)
        {
            objects[i] = _objects[i]->ToDevice();
        }
    }

    __host__ __device__  ~hittable_list() = default;
    __host__ __device__ void clear() {
        for (auto ptr : objects)
            cudaFree(ptr);
    }

    __host__ __device__ bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const;
    __host__ __device__ hittable_list* ToDevice();
    __host__ __device__ bool bounding_box(float time0, float time1, aabb& output_box) const;

public:
    object* objects[HITTABLE_COUNT];
};

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i != HITTABLE_COUNT;++i) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

hittable_list* hittable_list::ToDevice()
{
    hittable_list* device;

    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(hittable_list));
    cudaMemcpy(device, this, sizeof(hittable_list), cudaMemcpyHostToDevice);

    return device;
}


#endif