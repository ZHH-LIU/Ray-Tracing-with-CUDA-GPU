#ifndef BVH_H
#define BVH_H

#include "used.h"

#include "hittable.h"
#include "hittable_list.h"
#include <algorithm>

class bvh_node {
public:
    __host__ __device__ bvh_node() :
        left(nullptr), right(nullptr), box(nullptr), obj(nullptr) {};

    __host__ __device__ bvh_node(
        std::vector<object*>src_objects,
        size_t start, size_t end, double time0, double time1);

    __host__ __device__ bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const;

    __host__  bvh_node* ToDevice();
public:
    bvh_node* left;
    bvh_node* right;
    aabb* box;
    object* obj;
};

bvh_node* bvh_node::ToDevice()
{
    box = box!=nullptr?box->ToDevice():nullptr;
    obj = obj!=nullptr?obj->ToDevice():nullptr;

    bvh_node* device;
    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(bvh_node));
    cudaMemcpy(device, this, sizeof(bvh_node), cudaMemcpyHostToDevice);

    return device;
}

bool bvh_node::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
        return box->hit(r, t_min, t_max);
}

inline bool box_compare(const object* a, const object* b, int axis) {
    return Get(a->bounding_box()->min(), axis)< Get(b->bounding_box()->min(), axis);
}

bool box_x_compare(const object* a, const object* b) {
    return box_compare(a, b, 0);
}

bool box_y_compare(const object* a, const object* b) {
    return box_compare(a, b, 1);
}

bool box_z_compare(const object* a, const object* b) {
    return box_compare(a, b, 2);
}

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max + 1));
}

bvh_node::bvh_node(
    std::vector<object*>src_objects,
    size_t start, size_t end, double time0, double time1
) {
    auto objects = src_objects; // Create a modifiable array of the source scene objects
    size_t object_span = end - start;
    if (object_span == 1) {
        left = right = nullptr;
        obj = objects[start];
        box = obj->bounding_box();
    }
    else {
        int axis = random_int(0, 2);
        auto comparator = (axis == 0) ? box_x_compare
            : (axis == 1) ? box_y_compare
            : box_z_compare;

        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        auto mid = start + object_span / 2;
        auto leftnode = mid>start?(bvh_node*)new bvh_node(objects, start, mid, time0, time1):nullptr;
        auto rightnode = end>mid? (bvh_node*)new bvh_node(objects, mid, end, time0, time1):nullptr;
        obj = nullptr;
        box = surrounding_box(leftnode->box, rightnode->box);
        left = leftnode->ToDevice();
        right = rightnode->ToDevice();
    }
}

__host__ __device__ bool hit(bvh_node* root, const ray& r, double t_min, double t_max, hit_record& rec)
{
    if (!root || !root->box->hit(r, t_min, t_max)) return false;

    bvh_node* bvh[MAX_ONJECT_NUM] = {nullptr};
    int idx = 0, id=0;
    bvh[idx++] = root;

    hit_record temp_rec;
    auto closest_so_far = t_max;
    bool isHit=false;

    while (bvh[id])
    {
        if (bvh[id]->obj && bvh[id]->obj->hit(r, t_min, closest_so_far, temp_rec))
        {
            isHit = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
        else
        {
            if (bvh[id]->box->hit(r, t_min, isHit ? rec.t : t_max))
            {
                if (bvh[id]->left) bvh[idx++] = bvh[id]->left;
                if (bvh[id]->right) bvh[idx++] = bvh[id]->right;
            }
            
        }
        ++id;
    }
    return isHit;
}

#endif
