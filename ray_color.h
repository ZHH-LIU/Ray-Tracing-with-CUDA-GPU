#ifndef RAY_COLOR_H
#define RAY_COLOR_H
#include "used.h"
#include "bvh.h"
#include "cudarand.h"
#include "camera.h"
#include "scene.h"

__device__ void ray_indirect(float* out, float* rayData, const ray& r, bvh_node* world, float3 background, curandState* devStates, int id, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
    {
        out[4 * id] *= 0; out[4 * id + 1] *= 0; out[4 * id + 2] *= 0; out[4 * id + 3] = 1;
        return;
    }

    if (hit(world, r, 0.001, infinity, rec)) {
        ray scatterd;
        color attenuation;

        curandState localState = devStates[id];
        float3 unitVector = UnitVector(make_float3(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState)));
        int stop = -1;
        switch (rec.mat_ptr->mat_type)
        {
        case Diffuse:
            rec.mat_ptr->scatter_diffuse(r, rec, attenuation, scatterd, unitVector);
            break;
        case Specular:
            rec.mat_ptr->scatter_specular(r, rec, attenuation, scatterd, unitVector);
            break;
        case Dielectrics:
            rec.mat_ptr->scatter_dielectric(r, rec, attenuation, scatterd);
            break;
        case Light:
            rec.mat_ptr->scatter_light(attenuation);
            stop = 1;
            break;
        }

        //out[4 * id] = attenuation.x; out[4 * id + 1] = attenuation.y; out[4 * id + 2] = attenuation.z; out[4 * id + 3] = 1;
        out[4 * id] *= attenuation.x; out[4 * id + 1] *= attenuation.y; out[4 * id + 2] *= attenuation.z; out[4 * id + 3] = stop;
        rayData[6 * id] = scatterd.orig.x; rayData[6 * id + 1] = scatterd.orig.y; rayData[6 * id + 2] = scatterd.orig.z;
        rayData[6 * id + 3] = scatterd.dir.x; rayData[6 * id + 4] = scatterd.dir.y; rayData[6 * id + 5] = scatterd.dir.z;
        return;
    }
    
    out[4 * id] *= background.x; out[4 * id + 1] *= background.y; out[4 * id + 2] *= background.z; out[4 * id + 3] = 1;
}

__device__ void ray_indirect(float* out, float* rayData, const ray& r, hittable_list* world, float3 background, curandState* devStates, int id, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
    {
        out[4 * id] *= 0; out[4 * id + 1] *= 0; out[4 * id + 2] *= 0; out[4 * id + 3] = 1;
        return;
    }

    if (world->hit(r, 0.001, infinity, rec)) {
        ray scatterd;
        color attenuation;

        curandState localState = devStates[id];
        float3 unitVector = UnitVector(make_float3(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState)));

        int stop = -1;
        switch (rec.mat_ptr->mat_type)
        {
        case Diffuse:
            rec.mat_ptr->scatter_diffuse(r, rec, attenuation, scatterd, unitVector);
            break;
        case Specular:
            rec.mat_ptr->scatter_specular(r, rec, attenuation, scatterd, unitVector);
            break;
        case Dielectrics:
            rec.mat_ptr->scatter_dielectric(r, rec, attenuation, scatterd);
            break;
        case Light:
            rec.mat_ptr->scatter_light(attenuation);
            stop = 1;
            break;
        }
        //out[4 * id] = attenuation.x; out[4 * id + 1] = attenuation.y; out[4 * id + 2] = attenuation.z; out[4 * id + 3] = 1;
        out[4 * id] *= attenuation.x; out[4 * id + 1] *= attenuation.y; out[4 * id + 2] *= attenuation.z; out[4 * id + 3] = stop;
        rayData[6 * id] = scatterd.orig.x; rayData[6 * id + 1] = scatterd.orig.y; rayData[6 * id + 2] = scatterd.orig.z;
        rayData[6 * id + 3] = scatterd.dir.x; rayData[6 * id + 4] = scatterd.dir.y; rayData[6 * id + 5] = scatterd.dir.z;
        return;
    }

    out[4 * id] *= background.x; out[4 * id + 1] *= background.y; out[4 * id + 2] *= background.z; out[4 * id + 3] = 1;
}

__global__ void ray_color_firstPass(float* out, bvh_node* world, float3 background, float* rayData, camera* cam, curandState* devStates, int image_width, int image_height, int depth) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int id = idx + idy * blockDim.x * gridDim.x;

    out[4 * id] = 1; out[4 * id + 1] = 1; out[4 * id + 2] = 1; out[4 * id + 3] = -1;

    curandState localState = devStates[id];

    float rndu = curand_uniform(&localState);
    float rndv = curand_uniform(&localState);

    auto u = (double(idx) + rndu) / (image_width - 1);
    auto v = (double(idy) + rndv) / (image_height - 1);
    ray r = cam->get_ray(u, v);

    ray_indirect(out, rayData, r, world, background, devStates, id, depth);
    //printf("1:%f,%f,%f\n", out[4 * id], out[4 * id + 1], out[4 * id + 2]);
}

__global__ void ray_color_firstPass(float* out, hittable_list* world, float3 background, float* rayData, camera* cam, curandState* devStates, int image_width, int image_height, int depth) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int id = idx + idy * blockDim.x * gridDim.x;

    out[4 * id] = 1; out[4 * id + 1] = 1; out[4 * id + 2] = 1; out[4 * id + 3] = -1;

    curandState localState = devStates[id];

    float rndu = curand_uniform(&localState);
    float rndv = curand_uniform(&localState);

    auto u = (double(idx) + rndu) / (image_width - 1);
    auto v = (double(idy) + rndv) / (image_height - 1);
    ray r = cam->get_ray(u, v);

    ray_indirect(out, rayData, r, world, background, devStates, id, depth);
    //printf("1:%f,%f,%f\n", out[4 * id], out[4 * id + 1], out[4 * id + 2]);
}

__global__ void ray_color_otherPass(float* out, bvh_node* world, float3 background, float* rayData, curandState* devStates, int image_width, int image_height, int depth) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int id = idx + idy * blockDim.x * gridDim.x;

    if (out[4 * id + 3] > 0) return;

    curandState localState = devStates[id];

    float rndu = curand_uniform(&localState);
    float rndv = curand_uniform(&localState);

    float3 origion = make_float3(rayData[6 * id], rayData[6 * id + 1], rayData[6 * id + 2]);
    float3 direction = make_float3(rayData[6 * id + 3], rayData[6 * id + 4], rayData[6 * id + 5]);
    ray r = ray(origion, direction);

    ray_indirect(out, rayData, r, world, background, devStates, id, depth);

    //printf("other: %f,%f,%f\n", rayData[6 * id], rayData[6 * id + 1], rayData[6 * id + 2]);
}

__global__ void ray_color_otherPass(float* out, hittable_list* world, float3 background, float* rayData, curandState* devStates, int image_width, int image_height, int depth) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int id = idx + idy * blockDim.x * gridDim.x;

    if (out[4 * id + 3] > 0) return;

    curandState localState = devStates[id];

    float rndu = curand_uniform(&localState);
    float rndv = curand_uniform(&localState);

    float3 origion = make_float3(rayData[6 * id], rayData[6 * id + 1], rayData[6 * id + 2]);
    float3 direction = make_float3(rayData[6 * id + 3], rayData[6 * id + 4], rayData[6 * id + 5]);
    ray r = ray(origion, direction);

    ray_indirect(out, rayData, r, world, background, devStates, id, depth);

    //printf("other:%f,%f,%f\n", out[4 * id], out[4 * id + 1], out[4 * id + 2]);
}

__global__ void multisum(float* temp, float* out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int id = idx + idy * blockDim.x * gridDim.x;

    out[id * 4] += temp[id * 4];
    out[id * 4 + 1] += temp[id * 4 + 1];
    out[id * 4 + 2] += temp[id * 4 + 2];
    out[id * 4 + 3] = -1;
    //printf("out:%f,%f,%f\n", temp[4 * id], temp[4 * id + 1], temp[4 * id + 2]);
}

__host__ void ray_tracing(float* out, const scene ourscene, float* rayData, camera* cam, curandState* devStates)
{
    bvh_node ourbvh(ourscene.objects, 0, ourscene.objects.size(), 0.001, infinity);
    bvh_node* bvhDevice = ourbvh.ToDevice();

    hittable_list world(ourscene.objects);
    auto worlddevice = world.ToDevice();

    switch (BVH_USED)
    {
    case 0:
        for (int depth = numDepth; depth >= 0; depth--)
        {
            if (depth == numDepth)
            {
                 ray_color_firstPass << <grid, block >> > (out, worlddevice, ourscene.background, rayData, cam, devStates, image_width, image_height, depth);
                //cudaDeviceSynchronize();
            }
            else
            {
                ray_color_otherPass << <grid, block >> > (out, worlddevice, ourscene.background, rayData, devStates, image_width, image_height, depth);
                //cudaDeviceSynchronize();
            }
        }
        break;
    case 1:
        for (int depth = numDepth; depth >= 0; depth--)
        {
            if (depth == numDepth)
            {
                ray_color_firstPass << <grid, block >> > (out, bvhDevice, ourscene.background, rayData, cam, devStates, image_width, image_height, depth);
                //cudaDeviceSynchronize();
            }
            else
            {
                ray_color_otherPass << <grid, block >> > (out, bvhDevice, ourscene.background, rayData, devStates, image_width, image_height, depth);
                //cudaDeviceSynchronize();
            }
        }
        break;

    }

    world.clear();
    cudaFree(worlddevice);
    //cudaFree(bvhDevice);
    return;
}

__host__ void ray_multisamples(float* out, const scene& ourscene, float* rayData, camera* cam, curandState* devStates)
{
    int nElem = image_width * image_height * 4;
    int nByte = sizeof(float) * nElem;
    float* temp;
    CHECK(cudaMalloc((float**)&temp, nByte));

    std::cerr << "First Pass: GPU Computation is Started." << std::endl;
    for (int i = 0; i != numSamples; i++)
    {
        setup_kernel << <grid, block >> > (devStates, i);
        //cudaDeviceSynchronize();
        ray_tracing(temp, ourscene, rayData, cam, devStates);
        multisum << <grid, block >> > (temp, out);
        //cudaDeviceSynchronize();
    }
    std::cerr << "GPU Computation is Done." << std::endl;
    cudaFree(temp);
    return;
}

__host__ void drawPPM(float* outhost)
{
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int index = i + j * image_width;
            index *= 4;
            write_color(std::cout, make_color(outhost[index], outhost[index + 1], outhost[index + 2]));

        }
    }

    std::cerr << "Draw PPM is Done.\n";
}

#endif
