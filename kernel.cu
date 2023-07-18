
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector_functions.hpp>

#include "color.h"
#include "float3.h"
#include "color.h"
#include "ray.h"
#include"hittable_list.h"
#include "used.h"
#include "camera.h"
#include "object.h"
#include "cudarand.h"
#include"scene.h"
#include "bvh.h"
#include "ray_color.h"

int main() {

    int dev = 0;
    cudaSetDevice(dev);

    // World
    scene ourscene;

    // Camera
    auto cam = camera(LOOK_FROM, LOOK_AT, UP, FOV, ASPECT_RATIO).ToDevice();

    //Rand
    auto ourRand = cudaRand(image_width * image_height,0);

    // Render
    int nElem = image_width * image_height * 4;//RGBA
    int nByte = sizeof(float) * nElem;

    float* outHost = (float*)malloc(nByte);
    memset(outHost, 0, nByte);
    float* out;
    CHECK(cudaMalloc((float**)&out, nByte));

    int nElemRay = image_width * image_height * 6;
    int nByteRay = sizeof(float) * nElemRay;

    float* rayHost = (float*)malloc(nByteRay);
    memset(rayHost, 0, nByteRay);
    float* rayDevice;
    CHECK(cudaMalloc((float**)&rayDevice, nByteRay));

    ray_multisamples(out, ourscene, rayDevice, cam, ourRand.devStates);

    CHECK(cudaMemcpy(outHost, out, nByte, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(rayHost, rayDevice, nByte, cudaMemcpyDeviceToHost));

    drawPPM(outHost);

    cudaFree(out);
    cudaFree(rayDevice);

    cudaFree(cam);
    cudaFree(ourRand.devStates);
    free(outHost);

    cudaDeviceReset();

    return 0;
}



