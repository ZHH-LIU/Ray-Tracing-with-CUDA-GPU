#ifndef USED_H
#define USED_H

#include <cmath>
#include <limits>
#include <memory>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector_functions.hpp>

#include "color.h"
#include "float3.h"
#include "color.h"
#include "ray.h"
#include "define.h"
// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

__host__ __device__ double degrees_to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}
__host__ __device__ float clamp(float val, float minv, float maxv)
{
    return val > maxv ? maxv : val < minv ? minv : val;
}

__host__ __device__ float FloatMin(float v1, float v2)
{
    return v1 > v2 ? v2 : v1;
}

__host__ __device__ float FloatMax(float v1, float v2)
{
    return v1 > v2 ? v1 : v2;
}

// Common Headers

#include "ray.h"
#include "float3.h"

#endif
