#ifndef FLOAT3_H
#define FLOAT3_H

#include <vector_functions.hpp>
#include <math.h>
#include<iostream>

using point3 = float3;   // 3D point
using color = float3;    // RGB color

#define make_color make_float3
#define make_point3 make_float3

__host__ __device__ float3 operator+(const float3& v1, const float3& v2);
__host__ __device__ float3 operator+(const float3& v, const float& t);
__host__ __device__ float3 operator-(const float3& v1, const float3& v2);
__host__ __device__ float3 operator-(const float3& v, const float& t);
__host__ __device__ float3 operator*(const float3& v1, const float3& v2);
__host__ __device__ float3 operator*(float t, const float3& v);
__host__ __device__ float3 operator*(const float3& v, float t);
__host__ __device__ float3 operator/(const float3& v1, const float3& v2);
__host__ __device__ float3 operator/(float3 v, float t);


__host__ __device__ float Dot(const float3& v1, const float3& v2);
__host__ __device__ float3 Cross(const float3& v1, const float3& v2);
__host__ __device__ float3 Cross2(const float3& lhs, const float3& rhs);
__host__ __device__ float Length(float3 f);
__host__ __device__ float SquaredLength(float3 f);
__host__ __device__ bool IsZero(float3 f);
__host__ __device__ float3 UnitVector(float3 v);
__host__ __device__ float Distance(float3 a, float3 b);
__host__ __device__ void MakeUnitVector(float3* b);
__host__ __device__ float3 Reflect(float3 vin, float3 normal);
__host__ __device__ float3 Min(float3 a, float3 b);

__host__ __device__ float3 operator-(float3& a);
__host__ __device__ float3 operator-(const float3& a);

__host__ __device__ void Set(float3& f, float a, float b, float c);
__host__ __device__ void Set(float3& f, float a);

__host__ __device__ float3 Maximum(const float3& a, const float3& b);
__host__ __device__ float3 Minimum(const float3& a, const float3& b);

__host__ __device__ float Get(const float3& a, const int i);
__host__ __device__ std::ostream& operator<<(std::ostream& out, const float3& v);

__host__ __device__ float3 reflect(const float3& v, const float3& n);
__host__ __device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat);
__host__ __device__ bool nearZero(const float3& v);

#endif