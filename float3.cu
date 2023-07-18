#include "float3.h"
__host__ __device__  float3 operator+(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__host__ __device__  float3 operator+(const float3& v, const float& t)
{
	return make_float3(v.x + t, v.y + t, v.z + t);
}

__host__ __device__  float3 operator-(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__  float3 operator-(const float3& v, const float& t)
{
	return make_float3(v.x - t, v.y - t, v.z - t);
}

__host__ __device__  float3 operator*(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

__host__ __device__  float3 operator*(float t, const float3& v)
{
	return make_float3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__  float3 operator*(const float3& v, float t)
{
	return make_float3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__  float3 operator/(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

__host__ __device__  float3 operator/(float3 v, float t)
{
	return make_float3(v.x / t, v.y / t, v.z / t);
}

__host__ __device__  float Dot(const float3& v1, const float3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__  float3 Cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, (-(v1.x * v2.z - v1.z * v2.x)),
		(v1.x * v2.y - v1.y * v2.x));
}

__host__ __device__  float3 Cross2(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x);
}

__host__ __device__  float Length(float3 f)
{
	return sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
}

__host__ __device__  float SquaredLength(float3 f)
{
	return f.x * f.x + f.y * f.y + f.z * f.z;
}

__host__ __device__  bool IsZero(float3 f)
{
	return f.x == 0 && f.y == 0 && f.z == 0;
}

__host__ __device__  float3 UnitVector(float3 v)
{
	return v / Length(v);
}


__host__ __device__  float Distance(float3 a, float3 b)
{
	return Length(a - b);
}

__host__ __device__  void MakeUnitVector(float3* f)
{
	const float k = 1.0 / sqrt(f->x * f->x + f->y * f->y + f->z * f->z);
	f->x *= k; f->y *= k; f->z *= k;
}

__host__ __device__  float3 Reflect(float3 vin, float3 normal)
{
	return vin - 2 * Dot(vin, normal) * normal;
}

__host__ __device__  float3 Min(float3 a, float3 b)
{
	if (Length(a) < Length(b))return a;
	else return b;
}

__host__ __device__  float3 operator-(float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__  float3 operator-(const float3& a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__  void Set(float3& f, float a, float b, float c)
{
	f.x = a;
	f.y = b;
	f.z = c;
}

__host__ __device__  void Set(float3& f, float a)
{
	f.x = a;
	f.y = a;
	f.z = a;
}

__host__ __device__  float3 Maximum(const float3& a, const float3& b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__  float3 Minimum(const float3& a, const float3& b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__  float Get(const float3& a, const int i) {
	return i == 0 ? a.x : (i == 1 ? a.y : a.z);
}

__host__ __device__  std::ostream& operator<<(std::ostream& out, const float3& v) {
	return out << v.x << ' ' << v.y << ' ' << v.z;
}

__host__ __device__ float3 reflect(const float3& v, const float3& n) {
	return v - 2 * Dot(v, n) * n;
}

__host__ __device__ bool nearZero(const float3 & v)
{
	return Length(v) < 0.01;
}

__host__ __device__ float3 refract(const float3& uv, const float3& n, float etai_over_etat) {
	auto cos_theta = Dot(-1 * uv, n) > 1.0 ? 1.0 : Dot(-1 * uv, n);
	float3 r_out_perp =  etai_over_etat* (uv + cos_theta * n);
	float3 r_out_parallel =  -sqrt(fabs(1.0 - SquaredLength(r_out_perp))) * n;
	return r_out_perp + r_out_parallel;
}
