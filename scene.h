#ifndef SCENE_H
#define SCENE_H

#include"hittable.h"
#include"object.h"
#include<vector>
#include "used.h"
#define HITTABLE_COUNT 8

class scene
{
public:
	scene()
	{
		auto red = Material(Diffuse, color{ .65, .05, .05 });
		auto white = Material(Diffuse, color{ .73, .73, .73 });
		auto green = Material(Diffuse, color{ .12, .45, .15 });
		auto light = Material(Light, color{ 7, 7, 7 });
		auto spec = Material(Specular, color{ 1.0, 1.0, 1.0 }, 0.2, 1.0);
		auto dilec = Material(Dielectrics, color{ 1.0, 1.0, 1.0 }, 0.2, 1.5);

		objects.push_back((object*) new object(AARect, 0, 555, 0, 555, 555, faceX, green.ToDevice()));
		objects.push_back((object*) new object(AARect, 0, 555, 0, 555, 0, faceX, red.ToDevice()));
		objects.push_back((object*) new object(AARect, 113, 443, 127, 432, 554, faceY, light.ToDevice()));
		objects.push_back((object*) new object(AARect, 0, 555, 0, 555, 0, faceY, white.ToDevice()));
		objects.push_back((object*) new object(AARect, 0, 555, 0, 555, 555, faceY, white.ToDevice()));
		objects.push_back((object*) new object(AARect, 0, 555, 0, 555, 555, faceZ, white.ToDevice()));
		objects.push_back((object*) new object(Sphere, point3{ 120, 100, 120 }, 100, dilec.ToDevice()));
		objects.push_back((object*) new object(Sphere, point3{ 400, 120, 400 }, 120, spec.ToDevice()));
	}
	std::vector<object*> objects;
	color background{ 0.1,0.1,0.1 };
};

#endif
