#ifndef DEFINE_H
#define DEFINE_H

#define numSamples 200
#define M_PI  3.1415926535897932385
#define infinity  FLT_MAX
#define numDepth 50
#define MAX_ONJECT_NUM 50
#define BVH_USED 0
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#define LOOK_FROM float3{278, 278, -800}
#define LOOK_AT float3{278, 278, 0}
#define UP float3{0,1,0}
#define FOV 45.0
#define ASPECT_RATIO 1.0

const int image_width = 512;
const int image_height = static_cast<int>(image_width / ASPECT_RATIO);
const dim3 block(32, 32);
const dim3 grid(image_width / block.x, image_height / block.y);

#endif
