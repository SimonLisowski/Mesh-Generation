/*
    EdgeNet data preprocessing adaped to 360 degrees images
    Adapted to use with Python and numpy
    Author: Aloísio Dourado (jun, 2018)
    Original Caffe Code: Shuran Song (https://github.com/shurans/sscnet)
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef high_resolution_clock::time_point clock_tick;

// Voxel information
float vox_unit = 0.02;
float vox_margin = 0.24;
int vox_size_x = 240;
int vox_size_y = 144;
int vox_size_z = 240;

// Camera information
float f = 518.85;
float sensor_w = 640;
float sensor_h = 480;

// GPU parameters
int NUM_THREADS=1024;
int DEVICE = 0;
int debug = 0;

// GPU Variables
float *parameters_GPU;
#define VOX_UNIT (0)
#define VOX_MARGIN (1)
#define VOX_SIZE_X (2)
#define VOX_SIZE_Y (3)
#define VOX_SIZE_Z (4)
#define CAM_F (5)
#define SENSOR_W (6)
#define SENSOR_H (7)

#define NUM_CLASSES (256)
#define MAX_DOWN_SIZE (1000)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

clock_tick start_timer(){
    return (high_resolution_clock::now());
}

void end_timer(clock_tick t1, const char msg[]) {
  if (debug==1){
      clock_tick t2 = high_resolution_clock::now();
      auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
      printf("%s: %ld(ms)\n", msg, duration);
  }
}

//float cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};




void setup_CPP(int device, int num_threads, float v_unit, float v_margin,
               float focal_length, float s_w, float s_h,
               int vox_x, int vox_y, int vox_z,
               int debug_flag){
    DEVICE = device;
    NUM_THREADS = num_threads;
    vox_unit = v_unit;
    vox_margin = v_margin;
    f = focal_length;
    sensor_w = s_w;
    sensor_h = s_h;
    vox_size_x = vox_x;
    vox_size_y = vox_y;
    vox_size_z = vox_z;

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, DEVICE);
    cudaSetDevice(DEVICE);

    if (debug_flag==1) {

        printf("\nUsing GPU: %s - (device %d)\n", deviceProperties.name, DEVICE);
        printf("Total Memory: %ld\n", deviceProperties.totalGlobalMem);
        printf("Max threads per block: %d\n", deviceProperties.maxThreadsPerBlock);
        printf("Max threads dimension: (%d, %d, %d)\n", deviceProperties.maxGridSize[0],
                                                        deviceProperties.maxGridSize[1],
                                                        deviceProperties.maxGridSize[2]);
        printf("Major, Minor: (%d, %d)\n", deviceProperties.major, deviceProperties.minor);
        printf("Multiprocessor count: %d\n", deviceProperties.multiProcessorCount);
        printf("Threads per block: %d\n", NUM_THREADS);
    }

    debug = debug_flag;

    if (NUM_THREADS>deviceProperties.maxThreadsPerBlock){
        printf("Selected NUM_THREADS (%d) is greater than device's max threads per block (%d)\n",
               NUM_THREADS, deviceProperties.maxThreadsPerBlock);
        exit(0);
    }


    float parameters[8];

    cudaMalloc(&parameters_GPU, 8 * sizeof(float));

    parameters[VOX_UNIT] = vox_unit;
    parameters[VOX_MARGIN] = vox_margin;
    parameters[CAM_F] = f;
    parameters[SENSOR_W] = sensor_w;
    parameters[SENSOR_H] = sensor_h;
    parameters[VOX_SIZE_X] = (float)vox_size_x;
    parameters[VOX_SIZE_Y] = (float)vox_size_y;
    parameters[VOX_SIZE_Z] = (float)vox_size_z;


    cudaMemcpy(parameters_GPU, parameters, 8 * sizeof(float), cudaMemcpyHostToDevice);


}

void clear_parameters_GPU(){
    cudaFree(parameters_GPU);
}


__global__
void point_cloud_kernel(float *baseline, unsigned char *depth_data,
                        float *point_cloud, int *width, int *height){

  //if (threadIdx.x==0) printf("fwg %d  fwg %d", frame_width_GPU,frame_height_GPU);

  //Rerieve pixel coodinates
  int pixel_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (pixel_idx >= (*width * *height))
    return;

  int pixel_y = pixel_idx / *width;
  int pixel_x = pixel_idx % *width;

  //if (threadIdx.x==0 ) {printf("blockIdx.x:%d pidx:%d px:%d py:%d\n", blockIdx.x, pixel_idx, pixel_x, pixel_y );}

  float     CV_PI = 3.141592;

  int		max_radius = 30;
  int		inf_border = 160;		// Range (in pixel) from the pole to exclude from point cloud generation
  double	unit_h, unit_w;	//angular size of 1 pixel
  float		disp_scale = 2;
  float		disp_offset = -120;

  unit_h = 1.0 / (*height);
  unit_w = 2.0 / (*width);

  // Get point in world coordinate
  // Try to parallel later

  int point_disparity = depth_data[pixel_y * *width + pixel_x];

  if (point_disparity == 0)
	return;

  if (pixel_y<inf_border || pixel_y> *height - inf_border)
	return;

  float longitude, latitude, radius, angle_disp;

  latitude = pixel_y * unit_h * CV_PI;

  longitude = pixel_x * unit_w * CV_PI;

  angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;

  if (latitude + angle_disp <0)
    angle_disp = 0.01;

  if (angle_disp == 0)   {
	radius = max_radius;
	point_disparity = 0;
  }	else
	radius = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

  if (radius > max_radius || radius < 0.0) 	{
	radius = max_radius;
	point_disparity = 0;
  }

  //too close
  //if (latitude < CV_PI/4) || (latitude > CV_PI - CV_PI/4))
  //if (latitude < CV_PI/3)
  //  return;


  //world coordinates
  //float rx = radius*sin(latitude)*cos(CV_PI - longitude);
  //float ry = radius*sin(latitude)*sin(CV_PI - longitude);
  //float rz = radius*cos(latitude);
  //voxel coordinates
  //int z = (int)floor(rz / vox_unit_GPU + vox_size[2]/2);
  //int x = (int)floor(rx / vox_unit_GPU + vox_size[0]/2));
  //int y = (int)floor(ry / vox_unit_GPU);


  //float rx = -radius*sin(latitude)*cos(CV_PI - longitude);
  float rx = radius*sin(latitude)*cos(CV_PI - longitude);
  float rz = radius*sin(latitude)*sin(CV_PI - longitude);
  float ry = radius*cos(latitude); //+.20cm to get the floor

  //voxel coordinates
  //int z = (int)floor(rz / vox_unit_GPU);
  //int x = (int)floor(rx / vox_unit_GPU);// + vox_size[0]/2);
  //int y = (int)floor(ry / vox_unit_GPU);// + vox_size[1]/2);

  point_cloud[6 * pixel_idx + 0] = rx;
  point_cloud[6 * pixel_idx + 1] = ry;
  point_cloud[6 * pixel_idx + 2] = rz;
  point_cloud[6 * pixel_idx + 3] = latitude;
  point_cloud[6 * pixel_idx + 4] = longitude;
  point_cloud[6 * pixel_idx + 5] = radius;
  //if (threadIdx.x==0 ) {printf("blockIdx.x:%d pcx:%2.2f rx:%2.2f ry:%2.2f rz:%2.2f lat:%3.0f long:%3.0f \n",
  //                      blockIdx.x, point_cloud[6 * pixel_idx + 0], rx, ry, rz, latitude*180/CV_PI, longitude*180/CV_PI);}

}


void get_point_cloud_CPP(float baseline, unsigned char *depth_data, float *point_cloud, int width, int height) {

  clock_tick t1 = start_timer();

  float *baseline_GPU;
  int *width_GPU;
  int *height_GPU;
  unsigned char *depth_data_GPU;
  float *point_cloud_GPU;

  int num_pixels = width * height;


  gpuErrchk(cudaMalloc(&baseline_GPU, sizeof(float)));
  gpuErrchk(cudaMalloc(&width_GPU, sizeof(int)));
  gpuErrchk(cudaMalloc(&height_GPU, sizeof(int)));

  gpuErrchk(cudaMalloc(&depth_data_GPU, num_pixels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&point_cloud_GPU, 6 * num_pixels * sizeof(float)));
  gpuErrchk(cudaMemset(point_cloud_GPU, 0, 6 * num_pixels * sizeof(float)));

  gpuErrchk(cudaMemcpy(baseline_GPU, &baseline, sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(width_GPU, &width, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(height_GPU, &height, sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));

  end_timer(t1, "Prepare duration");

  if (debug==1) printf("frame width: %d   frame heigth: %d   num_pixels %d\n" , width,height, num_pixels);


  t1 = start_timer();
  // from depth map to binaray voxel representation
  //depth2Grid<<<frame_width,frame_height>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
  //                                         vox_grid_GPU, parameters_GPU);


  int NUM_BLOCKS = int((width*height + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  point_cloud_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(baseline_GPU, depth_data_GPU, point_cloud_GPU,
                                                  width_GPU, height_GPU);

  gpuErrchk( cudaPeekAtLastError() );

  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"depth2Grid duration");

  cudaMemcpy(point_cloud, point_cloud_GPU,  6* num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(baseline_GPU);
  cudaFree(width_GPU);
  cudaFree(height_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(point_cloud_GPU);

  end_timer(t1,"closeup duration");

}

__global__
void get_voxels_kernel(float *point_cloud_GPU, int *point_cloud_size_GPU,
                       float *boundaries_GPU, int *vol_number_GPU, unsigned char *vox_grid_GPU, float *parameters_GPU){

  //if (blockIdx.x!=2000)
  //   return;
  //printf("boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n" ,
  //                     boundaries_GPU[0], boundaries_GPU[1], boundaries_GPU[2], boundaries_GPU[3], boundaries_GPU[4], boundaries_GPU[5]);

  //if (blockIdx.x >40 && blockIdx.x <45) {printf("threadIdx.x: %d blockIdx.x:%d point_cloud_size:%d P0!!\n", threadIdx.x, blockIdx.x, *point_cloud_size_GPU);}

  //Rerieve pixel coodinates
  int point_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (point_idx >= *point_cloud_size_GPU)
    return;

  //if (blockIdx.x >40 && blockIdx.x <45) {printf("threadIdx.x: %d blockIdx.x:%d  point_idx:%d P1!!\n", threadIdx.x, blockIdx.x, point_idx);}

  int x_idx = point_idx * 6 + 0;
  int y_idx = point_idx * 6 + 1;
  int z_idx = point_idx * 6 + 2;
  int lat_idx = point_idx * 6 + 3;
  int long_idx = point_idx * 6 + 4;
  int rd_idx = point_idx * 6 + 5;

  float  min_x = boundaries_GPU[0];
  float  max_x = boundaries_GPU[1];
  float  min_y = boundaries_GPU[2];
  float  max_y = boundaries_GPU[3];
  float  min_z = boundaries_GPU[4];
  float  max_z = boundaries_GPU[5];

  float wx = point_cloud_GPU[x_idx];
  float wy = point_cloud_GPU[y_idx];
  float wz = point_cloud_GPU[z_idx];
  float latitude = point_cloud_GPU[lat_idx];
  float longitude = point_cloud_GPU[long_idx];
  float rd = point_cloud_GPU[rd_idx];

  float vox_unit_GPU = parameters_GPU[VOX_UNIT];
  float sensor_w_GPU = parameters_GPU[SENSOR_W];
  float sensor_h_GPU = parameters_GPU[SENSOR_H];
  float f_GPU = parameters_GPU[CAM_F];
  int vox_size_x_GPU = (int)parameters_GPU[VOX_SIZE_X];
  int vox_size_y_GPU = (int)parameters_GPU[VOX_SIZE_Y];
  int vox_size_z_GPU = (int)parameters_GPU[VOX_SIZE_Z];

  if ((wx == 0.) && (wy == 0.) && (wz == 0.)) {
    //if (blockIdx.x >40 && blockIdx.x <45) {printf("ZERO idx:%d rx:%f ry:%f rz:%f\n",point_idx, wx, wy, wz);}
    return;
  }


  if ((wx < min_x) || (wx > max_x) || (wy < min_y) || (wy > max_y) || (wz < min_z) || (wz > max_z) ) {
    //printf("OUT OF BOUNDARIES idx:%d rx: %2.2f (%2.2f %2.2f) ry: %2.2f (%2.2f %2.2f) rz: %2.2f (%2.2f %2.2f)\n",
    //        point_idx, wx, min_x, max_x, wy, min_y, max_y, wz, min_z, max_z);
    return;
  }

  /**/
  //if (blockIdx.x >40 && blockIdx.x <45) {printf("threadIdx.x:%d blockIdx.x:%d %2.2f %2.2f %2.2f P2!!\n", threadIdx.x, blockIdx.x, wx, wy, wz);}


  //Adjust to vol_number
  if (*vol_number_GPU == 1) {  //Vol 1 has no adjustments
  }



  //if ( (wx < -vox_unit_GPU*vox_size_x_GPU/2)  || (wx > vox_unit_GPU*vox_size_x_GPU/2) || (wz < 0)) {
    //printf("OUT OF VOLUME idx:%d rx:%f ry:%f rz:%f\n",point_idx, wx, wy, wz);
    //return; //out of volume
  //}


  //Calculating FOV
  float fov_x = wz * (sensor_w_GPU/2)/f_GPU;
  float fov_y = wz * (sensor_h_GPU/2)/f_GPU;

  if ((abs(wx)>fov_x) || (abs(wy)>fov_y)) {
    //printf("OUT OF FOV(%2.2f %2.2f %2.2f %2.2f %2.2f %2.2f) idx:%d rx:%2.2f ry:%2.2f rz:%2.2f\n",
    //sensor_w_GPU, sensor_h_GPU, f_GPU, rd, fov_x, fov_y,point_idx, wx, wy, wz);
    return;
  }

  //voxel coordinates
  int vz = (int)floor(wz/vox_unit_GPU);
  int vx = (int)floor(wx/vox_unit_GPU + vox_size_x_GPU/2);
  int vy = (int)floor(wy/vox_unit_GPU + vox_size_y_GPU/2);


  // mark vox_out with 1.0
  if( vx >= 0 && vx < vox_size_x_GPU && vy >= 0 && vy < vox_size_y_GPU && vz >= 0 && vz < vox_size_z_GPU){
      int vox_idx = vz * vox_size_x_GPU * vox_size_y_GPU + vy * vox_size_x_GPU + vx;
      vox_grid_GPU[vox_idx] = float(1.0);
      //printf("OK idx:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n",point_idx, wx, wy, wz, vx, vy, vz);

  } else {
      printf("OUT idx:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n", point_idx, wx, wy, wz, vx, vy, vz);
  }


}





void get_voxels_CPP(float *point_cloud, int point_cloud_size, float *boundaries, int vol_number, unsigned char *vox_grid) {

  clock_tick t1 = start_timer();

  float *point_cloud_GPU;
  int *point_cloud_size_GPU;
  float *boundaries_GPU;
  int *vol_number_GPU;
  unsigned char *vox_grid_GPU;

  int num_voxels = vox_size_x * vox_size_y * vox_size_z;

  if (debug==1) printf("get_voxels - point_cloud_size: %d   vol_number: %d  voxel_size: %d %d %d\n" ,
                       point_cloud_size, vol_number, vox_size_x , vox_size_y , vox_size_z);

  if (debug==1) printf("get_voxels - boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n" ,
                       boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], boundaries[5]);

  gpuErrchk(cudaMalloc(&point_cloud_GPU, point_cloud_size * 6 * sizeof(float)));
  gpuErrchk(cudaMalloc(&point_cloud_size_GPU, sizeof(int)));
  gpuErrchk(cudaMalloc(&boundaries_GPU, 6 * sizeof(float)));
  gpuErrchk(cudaMalloc(&vol_number_GPU, sizeof(int)));
  gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));

  gpuErrchk(cudaMemcpy(point_cloud_GPU, point_cloud, point_cloud_size * 6 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(point_cloud_size_GPU, &point_cloud_size, sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(boundaries_GPU, boundaries, 6 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vol_number_GPU, &vol_number, sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(vox_grid_GPU, 0, num_voxels * sizeof(unsigned char)));


  end_timer(t1, "Prepare duration");

  t1 = start_timer();
  int NUM_BLOCKS = int((point_cloud_size + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("get_voxels - NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  get_voxels_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(point_cloud_GPU, point_cloud_size_GPU,
                                                 boundaries_GPU, vol_number_GPU, vox_grid_GPU, parameters_GPU);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  end_timer(t1,"get_voxels duration");

  cudaMemcpy(vox_grid, vox_grid_GPU,  num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(point_cloud_GPU);
  cudaFree(point_cloud_size_GPU);
  cudaFree(boundaries_GPU);
  cudaFree(vol_number_GPU);
  cudaFree(vox_grid_GPU);

  end_timer(t1,"cleanup duration");

}

__global__
void downsample_grid_kernel( unsigned char *in_grid_GPU, unsigned char *out_grid_GPU, float *parameters_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    float downscale = 4;

    int in_vox_size_x = (int)parameters_GPU[VOX_SIZE_X];
    int in_vox_size_y = (int)parameters_GPU[VOX_SIZE_Y];
    int in_vox_size_z = (int)parameters_GPU[VOX_SIZE_Z];
    int out_vox_size_x = (int)in_vox_size_x/downscale;
    int out_vox_size_y = (int)in_vox_size_y/downscale;
    int out_vox_size_z = (int)in_vox_size_z/downscale;

    if (vox_idx >= out_vox_size_x * out_vox_size_y * out_vox_size_z){
      return;
    }

    int z = (vox_idx / ( out_vox_size_x * out_vox_size_y))%out_vox_size_z ;
    int y = (vox_idx / out_vox_size_x) % out_vox_size_y;
    int x = vox_idx % out_vox_size_x;

    int sum_occupied = 0;

    for (int tmp_x = x * downscale; tmp_x < (x + 1) * downscale; ++tmp_x) {
      for (int tmp_y = y * downscale; tmp_y < (y + 1) * downscale; ++tmp_y) {
        for (int tmp_z = z * downscale; tmp_z < (z + 1) * downscale; ++tmp_z) {

          int tmp_vox_idx = tmp_z * in_vox_size_x * in_vox_size_y + tmp_y * in_vox_size_z + tmp_x;

          if (in_grid_GPU[tmp_vox_idx]> 0){
            sum_occupied += 1;          }
        }
      }
    }
    if (sum_occupied>=8) {  //empty threshold
      out_grid_GPU[vox_idx] = 1;
    }

    for (int tmp_x = x * downscale; tmp_x < (x + 1) * downscale; ++tmp_x) {
      for (int tmp_y = y * downscale; tmp_y < (y + 1) * downscale; ++tmp_y) {
        for (int tmp_z = z * downscale; tmp_z < (z + 1) * downscale; ++tmp_z) {

          int tmp_vox_idx = tmp_z * in_vox_size_x * in_vox_size_y + tmp_y * in_vox_size_z + tmp_x;

          if (in_grid_GPU[tmp_vox_idx] > 0)
            out_grid_GPU[vox_idx] += 1;

        }
      }
    }
    if (out_grid_GPU[vox_idx]<8)   //empty threshold
      out_grid_GPU[vox_idx] = 0;
    else
      out_grid_GPU[vox_idx] = 1;

}

void downsample_grid_CPP(unsigned char *vox_grid, unsigned char *vox_grid_down) {

  clock_tick t1 = start_timer();

  unsigned char *vox_grid_GPU;
  unsigned char *vox_grid_down_GPU;

  int num_voxels = vox_size_x * vox_size_y * vox_size_z;
  int num_voxels_down = num_voxels/64;

  gpuErrchk(cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char)));

  gpuErrchk(cudaMemcpy(vox_grid_GPU, vox_grid, num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(vox_grid_down_GPU, 0, num_voxels_down * sizeof(unsigned char)));


  end_timer(t1, "Prepare duration");

  t1 = start_timer();
  int NUM_BLOCKS = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("downsample - NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  downsample_grid_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(vox_grid_GPU, vox_grid_down_GPU, parameters_GPU);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  end_timer(t1,"downsample duration");

  cudaMemcpy(vox_grid_down, vox_grid_down_GPU,  num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(vox_grid_GPU);
  cudaFree(vox_grid_down_GPU);

  end_timer(t1,"cleanup duration");

}



__global__
void SquaredDistanceTransform(unsigned char *depth_data, unsigned char *vox_grid,
                              float *vox_tsdf, unsigned char *vox_limits, float *baseline,
                              int *width, int *height, float *boundaries_GPU, int *vol_number, float *parameters_GPU) {

  float vox_unit_GPU = parameters_GPU[VOX_UNIT];
  float vox_margin_GPU = parameters_GPU[VOX_MARGIN];
  float sensor_w_GPU = parameters_GPU[SENSOR_W];
  float sensor_h_GPU = parameters_GPU[SENSOR_H];
  float f_GPU = parameters_GPU[CAM_F];
  int vox_size_x_GPU = (int)parameters_GPU[VOX_SIZE_X];
  int vox_size_y_GPU = (int)parameters_GPU[VOX_SIZE_Y];
  int vox_size_z_GPU = (int)parameters_GPU[VOX_SIZE_Z];

  float  min_x = boundaries_GPU[0];
  float  max_x = boundaries_GPU[1];
  float  min_y = boundaries_GPU[2];
  float  max_y = boundaries_GPU[3];
  float  min_z = boundaries_GPU[4];
  float  max_z = boundaries_GPU[5];

  //VOX_LIMITS
  int OUT_OF_FOV = 3;
  int OUT_OF_ROOM = 2;
  int EMPTY_VISIBLE = 1;
  int OCCUP_OR_OCLUDED = 0;

  int search_region = (int)roundf(vox_margin_GPU/vox_unit_GPU);

  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (vox_idx >= vox_size_x_GPU * vox_size_y_GPU * vox_size_z_GPU){
      return;
  }

  int z = (vox_idx / ( vox_size_x_GPU * vox_size_y_GPU))%vox_size_z_GPU ;
  int y = (vox_idx / vox_size_x_GPU) % vox_size_y_GPU;
  int x = vox_idx % vox_size_x_GPU;

  // Get point in world coordinates XYZ -> YZX
  float wz = float(z) * vox_unit_GPU;                    //point_base[0]
  float wx = (float(x)-vox_size_x_GPU/2) * vox_unit_GPU; //point_base[1]
  float wy = (float(y)-vox_size_y_GPU/2) * vox_unit_GPU;            //point_base[2]

  if (wx==0.0 && wy==0 && wz==0){
    vox_tsdf[vox_idx] = 2000.;
    vox_limits[vox_idx] = OUT_OF_FOV;
    return;
  }

  if (vox_idx==17400) {
    printf("\n3\n");
  }

  if (wx < min_x || wx > max_x || wy < min_y || wy > max_y || wz < min_z || wz > max_z ){
    // outside ROOM
    vox_tsdf[vox_idx] = 2000.;
    vox_limits[vox_idx] = OUT_OF_ROOM;
  if (vox_idx==17400) {
    printf("\n3b %4.2f idx:%d \n",vox_tsdf[vox_idx],vox_idx);
  }
    return;
  }

  //Calculating FOV
  float fov_x = wz * (sensor_w_GPU/2)/f_GPU;
  float fov_y = wz * (sensor_h_GPU/2)/f_GPU;

  if (vox_idx==17400) {
    printf("\n4\n");
  }
  if ((abs(wx)>fov_x) || (abs(wy)>fov_y)) {
     //outside FOV
    vox_tsdf[vox_idx] = 2000.;
    vox_limits[vox_idx] = OUT_OF_FOV;
    return;
  }



  float CV_PI = 3.141592;
  float longitude, latitude, point_depth, angle_disp;

  float hip1 = sqrtf(wx*wx + wz*wz);
  float hip2 = sqrtf(hip1*hip1 + wy*wy);

  float teta1, teta2;


  if (vox_idx==17400) {
    printf("\n wx:%2.2f wy:%2.2f wz:%2.2f hip1:%4.2f hip2:%4.2f\n ", wx, wy, wz, hip1, hip2);
  }


  /*
  if (wy>0)
     teta1 = asin(wy/hip2);
  else
     teta1 = CV_PI - asin(wy/hip2);
  */
  teta1 = asin(wy/hip2);

  latitude = CV_PI/2 - teta1;

  //longitude = 3*CV_PI/2 - teta2;
  if (wx<0)
     teta2 = asin(wz/hip1);
  else
     teta2 = CV_PI - asin(wz/hip1);

  longitude = teta2;

  if (vox_idx==17400) {
    printf("\n t1:%2.2f t2:%2.2f lat:%2.2f long:%4.2f\n ", teta1, teta2, latitude, longitude);
  }
  float  	unit_h, unit_w;	//angular size of 1 pixel
  float		disp_scale = 2;
  float		disp_offset = -120;
  int		max_radius = 30;


  unit_h = 1.0 / (*height);
  unit_w = 2.0 / (*width);

  int pixel_y = latitude /(unit_h * CV_PI);
  int pixel_x = longitude /(unit_w * CV_PI);

  if (vox_idx==17400) {
    printf("\n5\n");
  }
  int point_disparity = depth_data[pixel_y * *width + pixel_x];
  if (point_disparity == 0){ // mising depth
      vox_tsdf[vox_idx] = -1.0;
      return;
  }

  angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;

  if (latitude + angle_disp <0)
    angle_disp = 0.01;

  if (angle_disp == 0)   {
	point_depth = max_radius;
	point_disparity = 0;
  }	else
    point_depth = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

  if (point_depth > max_radius || point_depth < 0.0) 	{
	point_depth = max_radius;
	point_disparity = 0;
  }

  float vox_depth =hip2;

  if (x==0 && y==72 && z==120) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, vox_limits[vox_idx]);
  }

  if (vox_idx==17400) {
    printf("\n6\n");
  }
  //OCCUPIED
  if (vox_grid[vox_idx] >0 ){
     vox_tsdf[vox_idx] = 0;
     vox_limits[vox_idx] = OCCUP_OR_OCLUDED;
     return;
  }


  float sign;
  if (abs(point_depth - vox_depth) < 0.001){
      sign = -1; // avoid NaN
  }else{
      sign = (point_depth - vox_depth)/abs(point_depth - vox_depth);
  }
  vox_tsdf[vox_idx] = sign;
  if (sign >0.0) {
    vox_limits[vox_idx] = EMPTY_VISIBLE;
  } else {
    vox_limits[vox_idx] = OCCUP_OR_OCLUDED;

  }
  if (vox_idx==17400) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }

  /*
  if (x==115 && y==54 && z==80) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }

  if (x==117 && y==54 && z==80) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }
  if (x==120 && y==54 && z==80) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }
  if (x==123 && y==54 && z==80) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }
  if (x==125 && y==54 && z==80) {
    printf("\nx:%d\n wx:%2.2f wy:%2.2f wz:%2.2f\n lat:%2.2f  long:%2.2f px_x:%d px_y:%d pdisp:%d pdepth:%4.4f vox_depth:%4.4f sign:%4.4f vl:%d\n",
            x, wx, wy, wz, 180 *latitude/CV_PI, 180*longitude/CV_PI, pixel_x, pixel_y, point_disparity, point_depth, vox_depth, sign, vox_limits[vox_idx]);
  }
  */
    int radius=search_region; // out -> in
    int found = 0;
    //fixed y planes
    int iiy = max(0,y-radius);
    for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiy = min(y+radius,vox_size_y_GPU);
    for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed x planes
    int iix = max(0,x-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iix = min(x+radius,vox_size_x_GPU);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed z planes
    int iiz = max(0,z-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiz = min(z+radius,vox_size_z_GPU);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
            int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }

    if (found == 0)
        return;

    radius=1; // in -> out
    found = 0;
    while (radius < search_region) {
        //fixed y planes
        int iiy = max(0,y-radius);
        for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiy = min(y+radius,vox_size_y_GPU);
        for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed x planes
        int iix = max(0,x-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iix = min(x+radius,vox_size_x_GPU);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size_z_GPU,z+radius+1); iiz++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed z planes
        int iiz = max(0,z-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiz = min(z+radius,vox_size_z_GPU);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size_y_GPU,y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size_x_GPU,x+radius+1); iix++){
                int iidx = iiz * vox_size_x_GPU * vox_size_y_GPU + iiy * vox_size_x_GPU + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        if (found == 1)
          return;

        radius++;

    }
  if (vox_idx==17400) {
    printf("\ntsdf:%4.4f\n", vox_tsdf[vox_idx]);
  }
}

void FlipTSDF_CPP(float *vox_tsdf){

  clock_tick t1 = start_timer();

  for (int vox_idx=0; vox_idx< vox_size_x*vox_size_y*vox_size_x; vox_idx++) {

      float value = float(vox_tsdf[vox_idx]);
      if (value > 1)
          value =1;


      float sign;
      if (abs(value) < 0.001)
        sign = 1;
      else
        sign = value/abs(value);

      vox_tsdf[vox_idx] = sign*(max(0.001,(1.0-abs(value))));
  }
  end_timer(t1,"FlipTSDF");
}


void FTSDFDepth_CPP(unsigned char *depth_data,
                      unsigned char *vox_grid,
                      float *vox_tsdf,
                      unsigned char *vox_limits,
                      float baseline,
                      int width,
                      int height,
                      float *boundaries,
                      int vol_number) {

  clock_tick t1 = start_timer();

  float         *boundaries_GPU;
  unsigned char *vox_grid_GPU;
  unsigned char *depth_data_GPU;
  float         *vox_tsdf_GPU;
  unsigned char *vox_limits_GPU;
  float         *baseline_GPU;
  int           *width_GPU;
  int           *height_GPU;
  int           *vol_number_GPU;

  int num_voxels = vox_size_x * vox_size_y * vox_size_z;
  int num_pixels = width * height;

  if (debug==1) printf("FTSDFDepth - boundaries: (%2.2f %2.2f) (%2.2f %2.2f) (%2.2f %2.2f)\n" ,
                       boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], boundaries[5]);

  gpuErrchk(cudaMalloc(&boundaries_GPU, 6 * sizeof(float)));
  gpuErrchk(cudaMalloc(&vox_grid_GPU,   num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&depth_data_GPU, num_pixels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&vox_tsdf_GPU,   num_voxels * sizeof(float)));
  gpuErrchk(cudaMalloc(&vox_limits_GPU, num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&baseline_GPU,   sizeof(float)));
  gpuErrchk(cudaMalloc(&width_GPU,      sizeof(int)));
  gpuErrchk(cudaMalloc(&height_GPU,     sizeof(int)));
  gpuErrchk(cudaMalloc(&vol_number_GPU, sizeof(int)));

  gpuErrchk(cudaMemcpy(boundaries_GPU,       boundaries,    6 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vox_grid_GPU,         vox_grid,      num_voxels * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(depth_data_GPU,       depth_data,    num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(baseline_GPU,         &baseline,     sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(width_GPU,            &width,        sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(height_GPU,           &height,       sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(vol_number_GPU,       &vol_number,   sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(vox_limits_GPU,       0,             num_voxels * sizeof(unsigned char)));
  gpuErrchk(cudaMemset(vox_tsdf_GPU,         0,             num_voxels * sizeof(float)));


  end_timer(t1, "Prepare duration");

  t1 = start_timer();
  int NUM_BLOCKS = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

 /*  SquaredDistanceTransform(unsigned char *depth_data, float *vox_grid,
                              float *vox_tsdf, unsigned char *vox_limits, float *baseline,
                              int *width, int *height, float *boundaries_GPU, int *vol_number, float *parameters_GPU)
*/
  SquaredDistanceTransform<<<NUM_BLOCKS, NUM_THREADS>>>(depth_data_GPU, vox_grid_GPU, vox_tsdf_GPU, vox_limits_GPU,
                                                 baseline_GPU, width_GPU, height_GPU,
                                                 boundaries_GPU, vol_number_GPU, parameters_GPU);

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );


  end_timer(t1,"SquaredDistanceTransform duration");

  cudaMemcpy(vox_tsdf, vox_tsdf_GPU,      num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_limits, vox_limits_GPU,  num_voxels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  printf("\nvox_tsdf %4.4f\n",vox_tsdf[17400]);

  cudaFree(boundaries_GPU);
  cudaFree(vox_grid_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(vox_tsdf_GPU);
  cudaFree(vox_limits_GPU);
  cudaFree(baseline_GPU);
  cudaFree(width_GPU);
  cudaFree(height_GPU);
  cudaFree(vol_number_GPU);

  end_timer(t1,"cleanup duration");

  FlipTSDF_CPP(vox_tsdf);

}



extern "C" {
    void get_point_cloud  (float baseline,
                  unsigned char *depth_data,
                  float *point_cloud,
                  int width,
                  int height) {
                                 get_point_cloud_CPP  (baseline,
                                              depth_data,
                                              point_cloud,
                                              width,
                                              height) ;
                  }
    void get_voxels (float *point_cloud,
                     int point_cloud_size,
                     float *boundaries,
                     int vol_number,
                     unsigned char *vox_grid) {
                                 get_voxels_CPP(point_cloud,
                                                point_cloud_size,
                                                boundaries,
                                                vol_number,
                                                vox_grid) ;
                  }

    void FTSDFDepth(unsigned char *depth_data,
                      unsigned char *vox_grid,
                      float *vox_tsdf,
                      unsigned char *vox_limits,
                      float baseline,
                      int width,
                      int height,
                      float *boundaries,
                      int vol_number) {
                                 FTSDFDepth_CPP(depth_data,
                                                vox_grid,
                                                vox_tsdf,
                                                vox_limits,
                                                baseline,
                                                width,
                                                height,
                                                boundaries,
                                                vol_number) ;
                  }
    void downsample_grid (unsigned char *vox_grid,
                            unsigned char *vox_grid_down) {
                                 downsample_grid_CPP(vox_grid,
                                                vox_grid_down) ;
                  }
    void setup(int device, int num_threads,
               float v_unit, float v_margin,
               float f, float sensor_w, float sensor_h,
               int vox_size_x, int vox_size_y, int vox_size_z,
               int debug_flag){
                                  setup_CPP(device, num_threads,
                                            v_unit, v_margin,
                                            f, sensor_w, sensor_h,
                                            vox_size_x, vox_size_y, vox_size_z,
                                            debug_flag);
                  }



    void finish(){
                                  clear_parameters_GPU();
    }
/*    void ProcessEdges(int *vox_size
                  int out_scale,
                  unsigned char *depth_data,
                  unsigned char *edges_data,
                  float *vox_tsdf,
                  float *vox_edges,
                  float *tsdf_edges,
                  float *vox_limits,
                  int *segmentation_label_downscale) {
                                 ProcessEdges_CPP(vox_size,
                                             out_scale,
                                             depth_data,
                                             edges_data,
                                             vox_tsdf,
                                             vox_edges,
                                             tsdf_edges,
                                             vox_limits,
                                             segmentation_label_downscale) ;
                  }

    void get_grid(float baseline, int *vox_size,
                  unsigned char *depth_data,
                  unsigned char *vox_grid) {
                                 get_grid_CPP(baseline, vox_size,
                                             depth_data,
                                             vox_grid) ;
                  }

    void get_rgb_grid(float baseline, int *vox_size,
                  unsigned char *depth_data,
                  unsigned char *rgb_data,
                  unsigned char *vox_grid) {
                                 get_rgb_grid_CPP(baseline, vox_size,
                                             depth_data,
                                             rgb_data,
                                             vox_grid) ;
                  }
*/
}



/*
void destroy_parameters_GPU(float *parameters_GPU){

  cudaFree(parameters_GPU);

}


__global__
void depth2Grid(float *baseline, int *vox_size,  unsigned char *depth_data,
                unsigned char *vox_grid, float *parameters_GPU){

  //Get Parameters
  int frame_width_GPU, frame_height_GPU, total_width_GPU;
  float vox_unit_GPU, vox_margin_GPU;

  get_parameters_GPU(parameters_GPU, &frame_width_GPU, &frame_height_GPU, &total_width_GPU,
                                     &vox_unit_GPU, &vox_margin_GPU);

  //if (threadIdx.x==0) printf("fwg %d  fwg %d", frame_width_GPU,frame_height_GPU);

  //Rerieve pixel coodinates
  int pixel_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (pixel_idx >= frame_width_GPU * frame_height_GPU)
    return;

  int pixel_y = pixel_idx / frame_width_GPU;
  int pixel_x = pixel_idx % frame_width_GPU;


  float     CV_PI = 3.141592;

  int		max_radius = 30;
  int		inf_border = 160;		// Range (in pixel) from the pole to exclude from point cloud generation
  double	unit_h, unit_w;	//angular size of 1 pixel
  float		disp_scale = 2;
  float		disp_offset = -120;



  unit_h = 1.0 / (frame_height_GPU);
  unit_w = 2.0 / (total_width_GPU);




  // Get point in world coordinate
  // Try to parallel later


  int point_disparity = depth_data[pixel_y * frame_width_GPU + pixel_x];

  if (point_disparity == 0)
	return;

  if (pixel_y<inf_border || pixel_y> frame_height_GPU - inf_border)
	return;

  float longitude, latitude, radius, angle_disp;

  latitude = pixel_y * unit_h * CV_PI;

  longitude = pixel_x * unit_w * CV_PI;

  angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;

  if (latitude + angle_disp <0)
    angle_disp = 0.01;

  if (angle_disp == 0)   {
	radius = max_radius;
	point_disparity = 0;
  }	else
	radius = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

  if (radius > max_radius || radius < 0.0) 	{
	radius = max_radius;
	point_disparity = 0;
  }

  //world coordinates
  float rx = radius*sin(latitude)*cos(CV_PI - longitude);
  float ry = radius*sin(latitude)*sin(CV_PI - longitude);
  float rz = radius*cos(latitude);

  //voxel coordinates
  int z = (int)floor(rz / vox_unit_GPU + vox_size[2]/2);
  int x = (int)floor(rx / vox_unit_GPU + vox_size[0]/2);
  int y = (int)floor(ry / vox_unit_GPU);

  //too close
  if (z<.5)
    return;


  // mark vox_out with 1.0
  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
      int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
      vox_grid[vox_idx] = float(1.0);
      //printf("OK idx:%d d:%d px:%d py:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n",
      //    pixel_idx, point_disparity, pixel_x, pixel_y, rx, ry, rz, x, y, z);
  } else {
          printf("OUT idx:%d d:%d px:%d py:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n",
          pixel_idx, point_disparity, pixel_x, pixel_y, rx, ry, rz, x, y, z);
  }
}




void get_grid_CPP(float baseline, int *vox_size, unsigned char *depth_data, unsigned char *vox_grid_down) {

  clock_tick t1 = start_timer();
  int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];
  int vox_size_down[] = {vox_size[0]/4, vox_size[1]/4, vox_size[2]/4};
  int num_voxels_down = vox_size_down[0] * vox_size_down[1] * vox_size_down[2];

  float *baseline_GPU;
  unsigned char *depth_data_GPU;
  unsigned char *vox_grid_GPU;
  unsigned char *vox_grid_down_GPU;
  int *vox_size_GPU;
  int *vox_size_down_GPU;




  if (debug==1) printf("cudaMalloc1\n");
  gpuErrchk(cudaMalloc(&baseline_GPU, sizeof(float)));

  if (debug==1) printf("cudaMalloc2\n");
  gpuErrchk(cudaMalloc(&vox_size_GPU, 3 * sizeof(int)));

  if (debug==1) printf("cudaMallod depth_data_GPU\n");
  gpuErrchk(cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(unsigned char)));
  cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char));
  cudaMemset(vox_grid_GPU, 0, num_voxels * sizeof(unsigned char));

  cudaMemcpy(baseline_GPU, &baseline, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);

  if (debug==1) printf("cudaMencpy depth_data_GPU\n");
  gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice));

  end_timer(t1, "Prepare duration");

  if (debug==1) printf("frame width: %d   frame heigth: %d   num_voxels %d\n" , frame_width,frame_height, num_voxels);


  t1 = start_timer();
  // from depth map to binaray voxel representation
  //depth2Grid<<<frame_width,frame_height>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
  //                                         vox_grid_GPU, parameters_GPU);


  int NUM_BLOCKS = int((frame_width*frame_height + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  depth2Grid<<<NUM_BLOCKS, NUM_THREADS>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
                                           vox_grid_GPU, parameters_GPU);
  //depth2Grid<<<3, 1024>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
  //                                         vox_grid_GPU, parameters_GPU);

  if (debug==1) printf("depth2Grid\n");
  gpuErrchk( cudaPeekAtLastError() );

  if (debug==1) printf("cudaDeviceSynchronize\n");
  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"depth2Grid duration");




  gpuErrchk(cudaMalloc(&vox_size_down_GPU, 3 * sizeof(int)));
  cudaMalloc(&vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char));

  cudaMemcpy(vox_size_down_GPU, vox_size_down, 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(vox_grid_down_GPU, 0, num_voxels_down * sizeof(unsigned char));

  NUM_BLOCKS = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  grid_downsample_Kernel<<<NUM_BLOCKS, NUM_THREADS>>>(vox_size_GPU,  vox_size_down_GPU,
                                           vox_grid_GPU, vox_grid_down_GPU);

  if (debug==1) printf("grid_downsample_Kernel\n");
  gpuErrchk( cudaPeekAtLastError() );


  t1 = start_timer();
  cudaMemcpy(vox_grid_down, vox_grid_down_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(baseline_GPU);
  cudaFree(vox_size_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(vox_grid_GPU);

  end_timer(t1,"closeup duration");

  if (debug==1) printf("0 %d\n", depth_data[0]);
  if (debug==1) printf("1 %d\n", depth_data[1]);
  if (debug==1) printf("2 %d\n", depth_data[2]);
  if (debug==1) printf("0fw %d\n", depth_data[0+frame_width]);
  if (debug==1) printf("1fw %d\n", depth_data[1+frame_width]);
  if (debug==1) printf("2fw %d\n", depth_data[2+frame_width]);
  if (debug==1) printf("02fw %d\n", depth_data[0+2*frame_width]);
  if (debug==1) printf("12fw %d\n", depth_data[1+2*frame_width]);
  if (debug==1) printf("22fw %d\n", depth_data[2+2*frame_width]);

}


__global__
void rgb2Grid(float *baseline, int *vox_size,  unsigned char *depth_data, unsigned char *rgb_data,
                unsigned char *vox_grid, float *parameters_GPU){

  //Get Parameters
  int frame_width_GPU, frame_height_GPU, total_width_GPU;
  float vox_unit_GPU, vox_margin_GPU;

  get_parameters_GPU(parameters_GPU, &frame_width_GPU, &frame_height_GPU, &total_width_GPU,
                                     &vox_unit_GPU, &vox_margin_GPU);

  //if (threadIdx.x==0) printf("fwg %d  fwg %d", frame_width_GPU,frame_height_GPU);

  //Rerieve pixel coodinates
  int pixel_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (pixel_idx >= frame_width_GPU * frame_height_GPU)
    return;

  int pixel_y = pixel_idx / frame_width_GPU;
  int pixel_x = pixel_idx % frame_width_GPU;


  float     CV_PI = 3.141592;

  int		max_radius = 30;
  int		inf_border = 160;		// Range (in pixel) from the pole to exclude from point cloud generation
  double	unit_h, unit_w;	//angular size of 1 pixel
  float		disp_scale = 2;
  float		disp_offset = -120;



  unit_h = 1.0 / (frame_height_GPU);
  unit_w = 2.0 / (total_width_GPU);




  // Get point in world coordinate
  // Try to parallel later


  int point_disparity = depth_data[pixel_y * frame_width_GPU + pixel_x];
  int point_r = rgb_data[3* (pixel_y * frame_width_GPU + pixel_x) + 0];
  int point_g = rgb_data[3* (pixel_y * frame_width_GPU + pixel_x) + 1];
  int point_b = rgb_data[3* (pixel_y * frame_width_GPU + pixel_x) + 2];

  if (point_disparity == 0)
	return;

  if (pixel_y<inf_border || pixel_y> frame_height_GPU - inf_border)
	return;

  float longitude, latitude, radius, angle_disp;

  latitude = pixel_y * unit_h * CV_PI;

  longitude = pixel_x * unit_w * CV_PI;

  angle_disp = (point_disparity / disp_scale + disp_offset) * unit_h * CV_PI;

  if (latitude + angle_disp <0)
    angle_disp = 0.01;

  if (angle_disp == 0)   {
	radius = max_radius;
	point_disparity = 0;
  }	else
	radius = *baseline / ((sin(latitude) / tan(latitude + angle_disp)) - cos(latitude));

  if (radius > max_radius || radius < 0.0) 	{
	radius = max_radius;
	point_disparity = 0;
  }

  //too close
  //if (latitude < CV_PI/4) || (latitude > CV_PI - CV_PI/4))
  if (latitude < CV_PI/3)
    return;


  //world coordinates
  //float rx = radius*sin(latitude)*cos(CV_PI - longitude);
  //float ry = radius*sin(latitude)*sin(CV_PI - longitude);
  //float rz = radius*cos(latitude);
  //voxel coordinates
  //int z = (int)floor(rz / vox_unit_GPU + vox_size[2]/2);
  //int x = (int)floor(rx / vox_unit_GPU + vox_size[0]/2));
  //int y = (int)floor(ry / vox_unit_GPU);


  float rx = -radius*sin(latitude)*cos(CV_PI - longitude);
  float rz = radius*sin(latitude)*sin(CV_PI - longitude);
  float ry = radius*cos(latitude) + 1.45 +0.20; //+.20cm to get the floor

  //voxel coordinates
  int z = (int)floor(rz / vox_unit_GPU);
  int x = (int)floor(rx / vox_unit_GPU);// + vox_size[0]/2);
  int y = (int)floor(ry / vox_unit_GPU);// + vox_size[1]/2);



  // mark vox_out with 1.0
  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
      int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
      vox_grid[3 * vox_idx + 0] = point_r;
      vox_grid[3 * vox_idx + 1] = point_g;
      vox_grid[3 * vox_idx + 2] = point_b;
      //printf("RGB o.o:%d OK idx:%d d:%d px:%d py:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n",
      //    depth_data[0],pixel_idx, point_disparity, pixel_x, pixel_y, rx, ry, rz, x, y, z);
  } else {
          //printf("RGB OUT idx:%d d:%d px:%d py:%d rx:%f ry:%f rz:%f vx:%d vy:%d vz:%d\n",
          //pixel_idx, point_disparity, pixel_x, pixel_y, rx, ry, rz, x, y, z);
  }
}

__global__
void rgb_grid_downsample_Kernel( int *in_vox_size, int *out_vox_size,
                        unsigned char *in_grid_GPU, unsigned char *out_grid_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (vox_idx >= out_vox_size[0] * out_vox_size[1] * out_vox_size[2]){
      return;
    }

    float label_downscale = in_vox_size[0]/out_vox_size[0];

    //printf("down_size %d\n",down_size);

    int z = (vox_idx / ( out_vox_size[0] * out_vox_size[1]))%out_vox_size[2] ;
    int y = (vox_idx / out_vox_size[0]) % out_vox_size[1];
    int x = vox_idx % out_vox_size[0];

    int sum_occupied = 0;
    int r = 0;
    int g = 0;
    int b = 0;

    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {

          int tmp_vox_idx = tmp_z * in_vox_size[0] * in_vox_size[1] + tmp_y * in_vox_size[0] + tmp_x;

          if (in_grid_GPU[3 * tmp_vox_idx + 0] +
              in_grid_GPU[3 * tmp_vox_idx + 1] +
              in_grid_GPU[3 * tmp_vox_idx + 2]> 0){
            sum_occupied += 1;
            r +=  in_grid_GPU[3 * tmp_vox_idx + 0];
            g +=  in_grid_GPU[3 * tmp_vox_idx + 1];
            b +=  in_grid_GPU[3 * tmp_vox_idx + 2];
          }
        }
      }
    }
    if (sum_occupied<8) {  //empty threshold
      out_grid_GPU[3 * vox_idx + 0 ] = 0;
      out_grid_GPU[3 * vox_idx + 1 ] = 0;
      out_grid_GPU[3 * vox_idx + 2 ] = 0;
    }else{
      out_grid_GPU[3 * vox_idx + 0 ] = r/sum_occupied;
      out_grid_GPU[3 * vox_idx + 1 ] = g/sum_occupied;
      out_grid_GPU[3 * vox_idx + 2 ] = b/sum_occupied;
    }
}


float mae_calc_y(int y, unsigned char *vox_grid, int *vox_size, int top){
  int n = 0;
  float mae = 0.0;
  for (int x=0; x<vox_size[0]; x++) {
    for (int z=0; z<vox_size[2]; z++) {
       if (top) {
         //from top to bottom
         for (int y_try=vox_size[1] - 1; y_try>=vox_size[1]/2; y_try--){
           int vox_idx_try = z * vox_size[0] * vox_size[1] + y_try * vox_size[0] + x;
           if (vox_grid[3*vox_idx_try + 0] +  vox_grid[3*vox_idx_try + 1] + vox_grid[3*vox_idx_try + 2] > 0){
             n++;
             mae += abs(y_try - y);
           }
         }
       } else {
         //from bottom to top
         for (int y_try=0; y_try< vox_size[1]/2; y_try++){
           int vox_idx_try = z * vox_size[0] * vox_size[1] + y_try * vox_size[0] + x;
           if (vox_grid[3*vox_idx_try + 0] +  vox_grid[3*vox_idx_try + 1] + vox_grid[3*vox_idx_try + 2] > 0){
             n++;
             mae += abs(y_try - y);
           }
         }
       }
    }

  }

  mae /= n;
  //printf("top:%d y:%d  n:%d mae:%f vox_size: %d %d %d\n",top, y,n, mae, vox_size[0], vox_size[1], vox_size[2]);
  return(mae);

}

void find_vox_limits_CPP(unsigned char *vox_grid, unsigned char *vox_limits, int *vox_size, unsigned char *depth_data){
  //Top
  float min_mae = -1.;
  float mae;
  unsigned char ceil_y= vox_size[1] - 1;
  int top = 1;
  for (int y=vox_size[1] - 1; y>=vox_size[1]/2; y--){
    mae = mae_calc_y(y, vox_grid, vox_size, top);
    if ((mae<min_mae) || (min_mae==-1.)){
      min_mae = mae;
      ceil_y = y;
    }
  }
  printf("Top = %d\n", ceil_y);

  //adjust_top(vox_grid, vox_limits, vox_size, depth_data, ceil_y)


  unsigned char floor_y= 0;
  top = 0;
  min_mae = -1.;
  for (int y=0; y<vox_size[1]/2; y++){
    mae = mae_calc_y(y, vox_grid, vox_size, top);
    if ((mae<min_mae) || (min_mae==-1.)){
      min_mae = mae;
      floor_y = y;
    }
  }
  printf("Bottom = %d\n", floor_y);

}
*/


/*
void get_rgb_grid_CPP(float baseline, int *vox_size, unsigned char *depth_data, unsigned char *rgb_data, unsigned char *vox_grid_down) {

  clock_tick t1 = start_timer();
  int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];
  int vox_size_down[] = {vox_size[0]/4, vox_size[1]/4, vox_size[2]/4};
  int num_voxels_down = vox_size_down[0] * vox_size_down[1] * vox_size_down[2];

  float *baseline_GPU;
  unsigned char *depth_data_GPU;
  unsigned char *rgb_data_GPU;
  unsigned char *vox_grid_GPU;
  unsigned char *vox_grid_down_GPU;
  unsigned char *vox_grid;
  unsigned char *vox_limits;
  int *vox_size_GPU;
  int *vox_size_down_GPU;

  //for (int i=0; i< frame_width* 10; i++){
  //  if (depth_data[i] != 0) {

  //    printf("i:%d y%d x:%d val:%d\n", i, i/frame_width, i%frame_width, depth_data[i] );

  //  }
  //}



  //if (debug==1) printf("dd %d %d %d %d\n", depth_data[0],depth_data[1],depth_data[2],depth_data[3] );
  //if (debug==1) printf("dd %d %d %d\n", depth_data[0+ frame_width],depth_data[1+frame_width],depth_data[2+frame_width] );
  //if (debug==1) printf("baseline %f\n", baseline);
  //if (debug==1) printf("vox size %d %d %d \n", vox_size[0], vox_size[1], vox_size[2]);
  //if (debug==1) printf("fw %d 600-350 %d\n", frame_width, depth_data[350+600*frame_width]);
  //if (debug==1) printf("rgb data 600-350 %d %d %d\n", rgb_data[350+600*frame_width],
  //                                                    rgb_data[351+600*frame_width],
   //                                                   rgb_data[352+600*frame_width] );

  if (debug==1) printf("cudaMalloc1\n");
  gpuErrchk(cudaMalloc(&baseline_GPU, sizeof(float)));

  if (debug==1) printf("cudaMalloc2\n");
  gpuErrchk(cudaMalloc(&vox_size_GPU, 3 * sizeof(int)));

  if (debug==1) printf("cudaMallod depth_data_GPU\n");
  gpuErrchk(cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(unsigned char)));
  gpuErrchk(cudaMalloc(&rgb_data_GPU, 3 * frame_height * frame_width * sizeof(unsigned char)));



  cudaMalloc(&vox_grid_GPU, 3 * num_voxels * sizeof(unsigned char));
  cudaMemset(vox_grid_GPU, 0, 3 * num_voxels * sizeof(unsigned char));

  cudaMemcpy(baseline_GPU, &baseline, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);

  if (debug==1) printf("cudaMencpy depth_data_GPU\n");
  gpuErrchk(cudaMemcpy(depth_data_GPU, depth_data, frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(rgb_data_GPU, rgb_data, 3 * frame_height * frame_width * sizeof(unsigned char), cudaMemcpyHostToDevice));

  end_timer(t1, "Prepare duration");

  if (debug==1) printf("frame width: %d   frame heigth: %d   num_voxels %d\n" , frame_width,frame_height, num_voxels);


  t1 = start_timer();
  // from depth map to binaray voxel representation
  //depth2Grid<<<frame_width,frame_height>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
  //                                         vox_grid_GPU, parameters_GPU);


  int NUM_BLOCKS = int((frame_width*frame_height + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  rgb2Grid<<<NUM_BLOCKS, NUM_THREADS>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU, rgb_data_GPU,
                                           vox_grid_GPU, parameters_GPU);
  //depth2Grid<<<3, 1024>>>(baseline_GPU, vox_size_GPU,  depth_data_GPU,
  //                                         vox_grid_GPU, parameters_GPU);

  if (debug==1) printf("depth2Grid\n");
  gpuErrchk( cudaPeekAtLastError() );

  if (debug==1) printf("cudaDeviceSynchronize\n");
  gpuErrchk( cudaDeviceSynchronize() );

  end_timer(t1,"depth2Grid duration");

  vox_grid= (unsigned char *)malloc(num_voxels * 3 * sizeof(unsigned char));
  vox_limits= (unsigned char *)malloc(num_voxels * sizeof(unsigned char));

  cudaMemcpy(vox_grid, vox_grid_GPU, 3* num_voxels*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  find_vox_limits_CPP(vox_grid, vox_limits, vox_size, depth_data);


  gpuErrchk(cudaMalloc(&vox_size_down_GPU, 3 * sizeof(int)));
  cudaMalloc(&vox_grid_down_GPU, 3 *num_voxels_down * sizeof(unsigned char));

  cudaMemcpy(vox_size_down_GPU, vox_size_down, 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(vox_grid_down_GPU, 0, 3 * num_voxels_down * sizeof(unsigned char));

  NUM_BLOCKS = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  if (debug==1) printf("NUM_BLOCKS: %d   NUM_THREADS: %d\n" , NUM_BLOCKS,NUM_THREADS);

  rgb_grid_downsample_Kernel<<<NUM_BLOCKS, NUM_THREADS>>>(vox_size_GPU,  vox_size_down_GPU,
                                           vox_grid_GPU, vox_grid_down_GPU);

  if (debug==1) printf("grid_downsample_Kernel\n");
  gpuErrchk( cudaPeekAtLastError() );


  t1 = start_timer();
  cudaMemcpy(vox_grid_down, vox_grid_down_GPU, 3 * num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(baseline_GPU);
  cudaFree(vox_size_GPU);
  cudaFree(depth_data_GPU);
  cudaFree(vox_grid_GPU);

  end_timer(t1,"closeup duration");

  if (debug==1) printf("0 %d\n", depth_data[0]);
  if (debug==1) printf("1 %d\n", depth_data[1]);
  if (debug==1) printf("2 %d\n", depth_data[2]);
  if (debug==1) printf("0fw %d\n", depth_data[0+frame_width]);
  if (debug==1) printf("1fw %d\n", depth_data[1+frame_width]);
  if (debug==1) printf("2fw %d\n", depth_data[2+frame_width]);
  if (debug==1) printf("02fw %d\n", depth_data[0+2*frame_width]);
  if (debug==1) printf("12fw %d\n", depth_data[1+2*frame_width]);
  if (debug==1) printf("600-350 %d\n", depth_data[350+600*frame_width]);

}

*/

/*




__device__
float modeLargerZero(const int *values, int size) {
  int count_vector[NUM_CLASSES] = {0};

  for (int i = 0; i < size; ++i)
      if  (values[i] > 0)
          count_vector[values[i]]++;

  int md = 0;
  int freq = 0;

  for (int i = 0; i < NUM_CLASSES; i++)
      if (count_vector[i] > freq) {
          freq = count_vector[i];
          md = i;
      }
  return md;
}

// find mode of in an vector
__device__
float mode(const int *values, int size) {
  int count_vector[NUM_CLASSES] = {0};

  for (int i = 0; i < size; ++i)
          count_vector[values[i]]++;

  int md = 0;
  int freq = 0;

  for (int i = 0; i < NUM_CLASSES; i++)
      if (count_vector[i] > freq) {
          freq = count_vector[i];
          md = i;
      }
  return md;
}

__global__
void Downsample_Kernel( int *in_vox_size, int *out_vox_size,
                        int *in_labels, float *in_tsdf, float * in_grid_GPU,
                        int *out_labels, float *out_tsdf,
                        int label_downscale, float *out_grid_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (vox_idx >= out_vox_size[0] * out_vox_size[1] * out_vox_size[2]){
      return;
    }

    int down_size = label_downscale * label_downscale * label_downscale;

    //printf("down_size %d\n",down_size);

    int emptyT = int((0.95 * down_size)); //Empty Threshold

    int z = (vox_idx / ( out_vox_size[0] * out_vox_size[1]))%out_vox_size[2] ;
    int y = (vox_idx / out_vox_size[0]) % out_vox_size[1];
    int x = vox_idx % out_vox_size[0];

    //printf("x:%d, y:%d, z:%d\n", x, y, z);

    int label_vals[MAX_DOWN_SIZE] = {0};
    int count_vals=0;
    float tsdf_val = 0;

    int num_255 =0;

    int zero_count = 0;
    int zero_surface_count = 0;
    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {
          int tmp_vox_idx = tmp_z * in_vox_size[0] * in_vox_size[1] + tmp_y * in_vox_size[0] + tmp_x;
          label_vals[count_vals] = int(in_labels[tmp_vox_idx]);
          count_vals += 1;

          if (in_labels[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {
            if (in_labels[tmp_vox_idx]==255)
               num_255++;
            zero_count++;
          }
          if (in_grid_GPU[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {
            zero_surface_count++;
          }

          tsdf_val += in_tsdf[tmp_vox_idx];

        }
      }
    }


    if (zero_count > emptyT) {
      out_labels[vox_idx] = float(mode(label_vals, down_size));
    } else {
      out_labels[vox_idx] = float(modeLargerZero(label_vals, down_size)); // object label mode without zeros
    }

    if (zero_surface_count > emptyT) {
      out_grid_GPU[vox_idx] = 0;
    } else {
      out_grid_GPU[vox_idx] = 1.0;
    }

    out_tsdf[vox_idx] = tsdf_val /  down_size;

    //Encode weights into downsampled labels


}



void DownsampleLabel_CPP(int *vox_size,
                         int out_scale,
                         int *segmentation_label_fullscale,
                         float *vox_tsdf_fullscale,
                         int *segmentation_label_downscale,
                         float *vox_weights,float *vox_vol, float *vox_grid) {

  //downsample lable
  clock_tick t1 = start_timer();

  int num_voxels_in = vox_size[0] * vox_size[1] * vox_size[2];
  int label_downscale = 4;
  int num_voxels_down = num_voxels_in/(label_downscale*label_downscale*label_downscale);
  int out_vox_size[3];

  float *vox_tsdf = new float[num_voxels_down];
  float *vox_grid_downscale = new float[num_voxels_down];

  out_vox_size[0] = vox_size[0]/label_downscale;
  out_vox_size[1] = vox_size[1]/label_downscale;
  out_vox_size[2] = vox_size[2]/label_downscale;

  int *in_vox_size_GPU;
  int *out_vox_size_GPU;
  int *in_labels_GPU;
  int *out_labels_GPU;
  float *in_tsdf_GPU;
  float *out_tsdf_GPU;
  float *in_grid_GPU;
  float *out_grid_GPU;

  cudaMalloc(&in_vox_size_GPU, 3 * sizeof(int));
  cudaMalloc(&out_vox_size_GPU, 3 * sizeof(int));
  cudaMalloc(&in_labels_GPU, num_voxels_in * sizeof(int));
  cudaMalloc(&in_tsdf_GPU, num_voxels_in * sizeof(float));
  cudaMalloc(&in_grid_GPU, num_voxels_in * sizeof(float));
  cudaMalloc(&out_labels_GPU, num_voxels_down * sizeof(int));
  cudaMalloc(&out_tsdf_GPU, num_voxels_down * sizeof(float));
  cudaMalloc(&out_grid_GPU, num_voxels_down * sizeof(float));

  cudaMemcpy(in_vox_size_GPU, vox_size,  3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(out_vox_size_GPU, out_vox_size,  3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(in_labels_GPU, segmentation_label_fullscale, num_voxels_in * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(in_tsdf_GPU, vox_tsdf_fullscale, num_voxels_in * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(in_grid_GPU, vox_grid, num_voxels_in * sizeof(float), cudaMemcpyHostToDevice);


  int BLOCK_NUM = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  Downsample_Kernel<<< BLOCK_NUM, NUM_THREADS >>>(in_vox_size_GPU, out_vox_size_GPU,
                                                  in_labels_GPU, in_tsdf_GPU, in_grid_GPU,
                                                  out_labels_GPU, out_tsdf_GPU,
                                                  label_downscale, out_grid_GPU);

  cudaDeviceSynchronize();

  end_timer(t1,"Downsample duration");

  cudaMemcpy(segmentation_label_downscale, out_labels_GPU, num_voxels_down * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_tsdf, out_tsdf_GPU, num_voxels_down * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_grid_downscale, out_grid_GPU, num_voxels_down * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(in_vox_size_GPU);
  cudaFree(out_vox_size_GPU);
  cudaFree(in_labels_GPU);
  cudaFree(out_labels_GPU);
  cudaFree(in_tsdf_GPU);
  cudaFree(out_tsdf_GPU);
  cudaFree(in_grid_GPU);
  cudaFree(out_grid_GPU);


  // Find number of occupied voxels
  // Save voxel indices of background
  // Set label weights of occupied voxels as 1
  int num_occ_voxels = 0; //Occupied voxels in occluded regions
  std::vector<int> bg_voxel_idx;

  memset(vox_weights, 0, num_voxels_down * sizeof(float));
  memset(vox_vol, 0, num_voxels_down * sizeof(float));

  for (int i = 0; i < num_voxels_down; ++i) {
      if ((segmentation_label_downscale[i]) > 0 && (segmentation_label_downscale[i]<255)) { //Occupied voxels in the room
          vox_weights[i] = 1.0;
          num_occ_voxels++;
      } else {
          if ((vox_tsdf[i] < 0) && (segmentation_label_downscale[i]<255)) {
              bg_voxel_idx.push_back(i); // background voxels in unobserved region in the room
          }
      }

      if ((vox_grid_downscale[i] > 0) && (segmentation_label_downscale[i]>0) && (segmentation_label_downscale[i]<255)) { //Occupied voxels in the room
          vox_vol[i] = 0.5;
      } else {
          if ((vox_tsdf[i] < 0.1) && (segmentation_label_downscale[i]<255)) {
              if ((vox_tsdf[i] > -0.7) && (segmentation_label_downscale[i]>0))
                 vox_vol[i] = -0.5;
              else
                 vox_vol[i] = -1;
          } else {
                 vox_vol[i] = 1;
          }

      }

      if (vox_vol[i] == 0)
             vox_vol[i] = -3;
      if (vox_tsdf[i] > 1) {
             vox_weights[i] = 0;
             vox_vol[i] = -2;
      }
      if (segmentation_label_downscale[i] == 255){  //outside room
          segmentation_label_downscale[i] = 0;
          vox_vol[i] = -4;
      }


  }

  float occluded_empty_weight = num_occ_voxels * sample_neg_obj_ratio / bg_voxel_idx.size();

  for (int i = 0; i < bg_voxel_idx.size(); ++i) {
     vox_weights[bg_voxel_idx[i]] = occluded_empty_weight;
  }

  end_timer(t1,"Downsample duration + copy");

  delete [] vox_tsdf;


}

__global__
void depth2Grid_edges(float *cam_pose, int *vox_size,  float *vox_origin, float *depth_data, unsigned char *edges_data,
                      float *vox_edges, float *parameters_GPU){


  float *cam_K_GPU;
  int frame_width_GPU, frame_height_GPU;
  float vox_unit_GPU, vox_margin_GPU;

  get_parameters_GPU(parameters_GPU, &cam_K_GPU, &frame_width_GPU, &frame_height_GPU,
                                     &vox_unit_GPU, &vox_margin_GPU);


  // Get point in world coordinate
  // Try to parallel later

  // Get point in world coordinate
  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;



  unsigned char point_edges = edges_data[pixel_y * frame_width_GPU + pixel_x];

  if (point_edges > 0) {

      float min_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];
      int min_x = pixel_x;
      int min_y = pixel_y;

      //Search for the closest depth around the edge to get the object at the foreground
      for (int x =  pixel_x - 1; x<=pixel_x+1; x++) {
          if (x>=0 & x<frame_width_GPU) {
              for (int y = pixel_y -1; y<=pixel_y+1; y++) {
                   if (y>=0 & y<frame_height_GPU) {

                          float point_depth = depth_data[y * frame_width_GPU + x];
                          if (point_depth < min_depth) {
                                   min_depth = point_depth;
                                   min_x = x;
                                   min_y = y;
                          }
                   }
              }
          }

      }


      float point_cam[3] = {0};
      point_cam[0] =  (min_x - cam_K_GPU[2])*min_depth/cam_K_GPU[0];
      point_cam[1] =  (min_y - cam_K_GPU[5])*min_depth/cam_K_GPU[4];
      point_cam[2] =  min_depth;

      float point_base[3] = {0};

      point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
      point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
      point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

      point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
      point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
      point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];


      //printf("vox_origin: %f,%f,%f\n",vox_origin[0],vox_origin[1],vox_origin[2]);
      // World coordinate to grid coordinate
      int z = (int)floor((point_base[0] - vox_origin[0])/ vox_unit_GPU);
      int x = (int)floor((point_base[1] - vox_origin[1])/ vox_unit_GPU);
      int y = (int)floor((point_base[2] - vox_origin[2])/ vox_unit_GPU);
      //printf("point_base: %f,%f,%f, %d,%d,%d, %d,%d,%d \n",point_base[0],point_base[1],point_base[2], z, x, y, vox_size[0],vox_size[1],vox_size[2]);

      // mark vox_out with 1.0
      if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
          int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
          vox_edges[vox_idx] = float(1.0);
      }
  }
}







__global__
void SquaredDistanceTransform(float *cam_pose, int *vox_size,  float *vox_origin, float *depth_data, float *vox_grid,
                              float *vox_tsdf, float *parameters_GPU) {

    float *cam_K_GPU = parameters_GPU;
    int frame_width_GPU= int(parameters_GPU[9]), frame_height_GPU= int(parameters_GPU[10]);
    float vox_unit_GPU= parameters_GPU[11], vox_margin_GPU = parameters_GPU[12];

    int search_region = (int)round(vox_margin_GPU/vox_unit_GPU);

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]){
      return;
    }

    if (vox_grid[vox_idx] >0 ){
       vox_tsdf[vox_idx] = 0;
       return;
    }

    int z = (vox_idx / ( vox_size[0] * vox_size[1]))%vox_size[2] ;
    int y = (vox_idx / vox_size[0]) % vox_size[1];
    int x = vox_idx % vox_size[0];

    // Get point in world coordinates XYZ -> YZX
    float point_base[3] = {0};
    point_base[0] = float(z) * vox_unit_GPU + vox_origin[0];
    point_base[1] = float(x) * vox_unit_GPU + vox_origin[1];
    point_base[2] = float(y) * vox_unit_GPU + vox_origin[2];

    // Encode height from floor ??? check later

    // Get point in current camera coordinates
    float point_cam[3] = {0};
    point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
    point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
    point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
    point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
    point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
    point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
    if (point_cam[2] <= 0)
      return;

    // Project point to 2D
    int pixel_x = roundf(cam_K_GPU[0] * (point_cam[0] / point_cam[2]) + cam_K_GPU[2]);
    int pixel_y = roundf(cam_K_GPU[4] * (point_cam[1] / point_cam[2]) + cam_K_GPU[5]);
    if (pixel_x < 0 || pixel_x >= frame_width_GPU || pixel_y < 0 || pixel_y >= frame_height_GPU){ // outside FOV
      //vox_tsdf[vox_idx] = GPUCompute2StorageT(-1.0);
      vox_tsdf[vox_idx] = 2000;
      return;
    }

    // Get depth
    float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];
    if (point_depth < float(0.5f) || point_depth > float(8.0f))
    {
      vox_tsdf[vox_idx] = 1;
      return;
    }
    if (roundf(point_depth) == 0){ // mising depth
      vox_tsdf[vox_idx] = -1.0;
      return;
    }

    // Get depth difference
    float point_dist = (point_depth - point_cam[2]) * sqrtf(1 + powf((point_cam[0] / point_cam[2]), 2) + powf((point_cam[1] / point_cam[2]), 2));
    //float sign = point_dist/abs(point_dist);

    float sign;
    if (abs(point_depth - point_cam[2]) < 0.0001){
        sign = 1; // avoid NaN
    }else{
        sign = (point_depth - point_cam[2])/abs(point_depth - point_cam[2]);
    }
    vox_tsdf[vox_idx] = sign;

    int radius=search_region; // out -> in
    int found = 0;
    //fixed y planes
    int iiy = max(0,y-radius);
    for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiy = min(y+radius,vox_size[1]);
    for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed x planes
    int iix = max(0,x-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iix = min(x+radius,vox_size[0]);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    //fixed z planes
    int iiz = max(0,z-radius);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }
    iiz = min(z+radius,vox_size[2]);
    for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
            if (vox_grid[iidx] > 0){
              found = 1;
              float xd = abs(x - iix);
              float yd = abs(y - iiy);
              float zd = abs(z - iiz);
              float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
              if (tsdf_value < abs(vox_tsdf[vox_idx])){
                vox_tsdf[vox_idx] = tsdf_value*sign;
              }
            }
        }
    }


    if (found == 0)
        return;

    radius=1; // in -> out
    found = 0;
    while (radius < search_region) {
        //fixed y planes
        int iiy = max(0,y-radius);
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiy = min(y+radius,vox_size[1]);
        for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed x planes
        int iix = max(0,x-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iix = min(x+radius,vox_size[0]);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iiz = max(0,z-radius); iiz < min((int)vox_size[2],z+radius+1); iiz++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        //fixed z planes
        int iiz = max(0,z-radius);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        iiz = min(z+radius,vox_size[2]);
        for (int iiy = max(0,y-radius); iiy < min((int)vox_size[1],y+radius+1); iiy++){
            for (int iix = max(0,x-radius); iix < min((int)vox_size[0],x+radius+1); iix++){
                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0){
                  found = 1;
                  float xd = abs(x - iix);
                  float yd = abs(y - iiy);
                  float zd = abs(z - iiz);
                  float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/search_region;
                  if (tsdf_value < abs(vox_tsdf[vox_idx])){
                    vox_tsdf[vox_idx] = tsdf_value*sign;
                  }
                }
            }
        }
        if (found == 1)
          return;

        radius++;

    }
}



void ComputeTSDF_edges_CPP(int *vox_size,  unsigned char *depth_image, unsigned char *edges_image,
                     float *vox_grid, float *vox_tsdf, float *vox_edges, float *tsdf_edges) {

  //cout << "\nComputeTSDF_CPP\n";
  clock_tick t1 = start_timer();


  int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

  float *depth_data_GPU, *vox_grid_GPU, *vox_tsdf_GPU, *vox_edges_GPU, *tsdf_edges_GPU;
  unsigned char *edges_data_GPU;
  int *vox_size_GPU;

  cudaMalloc(&vox_size_GPU, 3 * sizeof(int));

  cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(float));
  //cudaMalloc(&edges_data_GPU, frame_height * frame_width * sizeof(float));
  cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(float));
  //cudaMalloc(&vox_tsdf_GPU, num_voxels * sizeof(float));
  //cudaMalloc(&vox_edges_GPU, num_voxels * sizeof(float));
  //cudaMalloc(&tsdf_edges_GPU, num_voxels * sizeof(float));
  //cudaMemset(vox_tsdf_GPU, 0, num_voxels * sizeof(float));
  //cudaMemset(tsdf_edges_GPU, 0, num_voxels * sizeof(float));
  //cudaMemset(vox_edges_GPU, 0, num_voxels * sizeof(float));

  cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(depth_data_GPU, depth_image, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(edges_data_GPU, edges_image, frame_height * frame_width * 1, cudaMemcpyHostToDevice);


  end_timer(t1, "Prepare duration");


  t1 = start_timer();
  // from depth map to binaray voxel representation
  depth2Grid<<<frame_width,frame_height>>>(vox_size_GPU,  depth_data_GPU,
                                           vox_grid_GPU, parameters_GPU);
  cudaDeviceSynchronize();


  depth2Grid_edges<<<frame_width,frame_height>>>(cam_pose_GPU, vox_size_GPU,  vox_origin_GPU, depth_data_GPU, edges_data_GPU,
                                           vox_edges_GPU, parameters_GPU);
  cudaDeviceSynchronize();
  end_timer(t1,"depth2Grid duration");

  // distance transform
  int BLOCK_NUM = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

  t1 = start_timer();

  SquaredDistanceTransform<<< BLOCK_NUM, NUM_THREADS >>>(cam_pose_GPU, vox_size_GPU,  vox_origin_GPU, depth_data_GPU, vox_grid_GPU, vox_tsdf_GPU, parameters_GPU);
  cudaDeviceSynchronize();

  SquaredDistanceTransform<<< BLOCK_NUM, NUM_THREADS >>>(cam_pose_GPU, vox_size_GPU,  vox_origin_GPU, depth_data_GPU, vox_edges_GPU, tsdf_edges_GPU, parameters_GPU);
  cudaDeviceSynchronize();

  end_timer(t1,"SquaredDistanceTransform duration");

  t1 = start_timer();
  cudaMemcpy(vox_grid, vox_grid_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_edges, vox_edges_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(vox_tsdf, vox_tsdf_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(tsdf_edges, tsdf_edges_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);




  cudaFree(vox_size_GPU);
  cudaFree(depth_data_GPU);

  //cudaFree(edges_data_GPU);
  cudaFree(vox_grid_GPU);
  //cudaFree(vox_edges_GPU);
  //cudaFree(vox_tsdf_GPU);
  //cudaFree(tsdf_edges_GPU);

  end_timer(t1,"closeup duration");

}

void FlipTSDF_CPP( int *vox_size, float *vox_tsdf){

  clock_tick t1 = start_timer();

  for (int vox_idx=0; vox_idx< vox_size[0]*vox_size[1]*vox_size[2]; vox_idx++) {

      float value = float(vox_tsdf[vox_idx]);
      if (value > 1)
          value =1;


      float sign;
      if (abs(value) < 0.001)
        sign = 1;
      else
        sign = value/abs(value);

      vox_tsdf[vox_idx] = sign*(max(0.001,(1.0-abs(value))));
  }
  end_timer(t1,"FlipTSDF");
}

void ProcessEdges_CPP(int *vox_size,
                 int out_scale,
                 unsigned char *depth_data,
                 unsigned char *edges_data,
                 float *vox_tsdf,
                 float *vox_edges,
                 float *tsdf_edges,
                 float *vox_limits,
                 int *segmentation_label_downscale) {


    int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    int *segmentation_label_fullscale;
    segmentation_label_fullscale= (int *) malloc((vox_size[0]*vox_size[1]*vox_size[2]) * sizeof(int));

    float *vox_grid = new float[num_voxels];
    memset(vox_grid, 0, num_voxels * sizeof(float));

    ComputeTSDF_edges_CPP(vox_size,  depth_data, edges_data, vox_grid, vox_tsdf, vox_edges, tsdf_edges);

  DownsampleLabel_CPP(vox_size,
                            out_scale,
                            segmentation_label_fullscale,
                            vox_tsdf,
                            segmentation_label_downscale,
                            vox_weights,vox_vol,vox_grid);


    FlipTSDF_CPP( vox_size, vox_tsdf);
    FlipTSDF_CPP( vox_size, tsdf_edges);

    delete [] vox_grid;

    free(segmentation_label_fullscale);
    //FlipTSDF_CPP( out_vox_size, vox_vol);


}
*/


