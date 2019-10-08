
#include "CudaSortTOP.h"

#include "thrust/device_ptr.h"
//#include "thrust/for_each.h"
//#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

//no performance difference if using float Mono input instead of float4 RGBA
//texture<float, cudaTextureType2D, cudaReadModeElementType> inTex;
//g_odata[offset] = tex2D(inTex, xc, yc);

texture<float4, cudaTextureType2D, cudaReadModeElementType> inTex;
surface<void, cudaSurfaceType2D> outputSurface;

__device__ float4 operator+(const float4 & a, const float4 & b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void
arrayToData(float *g_odata, uint* keys, int imgw, int imgh)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * imgw;

	if (x < imgw && y < imgh) {

		float xc = x + 0.5;
		float yc = y + 0.5;


		g_odata[offset] = tex2D(inTex, xc, yc).x;
		keys[offset] = offset;
	}


}

__global__ void
dataToTex(uint* indices, float4 *g_odata, int imgw, int imgh)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * imgw;

	if (x < imgw && y < imgh) {

		float res = indices[offset];
		g_odata[offset] = make_float4(res, 0, 0, 1);
	}

}

//https://stackoverflow.com/questions/27741888/writing-to-a-floating-point-opengl-texture-in-cuda-via-a-surface
//cudaBoundaryModeClamp

__global__ void
dataToArray(uint* indices, int imgw, int imgh)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * imgw;

	if (x < imgw && y < imgh) {

		float res = indices[offset];
		//surf2Dwrite(make_float4(res, 0, 0, 1), outputSurface, (int)sizeof(float4)*x, y);
		surf2Dwrite(res, outputSurface, (int)sizeof(float)*x, y);
	}

}



extern "C" void
launch_arrayToData(dim3 grid, dim3 block, cudaArray *g_data_array, float *g_odata, uint* keys, int imgw, int imgh) {

	cudaCheck(cudaBindTextureToArray(inTex, g_data_array));

	struct cudaChannelFormatDesc desc;
	cudaCheck(cudaGetChannelDesc(&desc, g_data_array));


	arrayToData << < grid, block >> >(g_odata, keys, imgw, imgh);

	cudaCheck(cudaUnbindTexture(inTex));

}

extern "C" void
launch_dataToTex(dim3 grid, dim3 block, uint *mIndices, float4 *g_odata, int imgw, int imgh) {


	dataToTex << < grid, block >> >(mIndices, g_odata, imgw, imgh);


}

extern "C" void
launch_dataToArray(dim3 grid, dim3 block, uint *mIndices, cudaArray *output, int imgw, int imgh) {

	cudaCheck(cudaBindSurfaceToArray(outputSurface, output));
	dataToArray << < grid, block >> >(mIndices, imgw, imgh);


}

extern "C" void
sortParticles(float *sortKeys, uint *indices, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
		thrust::device_ptr<float>(sortKeys + numParticles),
		thrust::device_ptr<uint>(indices));
}