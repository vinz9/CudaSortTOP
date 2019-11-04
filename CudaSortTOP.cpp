/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

#include "CudaSortTOP.h"

#include <assert.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <string.h>
#endif
#include <cstdio>
//#include "cuda_runtime.h"


// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{
DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CUDA;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Cudasort");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("CUDA Sort");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("CDA");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Vincent Houze");
	//info->customOPInfo.authorEmail->setString("email@email.com");

	// This TOP works with 0 or 1 inputs connected
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

    // Note we can't do any OpenGL work during instantiation

	return new CudaSortTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

    // We do some OpenGL teardown on destruction, so ask the TOP_Context
    // to set up our OpenGL context

	delete (CudaSortTOP*)instance;

}

};


CudaSortTOP::CudaSortTOP(const OP_NodeInfo* info, TOP_Context *context)
: myNodeInfo(info), myExecuteCount(0), myError(nullptr)
{

	previousWidth = -1;
	previousHeight = -1;

}

CudaSortTOP::~CudaSortTOP()
{

}

void
CudaSortTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved)
{
	// Setting cookEveryFrame to true causes the TOP to cook every frame even
	// if none of its inputs/parameters are changing. Set it to false if it
    // only needs to cook when inputs/parameters change.
	ginfo->cookEveryFrame = false;
}

bool
CudaSortTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs *inputs, void* reserved)
{
	
	format->redChannel = true;
	format->greenChannel = false;
	format->blueChannel = false;
	format->alphaChannel = false;

	format->bitsPerChannel = 32;
	format->floatPrecision = true;

	return true;
}

void CudaSortTOP::initBuffers(int width, int height) {


	if (previousWidth != -1) {
		deleteBuffers();
	}
	else {
		printf("***FIRST_INIT***\n");
	}

	printf("***CHANGE_RESOLUTION***\n");


	cudaMalloc((void**)&m_sortKeys, sizeof(float)*width*height);
	cudaMalloc((void**)&m_indices, sizeof(uint)*width*height);

	//cudaMalloc((void**)&out_data, sizeof(float4)*width*height);

	//init = 1;
	previousWidth = width;
	previousHeight = height;

}

void CudaSortTOP::deleteBuffers() {

	cudaFree(m_sortKeys);
	cudaFree(m_indices);
	//cudaFree(out_data);

}

//extern cudaError_t doCUDAOperation(int width, int height, cudaArray *input, cudaArray *output);

////////////////////////////////////////////////////////////////////////////////
extern "C" void
sortParticles(float *sortKeys, uint *indices, uint numParticles);
//launch_cudaProcess(dim3 grid, dim3 block, cudaArray *g_data_array, float4 *g_odata, int imgw, int imgh, float incr);

extern "C" void
launch_arrayToData(dim3 grid, dim3 block, cudaArray *g_data_array, float *g_odata, uint* keys, int imgw, int imgh);

extern "C" void
launch_dataToTex(dim3 grid, dim3 block, uint* in_indices, float4 *g_odata, int imgw, int imgh);

extern "C" void
launch_dataToArray(dim3 grid, dim3 block, uint *mIndices, cudaArray *output, int imgw, int imgh);

void
CudaSortTOP::execute(TOP_OutputFormatSpecs* outputFormat ,
							const OP_Inputs* inputs,
							TOP_Context* context,
							void* reserved)
{
	myExecuteCount++;


    int width = outputFormat->width;
    int height = outputFormat->height;


	cudaArray *inputMem = nullptr;
	if (inputs->getNumInputs() > 0)
	{
		const OP_TOPInput* topInput = inputs->getInputTOP(0);

		//const GLint format = outputFormat->pixelFormat;

		/*if (outputFormat->redBits != 32 ||
			outputFormat->greenBits != 0 ||
			outputFormat->blueBits != 0 ||
			outputFormat->alphaBits != 0)
		{
			myError = "TOP format should be set to 32-bit float (Mono)";
			return;
		}*/

		if (topInput->cudaInput == nullptr)
		{
			myError = "CUDA memory for input TOP was not mapped correctly.";
			return;
		}

		//matching texture<float4, cudaTextureType2D, cudaReadModeElementType> inTex;
		if (topInput->pixelFormat != GL_RGBA16F && topInput->pixelFormat != GL_RGBA32F) {
			myError = "Input Texture must be 16-bit float (RGBA) or 32-bit float (RGBA).";
			return;
		}

		inputMem = topInput->cudaInput;

		if (outputFormat->width != previousWidth || outputFormat->height != previousHeight) {
			initBuffers(outputFormat->width, outputFormat->height);
		}


		int nThreads = 16;

		// calculate grid size
		dim3 block(nThreads, nThreads, 1);
		int width = outputFormat->width;
		int height = outputFormat->height;

		//dim3 grid(outputFormat->width / block.x, outputFormat->width / block.y, 1);
		dim3 grid;
		grid.x = width / nThreads + (!(width%nThreads) ? 0 : 1);
		grid.y = height / nThreads + (!(height%nThreads) ? 0 : 1);
		grid.z = 1;

		// execute CUDA kernel
		launch_arrayToData(grid, block, inputMem, m_sortKeys, m_indices, outputFormat->width, outputFormat->height);

		int numKeys = outputFormat->width * outputFormat->height;

		sortParticles(m_sortKeys, m_indices, numKeys);

		//launch_dataToTex(grid, block, m_indices, out_data, outputFormat->width, outputFormat->height);
		//cudaMemcpyToArray(outputFormat->cudaOutput[0], 0, 0, out_data, outputFormat->width*outputFormat->height * sizeof(float4), cudaMemcpyDeviceToDevice);

		launch_dataToArray(grid, block, m_indices, outputFormat->cudaOutput[0], outputFormat->width, outputFormat->height);
	}
}

int32_t
CudaSortTOP::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 1;
}

void
CudaSortTOP::getInfoCHOPChan(int32_t index,
						OP_InfoCHOPChan* chan,
						void* reserved)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}
}

bool		
CudaSortTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 1;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
CudaSortTOP::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
        strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
        snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
CudaSortTOP::getErrorString(OP_String *error, void* reserved)
{
    error->setString(myError);
}

void
CudaSortTOP::setupParameters(OP_ParameterManager* manager, void* reserved)
{


}

void
CudaSortTOP::pulsePressed(const char* name, void* reserved)
{
	
}
