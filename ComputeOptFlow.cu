#pragma once
#include <opencv2/core.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

extern "C" {
#include <TH/TH.h>
}
#include <THC/THC.h>
#include "utils.h"

cv::cuda::GpuMat toGpuMat(THCState* state, THCudaTensor* tensor)
{
   int tensorDims = THCudaTensor_nDimension(state, tensor);
   int height = THCudaTensor_size(state, tensor, 0);
   int width = THCudaTensor_size(state, tensor, 1);
   int step = THCudaTensor_stride(state, tensor, 0);
   int type;

   if (tensorDims == 2) {
        // If such tensor is passed, assume that it is single-channel:
        type = CV_32F;
    } else if (tensorDims == 3) {
        // Otherwise depend on the 3rd dimension:
        int nChannels = THCudaTensor_size(state, tensor, 2);
        //printf("nChannels : %d\n", nChannels);
        switch(nChannels){
         case 1:
            type = CV_32FC1;
            //printf("CV_32FC1\n");
            break;
         case 2:
            type = CV_32FC2;
            //printf("CV_32FC2\n");
            break;
         case 3:
            type = CV_32FC3;
            //printf("CV_32FC3\n");
            break;
         case 4:
            type = CV_32FC4;
            //printf("CV_32FC4\n");
            break;
         default:
            THError("bad number of channels in toGpuMat, aborting");
        }
    }
    cv::cuda::GpuMat t(height, width, type, THCudaTensor_data(state, tensor), step*4);
    return t;
}

void computeOptFlow(THCState* state, THCudaTensor* image_a, THCudaTensor* image_b, THCudaTensor* flow, cv::Ptr<cv::cuda::DenseOpticalFlow> flowAlg, cv::cuda::Stream& s = cv::cuda::Stream::Null())
{
   cv::cuda::GpuMat img_a = toGpuMat(state, image_a);
   cv::cuda::GpuMat img_b = toGpuMat(state, image_b);
   cv::cuda::GpuMat flowout = toGpuMat(state, flow);

   flowAlg->calc(img_a,
               img_b,
               flowout,
               s);
}




// assume everything is pitched now and with correct sizes
static int cuof_computeOptFlow(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *input_a = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
   THCudaTensor *input_b = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
   THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

   cv::Ptr<cv::cuda::BroxOpticalFlow> flowAlg = cv::cuda::BroxOpticalFlow::create();
   computeOptFlow(state, input_a, input_b, output, flowAlg);

   // check for errors
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
     printf("error in computeOptFlow: %s\n", cudaGetErrorString(err));
     THError("aborting");
   }
   return 1;
}

static int cuof_computeOptFlowBatch(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *inputs = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
   THCudaTensor *outputs = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

   THCudaTensor *input_a = THCudaTensor_new(state);
   THCudaTensor *input_b = THCudaTensor_new(state);
   THCudaTensor *output_n = THCudaTensor_new(state);

   cv::Ptr<cv::cuda::BroxOpticalFlow> flowAlg = cv::cuda::BroxOpticalFlow::create();

   int batchSize = THCudaTensor_size(state, inputs, 0);
   int elt;

   for(elt=0; elt<batchSize-1; elt++)
   {
      THCudaTensor_select(state, input_a, inputs, 0, elt);
      THCudaTensor_select(state, input_b, inputs, 0, elt+1);
      THCudaTensor_select(state, output_n, outputs, 0, elt);
      computeOptFlow(state, input_a, input_b, output_n, flowAlg);
   }

   // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in computeOptFlowBatch: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static int cuof_computeOptFlowBatchStreamed(lua_State *L)
{
   THCState *state = getCutorchState(L);
   THCudaTensor *inputs = (THCudaTensor *)luaT_checkudata(L, 1, "torch.CudaTensor");
   THCudaTensor *outputs = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");

   THCudaTensor *input_a = THCudaTensor_new(state);
   THCudaTensor *input_b = THCudaTensor_new(state);
   THCudaTensor *output_n = THCudaTensor_new(state);

   cv::Ptr<cv::cuda::BroxOpticalFlow> flowAlg = cv::cuda::BroxOpticalFlow::create();

   int batchSize = THCudaTensor_size(state, inputs, 0);
   int elt;

   const int nStreams = 8;
   cv::cuda::Stream::Stream s[nStreams];

   for(elt=0; elt<batchSize-1; elt++)
   {
      THCudaTensor_select(state, input_a, inputs, 0, elt);
      THCudaTensor_select(state, input_b, inputs, 0, elt+1);
      THCudaTensor_select(state, output_n, outputs, 0, elt);
      computeOptFlow(state, input_a, input_b, output_n, flowAlg, s[elt % nStreams]);
   }

   // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in computeOptFlowBatch: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cuof_ComputeOptFlow__ [] = {
  {"computeOptFlow", cuof_computeOptFlow},
  {"computeOptFlowBatch", cuof_computeOptFlowBatch},
  {"computeOptFlowBatchStreamed", cuof_computeOptFlowBatchStreamed},
  {NULL, NULL}
};

static void cuof_ComputeOptFlow_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cuof_ComputeOptFlow__, "of");
  lua_pop(L,1);
}
