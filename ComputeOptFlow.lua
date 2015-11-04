local OptFlow = torch.class('of.BroxFlowGPU')

function OptFlow:__init()
   self.bufferLeft = torch.CudaTensor()
   self.bufferRight = torch.CudaTensor()
   self.bufferBatch = torch.CudaTensor()
   self.bufferOutput=torch.CudaTensor()
end

-- that function will return a non-contiguous cudatensor where the first element on pitchedDimension is aligned with pitchValue/elementSize
function pitchTensor(pitchedTensor, tensor, pitchedDimension, pitchValue, elementSize)
   assert(tensor:isContiguous())
   assert(pitchedDimension >= 2, "first element on dimension 1 is always aligned...")
   local pitchValue = pitchValue or 512 
   local floatSize = floatSize or 4
   local pitch = pitchValue/floatSize

   local currentStride = tensor:stride(pitchedDimension-1)
   local currentSize = tensor:nElement()/currentStride
   local newStride = torch.floor(( currentStride + pitch - 1 ) / pitch ) * pitch

   pitchedTensor:resize(currentSize, currentStride)

   local strides = tensor:stride()
   for i=1,pitchedDimension-1 do
      strides[i] = strides[i] * newStride / currentStride
   end
   --strides[pitchedDimension-1] = newStride
   local sizes = tensor:size()

   pitchedTensor:set(pitchedTensor:storage(), 1, sizes, strides)
   pitchedTensor:copy(tensor)
   return pitchedTensor
end

function OptFlow:computeOptFlowSingle(output, left, right)
   assert(left:isSameSizeAs(right))
   assert(left:nDimension()==2)

   pitchTensor(self.bufferLeft, left, 2)
   pitchTensor(self.bufferRight, right, 2)
   output:resize(left:size(1), left:size(2), 2)
   pitchTensor(self.bufferOutput, output, 2)

   torch.CudaTensor.of.computeOptFlow(self.bufferLeft, self.bufferRight, self.bufferOutput)
   output:copy(self.bufferOutput)
   return output
end

function OptFlow:computeOptFlowBatch(output, input)
   assert(input:nDimension()==3)

   pitchTensor(self.bufferBatch, input, 3)
   output:resize(input:size(1)-1, input:size(2), input:size(3), 2):zero()
   pitchTensor(self.bufferOutput, output[1], 2) --see that call to pitchedTensor:storage() ? it has to be output[1] .

   for i=1, input:size(1)-1 do
      torch.CudaTensor.of.computeOptFlow(self.bufferBatch[i], self.bufferBatch[i+1], self.bufferOutput)
      output:select(1,i):copy(self.bufferOutput)
   end

   return output
end

function OptFlow:computeOptFlowBatch2(output, input)
   assert(input:nDimension()==3)

   pitchTensor(self.bufferBatch, input, 3)
   output:resize(input:size(1)-1, input:size(2), input:size(3), 2):zero()
   pitchTensor(self.bufferOutput, output, 3) --see that call to pitchedTensor:storage() ? it has to be output[1] .

   torch.CudaTensor.of.computeOptFlowBatch(self.bufferBatch, self.bufferOutput)

   output:copy(self.bufferOutput)

   return output
end



