#include <include/device_manage.h>
#include <stdio.h>

static void HandleError( cudaError_t err,
	const char *file,
	int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
			file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

DeviceManager::DeviceManager() {
	this->host_data.cam_ptr = nullptr;
	this->host_data.scene_ptr = nullptr;
	this->host_data.img_ptr = nullptr;
	this->host_data.width = 0;
	this->host_data.height = 0;
	this->host_data.img_size = 0;

	this->dev_data.cam_ptr = nullptr;
	this->dev_data.scene_ptr = nullptr;
	this->dev_data.img_ptr = nullptr;
	this->dev_data.width = 0;
	this->dev_data.height = 0;
	this->dev_data.img_size = 0;
	
	this->isInit = false;
	this->bufferObj = 0;
	this->cudaObj = nullptr;
}

DeviceManager::~DeviceManager() {
	//TODO::cudaFree();
	if (!this->isInit) return;
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(this->cudaObj), NULL));
	HANDLE_ERROR(cudaFree(this->dev_data.cam_ptr));
	HANDLE_ERROR(cudaFree(this->dev_data.scene_ptr));
	HANDLE_ERROR(cudaFree(this->dev_data.img_ptr));
}

void DeviceManager::ToDevice() {
	//GPU内存分配
	//相机
	HANDLE_ERROR(cudaMalloc((void**)&(this->dev_data.cam_ptr), 
							sizeof(this->host_data.cam_ptr)));
	HANDLE_ERROR(cudaMemcpy(this->dev_data.cam_ptr, this->host_data.cam_ptr,
							sizeof(this->host_data.cam_ptr),
							cudaMemcpyHostToDevice));
	//场景
	HANDLE_ERROR(cudaMalloc((void**)&(this->dev_data.scene_ptr), 
							sizeof(this->host_data.scene_ptr)));
	HANDLE_ERROR(cudaMemcpy(this->dev_data.scene_ptr, this->host_data.scene_ptr,
							sizeof(this->host_data.scene_ptr),
							cudaMemcpyHostToDevice));

	//渲染图片
	//HANDLE_ERROR(cudaMalloc((void**)&(this->dev_data.img_ptr), this->dev_data.img_size));
	
	this->isInit = true;
}

void DeviceManager::ToHost() {
	HANDLE_ERROR(cudaMemcpy(this->host_data.img_ptr, this->dev_data.img_ptr,
							this->host_data.img_size,
							cudaMemcpyDeviceToHost));
}

//打印设备信息
void DeviceManager::PrintDeviceInfo() {
	auto device_count = 0;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0)
	{
		printf("没有支持CUDA的设备!\n");
		return;
	}
	for (auto dev = 0; dev < device_count; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp device_prop{};
		cudaGetDeviceProperties(&device_prop, dev);
		printf("设备 %d: \"%s\"\n", dev, device_prop.name);
		char msg[256];
		sprintf_s(msg, sizeof(msg),
			"global memory大小:        %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
			static_cast<unsigned long long>(device_prop.totalGlobalMem));
		printf("%s", msg);
		printf("SM数:                    %2d \n每SM CUDA核心数:           %3d \n总CUDA核心数:             %d \n",
			device_prop.multiProcessorCount,
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor),
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor) *
			device_prop.multiProcessorCount);
		printf("静态内存大小:             %zu bytes\n",
			device_prop.totalConstMem);
		printf("每block共享内存大小:      %zu bytes\n",
			device_prop.sharedMemPerBlock);
		printf("每block寄存器数:          %d\n",
			device_prop.regsPerBlock);
		printf("线程束大小:               %d\n",
			device_prop.warpSize);
		printf("每处理器最大线程数:       %d\n",
			device_prop.maxThreadsPerMultiProcessor);
		printf("每block最大线程数:        %d\n",
			device_prop.maxThreadsPerBlock);
		printf("线程块最大维度大小        (%d, %d, %d)\n",
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
			device_prop.maxThreadsDim[2]);
		printf("网格最大维度大小          (%d, %d, %d)\n",
			device_prop.maxGridSize[0], device_prop.maxGridSize[1],
			device_prop.maxGridSize[2]);
		printf("\n");
		HANDLE_ERROR(cudaChooseDevice(&dev, &device_prop));
		HANDLE_ERROR(cudaGLSetGLDevice(dev));
	}
	printf("************设备信息打印完毕************\n\n");
}

void DeviceManager::BindOpenGL() {
  glGenBuffers(1, &this->bufferObj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, this->bufferObj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, this->dev_data.img_size, NULL, GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&this->cudaObj, this->bufferObj, cudaGraphicsMapFlagsNone));
  HANDLE_ERROR(cudaGraphicsMapResources(1, &this->cudaObj, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&this->dev_data.img_ptr, &this->dev_data.img_size, this->cudaObj));

}

unsigned char* DeviceManager::get_img() {
	return this->dev_data.img_ptr;
}