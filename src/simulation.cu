/* *
 * Simple Molecular Dynamics for CUDAfdsa
 */
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>

class CudaTimer {
	cudaEvent_t start_event;
	cudaEvent_t end_event;
	float time;
public:

	void start() {
		cudaEventCreate(&start_event);
		cudaEventCreate(&end_event);
		cudaEventRecord(start_event, 0);}
	void stop() {
		cudaEventRecord(end_event,0);
		cudaEventSynchronize(end_event);}
	float elapsed() { cudaEventElapsedTime(&time, start_event, end_event); return time;}
	void reset() {
		cudaEventDestroy(start_event);
		cudaEventDestroy(end_event);
		time = 0.0f;
	}

};

#define EPS2 0.001f;

#define CHECK_ERROR(func) do { \
		if( func!=cudaSuccess )\
		printf("Cuda error in %s: %s\n",#func, cudaGetErrorString(func)); \
} while(0)


typedef   float3 (*atomicForceFunction)(float4,float4,float3);

__device__ float3 graviForce(float4 bi, float4 bj,float3 ai) {

	float3 r;
	// r_ij  [3 FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	// distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
	// invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = 1.0f/sqrtf(distSixth);
	// s = m_j * invDistCube [1 FLOP]
	float s = bj.w * invDistCube;
	// a_i =  a_i + s * r_ij [6 FLOPS]
	ai.x += r.x * s;
	ai.y += r.y * s;
	ai.z += r.z * s;
	return ai;

}


__device__ float3 calcAccByTile(float4 selfPosition,float3 acc) {
	extern __shared__ float4 positions [];
	for(int i = 0;i < blockDim.x;++i){
		acc = graviForce(selfPosition,positions[i],acc);
	}
	return acc;
}

__global__ void calcForces(void* X, void* A,int N) {
	extern __shared__ float4 positions [];
	float4* xf  = (float4*)X;
	float4* accf = (float4*)A;
	float4 selfPosition;


	int i, tile;
	float3 acc = {0.0f, 0.0f, 0.0f};
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	selfPosition = xf[gtid];

	for(i = 0,tile = 0; i < N; i += blockDim.x, ++tile) {
		int id = tile * blockDim.x + threadIdx.x;
		positions[threadIdx.x] = xf[id];
		__syncthreads();
		acc = calcAccByTile(selfPosition,acc);
		__syncthreads();
	}

	float4 resultAcc={acc.x,acc.y,acc.z,0.0f};
	accf[gtid] = resultAcc;


}


__global__ void velocityVerletIntegrator_I(void* const positions,
		void* const accelerations,
		void* const velocities,
		const float delta,
		const int N) {


	int id = blockDim.x * blockIdx.x + threadIdx.x;
	float4* x = (float4*)positions;
	float4* acc = (float4*)accelerations;
	float3* vel = (float3*)velocities;
	float deltaSq = delta * delta;
	if(id < N) {

		x[id].x =x[id].x + vel[id].x * delta + 0.5f * acc[id].x * deltaSq;
		x[id].y =x[id].y + vel[id].y * delta + 0.5f * acc[id].y * deltaSq;
		x[id].z =x[id].z + vel[id].z * delta + 0.5f * acc[id].z * deltaSq;

		/*v(0.5 * delta) - it is sub-step!
		 *NOTE: after this kernel finished work need to execute velocityVerletIntegrator_II
		 */

		vel[id].x = vel[id].x + 0.5f * acc[id].x * delta;
		vel[id].y = vel[id].y + 0.5f * acc[id].y * delta;
		vel[id].z = vel[id].z + 0.5f * acc[id].z * delta;
	}


}

__global__ void velocityVerletIntegrator_II(void* const velocities,
                                            void* const accelerations, 
                                            const float delta,
                                            const int N) {

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	float4* acc = (float4*)accelerations;
	float3* vel = (float3*)velocities;
	if(id < N) {
		vel[id].x = vel[id].x + 0.5f * acc[id].x * delta;
		vel[id].y = vel[id].y + 0.5f * acc[id].y * delta;
		vel[id].z = vel[id].z + 0.5f * acc[id].z * delta;
	}

}

static const int P = 256;




void molecularDynamicsWithVerlet(void* d_positions,
		void* d_velocities,
		void* d_accelerations,
		int N,
		const float delta,
		long int steps) {


	dim3 gridSize((N-1)/P +1,1,1);
	dim3 blockSize(P,1,1);
	CudaTimer timer;
	calcForces<<<gridSize,blockSize,P * sizeof(float4)>>>((void*)d_positions,(void*)d_accelerations,N);
	cudaThreadSynchronize();
	for(int i = 0; i < steps; ++i) {
		timer.start();
		velocityVerletIntegrator_I<<<gridSize,blockSize>>>(d_positions, d_accelerations, d_velocities, delta, N);
		cudaDeviceSynchronize();
		calcForces<<<gridSize,blockSize,P * sizeof(float4)>>>((void*)d_positions,(void*)d_accelerations,N);
		cudaThreadSynchronize();
		velocityVerletIntegrator_II<<<gridSize,blockSize>>>(d_accelerations, d_velocities, delta, N);
		cudaDeviceSynchronize();
		timer.stop();
	//	printf("Step elapsed time: %fms\n",timer.elapsed());
	}

}

void ringMassesCreator(float4* masses,float r,int N) {
	float angle = 2*3.1415 / N;
	for(int i = 0;i< N; i++) {
		masses[i].w = 1.0f;
		masses[i].x = r * cos(i * angle) + r;
		masses[i].y = r * sin(i * angle) + r;
		masses[i].z = 0.0f;
	}
}


const  int N_EL = 4096
void printToFile(float4* data, float time,int N,FILE* f) {
	for(int i =0;i< N;++i){
		fprintf(f," %f %f %f %f \n",time, data[i].x,data[i].y,data[i].z);
	}
}



int main(void) {
	void* h_masses = malloc(sizeof(float4) * N_EL);
	void* h_velocities = malloc(sizeof(float3) * N_EL);

	void* d_masses;
	void* d_velocities;
	void* d_accel;
	ringMassesCreator((float4*)h_masses,1000.0f,N_EL);


	CHECK_ERROR(cudaMalloc((void**)&d_masses,sizeof(float4) * N_EL));
	CHECK_ERROR(cudaMalloc((void**)&d_velocities,sizeof(float3) * N_EL));
	CHECK_ERROR(cudaMalloc((void**)&d_accel,sizeof(float4) * N_EL));
	CHECK_ERROR(cudaMemcpy(d_masses,h_masses,sizeof(float4) * N_EL, cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(d_velocities,h_velocities,sizeof(float3) * N_EL, cudaMemcpyHostToDevice));




	float simul_time = 0.0f;
	std::stringstream ss;
	for(int i = 0;i < 1000; ++i) {
		molecularDynamicsWithVerlet(d_masses, d_velocities,d_accel,N_EL,0.000001f, 1000);
		simul_time += 1.0f;
		CHECK_ERROR(cudaMemcpy(h_masses,d_masses,sizeof(float4) * N_EL, cudaMemcpyDeviceToHost));

		ss<<i;
		std::string fileName = "result" + ss.str() + ".dat";
		FILE* f = fopen(fileName.c_str(),"w");
		printToFile((float4*)h_masses,simul_time,N_EL,f);
		fclose(f);
		ss.clear();
		ss.str("");
	}

	free(h_masses);
	free(h_velocities);
	cudaFree(d_masses);
	cudaFree(d_velocities);
	cudaFree(d_accel);
	return 0;
}
