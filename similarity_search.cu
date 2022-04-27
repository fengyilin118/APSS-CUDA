#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include "sm_20_atomic_functions.h"
#include <iostream>
#include <fstream>
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define BLOCK_SIZE_NO_SHARED 128  //128,640

#define BLOCK_SIZE_INDEX 1024
#define BLOCK_SIZE_GENERATE_VAR 1024
#define BLOCK_SIZE_VERIFY_VAL 1024
#define BLOCK_SIZE_REST 512

const int max_dim = 198456;//
const float threhold=0.3;


 void checkCUDAError(const char *msg)
 {
     cudaError_t err = cudaGetLastError();
     if( cudaSuccess != err)
     {
         fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                   cudaGetErrorString( err) );
         exit(EXIT_FAILURE);
     }
 }


 __global__ void filter_candidate(int* dim, float* feature, int* tile_start,int* par_ptr,
		 	 	 	 	 	 	 	 	 	 	 	 	 	  float* col_max, int* flag,
		 	 	 	 	 	 	 	 	 	 	 	 	 	  int* flag_join_id, int* flag_par_id,
		 	 	 	 	 	 	 	 	 	 	 	 	 	  int* filter_block_start,int* par_sum_vec, int* left_num)
 {

	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;

	 int par_id=0;
	 float upper[10]; //该vector与对应par_i相交的最大相似性
	 char flag_c[10];//每个vector生成对应的flag,该flag目前是个四位二进制数,eg. 1011:1表示在对应的join中包含该vector,0表示已被提前过滤掉
	 int flag_ten=0; //flag的十进制表示
	 for(int i=0;i<10;i++)
	 {
		 upper[i]=0.0;
		 flag_c[i]='0';
	 }

	 if(b_id<filter_block_start[1]) //par0
		 par_id=0;
	 else if(b_id<filter_block_start[2])
		 par_id=1;
	 else if(b_id<filter_block_start[3])
		 par_id=2;
	 else if(b_id<filter_block_start[4])
		 par_id=3;
	 else if(b_id<filter_block_start[5])
		 par_id=4;

	int r_id = (b_id-filter_block_start[par_id]) * BLOCK_SIZE_INDEX + t_id; //vector在par0中的序号
	if(r_id<par_sum_vec[par_id])
	{
		int abs_id=par_id*10000+r_id;  //abs_id  10000 vectors/partition 如有变动需要修改!!!!!!

		int standard_size = (tile_start[par_ptr[par_id]+r_id/32 + 1] - tile_start[par_ptr[par_id]+r_id / 32])/32;
		int t_start = tile_start[par_ptr[par_id]+r_id / 32];

		for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
		{
			int d = dim[t_start + r_id % 32 + j * 32];
			float f = feature[t_start + r_id % 32 + j* 32];
			if(d!=-1)
			{
				for(int i=0;i<10;i++)
					upper[i]=upper[i]+f*col_max[i*max_dim+d];
			}
		}
		for(int i=0;i<10;i++)
			if(upper[i] > threhold)
			{
				atomicAdd(&left_num[par_id*10+i],1);
			}


		for(int i=0;i<4;i++)
		{
			int par_par_id=flag_par_id[par_id*4+i]; //与当前par join的par_id
			int join_id=flag_join_id[par_id*4+i]; //与当前par join的join id
			if(upper[par_par_id]>threhold)
				flag_c[join_id]='1';
		}
		for(int i=0;i<10;i++)
		{
			flag_ten=flag_ten+(flag_c[i]-'0')*powf(2.0,(float)i);
		}

		flag[abs_id]=flag_ten;

		}
 }

 __global__ void generate_index_ptr4(int* dim_A, int* tile_start_A, int* dim_B,  int* tile_start_B,
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	   int* dim_C,  int* tile_start_C, int* dim_D, int* tile_start_D,
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 int* index_A_dim_num, int* index_B_dim_num, int* index_C_dim_num, int* index_D_dim_num,
 	 	 	 	  	  	  	  	  	  	  	  	  	  	  	  	  	 int sum_vec_A, int sum_vec_B, int sum_vec_C, int sum_vec_D)
 {
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;
		if(t_id==0&&b_id==0)
			printf("---------------------------------\n");
 //   int block_num_A=sum_vec_A/BLOCK_SIZE_INDEX+1;
    int block_start_B=sum_vec_A/BLOCK_SIZE_INDEX+1;
    int block_start_C=sum_vec_A/BLOCK_SIZE_INDEX+1 +sum_vec_B/BLOCK_SIZE_INDEX+1 ;
    int block_start_D=sum_vec_A/BLOCK_SIZE_INDEX+1 +sum_vec_B/BLOCK_SIZE_INDEX+1 +sum_vec_C/BLOCK_SIZE_INDEX+1;
	if(b_id<block_start_B) //par A
	{
		int vec_id=b_id*BLOCK_SIZE_INDEX+t_id;
		if(vec_id<sum_vec_A)
		{

			int standard_size = (tile_start_A[vec_id/32 + 1] - tile_start_A[vec_id/ 32])/32;
			int t_start = tile_start_A[vec_id / 32];
			for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
			{
				int d = dim_A[t_start + vec_id % 32 + j * 32];

				if(d!=-1&&d>=4096)
				{

					atomicAdd(&index_A_dim_num[d],1);
				}
			}
		}
	}
	else if(b_id<block_start_C)
	{
		int vec_id = (b_id-block_start_B) * BLOCK_SIZE_INDEX + t_id; //vector在par_B中的序号
		if(vec_id<sum_vec_B)
		{

			int standard_size = (tile_start_B[vec_id/32 + 1] - tile_start_B[vec_id/32])/32;
			int t_start=tile_start_B[vec_id/32];
	for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
			{
				int d = dim_B[t_start + vec_id% 32 + j * 32];
				if(d!=-1&&d>=4096)
				{
					atomicAdd(&index_B_dim_num[d],1);
				}
			}
		}
	}
	else if(b_id<block_start_D)
	{
		int vec_id = (b_id-block_start_C) * BLOCK_SIZE_INDEX + t_id; //vector在par_C中的序号
		if(vec_id<sum_vec_C)
		{
			int standard_size = (tile_start_C[vec_id/32 + 1] - tile_start_C[vec_id/32])/32;
			int t_start=tile_start_C[vec_id/32];
			for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
			{
				int d = dim_C[t_start + vec_id% 32 + j * 32];

				if(d!=-1&&d>=4096)
				{
					atomicAdd(&index_C_dim_num[d],1);
				}
			}
		}
	}else
	{
		int vec_id = (b_id-block_start_D) * BLOCK_SIZE_INDEX + t_id; //vector在par_D中的序号
		if(vec_id<sum_vec_D)
		{
			int standard_size = (tile_start_D[vec_id/32 + 1] - tile_start_D[vec_id/32])/32;
			int t_start=tile_start_D[vec_id/32];
			for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
			{
				int d = dim_D[t_start + vec_id% 32 + j * 32];

				if(d!=-1&&d>=4096)
				{
					atomicAdd(&index_D_dim_num[d],1);
				}
			}
		}
	}
 }
 __global__ void generate_index_ptr1(int* dim_A, int* tile_start_A, int* index_A_dim_num, int sum_vec_A)
  {
 	 int b_id = blockIdx.x;
 	 int t_id = threadIdx.x;

  //   int block_num_A=sum_vec_A/BLOCK_SIZE_INDEX+1;
     int block_start_B=sum_vec_A/BLOCK_SIZE_INDEX+1;

 		int vec_id=b_id*BLOCK_SIZE_INDEX+t_id;
 		if(vec_id<sum_vec_A)
 		{

 			int standard_size = (tile_start_A[vec_id/32 + 1] - tile_start_A[vec_id/ 32])/32;
 			int t_start = tile_start_A[vec_id / 32];
 			for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
 			{
 				int d = dim_A[t_start + vec_id % 32 + j * 32];

 				if(d!=-1&&d>=4096)
 				{
 					atomicAdd(&index_A_dim_num[d],1);
 				}
 			}
 		}


  }

//同时为两个partition建立索引, 前一个为paritionA,后一个为partitionB
 __global__ void construct_Index2(int* dim_A, float* feature_A, int* tile_start_A,int* dim_B, float* feature_B, int* tile_start_B,
		 	 	 	 	 	 	 	 	 	 	 	 		   int* index_A_dim_ptr, int* index_A_id, float* index_A_val, int* index_B_dim_ptr, int* index_B_id, float* index_B_val,
		 	 	 	 	 	 	 	 	 	 	 	 	 	   int sum_vec_A, int sum_vec_B)
{
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;
		if(t_id==0&&b_id==0)
			printf("---------------construct------------------\n");
	  int block_start_B=sum_vec_A/BLOCK_SIZE_INDEX+1;

		if(b_id<block_start_B) //par A
		{

			int vec_id=b_id*BLOCK_SIZE_INDEX+t_id;
			if(vec_id<sum_vec_A)
			{

				int standard_size = (tile_start_A[vec_id/32 + 1] - tile_start_A[vec_id/ 32])/32;
				int t_start = tile_start_A[vec_id / 32];
				for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
				{
					int d = dim_A[t_start + vec_id % 32 + j * 32];
					float f=feature_A[t_start + vec_id% 32 + j * 32];
					if(d!=-1&&d>=4096)
					{
						int ptr = atomicAdd(&index_A_dim_ptr[d], 1);
						index_A_id[ptr]=vec_id;
						index_A_val[ptr]=f;
					}
				}
			}
		}else
		{
			int vec_id = (b_id-block_start_B) * BLOCK_SIZE_INDEX + t_id; //vector在par_B中的序号
			if(vec_id<sum_vec_B)
			{
				int standard_size = (tile_start_B[vec_id/32 + 1] - tile_start_B[vec_id/32])/32;
				int t_start=tile_start_B[vec_id/32];
				for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
				{
					int d = dim_B[t_start + vec_id% 32 + j * 32];
					float f=feature_B[t_start + vec_id% 32 + j * 32];
				//	if(vec_id==29999)
				//		printf(" %d %.8f\n",d,f);
					if(d!=-1&&d>=4096)
					{
						int ptr = atomicAdd(&index_B_dim_ptr[d], 1);
						index_B_id[ptr]=vec_id;
						index_B_val[ptr]=f;
					}
				}
			}
		}
 }
 __global__ void construct_Index1(int* dim_A, float* feature_A, int* tile_start_A,
		 	 	 	 	 	 	 	 	 	 	 	 		   int* index_A_dim_ptr, int* index_A_id, float* index_A_val,
		 	 	 	 	 	 	 	 	 	 	 	 	 	   int sum_vec_A)
{
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;

	int vec_id=b_id*BLOCK_SIZE_INDEX+t_id;
	if(vec_id<sum_vec_A)
	{
				int standard_size = (tile_start_A[vec_id/32 + 1] - tile_start_A[vec_id/ 32])/32;
				int t_start = tile_start_A[vec_id / 32];
				for (int j = 0; j < standard_size; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
				{
					int d = dim_A[t_start + vec_id % 32 + j * 32];
					float f=feature_A[t_start + vec_id% 32 + j * 32];
					if(d!=-1&&d>=4096)
					{
						int ptr = atomicAdd(&index_A_dim_ptr[d], 1);
						index_A_id[ptr]=vec_id;
						index_A_val[ptr]=f;
					}
				}
			}
		}

 __global__ void test(int* index_A_dim_ptr, int* index_A_id, float* index_A_val, int* index_B_dim_ptr,int* index_B_id, float* index_B_val,
		 	 	 	 	 	 	 	    int* index_C_dim_ptr, int* index_C_id, float* index_C_val, int* index_D_dim_ptr,int* index_D_id, float* index_D_val )
 {
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;

	 if(t_id==0&&b_id==0)
	 {
		 	int d=3000;
		   	for(int i=index_D_dim_ptr[d];i<index_D_dim_ptr[d+1];i++)
		   		printf(" %d %.8f\n",index_D_id[i],index_D_val[i]);

	//        printf("\n");

	 }

 }

 const int seg_size=32;
 const int SumBlockGenerate=(max_dim/BLOCK_SIZE_GENERATE_VAR)+1; //每个join需要的block数
 //generate dim_block_start[], dim_block_num[]

 __global__ void generate_auxiliary_variable(int* index_A_dim_ptr,int* index_B_dim_ptr,int* join_dim_start_block, int* join_dim_block_num, int*join_block_id)
 {
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;
	 int dim = b_id * BLOCK_SIZE_GENERATE_VAR + t_id;

     if(dim<max_dim)
     {
    	 int sum_segs_A=(index_A_dim_ptr[dim+1]-index_A_dim_ptr[dim])/seg_size;
    	 if((index_A_dim_ptr[dim+1]-index_A_dim_ptr[dim])%seg_size!=0)
    		 sum_segs_A++;
    	 int sum_segs_B=(index_B_dim_ptr[dim+1]-index_B_dim_ptr[dim])/seg_size;
    	 if((index_B_dim_ptr[dim+1]-index_B_dim_ptr[dim])%seg_size!=0)
    	 	 sum_segs_B++;
    	 int sum_block=sum_segs_A*sum_segs_B;
    	 join_dim_start_block[dim]=atomicAdd(&join_block_id[0],sum_block);
    	 join_dim_block_num[dim]=sum_block;
  /*  	 if(dim>=max_dim-50)
    		 printf("dim=%d A=%d B=%d sum=%d\n",dim,sum_segs_A,sum_segs_B,sum_block);*/
     }
 }

 __global__ void generate_auxiliary_variable2(int* index_B_dim_ptr,
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	    int* join_dim_start_block, int* join_dim_block_num,  //需要的辅助变量
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	    int* join_block_dim, int* join_block_seg_A, int* join_block_seg_B)  //生成的辅助变量
 {
	 int b_id = blockIdx.x;
	int t_id = threadIdx.x;
	int dim = b_id* BLOCK_SIZE_GENERATE_VAR + t_id;

	if(dim<max_dim)
	{
		int start_block=join_dim_start_block[dim]; //join_id中开始计算dim的block id
		//if(dim==50)
		//	printf("start=%d num=%d\n",start_block,join_dim_block_num[dim]);
		for(int i=start_block;i<start_block+join_dim_block_num[dim];i++) //遍历join_id中计算dim的block id
		{
			join_block_dim[i]=dim;
			int sum_segs_B=(index_B_dim_ptr[dim+1]-index_B_dim_ptr[dim])/seg_size;
			if((index_B_dim_ptr[dim+1]-index_B_dim_ptr[dim])%seg_size!=0)
				sum_segs_B++;
			join_block_seg_A[i]=(i-start_block)/sum_segs_B;
			join_block_seg_B[i]=(i-start_block)%sum_segs_B;
		}
	}
 }

 __global__ void similarity_join1(int* join_block_dim, int* join_block_seg_A, int* join_block_seg_B,
		 	 	 	 	 	 	 	 	 	 	 		   int* index_A_dim_ptr, int* index_A_id, float* index_A_val, int* index_B_dim_ptr, int* index_B_id, float* index_B_val,
		 	 	 	 	 	 	 	 	 	 	 		   int sum_vec_B, float* similarity_val)
 {
	 int b_id= blockIdx.x;
	 int t_id_x = threadIdx.x;
	 int t_id_y = threadIdx.y;

	 int dim = join_block_dim[b_id];
/*	if(dim==3000&&t_id_x==0&&t_id_y==0)
		 printf("b_id=%d\n",b_id);
*/
	 if (dim>=4096)
	{
		 int v_num_A = index_A_dim_ptr[dim + 1] - index_A_dim_ptr[dim]; //par_A的index中包含维数dim的vector总数
		 int v_num_B = index_B_dim_ptr[dim + 1] - index_B_dim_ptr[dim];

		 int seg_A = join_block_seg_A[b_id]; //A中要计算的seg_id
		 int seg_B = join_block_seg_B[b_id];//B

		 __shared__ int id_A[seg_size];
		 __shared__ float feature_A[seg_size];
		 __shared__  int id_B[seg_size];
		 __shared__ float feature_B[seg_size];


		 int x = (t_id_y % 1) * 32 + t_id_x;//A中相对偏移量
		 int y = (t_id_y % 1) * 32 + t_id_x;

		 int p_A = seg_A*seg_size + x;  //A中绝对偏移量
		 int p_B = seg_B*seg_size + y;

		 if (p_A < v_num_A)
		 {
			 id_A[x] = index_A_id[index_A_dim_ptr[dim] + p_A];
			 feature_A[x] = index_A_val[index_A_dim_ptr[dim] + p_A];
		 }
	     if (p_B < v_num_B)
	     {
			 id_B[y] = index_B_id[index_B_dim_ptr[dim] + p_B];
			 feature_B[y] = index_B_val[index_B_dim_ptr[dim] + p_B];
		 }

		 __syncthreads();

		 for (int m = 0; m < 1; m++)
			 for (int n = 0; n < 1; n++)
			 {
				int s_A= m * 32 + t_id_y;  //share id_A 中的索引
				 int s_B = n * 32 + t_id_x;  //share id_B中的索引

				 if ((seg_A*seg_size + s_A) < v_num_A && (seg_B*seg_size + s_B) < v_num_B)
				 {
					 float val = feature_A[s_A] * feature_B[s_B];
					 unsigned long p =id_A[s_A]* (unsigned long)sum_vec_B+ id_B[s_B];

				//	 unsigned int length=1250000000;
			//	 if(p<length)
				//	 if(similarity_val[p]>=0)
				//   if(dim<40000)
					atomicAdd(&similarity_val[p], val);

            /*       if(id_A[s_A]==3000&&id_B[s_B]==45000)
			        			printf("dim=%d p=%u feat_A=%.8f feat_B=%.8f val=%.8f similarity=%.8f\n",dim,p,feature_A[s_A],feature_B[s_B],val,similarity_val[p]);*/
				 }
			 }
		 }
 }



__global__ void similarity_join_rest(int* dim_A, float* feature_A, int* tile_start_A,int* dim_B, float* feature_B, int* tile_start_B,
																	int sum_vec_A, int sum_vec_B, float* similarity_val, int* sum_pair)
{
	int b_id = blockIdx.x;
	int t_id = threadIdx.x;

	__shared__ float vector_A[4096];

	register	int block_num=sum_vec_B/blockDim.x;
	if(sum_vec_B%blockDim.x!=0)
		block_num++;
	register int id_A=b_id/block_num;
	for(int i=t_id;i<4096;i=i+blockDim.x)
		vector_A[i]=0.0;

	 __syncthreads();
	 register int standard_size=0, t_start=0;
	 register int d=0;
	 register float f=0.0;
   //vector A初始化
	 if(id_A<sum_vec_A)
	 {
		 standard_size = (tile_start_A[id_A/32 + 1] - tile_start_A[id_A/ 32])/32;
		 t_start = tile_start_A[id_A / 32];
		 for(int j=t_id;j<standard_size&&j<4096;j=j+1024)
		 {
			 d = dim_A[t_start + id_A % 32 + j * 32];
			 f=feature_A[t_start + id_A% 32 + j * 32];
			 if(d!=-1&&d<4096)
				 vector_A[d]=f;
		 }

    __syncthreads();

    register   int b_id_start=block_num*id_A;  //A_id相同的block的开始id
    register int id_B=(b_id-b_id_start)*blockDim.x+t_id;

    if(id_B<sum_vec_B)
    {
    	register unsigned int p =id_A* (unsigned int)sum_vec_B+ id_B;
   	 if(similarity_val[p]>=0)
   	 {
   		register float similarity_val_rest=0.0;
   		standard_size = (tile_start_B[id_B/32 + 1] - tile_start_B[id_B/ 32])/32;
    	t_start = tile_start_B[id_B / 32];

    	for ( int j = 0; j < standard_size&&j<4096; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
    	{
    		d = dim_B[t_start + id_B % 32 + j * 32];
    		f=feature_B[t_start + id_B% 32 + j * 32];
    		if(d>=4096)
    			break;
    	 	if(d!=-1)
    	 		similarity_val_rest+=f*vector_A[d];
    	// 	int a=f*2.2;   //---------------------------------------------7.01S
    	}
      //	similarity_val[p]=similarity_val[p]+similarity_rest;
         if((similarity_val_rest+similarity_val[p])>=threhold)
       {
        	 //写入结果文件
        //	 atomicAdd(&sum_pair[0], 1);
        }
   	 }
    }
  }
}



 __global__ void filter_pair(float* similarity_val, unsigned int sum_cand, int sum_vec_B, int* sum_pair,
		 	 	 	 	 	 	 	 	 	 	 	float* pre_norm2_A, float* pre_norm2_B)
 {
	 unsigned int  b_id = blockIdx.x;
	 unsigned int  t_id = threadIdx.x;
	 unsigned int  id = b_id * BLOCK_SIZE_VERIFY_VAL + t_id;  //similarity_val的索引

	 if(id<sum_cand)
	 {
		 unsigned int  id_A=id/sum_vec_B;
		 unsigned int  id_B=id%sum_vec_B;

			 float upper1=pre_norm2_A[id_A]*pre_norm2_B[id_B];

			 float upper=similarity_val[id]+upper1;
			 if(upper<=threhold)
			 {
				 similarity_val[id]=-999;
		 }

	 }
 }

 __global__ void filter_pair2(float* similarity_val, unsigned int sum_cand, int sum_vec_B, int* sum_pair,
		 	 	 	 	 	 	 	 	 	 	 	float* pre_norm2_A, float* pre_norm2_B,
		 	 	 	 	 	 	 	 	 	 	 	float* pre_sum_A, float* pre_sum_B,
		 	 	 	 	 	 	 	 	 	 	 	float* pre_max_A, float* pre_max_B)
 {
	 unsigned int  b_id = blockIdx.x;
	 unsigned int  t_id = threadIdx.x;
	 unsigned int  id = b_id * BLOCK_SIZE_VERIFY_VAL + t_id;  //similarity_val的索引

	 if(id<sum_cand)
	 {
		 unsigned int  id_A=id/sum_vec_B;
		 unsigned int  id_B=id%sum_vec_B;


			 float upper2=pre_sum_A[id_A]*pre_max_B[id_B];
			 float upper3=pre_max_A[id_A]*pre_sum_B[id_B];
			 float upper=similarity_val[id]+fminf(upper2,upper3);
			 if(upper<=threhold)
			 {
				//  atomicAdd(&sum_pair[0], 1);
				 similarity_val[id]=-999;
		 }

	 }
 }
 __global__ void verify_val(float* similarity_val, unsigned int sum_cand, int sum_vec_B, int* sum_pair)
 {
	 unsigned int  b_id = blockIdx.x;
	 unsigned int  t_id = threadIdx.x;
	 unsigned int  id = b_id * BLOCK_SIZE_VERIFY_VAL + t_id;  //similarity_val的索引

	 if(id<sum_cand)
	 {
		 unsigned int  id_A=id/sum_vec_B;
		 unsigned int  id_B=id%sum_vec_B;

		 if(similarity_val[id] >= threhold)
		 {
			 atomicAdd(&sum_pair[0], 1);
		 }
	 }
 }

 __global__ void generate_auxiliary_variable_apss(int* index_dim_ptr, int* join_dim_start_block, int* join_dim_block_num, int*join_block_id)
 {
	 int b_id = blockIdx.x;
	 int t_id = threadIdx.x;
	 int dim = b_id * BLOCK_SIZE_GENERATE_VAR + t_id;

     if(dim<max_dim)
     {

         int sum_segs=(index_dim_ptr[dim+1]-index_dim_ptr[dim])/seg_size;
    	 if((index_dim_ptr[dim+1]-index_dim_ptr[dim])%seg_size!=0)
    		 sum_segs++;
    	 int sum_block=sum_segs*(sum_segs+1)/2;
    	 join_dim_start_block[dim]=atomicAdd(&join_block_id[0],sum_block);
    	 join_dim_block_num[dim]=sum_block;
    	/* if(dim==10000)
    		 printf("dim=%d sum_segs=%d sum_blocks=%d\n",dim,sum_segs,sum_block);*/
     }
 }

 __global__ void generate_auxiliary_variable2_apss(int* index_dim_ptr,
 		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	    int* join_dim_start_block, int* join_dim_block_num,  //需要的辅助变量
 		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	    int* apss_block_dim, int*apss_block_seg_A, int* apss_block_seg_B)  //生成的辅助变量
  {
 	 int b_id = blockIdx.x;
 	int t_id = threadIdx.x;
 	int dim = b_id* BLOCK_SIZE_GENERATE_VAR + t_id;

 	if(dim<max_dim)
 	{
 		int start_block=join_dim_start_block[dim]; //join_id中开始计算dim的block id
 		for(int i=start_block;i<start_block+join_dim_block_num[dim];i++) //遍历join_id中计算dim的block id
 		{
 			apss_block_dim[i]=dim;
 			int sum_segs=(index_dim_ptr[dim+1]-index_dim_ptr[dim])/seg_size;
 			if((index_dim_ptr[dim+1]-index_dim_ptr[dim])%seg_size!=0)
 				sum_segs++;
 			int r_b_id=i-start_block;//计算相同dim的相对block id
 			if(sum_segs%2!=0) //奇数
 			{
 				int num=(sum_segs-1)/2+1;
 				apss_block_seg_A[i]=r_b_id/num;
 				apss_block_seg_B[i]=(r_b_id%num+apss_block_seg_A[i])%sum_segs;
 			}else //偶数
 			{
 				int half=sum_segs/2*(sum_segs/2+1);
 				if(r_b_id<half)
 				{
 					int num=sum_segs/2+1;
 					apss_block_seg_A[i]=r_b_id/num;
 					apss_block_seg_B[i]=(r_b_id%num+apss_block_seg_A[i])%sum_segs;
 				}
 				else
 				{
 					apss_block_seg_A[i]=sum_segs/2+(r_b_id-half)/(sum_segs/2);
 					apss_block_seg_B[i]=((r_b_id-half)%(sum_segs/2)+apss_block_seg_A[i])%sum_segs;
 				}
 			}

 		/*	if(dim==600)
 				printf("sum_seg=%d i=%d seg_A=%d seg_B=%d\n",sum_segs,i,apss_block_seg_A[i],apss_block_seg_B[i]);*/
 		}
 	}
  }

 __global__ void similarity_apss1(int* apss_block_dim, int* apss_block_seg_A, int* apss_block_seg_B,
		 	 	 	 	 	 	 	 	 	 	 		   int* index_dim_ptr, int* index_id, float* index_val,
		 	 	 	 	 	 	 	 	 	 	 		   float* similarity_val)
 {
	 int b_id= blockIdx.x;
	 int t_id_x = threadIdx.x;
	 int t_id_y = threadIdx.y;

	 int dim = apss_block_dim[b_id];
	 if (dim>=4096)
	{
		 int v_num = index_dim_ptr[dim+1] - index_dim_ptr[dim]; //par_A的index中包含维数dim的vector总数

		 int seg_A = apss_block_seg_A[b_id]; //A中要计算的seg_id
		 int seg_B = apss_block_seg_B[b_id];//B

		 __shared__ int id_A[seg_size];
		 __shared__ float feature_A[seg_size];
		 __shared__  int id_B[seg_size];
		 __shared__ float feature_B[seg_size];

		 for(int i=0;i<seg_size;i++)
		{
			 id_A[i]=-1;
			 id_B[i]=-1;
		}
		 __syncthreads();

		 int x = (t_id_y % 1) * 32 + t_id_x;//A中相对偏移量
		 int y = (t_id_y % 1) * 32 + t_id_x;

		 int p_A = seg_A*seg_size + x;  //A中绝对偏移量
		 int p_B = seg_B*seg_size + y;

		 if (p_A < v_num)
		 {
			 id_A[x] = index_id[index_dim_ptr[dim] + p_A];
			 feature_A[x] = index_val[index_dim_ptr[dim] + p_A];
		 }
	     if (p_B < v_num)
	     {
			 id_B[y] = index_id[index_dim_ptr[dim] + p_B];
			 feature_B[y] = index_val[index_dim_ptr[dim] + p_B];
		 }

		 __syncthreads();

		 for (int m = 0; m < 1; m++)
			 for (int n = 0; n < 1; n++)
			 {

				int s_A= m * 32 + t_id_y;  //share id_A 中的索引
				 int s_B = n * 32 + t_id_x;  //share id_B中的索引

				 if ((seg_A*seg_size + s_A) < v_num && (seg_B*seg_size + s_B) < v_num)
				 {
					int m,n;//m: 大id, n: 小id
					 float fm,fn;
					 float val;
					unsigned long  p;
					 if(seg_A==seg_B)
					 {
						 if(id_A[s_A]>id_B[s_B])
						 {
							 m=id_A[s_A]; n=id_B[s_B];
							 fm= feature_A[s_A];fn=feature_B[s_B];
							 val = fm*fn;
							 p=(unsigned long)m*(m-1)/2+n;
							 atomicAdd(&similarity_val[p], val);
							/* if(n==100&&m==45000)
								 printf("dim=%d fn=%.8f fm=%.8f val=%.8f similar=%.8f \n",dim,fn,fm,val,similarity_val[p]);*/
						 }
					 }
					 else
					 {
						 if(id_A[s_A]>id_B[s_B])
						 {
							 m=id_A[s_A]; n=id_B[s_B];
						    fm= feature_A[s_A];fn=feature_B[s_B];
						 }else
						 {
							 m=id_B[s_B]; n=id_A[s_A];
							 fm=feature_B[s_B];fn=feature_A[s_A];
						 }
						 val = fm*fn;
						 p=(unsigned long)m*(m-1)/2+n;
						 atomicAdd(&similarity_val[p], val);
						/* if(n==100&&m==45000)
							 printf("dim=%d fn=%.8f fm=%.8f val=%.8f  similar=%.8f\n",dim,fn,fm,val,similarity_val[p]);*/
					 }
				 }
			 }
	}
 }

 __global__ void similarity_apss2(int* apss_block_dim, int* apss_block_seg_A, int* apss_block_seg_B,
		 	 	 	 	 	 	 	 	 	 	 		   int* index_dim_ptr, int* index_id, float* index_val,
		 	 	 	 	 	 	 	 	 	 	 		   float* similarity_val)
 {
	 int b_id= blockIdx.x;
	 int t_id_x = threadIdx.x;
	 int t_id_y = threadIdx.y;

	 int dim = apss_block_dim[b_id];
	 if (dim <1000&&dim>=0)
	{
		 int v_num = index_dim_ptr[dim+1] - index_dim_ptr[dim]; //par_A的index中包含维数dim的vector总数

		 int seg_A = apss_block_seg_A[b_id]; //A中要计算的seg_id
		 int seg_B = apss_block_seg_B[b_id];//B

		 __shared__ int id_A[seg_size];
		 __shared__ float feature_A[seg_size];
		 __shared__  int id_B[seg_size];
		 __shared__ float feature_B[seg_size];

		 for(int i=0;i<seg_size;i++)
		{
			 id_A[i]=-1;
			 id_B[i]=-1;
		}
		 __syncthreads();

		 int x = (t_id_y % 1) * 32 + t_id_x;//A中相对偏移量
		 int y = (t_id_y % 1) * 32 + t_id_x;

		 int p_A = seg_A*seg_size + x;  //A中绝对偏移量
		 int p_B = seg_B*seg_size + y;

		 if (p_A < v_num)
		 {
			 id_A[x] = index_id[index_dim_ptr[dim] + p_A];
			 feature_A[x] = index_val[index_dim_ptr[dim] + p_A];
		 }
	     if (p_B < v_num)
	     {
			 id_B[y] = index_id[index_dim_ptr[dim] + p_B];
			 feature_B[y] = index_val[index_dim_ptr[dim] + p_B];
		 }

		 __syncthreads();

		 for (int m = 0; m < 1; m++)
			 for (int n = 0; n < 1; n++)
			 {

				int s_A= m * 32 + t_id_y;  //share id_A 中的索引
				 int s_B = n * 32 + t_id_x;  //share id_B中的索引

				 if ((seg_A*seg_size + s_A) < v_num && (seg_B*seg_size + s_B) < v_num)
				 {
					int m,n;//m: 大id, n: 小id
					 float fm,fn;
					 float val;
					unsigned long  p;
					 if(seg_A==seg_B)
					 {
						 if(id_A[s_A]>id_B[s_B])
						 {
							 m=id_A[s_A]; n=id_B[s_B];
							 fm= feature_A[s_A];fn=feature_B[s_B];
							 val = fm*fn;
							 p=(unsigned long)m*(m-1)/2+n;
							 if(similarity_val[p]>=0)
								 atomicAdd(&similarity_val[p], val);
							/* if(n==100&&m==45000)
								 printf("dim=%d fn=%.8f fm=%.8f val=%.8f similar=%.8f \n",dim,fn,fm,val,similarity_val[p]);*/
						 }
					 }
					 else
					 {
						 if(id_A[s_A]>id_B[s_B])
						 {
							 m=id_A[s_A]; n=id_B[s_B];
						    fm= feature_A[s_A];fn=feature_B[s_B];
						 }else
						 {
							 m=id_B[s_B]; n=id_A[s_A];
							 fm=feature_B[s_B];fn=feature_A[s_A];
						 }
						 val = fm*fn;
						 p=(unsigned long)m*(m-1)/2+n;
						 if(similarity_val[p]>=0)
							 atomicAdd(&similarity_val[p], val);
						/* if(n==100&&m==45000)
							 printf("dim=%d fn=%.8f fm=%.8f val=%.8f  similar=%.8f\n",dim,fn,fm,val,similarity_val[p]);*/
					 }
				 }
			 }
	}
 }
 __global__ void filter_pair_apss(float* similarity_val, unsigned int sum_cand, int* b_id_a, float* pre_norm2)
 {
	 unsigned int  b_id = blockIdx.x;
	 unsigned int  t_id = threadIdx.x;
	 unsigned int  id = b_id * BLOCK_SIZE_VERIFY_VAL + t_id;  //similarity_val的索引

	 if(id<sum_cand)
	 {
		 int id_a=b_id_a[id];
		 int id_b=id-(unsigned int)id_a*(id_a-1)/2;
		 float upper=similarity_val[id]+pre_norm2[id_a]*pre_norm2[id_b];
	     if(upper<=threhold)
		 {

			// atomicAdd(&sum_pair[0], 1);
			 similarity_val[id]=-999;
		 }

	 }
 }

 __global__ void similarity_apss_rest(int* dim_A, float* feature_A, int* tile_start_A, int sum_vec_A,
		 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 int* apss_block_seg, int* apss_seg_start, float* similarity_val, int* sum_pair)
 {
 	int b_id = blockIdx.x;
 	int t_id = threadIdx.x;

 	int seg_id=apss_block_seg[b_id];
 	int seg_start=apss_seg_start[seg_id];
 	int r_b_id=b_id-seg_start; //相同seg_id内的相对block id

 	__shared__ float vector_A[4096];

 	int id_X=seg_id*1024+r_b_id%1024;
 	for(int i=t_id;i<4096;i=i+blockDim.x)
 		vector_A[i]=0.0;

 	 __syncthreads();
    //vector A初始化
 	 if(id_X<sum_vec_A)
 	 {
 	int standard_size = (tile_start_A[id_X/32 + 1] - tile_start_A[id_X/ 32])/32;
 	int t_start = tile_start_A[id_X / 32];
     for(int j=t_id;j<standard_size&&j<4096;j=j+1024)
     {
     	int d = dim_A[t_start + id_X % 32 + j * 32];
     	float f=feature_A[t_start + id_X% 32 + j * 32];
     	if(d!=-1&&d<4096)
     		vector_A[d]=f;
     }

     __syncthreads();

     int id_Y=(r_b_id/1024)*1024+t_id;

     if(id_Y<sum_vec_A&&id_Y<id_X)
     {
    	 unsigned long p =(unsigned long)id_X*(id_X-1)/2+id_Y;
    	 if(similarity_val[p]>=0)
    	 {
    	float similarity_val_rest=0.0;
     	int standard_size = (tile_start_A[id_Y/32 + 1] - tile_start_A[id_Y/ 32])/32;
     	int t_start = tile_start_A[id_Y / 32];
     	for (int j = 0; j < standard_size&&j<4096; j++) //every feature (dim[t_start + id % 32 + i * 32], feature[t_start + id % 32 + i * 32])
     	{
     		int d = dim_A[t_start + id_Y % 32 + j * 32];
     		float f=feature_A[t_start + id_Y% 32 + j * 32];
     		if(d>=4096)
     			break;
     	 	if(d!=-1)
     			similarity_val_rest+=f*vector_A[d];  //---------------------------------------------浮点数的运算最消耗时间
     	}

    // 	similarity_val[p]=similarity_val[p]+similarity_val_rest;
     	 if((similarity_val_rest+similarity_val[p])>=threhold)
     	       {
     		 	 	 //写入结果文件
     	        	// atomicAdd(&sum_pair[0], 1);
     	        }
    	 }
     }
   }
 }
 __global__ void verify_val_apss(float* similarity_val, unsigned int sum_cand,int* sum_pair)
 {
	 unsigned int  b_id = blockIdx.x;
	 unsigned int  t_id = threadIdx.x;
	 unsigned int  id = b_id * BLOCK_SIZE_VERIFY_VAL + t_id;  //similarity_val的索引

	 if(id<sum_cand)
	 {

		 if(similarity_val[id] >= threhold)
		 {
		//	 int id_a=b_id_a[id];
		//	 int id_b=id-(unsigned int)id_a*(id_a-1)/2;

				 atomicAdd(&sum_pair[0], 1);
		 }
	 }
 }
 void similarity_join_device(int* dim_A, float* feature_A, int* tile_start_A, int* dim_B, float* feature_B, int* tile_start_B,
													int* dim_C, float* feature_C, int* tile_start_C, int* dim_D, float* feature_D, int* tile_start_D,
													int* dim_E, float* feature_E, int* tile_start_E, int* dim_F, float* feature_F, int* tile_start_F,
													int* dim_G, float* feature_G, int* tile_start_G, int* dim_H, float* feature_H, int* tile_start_H,
													float* pre_norm_A1, float* pre_norm_B1, float* pre_norm_C1, float* pre_norm_D1,
													float* pre_norm_E1, float* pre_norm_F1, float* pre_norm_G1, float* pre_norm_H1,
													int sum_vec_A, int sum_feat_A, int sum_tile_A, int sum_vec_B, int sum_feat_B, int sum_tile_B,
													int sum_vec_C, int sum_feat_C, int sum_tile_C, int sum_vec_D, int sum_feat_D, int sum_tile_D,
													int sum_vec_E, int sum_feat_E, int sum_tile_E, int sum_vec_F, int sum_feat_F, int sum_tile_F,
													int sum_vec_G, int sum_feat_G, int sum_tile_G, int sum_vec_H, int sum_feat_H, int sum_tile_H,
													int* b_id_a, int* apss_block_seg, int* apss_seg_start
													)
 {

		cudaSetDevice(0);
		//----------------------------------------------------------------define partition pointers
	    int* dim_A_d;
	    float* feature_A_d;
	    int* tile_start_A_d;
	    int* dim_B_d;
	    float* feature_B_d;
	    int* tile_start_B_d;
	    int* dim_C_d;
	    float* feature_C_d;
	    int* tile_start_C_d;
	    int* dim_D_d;
	    float* feature_D_d;
	   int* tile_start_D_d;
	   	   int* dim_E_d;
		    float* feature_E_d;
		    int* tile_start_E_d;
		    int* dim_F_d;
		    float* feature_F_d;
		    int* tile_start_F_d;
		    int* dim_G_d;
		    float* feature_G_d;
		    int* tile_start_G_d;
		    int* dim_H_d;
		    float* feature_H_d;
		   int* tile_start_H_d;

	    float* pre_norm_A1_d;
	    float* pre_norm_B1_d;
	    float* pre_norm_C1_d;
	    float* pre_norm_D1_d;
	    	float* pre_norm_E1_d;
	    	float* pre_norm_F1_d;
	    	float* pre_norm_G1_d;
	    	float* pre_norm_H1_d;

        int* index_A_dim_num_d;
        int* index_B_dim_num_d;
        int* index_C_dim_num_d;
       int* index_D_dim_num_d;
       	   int* index_E_dim_num_d;
       	   int* index_F_dim_num_d;
       	   int* index_G_dim_num_d;
       	   int* index_H_dim_num_d;

        int* apss_block_seg_d;
        int* apss_seg_start_d;

		//----------------------------------------------------------------allocate space for partition pointers
		cudaMalloc((void**)&dim_A_d,sizeof(int)*(unsigned int)sum_feat_A);
		cudaMemcpy(dim_A_d,dim_A,sizeof(int)*(unsigned int)sum_feat_A,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&feature_A_d,sizeof(float)*(unsigned int)sum_feat_A);
		cudaMemcpy(feature_A_d,feature_A,sizeof(float)*(unsigned int)sum_feat_A,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&tile_start_A_d,sizeof(int)*(sum_tile_A+1));
		cudaMemcpy(tile_start_A_d,tile_start_A,sizeof(int)*(sum_tile_A+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dim_B_d,sizeof(int)*(unsigned int)sum_feat_B);
		cudaMemcpy(dim_B_d,dim_B,sizeof(int)*(unsigned int)sum_feat_B,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&feature_B_d,sizeof(float)*(unsigned int)sum_feat_B);
		cudaMemcpy(feature_B_d,feature_B,sizeof(float)*(unsigned int)sum_feat_B,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&tile_start_B_d,sizeof(int)*(sum_tile_B+1));
		cudaMemcpy(tile_start_B_d,tile_start_B,sizeof(int)*(sum_tile_B+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dim_C_d,sizeof(int)*sum_feat_C);
		cudaMemcpy(dim_C_d,dim_C,sizeof(int)*sum_feat_C,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&feature_C_d,sizeof(float)*sum_feat_C);
		cudaMemcpy(feature_C_d,feature_C,sizeof(float)*sum_feat_C,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&tile_start_C_d,sizeof(int)*(sum_tile_C+1));
		cudaMemcpy(tile_start_C_d,tile_start_C,sizeof(int)*(sum_tile_C+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&dim_D_d,sizeof(int)*sum_feat_D);
		cudaMemcpy(dim_D_d,dim_D,sizeof(int)*sum_feat_D,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&feature_D_d,sizeof(float)*sum_feat_D);
		cudaMemcpy(feature_D_d,feature_D,sizeof(float)*sum_feat_D,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&tile_start_D_d,sizeof(int)*(sum_tile_D+1));
		cudaMemcpy(tile_start_D_d,tile_start_D,sizeof(int)*(sum_tile_D+1),cudaMemcpyHostToDevice);
				cudaMalloc((void**)&dim_E_d,sizeof(int)*(unsigned int)sum_feat_E);
				cudaMemcpy(dim_E_d,dim_E,sizeof(int)*(unsigned int)sum_feat_E,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&feature_E_d,sizeof(float)*(unsigned int)sum_feat_E);
				cudaMemcpy(feature_E_d,feature_E,sizeof(float)*(unsigned int)sum_feat_E,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&tile_start_E_d,sizeof(int)*(sum_tile_E+1));
				cudaMemcpy(tile_start_E_d,tile_start_E,sizeof(int)*(sum_tile_E+1),cudaMemcpyHostToDevice);
				cudaMalloc((void**)&dim_F_d,sizeof(int)*(unsigned int)sum_feat_F);
				cudaMemcpy(dim_F_d,dim_F,sizeof(int)*(unsigned int)sum_feat_F,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&feature_F_d,sizeof(float)*(unsigned int)sum_feat_F);
				cudaMemcpy(feature_F_d,feature_F,sizeof(float)*(unsigned int)sum_feat_F,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&tile_start_F_d,sizeof(int)*(sum_tile_F+1));
				cudaMemcpy(tile_start_F_d,tile_start_F,sizeof(int)*(sum_tile_F+1),cudaMemcpyHostToDevice);
				cudaMalloc((void**)&dim_G_d,sizeof(int)*(unsigned int)sum_feat_G);
				cudaMemcpy(dim_G_d,dim_G,sizeof(int)*(unsigned int)sum_feat_G,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&feature_G_d,sizeof(float)*(unsigned int)sum_feat_G);
				cudaMemcpy(feature_G_d,feature_G,sizeof(float)*(unsigned int)sum_feat_G,cudaMemcpyHostToDevice);
				cudaMalloc((void**)&tile_start_G_d,sizeof(int)*(sum_tile_G+1));
				cudaMemcpy(tile_start_G_d,tile_start_G,sizeof(int)*(sum_tile_G+1),cudaMemcpyHostToDevice);
				cudaMalloc((void**)&dim_H_d,sizeof(int)*(unsigned int)sum_feat_H);
						cudaMemcpy(dim_H_d,dim_H,sizeof(int)*(unsigned int)sum_feat_H,cudaMemcpyHostToDevice);
						cudaMalloc((void**)&feature_H_d,sizeof(float)*(unsigned int)sum_feat_H);
						cudaMemcpy(feature_H_d,feature_H,sizeof(float)*(unsigned int)sum_feat_H,cudaMemcpyHostToDevice);
						cudaMalloc((void**)&tile_start_H_d,sizeof(int)*(sum_tile_H+1));
						cudaMemcpy(tile_start_H_d,tile_start_H,sizeof(int)*(sum_tile_H+1),cudaMemcpyHostToDevice);

		cudaMalloc((void**)&pre_norm_A1_d,sizeof(float)*sum_vec_A);
		cudaMemcpy(pre_norm_A1_d,pre_norm_A1,sizeof(float)*sum_vec_A,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_B1_d,sizeof(float)*sum_vec_B);
		cudaMemcpy(pre_norm_B1_d,pre_norm_B1,sizeof(float)*sum_vec_B,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_C1_d,sizeof(float)*sum_vec_C);
		cudaMemcpy(pre_norm_C1_d,pre_norm_C1,sizeof(float)*sum_vec_C,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_D1_d,sizeof(float)*sum_vec_D);
		cudaMemcpy(pre_norm_D1_d,pre_norm_D1,sizeof(float)*sum_vec_D,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_E1_d,sizeof(float)*sum_vec_E);
		cudaMemcpy(pre_norm_E1_d,pre_norm_E1,sizeof(float)*sum_vec_E,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_F1_d,sizeof(float)*sum_vec_F);
		cudaMemcpy(pre_norm_F1_d,pre_norm_F1,sizeof(float)*sum_vec_F,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_G1_d,sizeof(float)*sum_vec_G);
		cudaMemcpy(pre_norm_G1_d,pre_norm_G1,sizeof(float)*sum_vec_G,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&pre_norm_H1_d,sizeof(float)*sum_vec_H);
		cudaMemcpy(pre_norm_H1_d,pre_norm_H1,sizeof(float)*sum_vec_H,cudaMemcpyHostToDevice);

		cudaMalloc((void**)&index_A_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_B_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_C_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_D_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_E_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_F_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_G_dim_num_d,sizeof(int)*max_dim);
		cudaMalloc((void**)&index_H_dim_num_d,sizeof(int)*max_dim);

		cudaMalloc((void**)&apss_block_seg_d,sizeof(int)*1254400);
		cudaMemcpy(apss_block_seg_d,apss_block_seg,sizeof(int)*1254400,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&apss_seg_start_d,sizeof(int)*49);
		cudaMemcpy(apss_seg_start_d,apss_seg_start,sizeof(int)*49,cudaMemcpyHostToDevice);

		int GridSizeIndex1=(sum_vec_A/BLOCK_SIZE_INDEX)+1   +(sum_vec_B/BLOCK_SIZE_INDEX)+1 +
										 (sum_vec_C/BLOCK_SIZE_INDEX)+1   +(sum_vec_D/BLOCK_SIZE_INDEX)+1 ;
		int GridSizeIndex2=(sum_vec_E/BLOCK_SIZE_INDEX)+1   +(sum_vec_F/BLOCK_SIZE_INDEX)+1 +
										 (sum_vec_G/BLOCK_SIZE_INDEX)+1   +(sum_vec_H/BLOCK_SIZE_INDEX)+1 ;
		generate_index_ptr4<<<GridSizeIndex1,BLOCK_SIZE_INDEX>>>(dim_A_d, tile_start_A_d,dim_B_d,tile_start_B_d,
																														dim_C_d,  tile_start_C_d,dim_D_d,tile_start_D_d,
																														index_A_dim_num_d,index_B_dim_num_d,index_C_dim_num_d,index_D_dim_num_d,
																														sum_vec_A,sum_vec_B,sum_vec_C,sum_vec_D);
		cudaThreadSynchronize();
		generate_index_ptr4<<<GridSizeIndex1,BLOCK_SIZE_INDEX>>>(dim_E_d, tile_start_E_d,dim_F_d,tile_start_F_d,
																															dim_G_d,  tile_start_G_d,dim_H_d,tile_start_H_d,
																															index_E_dim_num_d,index_F_dim_num_d,index_G_dim_num_d,index_H_dim_num_d,
																															sum_vec_E,sum_vec_F,sum_vec_G,sum_vec_H);
		cudaThreadSynchronize();
		int* index_A_dim_num=new int[max_dim];
		int* index_B_dim_num=new int[max_dim];
		int* index_C_dim_num=new int[max_dim];
		int* index_D_dim_num=new int[max_dim];
			int* index_E_dim_num=new int[max_dim];
			int* index_F_dim_num=new int[max_dim];
			int* index_G_dim_num=new int[max_dim];
			int* index_H_dim_num=new int[max_dim];


		cudaMemcpy(index_A_dim_num,index_A_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
		cudaMemcpy(index_B_dim_num,index_B_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
		cudaMemcpy(index_C_dim_num,index_C_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
	cudaMemcpy(index_D_dim_num,index_D_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
	cudaMemcpy(index_E_dim_num,index_E_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
		cudaMemcpy(index_F_dim_num,index_F_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
		cudaMemcpy(index_G_dim_num,index_G_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
	cudaMemcpy(index_H_dim_num,index_H_dim_num_d, sizeof(int)*max_dim,cudaMemcpyDeviceToHost);
	/*	for(int d=0;d<100;d++)
			printf("d=%d A=%d B=%d C=%d D=%d E=%d\n",d,index_A_dim_num[d],index_B_dim_num[d],index_C_dim_num[d],index_D_dim_num[d],index_E_dim_num[d]);
*/
        //----------------------------------------------------------------------define index pointers
        int* index_A_dim_ptr_d;
        int* index_A_id_d;
        float* index_A_val_d;
        int* index_B_dim_ptr_d;
        int* index_B_id_d;
        float* index_B_val_d;
        int* index_C_dim_ptr_d;
        int* index_C_id_d;
        float* index_C_val_d;
       int* index_D_dim_ptr_d;
       int* index_D_id_d;
        float* index_D_val_d;
        	int* index_E_dim_ptr_d;
              int* index_E_id_d;
              float* index_E_val_d;
              int* index_F_dim_ptr_d;
              int* index_F_id_d;
              float* index_F_val_d;
              int* index_G_dim_ptr_d;
              int* index_G_id_d;
              float* index_G_val_d;
              int* index_H_dim_ptr_d;
              int* index_H_id_d;
              float* index_H_val_d;

        //--------------------------------------------------------------------allocate space for index pointers
        int* index_A_dim_ptr=new int[max_dim+1];
        int* index_B_dim_ptr=new int[max_dim+1];
        int* index_C_dim_ptr=new int[max_dim+1];
       int* index_D_dim_ptr=new int[max_dim+1];
       int* index_E_dim_ptr=new int[max_dim+1];
       int* index_F_dim_ptr=new int[max_dim+1];
       int* index_G_dim_ptr=new int[max_dim+1];
      int* index_H_dim_ptr=new int[max_dim+1];

        int index_A_p=0;
        int index_B_p=0;
        int index_C_p=0;
        int index_D_p=0;
        int index_E_p=0;
        int index_F_p=0;
        int index_G_p=0;
        int index_H_p=0;

        index_A_dim_ptr[0]=index_A_p;
        index_B_dim_ptr[0]=index_B_p;
        index_C_dim_ptr[0]=index_C_p;
      index_D_dim_ptr[0]=index_D_p;
      index_E_dim_ptr[0]=index_E_p;
         index_F_dim_ptr[0]=index_F_p;
         index_G_dim_ptr[0]=index_G_p;
       index_H_dim_ptr[0]=index_H_p;



        for(int i=1;i<=max_dim;i++)
        {
        	index_A_p=index_A_p+index_A_dim_num[i-1];
        	index_A_dim_ptr[i]=index_A_p;
        	index_B_p=index_B_p+index_B_dim_num[i-1];
        	index_B_dim_ptr[i]=index_B_p;
        	index_C_p=index_C_p+index_C_dim_num[i-1];
        	index_C_dim_ptr[i]=index_C_p;
        	index_D_p=index_D_p+index_D_dim_num[i-1];
        	index_D_dim_ptr[i]=index_D_p;
        		index_E_p=index_E_p+index_E_dim_num[i-1];
                	index_E_dim_ptr[i]=index_E_p;
                	index_F_p=index_F_p+index_F_dim_num[i-1];
                	index_F_dim_ptr[i]=index_F_p;
                	index_G_p=index_G_p+index_G_dim_num[i-1];
                	index_G_dim_ptr[i]=index_G_p;
                	index_H_p=index_H_p+index_H_dim_num[i-1];
                	index_H_dim_ptr[i]=index_H_p;
        }

		cudaMalloc((void**)&index_A_dim_ptr_d,sizeof(int)*(max_dim+1));
		cudaMemcpy(index_A_dim_ptr_d, index_A_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&index_A_id_d,sizeof(int)*index_A_dim_ptr[max_dim]);
		cudaMalloc((void**)&index_A_val_d,sizeof(float)*index_A_dim_ptr[max_dim]);

		cudaMalloc((void**)&index_B_dim_ptr_d,sizeof(int)*(max_dim+1));
		cudaMemcpy(index_B_dim_ptr_d, index_B_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&index_B_id_d,sizeof(int)*index_B_dim_ptr[max_dim]);
		cudaMalloc((void**)&index_B_val_d,sizeof(float)*index_B_dim_ptr[max_dim]);

		cudaMalloc((void**)&index_C_dim_ptr_d,sizeof(int)*(max_dim+1));
		cudaMemcpy(index_C_dim_ptr_d, index_C_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&index_C_id_d,sizeof(int)*index_C_dim_ptr[max_dim]);
		cudaMalloc((void**)&index_C_val_d,sizeof(float)*index_C_dim_ptr[max_dim]);

		cudaMalloc((void**)&index_D_dim_ptr_d,sizeof(int)*(max_dim+1));
	cudaMemcpy(index_D_dim_ptr_d, index_D_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMalloc((void**)&index_D_id_d,sizeof(int)*index_D_dim_ptr[max_dim]);
	cudaMalloc((void**)&index_D_val_d,sizeof(float)*index_D_dim_ptr[max_dim]);

	cudaMalloc((void**)&index_E_dim_ptr_d,sizeof(int)*(max_dim+1));
	cudaMemcpy(index_E_dim_ptr_d, index_E_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&index_E_id_d,sizeof(int)*index_E_dim_ptr[max_dim]);
	cudaMalloc((void**)&index_E_val_d,sizeof(float)*index_E_dim_ptr[max_dim]);

	cudaMalloc((void**)&index_F_dim_ptr_d,sizeof(int)*(max_dim+1));
	cudaMemcpy(index_F_dim_ptr_d, index_F_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&index_F_id_d,sizeof(int)*index_F_dim_ptr[max_dim]);
	cudaMalloc((void**)&index_F_val_d,sizeof(float)*index_F_dim_ptr[max_dim]);

	cudaMalloc((void**)&index_G_dim_ptr_d,sizeof(int)*(max_dim+1));
	cudaMemcpy(index_G_dim_ptr_d, index_G_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&index_G_id_d,sizeof(int)*index_G_dim_ptr[max_dim]);
	cudaMalloc((void**)&index_G_val_d,sizeof(float)*index_G_dim_ptr[max_dim]);

	cudaMalloc((void**)&index_H_dim_ptr_d,sizeof(int)*(max_dim+1));
	cudaMemcpy(index_H_dim_ptr_d, index_H_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&index_H_id_d,sizeof(int)*index_H_dim_ptr[max_dim]);
	cudaMalloc((void**)&index_H_val_d,sizeof(float)*index_H_dim_ptr[max_dim]);

		GridSizeIndex1=(sum_vec_A/BLOCK_SIZE_INDEX)+1   +(sum_vec_B/BLOCK_SIZE_INDEX)+1;
	    GridSizeIndex2=(sum_vec_C/BLOCK_SIZE_INDEX)+1   +(sum_vec_D/BLOCK_SIZE_INDEX)+1 ;
	   int  GridSizeIndex3=(sum_vec_E/BLOCK_SIZE_INDEX)+1   +(sum_vec_F/BLOCK_SIZE_INDEX)+1 ;
	   int  GridSizeIndex4=(sum_vec_G/BLOCK_SIZE_INDEX)+1   +(sum_vec_H/BLOCK_SIZE_INDEX)+1 ;

        construct_Index2<<<GridSizeIndex1,BLOCK_SIZE_INDEX>>>(dim_A_d, feature_A_d, tile_start_A_d,dim_B_d,feature_B_d,tile_start_B_d,
        																										 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_B_dim_ptr_d,index_B_id_d,index_B_val_d,
        																										 sum_vec_A,sum_vec_B);
        cudaThreadSynchronize();


        construct_Index2<<<GridSizeIndex2,BLOCK_SIZE_INDEX>>>(dim_C_d, feature_C_d, tile_start_C_d,dim_D_d,feature_D_d,tile_start_D_d,
        																										 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d,
        																										 sum_vec_C,sum_vec_D);


        cudaThreadSynchronize();

        construct_Index2<<<GridSizeIndex3,BLOCK_SIZE_INDEX>>>(dim_E_d, feature_E_d, tile_start_E_d,dim_F_d,feature_F_d,tile_start_F_d,
        																										 index_E_dim_ptr_d,index_E_id_d,index_E_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
        																										 sum_vec_E,sum_vec_F);
        cudaThreadSynchronize();

        construct_Index2<<<GridSizeIndex4,BLOCK_SIZE_INDEX>>>(dim_G_d, feature_G_d, tile_start_G_d,dim_H_d,feature_H_d,tile_start_H_d,
        																										 index_G_dim_ptr_d,index_G_id_d,index_G_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
        																										 sum_vec_G,sum_vec_H);
        cudaThreadSynchronize();

        cudaMemcpy(index_A_dim_ptr_d, index_A_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMemcpy(index_B_dim_ptr_d, index_B_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMemcpy(index_C_dim_ptr_d, index_C_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		cudaMemcpy(index_D_dim_ptr_d, index_D_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
		   cudaMemcpy(index_E_dim_ptr_d, index_E_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
				cudaMemcpy(index_F_dim_ptr_d, index_F_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
				cudaMemcpy(index_G_dim_ptr_d, index_G_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);
				cudaMemcpy(index_H_dim_ptr_d, index_H_dim_ptr,sizeof(int)*(max_dim+1),cudaMemcpyHostToDevice);


   /*   test<<<10,10>>>(index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_B_dim_ptr_d,index_B_id_d,index_B_val_d,
    		  	  	  	  	  	  	  index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d);
       cudaThreadSynchronize();*/

		   clock_t startTime,endTime;
		   startTime = clock();

		//JOIN (par_A, par_B)
		//---------------------------------------------------------------- "generate_auxiliary_variable"生成的辅助变量
		int* join_dim_start_block_d;//join中开始计算每个维度的block id
		int* join_dim_block_num_d; //计算每个维度的block数目
		int* join_block_id_d;

        //----------------------------------------------------------------allocate space for join pointer
	   cudaMalloc((void**)&join_dim_start_block_d, sizeof(int)*max_dim);
		cudaMalloc((void**)&join_dim_block_num_d, sizeof(int)*max_dim);
		cudaMalloc((void**)&join_block_id_d, sizeof(int));
	//	cudaMalloc((void**)&join_similarity_val_d,sizeof(float)*join_sum_cand[0]);

		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_B_dim_ptr_d,
																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
        cudaThreadSynchronize();
		int* join_block_sum=new int[1];
		cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
//		printf("sum block=%d\n",join_block_sum[0]);

        //------------------------------------------------------------------用于下面kernel的辅助变量
		int* join_block_dim_d; //每个block计算的dim
		int* join_block_seg_a_d; //每个block计算的seg_a
		int* join_block_seg_b_d; //每个block计算的seg_b

		cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);

		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,
																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
		 cudaThreadSynchronize();

		float* similarity_val_d; //similarity value

		unsigned long size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_B;
		cudaMalloc((void**)&similarity_val_d, size);
		printf("malloc size=%u\n",size);
         printf("sum A=%d, sum B=%d cand size=%u\n",sum_vec_A,sum_vec_B,(unsigned int)sum_vec_A*(unsigned int)sum_vec_B);
	     dim3 BLOCK_SIZE_JOIN(32, 32);

	     similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_B_dim_ptr_d,index_B_id_d,index_B_val_d,
				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,similarity_val_d);
		 cudaThreadSynchronize();

		 unsigned int sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_B;
         int* sum_pair_d;
         cudaMalloc((void**)&sum_pair_d,sizeof(int));

         unsigned long GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;


         filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_B, sum_pair_d,
																													pre_norm_A1_d,pre_norm_B1_d);

    	 cudaThreadSynchronize();
    	int block_num=sum_vec_B/BLOCK_SIZE_REST;
    	if(sum_vec_B%BLOCK_SIZE_REST!=0)
    		block_num++;
    	 int GridSizeRest=block_num*sum_vec_A;
    	 int* sum_pair=new int[1];
   	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_B_d,feature_B_d,tile_start_B_d,
    				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_B,similarity_val_d,sum_pair_d);

    	 	 cudaThreadSynchronize();

     //        verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_B, sum_pair_d);
             cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
             printf("par_A join par_B: sum pair=%d\n\n",sum_pair[0]);
             cudaFree(similarity_val_d);

    	 //JOIN (par_A, par_C)
    	 join_block_sum[0]=0;
    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
 		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_C_dim_ptr_d,
 																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
         cudaThreadSynchronize();
         cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
         printf("sum_block=%d\n",join_block_sum[0]);
         cudaFree(join_block_dim_d);
         cudaFree(join_block_seg_a_d);
         cudaFree(join_block_seg_b_d);
	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,
																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
		 cudaThreadSynchronize();

		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_C;
		cudaMalloc((void**)&similarity_val_d, size);
		printf("malloc size=%lu\n",size);
	    printf("sum A=%d, sum C=%d cand size=%u\n",sum_vec_A,sum_vec_C,(unsigned int)sum_vec_A*(unsigned int)sum_vec_C);
		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_C_dim_ptr_d,index_C_id_d,index_C_val_d,
					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,similarity_val_d);
		cudaThreadSynchronize();
	 	sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_C;
		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_C, sum_pair_d,
																												pre_norm_A1_d,pre_norm_C1_d);
		cudaThreadSynchronize();

  	 block_num=sum_vec_C/BLOCK_SIZE_REST;
    	 if(sum_vec_C%BLOCK_SIZE_REST!=0)
    		 block_num++;
    	 GridSizeRest=block_num*sum_vec_A;
    	 sum_pair[0]=0;
    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
    	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_C_d,feature_C_d,tile_start_C_d,
    				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_C,similarity_val_d,sum_pair_d);

	//	verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_C, sum_pair_d);
	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	     printf("par_A join par_C: sum pair=%d\n\n",sum_pair[0]);
	     cudaFree(similarity_val_d);

	     //JOIN (par_A, par_D)
	         	 join_block_sum[0]=0;
	         	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	      		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_D_dim_ptr_d,
	      																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	              cudaThreadSynchronize();
	              cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	              printf("sum_block=%d\n",join_block_sum[0]);
	              cudaFree(join_block_dim_d);
	              cudaFree(join_block_seg_a_d);
	              cudaFree(join_block_seg_b_d);
	     	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	     		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	     		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	     		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,
	     																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	     																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	     		 cudaThreadSynchronize();

	     		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_D;
	     		cudaMalloc((void**)&similarity_val_d, size);
	     		printf("malloc size=%lu\n",size);
	     	 printf("sum A=%d, sum D=%d cand size=%u\n",sum_vec_A,sum_vec_D,(unsigned int)sum_vec_A*(unsigned int)sum_vec_D);
	     		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d,
	     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,similarity_val_d);
	     	    cudaThreadSynchronize();
	   		sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_D;
	     		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	            filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d,
	     			     																												pre_norm_A1_d,pre_norm_D1_d);
	     		cudaThreadSynchronize();
	        	 block_num=sum_vec_D/BLOCK_SIZE_REST;
	       	 if(sum_vec_D%BLOCK_SIZE_REST!=0)
	       		 block_num++;
	       	 GridSizeRest=block_num*sum_vec_A;
	       	 sum_pair[0]=0;
	       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_D_d,feature_D_d,tile_start_D_d,
	       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_D,similarity_val_d,sum_pair_d);

	   //  		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d);
	     		cudaThreadSynchronize();
	     	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	     	     printf("par_A join par_D: sum pair=%d\n\n",sum_pair[0]);
	     	    cudaFree(similarity_val_d);

	   	     //JOIN (par_A, par_E)
	   	         	 join_block_sum[0]=0;
	   	         	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   	      		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_E_dim_ptr_d,
	   	      																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	   	              cudaThreadSynchronize();
	   	              cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	              printf("sum_block=%d\n",join_block_sum[0]);
	   	              cudaFree(join_block_dim_d);
	   	              cudaFree(join_block_seg_a_d);
	   	              cudaFree(join_block_seg_b_d);
	   	     	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	   	     		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	   	     		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	   	     		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,
	   	     																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	   	     																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	   	     		 cudaThreadSynchronize();

	   	     		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_E;
	   	     		cudaMalloc((void**)&similarity_val_d, size);
	   	     		printf("malloc size=%lu\n",size);
	   	     	 printf("sum A=%d, sum E=%d cand size=%u\n",sum_vec_A,sum_vec_E,(unsigned int)sum_vec_A*(unsigned int)sum_vec_E);
	   	     		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	   	     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_E_dim_ptr_d,index_E_id_d,index_E_val_d,
	   	     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,similarity_val_d);
	   	     	    cudaThreadSynchronize();
	   	     		sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_E;
	   	     		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	   	            filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d,
	   	     			     																												pre_norm_A1_d,pre_norm_E1_d);
	   	     		cudaThreadSynchronize();
	   	     	 block_num=sum_vec_E/BLOCK_SIZE_REST;
	   	       	 if(sum_vec_E%BLOCK_SIZE_REST!=0)
	   	       		 block_num++;
	   	       	 GridSizeRest=block_num*sum_vec_A;
	   	       	 sum_pair[0]=0;
	   	       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   	       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_E_d,feature_E_d,tile_start_E_d,
	   	       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_E,similarity_val_d,sum_pair_d);

	//   	     		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d);
	   	     		cudaThreadSynchronize();
	   	     	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	     	     printf("par_A join par_E: sum pair=%d\n\n",sum_pair[0]);
	   	     	    cudaFree(similarity_val_d);

	   		     //JOIN (par_A, par_F)
	   		         	 join_block_sum[0]=0;
	   		         	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   		      		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_F_dim_ptr_d,
	   		      																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	   		              cudaThreadSynchronize();
	   		              cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   		              printf("sum_block=%d\n",join_block_sum[0]);
	   		              cudaFree(join_block_dim_d);
	   		              cudaFree(join_block_seg_a_d);
	   		              cudaFree(join_block_seg_b_d);
	   		     	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	   		     		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	   		     		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	   		     		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,
	   		     																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	   		     																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	   		     		 cudaThreadSynchronize();

	   		     		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_F;
	   		     		cudaMalloc((void**)&similarity_val_d, size);
	   		     		printf("malloc size=%lu\n",size);
	   		     	 printf("sum A=%d, sum F=%d cand size=%u\n",sum_vec_A,sum_vec_F,(unsigned int)sum_vec_A*(unsigned int)sum_vec_F);
	   		     		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	   		     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
	   		     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,similarity_val_d);
	   		     	    cudaThreadSynchronize();
	   	     		sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_F;
	   		     		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	   		            filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d,
	   		     			     																												pre_norm_A1_d,pre_norm_F1_d);
	   		     		cudaThreadSynchronize();
	   		    	 block_num=sum_vec_F/BLOCK_SIZE_REST;
	   		       	 if(sum_vec_F%BLOCK_SIZE_REST!=0)
	   		       		 block_num++;
	   		       	 GridSizeRest=block_num*sum_vec_A;
	   		       	 sum_pair[0]=0;
	   		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   		       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_F_d,feature_F_d,tile_start_F_d,
	   		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_F,similarity_val_d,sum_pair_d);

//	   		  		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d);
	   		     		cudaThreadSynchronize();
	   		     	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   		     	     printf("par_A join par_F: sum pair=%d\n\n",sum_pair[0]);
	   		     	    cudaFree(similarity_val_d);

	   			     //JOIN (par_A, par_G)
	   			         	 join_block_sum[0]=0;
	   			         	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   			      		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_G_dim_ptr_d,
	   			      																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	   			              cudaThreadSynchronize();
	   			              cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   			              printf("sum_block=%d\n",join_block_sum[0]);
	   			              cudaFree(join_block_dim_d);
	   			              cudaFree(join_block_seg_a_d);
	   			              cudaFree(join_block_seg_b_d);
	   			     	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	   			     		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	   			     		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	   			     		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	   			     																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	   			     																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	   			     		 cudaThreadSynchronize();

	   			     		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_G;
	   			     		cudaMalloc((void**)&similarity_val_d, size);
	   			     		printf("malloc size=%lu\n",size);
	   			     	 printf("sum A=%d, sum G=%d cand size=%u\n",sum_vec_A,sum_vec_G,(unsigned int)sum_vec_A*(unsigned int)sum_vec_G);
	   			     		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	   			     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	   			     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	   			     	    cudaThreadSynchronize();
	   	    		sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_G;
	   			     		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	   			            filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	   			     			     																												pre_norm_A1_d,pre_norm_G1_d);
	   			     		cudaThreadSynchronize();
	   			      	 block_num=sum_vec_G/BLOCK_SIZE_REST;
	   			       	 if(sum_vec_G%BLOCK_SIZE_REST!=0)
	   			       		 block_num++;
	   			       	 GridSizeRest=block_num*sum_vec_A;
	   			       	 sum_pair[0]=0;
	   			       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   			       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_G_d,feature_G_d,tile_start_G_d,
	   			       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_G,similarity_val_d,sum_pair_d);

	   			//     		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	   			     		cudaThreadSynchronize();
	   			     	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   			     	     printf("par_A join par_G: sum pair=%d\n\n",sum_pair[0]);
	   			     	    cudaFree(similarity_val_d);

	   				     //JOIN (par_A, par_H)
	   				         	 join_block_sum[0]=0;
	   				         	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   				      		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,index_H_dim_ptr_d,
	   				      																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	   				              cudaThreadSynchronize();
	   				              cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   				              printf("sum_block=%d\n",join_block_sum[0]);
	   				              cudaFree(join_block_dim_d);
	   				              cudaFree(join_block_seg_a_d);
	   				              cudaFree(join_block_seg_b_d);
	   				     	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	   				     		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	   				     		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	   				     		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	   				     																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	   				     																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	   				     		 cudaThreadSynchronize();

	   				     		size= (unsigned long)sizeof(float)*sum_vec_A*sum_vec_H;
	   				     		cudaMalloc((void**)&similarity_val_d, size);
	   				     		printf("malloc size=%lu\n",size);
	   				     	 printf("sum A=%d, sum H=%d cand size=%u\n",sum_vec_A,sum_vec_H,(unsigned int)sum_vec_A*(unsigned int)sum_vec_H);
	   				     		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	   				     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_A_dim_ptr_d,index_A_id_d,index_A_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	   				     					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	   				     	    cudaThreadSynchronize();
	   			     		sum_cand=(unsigned int)sum_vec_A*(unsigned int)sum_vec_H;
	   				     		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	   				            filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	   				     			     																												pre_norm_A1_d,pre_norm_H1_d);
	   				     		cudaThreadSynchronize();
	   		       	 block_num=sum_vec_H/BLOCK_SIZE_REST;
	   				       	 if(sum_vec_H%BLOCK_SIZE_REST!=0)
	   				       		 block_num++;
	   				       	 GridSizeRest=block_num*sum_vec_A;
	   				       	 sum_pair[0]=0;
	   				       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   				       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,dim_H_d,feature_H_d,tile_start_H_d,
	   				       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_A,sum_vec_H,similarity_val_d,sum_pair_d);

	   		//		     		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	   				     		cudaThreadSynchronize();
	   				     	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   				     	     printf("par_A join par_H: sum pair=%d\n\n",sum_pair[0]);
	   				     	    cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_C)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_C_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_C;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum C=%d cand size=%u\n",sum_vec_B,sum_vec_C,(unsigned int)sum_vec_B*(unsigned int)sum_vec_C);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_C_dim_ptr_d,index_C_id_d,index_C_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	    		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_C;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_C, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_C1_d);
	    	    		cudaThreadSynchronize();
	    	      block_num=sum_vec_C/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_C%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	        	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_C_d,feature_C_d,tile_start_C_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_C,similarity_val_d,sum_pair_d);

//	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_C, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_C: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_D)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_D_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_D;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum D=%d cand size=%u\n",sum_vec_B,sum_vec_D,(unsigned int)sum_vec_B*(unsigned int)sum_vec_D);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	 		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_D;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_D1_d);
	    	    		cudaThreadSynchronize();

	    		          block_num=sum_vec_D/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_D%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    		       	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_D_d,feature_D_d,tile_start_D_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_D,similarity_val_d,sum_pair_d);

//	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_D: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_E)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_E_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_E;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum E=%d cand size=%u\n",sum_vec_B,sum_vec_E,(unsigned int)sum_vec_B*(unsigned int)sum_vec_E);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_E_dim_ptr_d,index_E_id_d,index_E_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	  		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_E;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_E1_d);
	    	    		cudaThreadSynchronize();
	    	     block_num=sum_vec_E/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_E%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	      	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_E_d,feature_E_d,tile_start_E_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_E,similarity_val_d,sum_pair_d);

//	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_E: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_F)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_F_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_F;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum F=%d cand size=%u\n",sum_vec_B,sum_vec_F,(unsigned int)sum_vec_B*(unsigned int)sum_vec_F);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,similarity_val_d);
	    	    		cudaThreadSynchronize();

	 	    		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_F;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_F1_d);
	    	    		cudaThreadSynchronize();
	    	       block_num=sum_vec_F/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_F%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	      	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_F_d,feature_F_d,tile_start_F_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_F,similarity_val_d,sum_pair_d);

//	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_F: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_G)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_G_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_G;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum G=%d cand size=%u\n",sum_vec_B,sum_vec_G,(unsigned int)sum_vec_B*(unsigned int)sum_vec_G);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	    	    		cudaThreadSynchronize();

	  		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_G;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_G1_d);
	    	    		cudaThreadSynchronize();
	     	   		        block_num=sum_vec_G/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_G%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	      	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_G_d,feature_G_d,tile_start_G_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_G,similarity_val_d,sum_pair_d);

//	      		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_G: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_B, par_H)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,index_H_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_B*sum_vec_H;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum B=%d, sum H=%d cand size=%u\n",sum_vec_B,sum_vec_H,(unsigned int)sum_vec_B*(unsigned int)sum_vec_H);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_B_dim_ptr_d,index_B_id_d,index_B_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	    		sum_cand=(unsigned int)sum_vec_B*(unsigned int)sum_vec_H;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    																												pre_norm_B1_d,pre_norm_H1_d);
	    	    		cudaThreadSynchronize();
	    		       block_num=sum_vec_H/BLOCK_SIZE_REST;
	    		       	 if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    		       		 block_num++;
	    		       	 GridSizeRest=block_num*sum_vec_B;
	    		       	 sum_pair[0]=0;
	    		       	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	      	 similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,dim_H_d,feature_H_d,tile_start_H_d,
	    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_B,sum_vec_H,similarity_val_d,sum_pair_d);

	//    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_B join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_C, par_D)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,index_D_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_C*sum_vec_D;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum C=%d, sum D=%d cand size=%u\n",sum_vec_C,sum_vec_D,(unsigned int)sum_vec_C*(unsigned int)sum_vec_D);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	  sum_cand=(unsigned int)sum_vec_C*(unsigned int)sum_vec_D;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d,
	    	    																												pre_norm_C1_d,pre_norm_D1_d);
	    	    		cudaThreadSynchronize();
	    		   		block_num=sum_vec_D/BLOCK_SIZE_REST;
	    	    		if(sum_vec_D%BLOCK_SIZE_REST!=0)
	    	    			block_num++;
	    	    		GridSizeRest=block_num*sum_vec_C;
	    	    		sum_pair[0]=0;
	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,dim_D_d,feature_D_d,tile_start_D_d,
	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,sum_vec_D,similarity_val_d,sum_pair_d);
	    	        		 cudaThreadSynchronize();

//	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_D, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_C join par_D: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_C, par_E)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,index_E_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_C*sum_vec_E;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum C=%d, sum E=%d cand size=%u\n",sum_vec_C,sum_vec_E,(unsigned int)sum_vec_C*(unsigned int)sum_vec_E);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_E_dim_ptr_d,index_E_id_d,index_E_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	   	sum_cand=(unsigned int)sum_vec_C*(unsigned int)sum_vec_E;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d,
	    	    																												pre_norm_C1_d,pre_norm_E1_d);
	    	    		cudaThreadSynchronize();
	    	   		block_num=sum_vec_E/BLOCK_SIZE_REST;
	    	    		if(sum_vec_E%BLOCK_SIZE_REST!=0)
	    	    			block_num++;
	    	    		GridSizeRest=block_num*sum_vec_C;
	    	    		sum_pair[0]=0;
	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,dim_E_d,feature_E_d,tile_start_E_d,
	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,sum_vec_E,similarity_val_d,sum_pair_d);
	    	        		 cudaThreadSynchronize();

	 //   	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_C join par_E: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_C, par_F)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,index_F_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_C*sum_vec_F;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum C=%d, sum F=%d cand size=%u\n",sum_vec_C,sum_vec_F,(unsigned int)sum_vec_C*(unsigned int)sum_vec_F);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,similarity_val_d);
	    	    		cudaThreadSynchronize();

	    	    		sum_cand=(unsigned int)sum_vec_C*(unsigned int)sum_vec_F;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d,
	    	    																												pre_norm_C1_d,pre_norm_F1_d);
	    	    		cudaThreadSynchronize();
	    	 	block_num=sum_vec_F/BLOCK_SIZE_REST;
	    	    		if(sum_vec_F%BLOCK_SIZE_REST!=0)
	    	    			block_num++;
	    	    		GridSizeRest=block_num*sum_vec_C;
	    	    		sum_pair[0]=0;
	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,dim_F_d,feature_F_d,tile_start_F_d,
	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,sum_vec_F,similarity_val_d,sum_pair_d);
	    	        cudaThreadSynchronize();

	    	//    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_C join par_F: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_C, par_G)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,index_G_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_C*sum_vec_G;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum C=%d, sum G=%d cand size=%u\n",sum_vec_C,sum_vec_G,(unsigned int)sum_vec_C*(unsigned int)sum_vec_G);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	    	    		cudaThreadSynchronize();

   	    		sum_cand=(unsigned int)sum_vec_C*(unsigned int)sum_vec_G;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	    	    																												pre_norm_C1_d,pre_norm_G1_d);
	    	    		cudaThreadSynchronize();
	   	    		block_num=sum_vec_G/BLOCK_SIZE_REST;
	    	    		if(sum_vec_G%BLOCK_SIZE_REST!=0)
	    	    			block_num++;
	    	    		GridSizeRest=block_num*sum_vec_C;
	    	    		sum_pair[0]=0;
	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,dim_G_d,feature_G_d,tile_start_G_d,
	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,sum_vec_G,similarity_val_d,sum_pair_d);
	    	        		 cudaThreadSynchronize();

	//    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_C join par_G: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	        	 //JOIN (par_C, par_H)
	    	        	 join_block_sum[0]=0;
	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,index_H_dim_ptr_d,
	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	             cudaThreadSynchronize();
	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	             cudaFree(join_block_dim_d);
	    	             cudaFree(join_block_seg_a_d);
	    	             cudaFree(join_block_seg_b_d);
	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    		 cudaThreadSynchronize();
	    	    		size= (unsigned long)sizeof(float)*sum_vec_C*sum_vec_H;
	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    		printf("malloc size=%lu\n",size);
	    	    	    printf("sum C=%d, sum H=%d cand size=%u\n",sum_vec_C,sum_vec_H,(unsigned int)sum_vec_C*(unsigned int)sum_vec_H);
	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_C_dim_ptr_d,index_C_id_d,index_C_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	    	    		cudaThreadSynchronize();

	      		sum_cand=(unsigned int)sum_vec_C*(unsigned int)sum_vec_H;

	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    																												pre_norm_C1_d,pre_norm_H1_d);
	    	    		cudaThreadSynchronize();
	    	    		block_num=sum_vec_H/BLOCK_SIZE_REST;
	    	    		if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    	    			block_num++;
	    	    		GridSizeRest=block_num*sum_vec_C;
	    	    		sum_pair[0]=0;
	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,dim_H_d,feature_H_d,tile_start_H_d,
	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_C,sum_vec_H,similarity_val_d,sum_pair_d);
	    	        cudaThreadSynchronize();

	 //  	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	     printf("par_C join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	     cudaFree(similarity_val_d);

	    	          	 //JOIN (par_D, par_E)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,index_E_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_D*sum_vec_E;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum D=%d, sum E=%d cand size=%u\n",sum_vec_D,sum_vec_E,(unsigned int)sum_vec_D*(unsigned int)sum_vec_E);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_D_dim_ptr_d,index_D_id_d,index_D_val_d,index_E_dim_ptr_d,index_E_id_d,index_E_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	    	   		sum_cand=(unsigned int)sum_vec_D*(unsigned int)sum_vec_E;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d,
	    	    	    	    																												pre_norm_D1_d,pre_norm_E1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	    	    		block_num=sum_vec_E/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_E%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_D;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_D_d,feature_D_d,tile_start_D_d,dim_E_d,feature_E_d,tile_start_E_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,sum_vec_E,similarity_val_d,sum_pair_d);
	    	    	            		 cudaThreadSynchronize();

	    	    	//    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_E, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_D join par_E: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	        	 //JOIN (par_D, par_F)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,index_F_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_D*sum_vec_F;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum D=%d, sum F=%d cand size=%u\n",sum_vec_D,sum_vec_F,(unsigned int)sum_vec_D*(unsigned int)sum_vec_F);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_D_dim_ptr_d,index_D_id_d,index_D_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	    	    	    		sum_cand=(unsigned int)sum_vec_D*(unsigned int)sum_vec_F;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d,
	    	    	    	    																												pre_norm_D1_d,pre_norm_F1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	    		block_num=sum_vec_F/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_F%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_D;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_D_d,feature_D_d,tile_start_D_d,dim_F_d,feature_F_d,tile_start_F_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,sum_vec_F,similarity_val_d,sum_pair_d);
	    	    	    	        		 cudaThreadSynchronize();

	    	    //	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_D join par_F: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	        	 //JOIN (par_D, par_G)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,index_G_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_D*sum_vec_G;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum D=%d, sum G=%d cand size=%u\n",sum_vec_D,sum_vec_G,(unsigned int)sum_vec_D*(unsigned int)sum_vec_G);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_D_dim_ptr_d,index_D_id_d,index_D_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	    	    	    		sum_cand=(unsigned int)sum_vec_D*(unsigned int)sum_vec_G;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	    	    	    	    																												pre_norm_D1_d,pre_norm_G1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	    	  		block_num=sum_vec_G/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_G%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_D;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_D_d,feature_D_d,tile_start_D_d,dim_G_d,feature_G_d,tile_start_G_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,sum_vec_G,similarity_val_d,sum_pair_d);
	    	    	    	        		 cudaThreadSynchronize();

	    	 //   	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_D join par_G: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);


	    	    	    	        	 //JOIN (par_D, par_H)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,index_H_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_D*sum_vec_H;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum D=%d, sum H=%d cand size=%u\n",sum_vec_D,sum_vec_H,(unsigned int)sum_vec_D*(unsigned int)sum_vec_H);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_D_dim_ptr_d,index_D_id_d,index_D_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	     	    		sum_cand=(unsigned int)sum_vec_D*(unsigned int)sum_vec_H;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    	    	    																												pre_norm_D1_d,pre_norm_H1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	  	    	    		block_num=sum_vec_H/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_D;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_D_d,feature_D_d,tile_start_D_d,dim_H_d,feature_H_d,tile_start_H_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_D,sum_vec_H,similarity_val_d,sum_pair_d);
	    	    	    	        		 cudaThreadSynchronize();

//	    	    	    	    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_D join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	        	 //JOIN (par_E, par_F)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,index_F_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_E*sum_vec_F;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum E=%d, sum F=%d cand size=%u\n",sum_vec_E,sum_vec_F,(unsigned int)sum_vec_E*(unsigned int)sum_vec_F);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_E_dim_ptr_d,index_E_id_d,index_E_val_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	    	    	    		sum_cand=(unsigned int)sum_vec_E*(unsigned int)sum_vec_F;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d,
	    	    	    	    																												pre_norm_E1_d,pre_norm_F1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	 	    		block_num=sum_vec_F/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_F%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_E;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_E_d,feature_E_d,tile_start_E_d,dim_F_d,feature_F_d,tile_start_F_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,sum_vec_F,similarity_val_d,sum_pair_d);
	    	    	    	        	cudaThreadSynchronize();

	    	    	    	    //		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_F, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_E join par_F: sum pair=%d\n\n",sum_pair[0]);

	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	        	 //JOIN (par_E, par_G)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,index_G_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_E*sum_vec_G;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum E=%d, sum G=%d cand size=%u\n",sum_vec_E,sum_vec_G,(unsigned int)sum_vec_E*(unsigned int)sum_vec_G);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_E_dim_ptr_d,index_E_id_d,index_E_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	    	    	   		sum_cand=(unsigned int)sum_vec_E*(unsigned int)sum_vec_G;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	    	    	    	    																												pre_norm_E1_d,pre_norm_G1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	    	   		block_num=sum_vec_G/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_G%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_E;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_E_d,feature_E_d,tile_start_E_d,dim_G_d,feature_G_d,tile_start_G_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,sum_vec_G,similarity_val_d,sum_pair_d);
	    	    	    	        		 cudaThreadSynchronize();

	    	    	    	//    		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_E join par_G: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	        	 //JOIN (par_E, par_H)
	    	    	    	        	 join_block_sum[0]=0;
	    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,index_H_dim_ptr_d,
	    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	             cudaThreadSynchronize();
	    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_E*sum_vec_H;
	    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    	    printf("sum E=%d, sum H=%d cand size=%u\n",sum_vec_E,sum_vec_H,(unsigned int)sum_vec_E*(unsigned int)sum_vec_H);
	    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_E_dim_ptr_d,index_E_id_d,index_E_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	    	    	    	    		cudaThreadSynchronize();

	    	     	    		sum_cand=(unsigned int)sum_vec_E*(unsigned int)sum_vec_H;

	    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    	    	    																												pre_norm_E1_d,pre_norm_H1_d);
	    	    	    	    		cudaThreadSynchronize();
	    	    	   	    		block_num=sum_vec_H/BLOCK_SIZE_REST;
	    	    	    	    		if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    	    	    	    			block_num++;
	    	    	    	    		GridSizeRest=block_num*sum_vec_E;
	    	    	    	    		sum_pair[0]=0;
	    	    	    	    		cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_E_d,feature_E_d,tile_start_E_d,dim_H_d,feature_H_d,tile_start_H_d,
	    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_E,sum_vec_H,similarity_val_d,sum_pair_d);
	    	    	    	        		 cudaThreadSynchronize();

	    	    	    	    	//	verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    	     printf("par_E join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	    		 //JOIN (par_F, par_G)
	    	    	    	    		    	    	    	        	 join_block_sum[0]=0;
	    	    	    	    		    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,index_G_dim_ptr_d,
	    	    	    	    		    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	    		    	    	    	             cudaThreadSynchronize();
	    	    	    	    		    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	    		    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    		    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,
	    	    	    	    		    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    		    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_F*sum_vec_G;
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    		    	    	    	    	    printf("sum F=%d, sum G=%d cand size=%u\n",sum_vec_F,sum_vec_G,(unsigned int)sum_vec_F*(unsigned int)sum_vec_G);
	    	    	    	    		    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    		    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_F_dim_ptr_d,index_F_id_d,index_F_val_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
	    	    	    	    		    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,similarity_val_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();

	    	    	    	    		    	    	    		sum_cand=(unsigned int)sum_vec_F*(unsigned int)sum_vec_G;

	    	    	    	    		    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d,
	    	    	    	    		    	    	    	    																												pre_norm_F1_d,pre_norm_G1_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();
	    	    	    	    		    	       	    		block_num=sum_vec_G/BLOCK_SIZE_REST;
	    	    	    	    		    	    	    	    		if(sum_vec_G%BLOCK_SIZE_REST!=0)
	    	    	    	    		    	    	    	    			block_num++;
	    	    	    	    		    	    	    	    		GridSizeRest=block_num*sum_vec_F;
	    	    	    	    		    	    	    	    	sum_pair[0]=0;
	    	    	    	    		    	    	    	    	cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_F_d,feature_F_d,tile_start_F_d,dim_G_d,feature_G_d,tile_start_G_d,
	    	    	    	    		    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,sum_vec_G,similarity_val_d,sum_pair_d);
	    	    	    	    		    	    	    	        		 cudaThreadSynchronize();

	    	    	    	    		    	    	    	   // 		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_G, sum_pair_d);
	    	    	    	    		    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	    	     printf("par_F join par_G: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    		    	    	    	    	     cudaFree(similarity_val_d);

	    	    	    	    		    	    	    	        	 //JOIN (par_F, par_H)
	    	    	    	    		    	    	    	        	 join_block_sum[0]=0;
	    	    	    	    		    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,index_H_dim_ptr_d,
	    	    	    	    		    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	    		    	    	    	             cudaThreadSynchronize();
	    	    	    	    		    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	    		    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    		    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    	    	    		    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    		    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_F*sum_vec_H;
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    		    	    	    	    	    printf("sum F=%d, sum H=%d cand size=%u\n",sum_vec_F,sum_vec_H,(unsigned int)sum_vec_F*(unsigned int)sum_vec_H);
	    	    	    	    		    	    	    	    		similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    		    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 index_F_dim_ptr_d,index_F_id_d,index_F_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    	    	    		    	    	    	    					 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_H,similarity_val_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();

	    	    	    	    		    	    	    	    		sum_cand=(unsigned int)sum_vec_F*(unsigned int)sum_vec_H;

	    	    	    	    		    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    	    	    		    	    	    	    																												pre_norm_F1_d,pre_norm_H1_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();
	    	    	    	    		    	    	    	block_num=sum_vec_H/BLOCK_SIZE_REST;
	    	    	    	    		    	    	    	    		if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    	    	    	    		    	    	    	    			block_num++;
	    	    	    	    		    	    	    	    		GridSizeRest=block_num*sum_vec_F;
	    	    	    	    		    	    	    	    		sum_pair[0]=0;
	    	    	    	    		    	    	    	    	cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_F_d,feature_F_d,tile_start_F_d,dim_H_d,feature_H_d,tile_start_H_d,
	    	    	    	    		    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_F,sum_vec_H,similarity_val_d,sum_pair_d);
	    	    	    	    		    	    	    	        	cudaThreadSynchronize();
	    	    	    	    		    	    	    	    		//verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	    	    		    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	    	     printf("par_F join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    		    	    	    	    	     cudaFree(similarity_val_d);


	    	    	    	    		    	    	    	        	 //JOIN (par_G, par_H)66
	    	    	    	    		    	    	    	        	 join_block_sum[0]=0;
	    	    	    	    		    	    	    	        	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	     		generate_auxiliary_variable<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,index_H_dim_ptr_d,
	    	    	    	    		    	    	    	     																																					join_dim_start_block_d, join_dim_block_num_d,join_block_id_d); //生成的辅助变量
	    	    	    	    		    	    	    	             cudaThreadSynchronize();
	    	    	    	    		    	    	    	             cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	             printf("sum_block=%d\n",join_block_sum[0]);
	    	    	    	    		    	    	    	             cudaFree(join_block_dim_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_a_d);
	    	    	    	    		    	    	    	             cudaFree(join_block_seg_b_d);
	    	    	    	    		    	    	    	    	 	cudaMalloc((void**)&join_block_dim_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_a_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&join_block_seg_b_d,sizeof(int)*join_block_sum[0]);
	    	    	    	    		    	    	    	    		generate_auxiliary_variable2<<<SumBlockGenerate, BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,
	    	    	    	    		    	    	    	    																															join_dim_start_block_d,join_dim_block_num_d,//需要的辅助变量
	    	    	    	    		    	    	    	    																															join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d); //生成的辅助变量
	    	    	    	    		    	    	    	    		 cudaThreadSynchronize();
	    	    	    	    		    	    	    	    		size= (unsigned long)sizeof(float)*sum_vec_G*sum_vec_H;
	    	    	    	    		    	    	    	    		cudaMalloc((void**)&similarity_val_d, size);
	    	    	    	    		    	    	    	    		printf("malloc size=%lu\n",size);
	    	    	    	    		    	    	    	    	    printf("sum G=%d, sum H=%d cand size=%u\n",sum_vec_G,sum_vec_H,(unsigned int)sum_vec_G*(unsigned int)sum_vec_H);
	    	    	    	    		    	    	    	    	    similarity_join1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>( join_block_dim_d,join_block_seg_a_d,join_block_seg_b_d,
	    	    	    	    		    	    	    	    	    																											index_G_dim_ptr_d,index_G_id_d,index_G_val_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
	    	    	    	    		    	    	    	    	    																											sum_vec_H,similarity_val_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();


	    	    	    	    		    		    		sum_cand=(unsigned int)sum_vec_G*(unsigned int)sum_vec_H;

	    	    	    	    		    	    	    	    		GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;
	    	    	    	    		    	    	    	    		filter_pair<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d,
	    	    	    	    		    	    	    	    																												pre_norm_G1_d,pre_norm_H1_d);
	    	    	    	    		    	    	    	    		cudaThreadSynchronize();
	    	    	    	    		    	    	         	    		block_num=sum_vec_H/BLOCK_SIZE_REST;
	    	    	    	    		    	    	    	    		if(sum_vec_H%BLOCK_SIZE_REST!=0)
	    	    	    	    		    	    	    	    			block_num++;
	    	    	    	    		    	    	    	    		GridSizeRest=block_num*sum_vec_G;
	    	    	    	    		    	    	    	    	sum_pair[0]=0;
	    	    	    	    		    	    	    	    	cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	    	    		    	    	    	    	similarity_join_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_G_d,feature_G_d,tile_start_G_d,dim_H_d,feature_H_d,tile_start_H_d,
	    	    	    	    		    	    	    				    		       				 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 sum_vec_G,sum_vec_H,similarity_val_d,sum_pair_d);
	    	    	    	    		    	    	    	        		 cudaThreadSynchronize();

	    	    	    	    		    	    	    	    //		verify_val<<<GridSizeVerify, BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,sum_cand, sum_vec_H, sum_pair_d);
	    	    	    	    		    	    	    	    	     cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	    	    		    	    	    	    	     printf("par_G join par_H: sum pair=%d\n\n",sum_pair[0]);
	    	    	    	    		    	    	    	    	     cudaFree(similarity_val_d);



	  cudaFree(index_A_dim_num_d);
	  cudaFree(index_B_dim_num_d);
	  cudaFree(index_C_dim_num_d);
	  cudaFree(index_D_dim_num_d);

	   endTime = clock();
	    printf("Cuda Time(join) : %.6f\n\n", (double)(endTime - startTime)/CLOCKS_PER_SEC );


	//   clock_t startTime,endTime;
	   startTime = clock();
	    int* b_id_a_d;// apss_verify阶段每个block计算对应的vector_a_id
	    cudaMalloc((void**)&b_id_a_d, sizeof(int)*1249975000);
		cudaMemcpy(b_id_a_d,b_id_a,sizeof(int)*1249975000,cudaMemcpyHostToDevice);

    	 //APSS partition_A
    	 join_block_sum[0]=0;
    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
         //------------------------------------------------------------------用于下面kernel的辅助变量
 		int* apss_block_dim_d; //每个block计算的dim
 		int* apss_block_seg_A_d; //每个block计算的seg_A
 		int* apss_block_seg_B_d; //每个block计算的seg_B
    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
    	 printf("sum_block=%d\n",join_block_sum[0]);

	 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
		cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_A_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
    	 cudaThreadSynchronize();
    	 size= (unsigned long)sizeof(float)*sum_vec_A*(sum_vec_A-1)/2;
    	 cudaMalloc((void**)&similarity_val_d, size);
	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_A_dim_ptr_d,index_A_id_d,index_A_val_d,
	 	 	 	 	 	 	 	 	 	      similarity_val_d);
	    	 cudaThreadSynchronize();
	    	sum_cand=(unsigned long)sum_vec_A*(sum_vec_A-1)/2;
	    	 printf("apss sum_cand=%u\n",sum_cand);
	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

	    	filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_A1_d);
	    	 cudaThreadSynchronize();
	    	 int seg_sum=sum_vec_A/1024+1;
	    	GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
	     	printf("Grid size Rest=%d\n",GridSizeRest);
	     	sum_pair[0]=0;
	    	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   	     similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_A_d,feature_A_d,tile_start_A_d,sum_vec_A, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
	         cudaThreadSynchronize();
	   // 	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand, sum_pair_d);
	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	 printf("par_A apss: sum pair=%d\n\n",sum_pair[0]);
	    	 cudaFree(dim_A_d);
	    	 cudaFree(feature_A_d);
	    	  cudaFree(tile_start_A_d);
	    	  cudaFree(similarity_val_d);

	    	 //APSS partition_B
	    	    	 join_block_sum[0]=0;
	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
	    	         cudaFree(apss_block_dim_d);
	    	         cudaFree(apss_block_seg_A_d);
	    	         cudaFree(apss_block_seg_B_d);
	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_B_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
	    	    	 cudaThreadSynchronize();
	    	    	 size= sizeof(float)*(unsigned long)sum_vec_B*(sum_vec_B-1)/2;
	    	    	 cudaMalloc((void**)&similarity_val_d, size);
	    	    	 printf("malloc size=%lu\n",size);
	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_B_dim_ptr_d,index_B_id_d,index_B_val_d,
	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
	    	    	 cudaThreadSynchronize();
	    	    	sum_cand=(unsigned long)sum_vec_B*(sum_vec_B-1)/2;
	    	    	 printf("apss sum_cand=%u\n",sum_cand);
	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

	    	    	 filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_B1_d);
	    	    	 cudaThreadSynchronize();
	    	    	 seg_sum=sum_vec_B/1024+1;
	    	    	 GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
	    	    	 printf("Grid size Rest=%d\n",GridSizeRest);
	    	          sum_pair[0]=0;
	    	    	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	    	    	 similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_B_d,feature_B_d,tile_start_B_d,sum_vec_B, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
	    	    	 cudaThreadSynchronize();
	    //	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	    	    	 printf("par_B apss: sum pair=%d\n\n",sum_pair[0]);
	    	    	cudaFree(dim_B_d);
	    	    	cudaFree(feature_B_d);
	    	    	cudaFree(tile_start_B_d);
	    	    	 cudaFree(similarity_val_d);

	   	    	 //APSS partition_C
	   	    	    	 join_block_sum[0]=0;
	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
	   	    	         cudaFree(apss_block_dim_d);
	   	    	         cudaFree(apss_block_seg_A_d);
	   	    	         cudaFree(apss_block_seg_B_d);
	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_C_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
	   	    	    	 cudaThreadSynchronize();
	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_C*(sum_vec_C-1)/2;
	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
	   	    	    	 printf("malloc size=%lu\n",size);
	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_C_dim_ptr_d,index_C_id_d,index_C_val_d,
	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
	   	    	    	 cudaThreadSynchronize();
	   	    	    	sum_cand=(unsigned long)sum_vec_C*(sum_vec_C-1)/2;
	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_C1_d);
	   	    	    	 cudaThreadSynchronize();
	   	    	    	 seg_sum=sum_vec_C/1024+1;
	   	    	    	GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
	   	    	    	printf("Grid size Rest=%d\n",GridSizeRest);
	   	    	          sum_pair[0]=0;
	   	    	    	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   	    	    	similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_C_d,feature_C_d,tile_start_C_d,sum_vec_C, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
	   	    	    	cudaThreadSynchronize();
	   	   // 	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	    	    	 printf("par_C apss: sum pair=%d\n\n",sum_pair[0]);
	   	    	    	cudaFree(dim_C_d);
	   	    	    	cudaFree(feature_C_d);
	   	    	    	cudaFree(tile_start_C_d);
	   	    	     cudaFree(similarity_val_d);

	   	   	    	 //APSS partition_D
	   	   	    	    	 join_block_sum[0]=0;
	   	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
	   	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
	   	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
	   	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
	   	   	    	         cudaFree(apss_block_dim_d);
	   	   	    	         cudaFree(apss_block_seg_A_d);
	   	   	    	         cudaFree(apss_block_seg_B_d);
	   	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
	   	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
	   	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
	   	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_D_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
	   	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
	   	   	    	    	 cudaThreadSynchronize();
	   	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_D*(sum_vec_D-1)/2;
	   	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
	   	   	    	    	 printf("malloc size=%lu\n",size);
	   	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_D_dim_ptr_d,index_D_id_d,index_D_val_d,
	   	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
	   	   	    	    	 cudaThreadSynchronize();
	   	   	    	    	sum_cand=(unsigned long)sum_vec_D*(sum_vec_D-1)/2;
	   	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
	   	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

	   	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_D1_d);
	   	   	    	    	 cudaThreadSynchronize();
	   	   	    	    	 seg_sum=sum_vec_D/1024+1;
	   	   	    	  	    	GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
	   	   	    	  	     	printf("Grid size Rest=%d\n",GridSizeRest);
		   	   	    	          sum_pair[0]=0;
		   	   	    	    	 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
	   	   	    	  	   	     similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_D_d,feature_D_d,tile_start_D_d,sum_vec_D, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
	   	   	    	  	         cudaThreadSynchronize();
	   	   	 //   	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
	   	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
	   	   	    	    	 printf("par_D apss: sum pair=%d\n\n",sum_pair[0]);
	   	   	    	    	cudaFree(dim_D_d);
	   	   	    	    	cudaFree(feature_D_d);
	   	   	    	    	cudaFree(tile_start_D_d);
	   	   	    	  cudaFree(similarity_val_d);

		   	   	    	 //APSS partition_E
		   	   	    	    	 join_block_sum[0]=0;
		   	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
		   	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
		   	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
		   	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
		   	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
		   	   	    	         cudaFree(apss_block_dim_d);
		   	   	    	         cudaFree(apss_block_seg_A_d);
		   	   	    	         cudaFree(apss_block_seg_B_d);
		   	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
		   	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
		   	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
		   	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_E_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
		   	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
		   	   	    	    	 cudaThreadSynchronize();
		   	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_E*(sum_vec_E-1)/2;
		   	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
		   	   	    	    	 printf("malloc size=%lu\n",size);
		   	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_E_dim_ptr_d,index_E_id_d,index_E_val_d,
		   	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
		   	   	    	    	 cudaThreadSynchronize();
		   	   	    	    	sum_cand=(unsigned long)sum_vec_E*(sum_vec_E-1)/2;
		   	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
		   	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

		   	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_E1_d);
		   	   	    	    	 cudaThreadSynchronize();
		   	   	    	    	 seg_sum=sum_vec_E/1024+1;
		   	   	    	  	    GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
		   	   	    	  	     printf("Grid size Rest=%d\n",GridSizeRest);
			   	   	    	     sum_pair[0]=0;
			   	   	    	     cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
		   	   	    	  	   	 similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_E_d,feature_E_d,tile_start_E_d,sum_vec_E, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
		   	   	    	  	      cudaThreadSynchronize();
		   	   	 //   	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
		   	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
		   	   	    	    	 printf("par_E apss: sum pair=%d\n\n",sum_pair[0]);
		   	   	    	    	cudaFree(dim_E_d);
		   	   	    	    	cudaFree(feature_E_d);
		   	   	    	    	cudaFree(tile_start_E_d);
		   	   	    	  cudaFree(similarity_val_d);

			   	   	    	 //APSS partition_F
			   	   	    	    	 join_block_sum[0]=0;
			   	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
			   	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
			   	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
			   	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
			   	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
			   	   	    	         cudaFree(apss_block_dim_d);
			   	   	    	         cudaFree(apss_block_seg_A_d);
			   	   	    	         cudaFree(apss_block_seg_B_d);
			   	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
			   	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
			   	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
			   	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_F_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
			   	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
			   	   	    	    	 cudaThreadSynchronize();
			   	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_F*(sum_vec_F-1)/2;
			   	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
			   	   	    	    	 printf("malloc size=%lu\n",size);
			   	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_F_dim_ptr_d,index_F_id_d,index_F_val_d,
			   	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
			   	   	    	    	 cudaThreadSynchronize();
			   	   	    	    	sum_cand=(unsigned long)sum_vec_F*(sum_vec_F-1)/2;
			   	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
			   	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

			   	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_F1_d);
			   	   	    	    	 cudaThreadSynchronize();
			   	   	    	    	 seg_sum=sum_vec_F/1024+1;
			   	   	    	  	    GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
			   	   	    	  	     printf("Grid size Rest=%d\n",GridSizeRest);
			   	   	    	  	     sum_pair[0]=0;
			   	   	    		     cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
			   	   	    	  	   	 similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_F_d,feature_F_d,tile_start_F_d,sum_vec_F, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
			   	   	    	  	     cudaThreadSynchronize();
			   	   	 //   	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
			   	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
			   	   	    	    	 printf("par_F apss: sum pair=%d\n\n",sum_pair[0]);
			   	   	    	    	cudaFree(dim_F_d);
			   	   	    	    	cudaFree(feature_F_d);
			   	   	    	    	cudaFree(tile_start_F_d);
			   	   	    	  cudaFree(similarity_val_d);

				   	   	    	 //APSS partition_G
				   	   	    	    	 join_block_sum[0]=0;
				   	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
				   	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
				   	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
				   	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
				   	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
				   	   	    	         cudaFree(apss_block_dim_d);
				   	   	    	         cudaFree(apss_block_seg_A_d);
				   	   	    	         cudaFree(apss_block_seg_B_d);
				   	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
				   	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
				   	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
				   	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_G_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
				   	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
				   	   	    	    	 cudaThreadSynchronize();
				   	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_G*(sum_vec_G-1)/2;
				   	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
				   	   	    	    	 printf("malloc size=%lu\n",size);
				   	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_G_dim_ptr_d,index_G_id_d,index_G_val_d,
				   	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
				   	   	    	    	 cudaThreadSynchronize();
				   	   	    	    	sum_cand=(unsigned long)sum_vec_G*(sum_vec_G-1)/2;
				   	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
				   	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

				   	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_G1_d);
				   	   	    	    	cudaThreadSynchronize();
				   	   	    	    	seg_sum=sum_vec_G/1024+1;
				   	   	    	  	    GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
				   	   	    	  	     printf("Grid size Rest=%d\n",GridSizeRest);
				   	   	    	  	     sum_pair[0]=0;
				   	   	    	  		 cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
				   	   	    	  	   	 similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_G_d,feature_G_d,tile_start_G_d,sum_vec_G, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
				   	   	    	  	     cudaThreadSynchronize();
				   	   	 //   	    	 verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
				   	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
				   	   	    	    	 printf("par_G apss: sum pair=%d\n\n",sum_pair[0]);
				   	   	    	    	cudaFree(dim_G_d);
				   	   	    	    	cudaFree(feature_G_d);
				   	   	    	    	cudaFree(tile_start_G_d);
				   	   	    	  cudaFree(similarity_val_d);

					   	   	    	 //APSS partition_H
					   	   	    	    	 join_block_sum[0]=0;
					   	   	    	    	 cudaMemcpy(join_block_id_d,join_block_sum,sizeof(int),cudaMemcpyHostToDevice);
					   	   	    	    	 generate_auxiliary_variable_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d, join_dim_start_block_d, join_dim_block_num_d, join_block_id_d);
					   	   	    	         //------------------------------------------------------------------用于下面kernel的辅助变量
					   	   	    	    	 cudaMemcpy(join_block_sum,join_block_id_d,sizeof(int),cudaMemcpyDeviceToHost);
					   	   	    	    	 printf("sum_block=%d\n",join_block_sum[0]);
					   	   	    	         cudaFree(apss_block_dim_d);
					   	   	    	         cudaFree(apss_block_seg_A_d);
					   	   	    	         cudaFree(apss_block_seg_B_d);
					   	   	    		 	cudaMalloc((void**)&apss_block_dim_d,sizeof(int)*join_block_sum[0]);
					   	   	    			cudaMalloc((void**)&apss_block_seg_A_d,sizeof(int)*join_block_sum[0]);
					   	   	    			cudaMalloc((void**)&apss_block_seg_B_d,sizeof(int)*join_block_sum[0]);
					   	   	    	    	generate_auxiliary_variable2_apss<<<SumBlockGenerate,BLOCK_SIZE_GENERATE_VAR>>>(index_H_dim_ptr_d,join_dim_start_block_d,join_dim_block_num_d,
					   	   	    	    			 	 	 	 	 	 	 	 	 	 	 apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d);
					   	   	    	    	 cudaThreadSynchronize();
					   	   	    	    	 size= sizeof(float)*(unsigned long)sum_vec_H*(sum_vec_H-1)/2;
					   	   	    	    	 cudaMalloc((void**)&similarity_val_d, size);
					   	   	    	    	 printf("malloc size=%lu\n",size);
					   	   	    	     similarity_apss1<<<join_block_sum[0],BLOCK_SIZE_JOIN>>>(apss_block_dim_d,apss_block_seg_A_d,apss_block_seg_B_d,index_H_dim_ptr_d,index_H_id_d,index_H_val_d,
					   	   	    	 	 	 	 	 	 	 	 	 	      similarity_val_d);
					   	   	    	    	 cudaThreadSynchronize();
					   	   	    	    	sum_cand=(unsigned long)sum_vec_H*(sum_vec_H-1)/2;
					   	   	    	    	 printf("apss sum_cand=%u\n",sum_cand);
					   	   	    	    	 GridSizeVerify=(sum_cand/BLOCK_SIZE_VERIFY_VAL)+1;

					   	   	    	       filter_pair_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d,  sum_cand,  b_id_a_d, pre_norm_H1_d);
					   	   	    	    	cudaThreadSynchronize();
					   	   	    	    	seg_sum=sum_vec_H/1024+1;
					   	   	    	  	    GridSizeRest=1024*(1+seg_sum)*seg_sum/2;
					   	   	    	  	    printf("Grid size Rest=%d\n",GridSizeRest);
						   	   	    	    sum_pair[0]=0;
						   	   	    	    cudaMemcpy(sum_pair_d,sum_pair,sizeof(int),cudaMemcpyHostToDevice);
					   	   	    	  	   	similarity_apss_rest<<<GridSizeRest,BLOCK_SIZE_REST>>>(dim_H_d,feature_H_d,tile_start_H_d,sum_vec_H, apss_block_seg_d,apss_seg_start_d,similarity_val_d,sum_pair_d);
					   	   	    	  	    cudaThreadSynchronize();
					   	   //	    	    verify_val_apss<<<GridSizeVerify,BLOCK_SIZE_VERIFY_VAL>>>(similarity_val_d, sum_cand,  sum_pair_d);
					   	   	    	    	 cudaMemcpy(sum_pair,sum_pair_d,sizeof(int),cudaMemcpyDeviceToHost);
					   	   	    	    	 printf("par_H apss: sum pair=%d\n\n",sum_pair[0]);
					   	   	    	    	cudaFree(dim_H_d);
					   	   	    	    	cudaFree(feature_H_d);
					   	   	    	    	cudaFree(tile_start_H_d);
					   	   	    	    	cudaFree(similarity_val_d);

	   	   	   		   endTime = clock();
	   	   	   		   printf("Cuda Time(apss) : %.6f\n", (double)(endTime - startTime)/CLOCKS_PER_SEC );
		cudaFree(index_A_dim_ptr_d);
		cudaFree(index_A_id_d);
		cudaFree(index_A_val_d);
		cudaFree(index_B_dim_ptr_d);
		cudaFree(index_B_id_d);
		cudaFree(index_B_val_d);
     	cudaFree(join_dim_start_block_d);
		cudaFree(join_dim_block_num_d);
		cudaFree(join_block_id_d);
		cudaFree(join_block_dim_d);
		cudaFree(join_block_seg_a_d);
		cudaFree(join_block_seg_b_d);
	    cudaFree(apss_block_dim_d);
		cudaFree(apss_block_seg_A_d);
		cudaFree(apss_block_seg_B_d);
		cudaFree(apss_block_seg_d);
		cudaFree(apss_seg_start_d);
		 cudaFree(similarity_val_d);
		cudaThreadSynchronize();
 }
 static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
 {
 	if (err == cudaSuccess)
 		return;
 	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
 	exit (1);
 }
