#include <stdio.h> 
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 
#include "similarity_search.h"
using  std::string;

#pragma comment( lib,"winmm.lib" )
#define DATA_SIZE 10

const int max_dims = 198456;//41118,174954


//const int sum_seg = 98;//25,40

int sum_index_terms = 6498195;//88276,552295

struct Term{
	Term(int id, float val)	{ this->id = id; this->val = val; };
	Term(int id,int abs_id, float val)	{ this->id = id; this->abs_id=abs_id; this->val = val; };
	Term(int id, float val, float norm){ this->id = id; this->val = val; this->norm = norm; };
	int id;//相对id
	int abs_id;//绝对id
	float val;
	float norm;
};

struct InvertedList {
	std::vector<Term*> terms;   //一组 vector
};

struct Vector{
	int vec_id;
	int vec_size;
	double vec_max;
	double norm1;
	std::pair<int,double>* feature;

	Vector(int id, int size, double max, double norm1,std::pair<int,double>* begin)
	{
		this->vec_id=id;
		this->vec_size=size;
		this->vec_max=max;
		this->norm1=norm1;
		this->feature=new std::pair<int,double>[size];
		memcpy(feature,begin,sizeof(std::pair<int,double>)*size);
	}
};

struct Partition{
	int id;
	double* col_max;
	std::vector<Vector*> vector_list;
	Partition(int i)
	{
		this->id=i;
		col_max=new double[max_dims];
		for(int j=0;j<max_dims;j++)
			col_max[j]=0.0;
	}
	void addVector(int vec_id, int size, double max, double norm1, std::pair<int,double>* feature)
	{
		Vector* vector= new Vector(vec_id, size, max, norm1,feature);
		this->vector_list.push_back(vector);
	}
};

Partition pA(0), pB(1), pC(2), pD(3);
std::vector<InvertedList> inverted_index_A;
std::vector<InvertedList> inverted_index_B;
std::vector<InvertedList> inverted_index_C;
std::vector<InvertedList> inverted_index_D;
std::vector<InvertedList> inverted_index_E;
std::vector<InvertedList> index_lists;

std::vector<Vector*> partition_A;
std::vector<InvertedList> index_list_all;

std::multimap<double,Vector*> norm1_vec_map;//<norm1,vec>

void initial()
{
	inverted_index_A.clear();
	inverted_index_A.resize(max_dims);
	inverted_index_B.clear();
	inverted_index_B.resize(max_dims);
	inverted_index_C.clear();
	inverted_index_C.resize(max_dims);
	inverted_index_D.clear();
	inverted_index_D.resize(max_dims);
	inverted_index_E.clear();
	inverted_index_E.resize(max_dims);
	index_lists.clear();
	index_lists.resize(max_dims);

}

void InvertedIndex_search()
{
	int sum_pair=0;
	std::map<int,float> cand;

	for(int i=0;i<pB.vector_list.size();i++)
	{

			printf("*id=%d\n",i);
			Vector* vector=pB.vector_list[i];
			cand.clear();

			for(int j=0;j<vector->vec_size;j++) //every feature of vector
			{

				for(int t=0;t<inverted_index_E[vector->feature[j].first].terms.size();t++)
				{
					int vec_id=inverted_index_E[vector->feature[j].first].terms[t]->id;
					//if(vec_id>=i)
						//break;
					float vec_fea=inverted_index_E[vector->feature[j].first].terms[t]->val;
					cand[vec_id]=cand[vec_id]+vec_fea*vector->feature[j].second;

				 /* if(vector->vec_id==10568&&vec_id==9927)
					printf("d=%d f_A=%.8f f_B=%.8f cand[]=%.8f\n ",vector->feature[j].first,vector->feature[j].second,vec_fea,cand[vec_id]);*/
			}
			}
		for(std::map<int,float>::iterator it=cand.begin();it!=cand.end();it++)
		{
			 if(it->second>=0.8)
				 sum_pair++;
		}

	}
	//output.clear();
	//output.close();
	printf("sum_pair=%d\n",sum_pair);
}

void compare()
{
	std::ifstream input1,input2;
	input1.open("p1-p0-result-parallel.txt");
	input2.open("p1-p0-result-serial.txt");
	char ch1, ch2;
	int id_A_p,id_B_p,id_A_s,id_B_s;
	for(int i=0;i<22000;i++)
	{
			input1>>id_A_p>> id_B_p;
			input1.get(ch1);
			while(ch1!='\n')
			{
				input1.get(ch1);
			}

		input2>>id_A_s>>id_B_s;
		input2.get(ch2);
		while (ch2 != '\n')
		{
			input2.get(ch2);
		}

		if(id_A_p!=id_A_s||id_B_p!=id_B_s)
			printf("i=%d %d %d\n",i,id_A_p,id_B_p);
	}
	input1.close();
	input2.close();
}



int main()
{
	initial();


	int sum_vec_A=50000;//每个partition的vector数目
	int sum_vec_B=50000;
	int sum_vec_C=50000;
	int sum_vec_D=50000;
	int sum_vec_E=50000;
	int sum_vec_F=50000;
	int sum_vec_G=50000;
	int sum_vec_H=48667;

	int sum_tile_A=1563;
	int sum_tile_B=1563;
	int sum_tile_C=1563;
	int sum_tile_D=1563;
	int sum_tile_E=1563;
	int sum_tile_F=1563;
	int sum_tile_G=1563;
	int sum_tile_H=1521;

	int sum_feat_A=9507360;
	int sum_feat_B=6179104;
	int sum_feat_C=4520896;
	int sum_feat_D=3472544;  //sum feature=15359808
	int sum_feat_E=2716928;
	int sum_feat_F=2063968;
	int sum_feat_G=1513408;
	int sum_feat_H=943872;

	int* dim_A=new int[sum_feat_A];
	float* feature_A=new float[sum_feat_A];
	int* tile_start_A=new int[sum_tile_A+1];
	int* dim_B=new int[sum_feat_B];
	float* feature_B=new float[sum_feat_B];
	int* tile_start_B=new int[sum_tile_B+1];
	int* dim_C=new int[sum_feat_C];
	float* feature_C=new float[sum_feat_C];
	int* tile_start_C=new int[sum_tile_C+1];
	int* dim_D=new int[sum_feat_D];
	float* feature_D=new float[sum_feat_D];
	int* tile_start_D=new int[sum_tile_D+1];
	int* dim_E=new int[sum_feat_E];
	float* feature_E=new float[sum_feat_E];
	int* tile_start_E=new int[sum_tile_E+1];
	int* dim_F=new int[sum_feat_F];
	float* feature_F=new float[sum_feat_F];
	int* tile_start_F=new int[sum_tile_F+1];
	int* dim_G=new int[sum_feat_G];
	float* feature_G=new float[sum_feat_G];
	int* tile_start_G=new int[sum_tile_G+1];
	int* dim_H=new int[sum_feat_H];
	float* feature_H=new float[sum_feat_H];
	int* tile_start_H=new int[sum_tile_H+1];

	int* b_id_a=new int[1249975000]; //apss_verify中每个block计算的vector_a的id

	float* pre_norm_A=new float[sum_vec_A];
	float* pre_norm_B=new float[sum_vec_B];
	float* pre_norm_C=new float[sum_vec_C];
	float* pre_norm_D=new float[sum_vec_D];
	float* pre_norm_E=new float[sum_vec_E];
	float* pre_norm_F=new float[sum_vec_F];
	float* pre_norm_G=new float[sum_vec_G];
	float* pre_norm_H=new float[sum_vec_H];

	for(int i=0;i<=sum_tile_A;i++)
		tile_start_A[i]=0;
	for(int i=0;i<=sum_tile_B;i++)
		tile_start_B[i]=0;
	for(int i=0;i<=sum_tile_C;i++)
		tile_start_C[i]=0;
	for(int i=0;i<=sum_tile_D;i++)
		tile_start_D[i]=0;
	for(int i=0;i<=sum_tile_E;i++)
		tile_start_E[i]=0;
	for(int i=0;i<=sum_tile_F;i++)
		tile_start_F[i]=0;
	for(int i=0;i<=sum_tile_G;i++)
		tile_start_G[i]=0;
	for(int i=0;i<=sum_tile_H;i++)
		tile_start_H[i]=0;

	for(unsigned int a=1;a<50000;a++)
	{
		for(unsigned int b_id=a*(a-1)/2; b_id<(a+1)*a/2; b_id++)
		{
			b_id_a[b_id]=a;
		}
	}


	std::ifstream input;
	input.open("test-pt0-pt1-tf-idf-normalized-by-size.txt");
	int standard_size = 0;
	int tile_start_index = 0;
	int size=0;
	int j=0; //feature 序号
	double max;
	double norm1;
	double pre_norm_t1=0;
	double pre_norm_1=0;

	int d;
	float v;
	char ch;
	int index=0;
	std::vector<std::pair<int, double> > vector;
	//partition A
	for(int i=0;i<sum_feat_A;i++)
		dim_A[i]=-1;
	for (int vec_id = 0; vec_id < sum_vec_A; vec_id++)
	{
		size=0;
		j=0;
		max=-9999;
		pre_norm_t1=0.0;
		pre_norm_1=0.0;

		vector.clear();
		if (vec_id % 32 == 0)
		{
			tile_start_A[vec_id / 32] = tile_start_index;
		}
		input.get(ch);
		while (ch != '\n')
		{
			input >> d >> v;
			vector.push_back(std::pair<int,double>(d,v));
			dim_A[tile_start_index+(vec_id%32+j*32)]=d;
			feature_A[tile_start_index+(vec_id%32+j*32)]=v;
			if(d<4096)
			{
				pre_norm_t1=pre_norm_t1+pow(v,2);

			}

			if(v>max)
					max=v;
			norm1=norm1+v;
			if(v>pA.col_max[d])
					pA.col_max[d]=v;
		//	inverted_index_A[d].terms.push_back(new Term(vec_id, v));
			size++;
			j++;
			input.get(ch);
		}
		pre_norm_1=sqrt(pre_norm_t1);
		pre_norm_A[vec_id]=pre_norm_1;

		pA.addVector(vec_id, size, max, norm1, &vector[0]);

		if (vec_id % 32 == 0)
		{
			standard_size = size;
		}

		else if (vec_id % 32 == 31)
			tile_start_index += standard_size * 32;
	//	index++;
	}
	tile_start_A[sum_tile_A] = tile_start_A[sum_tile_A - 1] + 32 * standard_size;
	printf("sum feat A=%d\n",tile_start_A[sum_tile_A]);

	//partition B
	tile_start_index=0;
	for(int i=0;i<sum_feat_B;i++)
			dim_B[i]=-1;
	for (int vec_id = 0; vec_id < sum_vec_B; vec_id++)
	{
		size=0;
		j=0;
		max=-9999;
		norm1=0.0;
		pre_norm_t1=0.0;
		pre_norm_1=0.0;

		vector.clear();
		if (vec_id % 32 == 0)
		{
			tile_start_B[vec_id / 32] = tile_start_index;
		}
		input.get(ch);
		while (ch != '\n')
		{
			input >> d >> v;
			dim_B[tile_start_index+(vec_id%32+j*32)]=d;
			feature_B[tile_start_index+(vec_id%32+j*32)]=v;
			vector.push_back(std::pair<int,double>(d,v));
			if(d<4096)
			{
				pre_norm_t1=pre_norm_t1+pow(v,2);
			}

			if(v>max)
					max=v;
			norm1=norm1+v;
			if(v>pB.col_max[d])
					pB.col_max[d]=v;
		   	inverted_index_B[d].terms.push_back(new Term(vec_id, v));
			size++;
			j++;
			input.get(ch);
		}
		pre_norm_1=sqrt(pre_norm_t1);
		pre_norm_B[vec_id]=pre_norm_1;

		pB.addVector(vec_id, size, max, norm1, &vector[0]);
		if (vec_id % 32 == 0)
		{
			standard_size = size;
		}

		else if (vec_id % 32 == 31)
			tile_start_index += standard_size * 32;
		//index++;
	}
	tile_start_B[sum_tile_B] = tile_start_B[sum_tile_B - 1] + 32 * standard_size;
	printf("sum feat B=%d\n",tile_start_B[sum_tile_B]);

	//partition C
	tile_start_index=0;
	for(int i=0;i<sum_feat_C;i++)
		dim_C[i]=-1;
	for (int vec_id = 0; vec_id < sum_vec_C; vec_id++)
	{
		size=0;
		j=0;
		max=-9999;
		norm1=0.0;
		pre_norm_t1=0.0;
		pre_norm_1=0.0;

		vector.clear();
		if (vec_id % 32 == 0)
		{
			tile_start_C[vec_id / 32] = tile_start_index;
		}
		input.get(ch);
		while (ch != '\n')
		{
			input >> d >> v;
			dim_C[tile_start_index+(vec_id%32+j*32)]=d;
			feature_C[tile_start_index+(vec_id%32+j*32)]=v;
			vector.push_back(std::pair<int,double>(d,v));
			if(d<4096)
			{
				pre_norm_t1=pre_norm_t1+pow(v,2);

			}

			if(v>max)
				max=v;
			norm1=norm1+v;
			if(v>pC.col_max[d])
				pC.col_max[d]=v;
			//	inverted_index_C[d].terms.push_back(new Term(vec_id, v));
			size++;
			j++;
			input.get(ch);
		}
		pC.addVector(vec_id, size, max, norm1, &vector[0]);
		pre_norm_1=sqrt(pre_norm_t1);
		pre_norm_C[vec_id]=pre_norm_1;

		if (vec_id % 32 == 0)
		{
			standard_size = size;
		}

		else if (vec_id % 32 == 31)
			tile_start_index += standard_size * 32;
			//index++;
	}
	tile_start_C[sum_tile_C] = tile_start_C[sum_tile_C - 1] + 32 * standard_size;
	printf("sum feat C=%d\n",tile_start_C[sum_tile_C]);

	//partition D
	tile_start_index=0;
	for(int i=0;i<sum_feat_D;i++)
		dim_D[i]=-1;
	for (int vec_id = 0; vec_id < sum_vec_D; vec_id++)
	{
		size=0;
		j=0;
		max=-9999;
		norm1=0.0;
		pre_norm_t1=0.0;
			pre_norm_1=0.0;
		vector.clear();
		if (vec_id % 32 == 0)
		{
			tile_start_D[vec_id / 32] = tile_start_index;

		}
		input.get(ch);
		while (ch != '\n')
		{
			input >> d >> v;
			dim_D[tile_start_index+(vec_id%32+j*32)]=d;
			feature_D[tile_start_index+(vec_id%32+j*32)]=v;
			vector.push_back(std::pair<int,double>(d,v));
			if(d<4096)
			{
				pre_norm_t1=pre_norm_t1+pow(v,2);
			}

			if(v>max)
				max=v;
			norm1=norm1+v;
			if(v>pD.col_max[d])
				pD.col_max[d]=v;
			size++;
			j++;
			input.get(ch);
		}
		pD.addVector(vec_id, size, max, norm1, &vector[0]);

		pre_norm_1=sqrt(pre_norm_t1);
		pre_norm_D[vec_id]=pre_norm_1;

		if (vec_id % 32 == 0)
		{
			standard_size = size;
		}

		else if (vec_id % 32 == 31)
			tile_start_index += standard_size * 32;
			//index++;
	}
	tile_start_D[sum_tile_D] = tile_start_D[sum_tile_D - 1] + 32 * standard_size;
	printf("sum feat D=%d\n",tile_start_D[sum_tile_D]);

	//partition E
		tile_start_index=0;
		for(int i=0;i<sum_feat_E;i++)
				dim_E[i]=-1;
		for (int vec_id = 0; vec_id < sum_vec_E; vec_id++)
		{
			size=0;
			j=0;
			max=-9999;
			norm1=0.0;
			pre_norm_t1=0.0;
			pre_norm_1=0.0;

			vector.clear();
			if (vec_id % 32 == 0)
			{
				tile_start_E[vec_id / 32] = tile_start_index;
			}
			input.get(ch);
			while (ch != '\n')
			{
				input >> d >> v;
				dim_E[tile_start_index+(vec_id%32+j*32)]=d;
				feature_E[tile_start_index+(vec_id%32+j*32)]=v;
				vector.push_back(std::pair<int,double>(d,v));
				if(d<4096)
				{
					pre_norm_t1=pre_norm_t1+pow(v,2);
				}

				if(v>max)
						max=v;
				norm1=norm1+v;

			   	inverted_index_E[d].terms.push_back(new Term(vec_id, v));
				size++;
				j++;
				input.get(ch);
			}
			pre_norm_1=sqrt(pre_norm_t1);
			pre_norm_E[vec_id]=pre_norm_1;

	//		pE.addVector(vec_id, size, max, norm1, &vector[0]);
			if (vec_id % 32 == 0)
			{
				standard_size = size;
			}

			else if (vec_id % 32 == 31)
				tile_start_index += standard_size * 32;
			//index++;
		}
		tile_start_E[sum_tile_E] = tile_start_E[sum_tile_E - 1] + 32 * standard_size;
		printf("sum feat E=%d\n",tile_start_E[sum_tile_E]);

		//partition F
			tile_start_index=0;
			for(int i=0;i<sum_feat_F;i++)
					dim_F[i]=-1;
			for (int vec_id = 0; vec_id < sum_vec_F; vec_id++)
			{
				size=0;
				j=0;
				max=-9999;
				norm1=0.0;
				pre_norm_t1=0.0;
				pre_norm_1=0.0;

				vector.clear();
				if (vec_id % 32 == 0)
				{
					tile_start_F[vec_id / 32] = tile_start_index;
				}
				input.get(ch);
				while (ch != '\n')
				{
					input >> d >> v;
					dim_F[tile_start_index+(vec_id%32+j*32)]=d;
					feature_F[tile_start_index+(vec_id%32+j*32)]=v;
					vector.push_back(std::pair<int,double>(d,v));
					if(d<4096)
					{
						pre_norm_t1=pre_norm_t1+pow(v,2);
					}

					if(v>max)
							max=v;
					norm1=norm1+v;

				//   	inverted_index_F[d].terms.push_back(new Term(vec_id, v));
					size++;
					j++;
					input.get(ch);
				}
				pre_norm_1=sqrt(pre_norm_t1);
				pre_norm_F[vec_id]=pre_norm_1;

			//	pF.addVector(vec_id, size, max, norm1, &vector[0]);
				if (vec_id % 32 == 0)
				{
					standard_size = size;
				}

				else if (vec_id % 32 == 31)
					tile_start_index += standard_size * 32;
				//index++;
			}
			tile_start_F[sum_tile_F] = tile_start_F[sum_tile_F - 1] + 32 * standard_size;
			printf("sum feat F=%d\n",tile_start_F[sum_tile_F]);

			//partition G
				tile_start_index=0;
				for(int i=0;i<sum_feat_G;i++)
						dim_G[i]=-1;
				for (int vec_id = 0; vec_id < sum_vec_G; vec_id++)
				{
					size=0;
					j=0;
					max=-9999;
					norm1=0.0;
					pre_norm_t1=0.0;
					pre_norm_1=0.0;

					vector.clear();
					if (vec_id % 32 == 0)
					{
						tile_start_G[vec_id / 32] = tile_start_index;
					}
					input.get(ch);
					while (ch != '\n')
					{
						input >> d >> v;
						dim_G[tile_start_index+(vec_id%32+j*32)]=d;
						feature_G[tile_start_index+(vec_id%32+j*32)]=v;
						vector.push_back(std::pair<int,double>(d,v));
						if(d<4096)
						{
							pre_norm_t1=pre_norm_t1+pow(v,2);
						}

						if(v>max)
								max=v;
						norm1=norm1+v;

					//   	inverted_index_G[d].terms.push_back(new Term(vec_id, v));
						size++;
						j++;
						input.get(ch);
					}
					pre_norm_1=sqrt(pre_norm_t1);
					pre_norm_G[vec_id]=pre_norm_1;

				//	pG.addVector(vec_id, size, max, norm1, &vector[0]);
					if (vec_id % 32 == 0)
					{
						standard_size = size;
					}

					else if (vec_id % 32 == 31)
						tile_start_index += standard_size * 32;
					//index++;
				}
				tile_start_G[sum_tile_G] = tile_start_G[sum_tile_G - 1] + 32 * standard_size;
				printf("sum feat G=%d\n",tile_start_G[sum_tile_G]);

				//partition H
					tile_start_index=0;
					for(int i=0;i<sum_feat_H;i++)
							dim_H[i]=-1;
					for (int vec_id = 0; vec_id < sum_vec_H; vec_id++)
					{
						size=0;
						j=0;
						max=-9999;
						norm1=0.0;
						pre_norm_t1=0.0;
						pre_norm_1=0.0;

						vector.clear();
						if (vec_id % 32 == 0)
						{
							tile_start_H[vec_id / 32] = tile_start_index;
						}
						input.get(ch);
						while (ch != '\n')
						{
							input >> d >> v;
							dim_H[tile_start_index+(vec_id%32+j*32)]=d;
							feature_H[tile_start_index+(vec_id%32+j*32)]=v;
							vector.push_back(std::pair<int,double>(d,v));
							if(d<4096)
							{
								pre_norm_t1=pre_norm_t1+pow(v,2);
							}

							if(v>max)
									max=v;
							norm1=norm1+v;

						//   	inverted_index_H[d].terms.push_back(new Term(vec_id, v));
							size++;
							j++;
							input.get(ch);
						}
						pre_norm_1=sqrt(pre_norm_t1);
						pre_norm_H[vec_id]=pre_norm_1;

					//	pH.addVector(vec_id, size, max, norm1, &vector[0]);
						if (vec_id % 32 == 0)
						{
							standard_size = size;
						}

						else if (vec_id % 32 == 31)
							tile_start_index += standard_size * 32;
						//index++;
					}
					tile_start_H[sum_tile_H] = tile_start_H[sum_tile_H - 1] + 32 * standard_size;
					printf("sum feat H=%d\n",tile_start_H[sum_tile_H]);



	input.clear();
	input.close();

    int seg_num=49;//50000/1024
	int apss_block_sum=1254400; //1024*(1+2+3+....49), 49 segments
	int* apss_block_seg=new int[apss_block_sum];
	int* apss_seg_start=new int[seg_num];
	int block_id=0;
	for(int s=0;s<seg_num;s++)
	{
		apss_seg_start[s]=block_id;
		for(int b=block_id;b<block_id+1024*(s+1);b++)
		{
			apss_block_seg[b]=s;
		}
		block_id=block_id+1024*(s+1);
	}

//InvertedIndex_search();

	   clock_t startTime,endTime;
	   startTime = clock();
	   similarity_join_device(dim_A, feature_A,  tile_start_A, dim_B, feature_B, tile_start_B,
			   	   	   	   	   	   	   	   	   dim_C, feature_C,  tile_start_C, dim_D, feature_D, tile_start_D,
			   	   	   	   	   	   	   	   	   dim_E, feature_E,  tile_start_E, dim_F, feature_F, tile_start_F,
			   	   	   	   	   	   	   	   	   dim_G, feature_G,  tile_start_G, dim_H, feature_H, tile_start_H,
			   	   	   	   	   	   	   	   	   pre_norm_A, pre_norm_B,pre_norm_C,pre_norm_D,
			   	   	   	   	   	   	   	   	   pre_norm_E, pre_norm_F,pre_norm_G,pre_norm_H,
			   	   	   	   	   	   	   	   	   sum_vec_A, sum_feat_A, sum_tile_A,sum_vec_B, sum_feat_B, sum_tile_B,
			   	   	   	   	   	   	   	   	   sum_vec_C, sum_feat_C, sum_tile_C,sum_vec_D, sum_feat_D, sum_tile_D,
			   	   	   	   	   	   	   	   	   sum_vec_E, sum_feat_E, sum_tile_E,sum_vec_F, sum_feat_F, sum_tile_F,
			   	   	   	   	   	   	   	   	   sum_vec_G, sum_feat_G, sum_tile_G,sum_vec_H, sum_feat_H, sum_tile_H,
			   	   	   	   	   	   	   	   	   b_id_a, apss_block_seg, apss_seg_start
			   	   	   	   	   	   	   	   	 );
	   endTime = clock();
	   printf("Cuda Time : %.6f\n", (double)(endTime - startTime)/CLOCKS_PER_SEC );

	getchar();
	return 0;
}
