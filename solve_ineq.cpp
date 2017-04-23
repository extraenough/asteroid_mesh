#include <stdio.h>
// ?
#include <Eigen\Dense>

#include <cmath>

bool check_ineq(
	Eigen::Matrix3n A,
	vector<int> b,
	Vertex x)
{
	double result = 0.0;

	for(int i=0; i< A.sizeX(); i++)
	{
		double ax_sum = 0.0;
		for(int j=0; j< A.sizeY(); j++)
		{
			ax_sum += A[i][j] * x[j];
		}
		result += std::power(std::max(0, (b[i]-ax_sum)), 2);
	}

	return result == 0 ? true : false;
}

bool solve_ineq(
	PointCloud &cloud,
	// neighbours
	VertexSet &search
)
{
	Eigen::Vertex3n p1 = search[0];
	Eigen::Matrix3n A;

	Eigen::Vector3n b;
	for(int i=1; i<search.size(); i++)
	{
		A[i] = {search[i][0] - search[i][0], search[i][1] - search[i][1], search[i][2] - search[i][2]};
		b[i] = L2;
	}
	A = A.transpose();

	bool solved = false;
	bool finish = false;

	int i = 0;
	int j = 0;
	Vertex n = {0, 0, 0}

	while(!solved && !finish)
	{
		n[0] = i; n[1] = j; 
		n[2] = std::sqrt(std::power(n[0], 2) + std::power(n[1], 2));
		
		if(check_ineq(A, b, n))
			solved = true;

		if(!solved)
		{
			if(i < 180) i++;
			else 
			{
				if(j < 180) j++;
				else
				{
					finish = true;
				}
		}
	}

	return solved;
}
