#include <stdio.h>
// ?
#include <Eigen\Dense>

#include <cmath>
#include <math.h>
#include <vector>

#include "PointCloud.h"
#include "nanoflann.hpp"

using namespace nanoflann;

bool check_ineq(
	Eigen::MatrixXf A,
	//std::vector<int> b,
	Eigen::VectorXf b,
	//PointCloud<float>::Point x)
	Eigen::VectorXf x)
{
	double result = 0.0;

	for(int i=0; i< A.cols(); i++)
	{
		double ax_sum = 0.0;
		for(int j=0; j< A.rows(); j++)
		{
			ax_sum += A(i, j) * x(j);
		}
		result += std::pow(std::max(0.0, (b[i]-ax_sum)), 2);
	}

	return result == 0 ? true : false;
}

bool solve_ineq(
	PointCloud<float> &cloud,
	// neighbours
	RadiusResultSet<float, size_t> &resultSet,
	double L2
)
{
	/*
	A(i, 0) = cloud.pts[resultSet.m_indices_dists[i].first].x;
	A(i, 1) = cloud.pts[resultSet.m_indices_dists[i].first].y;
	A(i, 2) = cloud.pts[resultSet.m_indices_dists[i].first].z;
	*/

	Eigen::Vector3f p1 = resultSet[0];
	Eigen::Matrix3f A;

	Eigen::VectorXf b;
	for(int i=1; i<resultSet.size(); i++)
	{
		//A(i) = {resultSet[i][0] - resultSet[i][0], resultSet[i][1] - resultSet[i][1], resultSet[i][2] - resultSet[i][2]};
		A(i, 0) = resultSet[i][0] - resultSet[0][0];
		A(i, 1) = resultSet[i][1] - resultSet[0][1];
		A(i, 2) = resultSet[i][2] - resultSet[0][2];
		b(i) = L2;
	}
	A = A.transpose();

	bool solved = false;
	bool finish = false;

	int i = 0;
	int j = 0;
	//Vector n = {0, 0, 0}
	Eigen::Vector3f n;
	n(0) = 0; n(1) = 0; n(2) = 0;

	while(!solved && !finish)
	{
		n[0] = i; n[1] = j; 
		n[2] = std::sqrt(std::pow(n(0), 2) + std::pow(n(1), 2));
		
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
