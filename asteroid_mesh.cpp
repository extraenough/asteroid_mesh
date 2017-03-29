// asteroid_mesh.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

#include <thread>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <stdlib.h> 
#include <string.h> 
#include <stdarg.h> 

#include <locale>
#include <codecvt>
#include <io.h>
#include <fcntl.h>

#include <random>

// ply lib
#include "tinyply.h"

// kdtree lib
#include "PointCloud.h"
#include "nanoflann.hpp"
//#include "Eigen\src\Core\DenseBase.h"
#include "Eigen\Dense"
#include "Eigen\Geometry"

//#include "rply_service.c"

#include <boost\thread\thread.hpp>
#include <boost\thread\shared_mutex.hpp>
#include <boost\interprocess\sync\named_semaphore.hpp>

using namespace tinyply;
using namespace nanoflann;

boost::shared_mutex __access;
boost::interprocess::named_semaphore threadCount(boost::interprocess::open_or_create, "threadCount", -1);

typedef std::chrono::time_point<std::chrono::high_resolution_clock> timepoint;
std::chrono::high_resolution_clock c;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
	return c.now();
}

inline double difference_micros(timepoint start, timepoint end)
{
	return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000;
}

void read_ply_file(const std::string & filename, PointCloud<float> &cloud)
{
	// Tinyply can and will throw exceptions at you!
	try
	{
		// Read the file and create a std::istringstream suitable
		// for the lib -- tinyply does not perform any file i/o.
		std::ifstream ss(filename, std::ios::binary);

		// Parse the ASCII header fields
		PlyFile file(ss);

		for (auto e : file.get_elements())
		{
			std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
			for (auto p : e.properties)
			{
				std::cout << "\tproperty - " << p.name << " (" << PropertyTable[p.propertyType].str << ")" << std::endl;
			}
		}
		std::cout << std::endl;

		for (auto c : file.comments)
		{
			std::cout << "Comment: " << c << std::endl;
		}

		// Define containers to hold the extracted data. The type must match
		// the property type given in the header. Tinyply will interally allocate the
		// the appropriate amount of memory.
		std::vector<float> verts;
		std::vector<float> norms;
		std::vector<uint8_t> colors;

		std::vector<uint32_t> faces;
		std::vector<float> uvCoords;

		uint32_t vertexCount, normalCount, colorCount, faceCount, faceTexcoordCount, faceColorCount;
		vertexCount = normalCount = colorCount = faceCount = faceTexcoordCount = faceColorCount = 0;

		// The count returns the number of instances of the property group. The vectors
		// above will be resized into a multiple of the property group size as
		// they are "flattened"... i.e. verts = {x, y, z, x, y, z, ...}
		vertexCount = file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
		normalCount = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }, norms);
		colorCount = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }, colors);

		// For properties that are list types, it is possibly to specify the expected count (ideal if a
		// consumer of this library knows the layout of their format a-priori). Otherwise, tinyply
		// defers allocation of memory until the first instance of the property has been found
		// as implemented in file.read(ss)
		faceCount = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);
		faceTexcoordCount = file.request_properties_from_element("face", { "texcoord" }, uvCoords, 6);

		// Now populate the vectors...
		timepoint before = now();
		file.read(ss);
		timepoint after = now();

		// fill PointCloud structure
		cloud.pts.resize(vertexCount);
		for (int i = 0; i < vertexCount; i++)
		{
			cloud.pts[i].x = verts[i * 3];
			cloud.pts[i].y = verts[i * 3 + 1];
			cloud.pts[i].z = verts[i * 3 + 2];
		}

		// Good place to put a breakpoint!
		std::cout << "Parsing took " << difference_micros(before, after) << "ms: " << std::endl;
		std::cout << "\tRead " << verts.size() << " total vertices (" << vertexCount << " properties)." << std::endl;
		std::cout << "\tRead " << norms.size() << " total normals (" << normalCount << " properties)." << std::endl;
		std::cout << "\tRead " << colors.size() << " total vertex colors (" << colorCount << " properties)." << std::endl;
		std::cout << "\tRead " << faces.size() << " total faces (triangles) (" << faceCount << " properties)." << std::endl;
		std::cout << "\tRead " << uvCoords.size() << " total texcoords (" << faceTexcoordCount << " properties)." << std::endl;

	}

	catch (const std::exception & e)
	{
		std::cerr << "Caught exception: " << e.what() << std::endl;
	}
}

void fillPointCloud(const std::string & filename, PointCloud<float> &cloud)
{
	read_ply_file(filename, cloud);
}

RadiusResultSet<float, size_t> findNeighborsSphere(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	size_t _leaf_max_size,
	int dim,
	int index,
	const float radius = 1
	)
{	
	//std::srand(std::time(NULL));

	//int query_pt_index = std::rand() % cloud.pts.size();
	// опорная точка
	//float query_pt[3] = { 0.5, 0.5, 0.5 };

	float query_pt[3];
	query_pt[0] = cloud.pts[index].x;
	query_pt[1] = cloud.pts[index].y;
	query_pt[2] = cloud.pts[index].z;

	// construct a kd-tree index:
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<float, PointCloud<float> >,
		PointCloud<float>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t   indexSearch(dim, cloud, KDTreeSingleIndexAdaptorParams(_leaf_max_size));
	indexSearch.buildIndex();

	//std::vector<std::pair<size_t, float> > indices_dists;
	//RadiusResultSet<float, size_t> resultSet(radius, indices_dists);

	//index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
	indexSearch.radiusSearch(query_pt, radius, resultSet.m_indices_dists, nanoflann::SearchParams());

	/*
	// Get worst (furthest) point, without sorting:
	std::pair<size_t, float> worst_pair = resultSet.worst_item();
	std::cout << "Worst pair: idx=" << worst_pair.first << " dist=" << worst_pair.second << std::endl;
	*/	

	return resultSet;
}

void computeSVD(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	int index
	)
{
	/*
		Example:
		https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Eigen/SVD/SVD.cpp
		По ссылке - полезный вопрос на stackoverflow:
		http://stackoverflow.com/questions/39370370/eigen-and-svd-to-find-best-fitting-plane-given-a-set-of-points
	*/


	//Eigen::MatrixXf A = Eigen::MatrixXf::Random(50, 3);
	Eigen::MatrixXf A(resultSet.size(), 3);
	for (int i = 0; i < resultSet.size(); i++)
	{
		A(i, 0) = cloud.pts[resultSet.m_indices_dists[i].first].x;
		A(i, 1) = cloud.pts[resultSet.m_indices_dists[i].first].y;
		A(i, 2) = cloud.pts[resultSet.m_indices_dists[i].first].z;
	}

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXf U_3 = svd.matrixU().col(2);
	//U_3 = U_3.transpose();
	Eigen::Vector3f U_3_cleaned(U_3(0), U_3(1), U_3(2));
	Eigen::Vector3f A_0(A(0, 0), A(0, 1), A(0, 2));

	Eigen::Hyperplane<float, 3> plane = Eigen::Hyperplane<float, 3>(U_3_cleaned, A_0);
	
/*
	std::cout << "A          : " << std::endl << A			    << std::endl << std::endl;
//  std::cout << "U          : " << std::endl << svd.matrixU()  << std::endl << std::endl;
//  std::cout << "V          : " << std::endl << svd.matrixV()  << std::endl << std::endl;
	std::cout << "U_3        : " << std::endl << U_3		    << std::endl << std::endl;
	std::cout << "U_3_cleaned: " << std::endl << U_3_cleaned    << std::endl << std::endl;
	std::cout << "A_0        : " << std::endl << A_0		    << std::endl << std::endl;
	std::cout << "fitting (or median) plane coeffs:" 
									<< std::endl << plane.coeffs() << std::endl << std::endl;

	float* data = new float[4]; 
//	std::cout << "data       : " << std::endl;
	for (int j = 0; j < 4; j++){
		data[j] = plane.coeffs()(j);
//		std::cout << data[j] << std::endl;
	}
//	std::cout << std::endl;

*/

	boost::unique_lock<boost::shared_mutex> uniqueLock(__access);
	cloud.pts[index].coeffs = new float[4];
	for (int j = 0; j < 4; j++)
	{
		cloud.pts[index].coeffs[j] = plane.coeffs()(j);
	}

}

Eigen::Vector4f computeSVD_n0(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	int index,
	Eigen::MatrixXf &p
	)
{

	Eigen::MatrixXf A = p;
	for (int i = 0; i < A.cols(); i++)
		for (int j = 0; j < A.rows(); j++)
			A(j, i) = A(j, i) - A.col(i).sum()/A.rows();

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXf U_3 = svd.matrixU().col(2);
	//U_3 = U_3.transpose();
	Eigen::Vector3f U_3_cleaned(U_3(0), U_3(1), U_3(2));
	Eigen::Vector3f A_0(A(0, 0), A(0, 1), A(0, 2));

	Eigen::Hyperplane<float, 3> plane = Eigen::Hyperplane<float, 3>(U_3_cleaned, A_0);
	Eigen::Vector4f result;

	for (int j = 0; j < 4; j++)
	{
		result(j) = plane.coeffs()(j);
	}

	std::cout << index << " n0:" << std::endl << result.transpose() << std::endl;

	return result;
}

Eigen::Vector4f computeSVD_LMM(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	int index,
	Eigen::MatrixXf &A,
	Eigen::Vector4f &n_prev
	)
{
	/*
	Eigen::MatrixXf A(resultSet.size(), 3);
	for (int i = 0; i < resultSet.size(); i++)
	{
		A(i, 0) = cloud.pts[resultSet.m_indices_dists[i].first].x;
		A(i, 1) = cloud.pts[resultSet.m_indices_dists[i].first].y;
		A(i, 2) = cloud.pts[resultSet.m_indices_dists[i].first].z;
	}
	*/

	//Eigen::MatrixXf A_d = A / (A.transpose() * n_prev).sqrt();
	Eigen::MatrixXf A_d = A;
	//Eigen::VectorXf A_t = (A.transpose() * n_prev);
	Eigen::Vector3f n_prev_3(n_prev(0), n_prev(1), n_prev(2));
	//Eigen::MatrixXf A_t_m = (A.transpose() * n_prev_3);
	//Eigen::VectorXf A_t = A_t_m.col(0);
	//std::cout << "A:" << std::endl << A << std::endl << "n_prev:" << std::endl << n_prev << std::endl;
	Eigen::VectorXf A_t = (A * n_prev_3);
	//std::cout << "A_t:" << std::endl << A_t << std::endl;
	Eigen::Vector3f c(A.col(0).sum() / A.rows(), A.col(1).sum() / A.rows(), A.col(2).sum() / A.rows());

	for (int i = 0; i < A_d.rows(); i++)
	{
		//A_d.row(i) /= std::sqrt(A_t(i));
		//std::cout << "row:" << std::endl << A_d.row(i) << std::endl << "c.tranpose():" << std::endl << c.transpose() << std::endl << "A_t(i):" << std::endl << A_t(i) << std::endl << "std::sqrt(std::abs(A_t(i))):" << std::endl << std::sqrt(std::abs(A_t(i))) << std::endl;
		A_d.row(i) = (A_d.row(i) - c.transpose()) / std::sqrt(std::abs(A_t(i)));
	}

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A_d, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::VectorXf U_3 = svd.matrixU().col(2);
	//U_3 = U_3.transpose();
	Eigen::Vector3f U_3_cleaned(U_3(0), U_3(1), U_3(2));
	Eigen::Vector3f A_0(A_d(0, 0), A_d(0, 1), A_d(0, 2));

	Eigen::Hyperplane<float, 3> plane = Eigen::Hyperplane<float, 3>(U_3_cleaned, A_0);
	Eigen::Vector4f result;

	for (int j = 0; j < 4; j++)
	{
		result(j) = plane.coeffs()(j);
	}

	return result;
}

/*
	Lowest modules method
*/
void computeLMM(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	int index
	)
{
	Eigen::MatrixXf A(resultSet.size(), 3);
	for (int i = 0; i < resultSet.size(); i++)
	{
		A(i, 0) = cloud.pts[resultSet.m_indices_dists[i].first].x;
		A(i, 1) = cloud.pts[resultSet.m_indices_dists[i].first].y;
		A(i, 2) = cloud.pts[resultSet.m_indices_dists[i].first].z;
	}

	Eigen::Vector4f n_current = computeSVD_n0(cloud, resultSet, index, A);
	Eigen::Vector4f n_prev(0, 0, 0, 0);

	float eps = 0.000000001;
	std::cout << "n_current: " << n_current.transpose() << std::endl;
	// не детерминант, там должно быть перемножение векторов
	while (((n_current - n_prev).transpose() * (n_current - n_prev)).determinant() > eps)
	{
		n_prev = n_current;
		n_current = computeSVD_LMM(cloud, resultSet, index, A, n_prev);
		std::cout << "n_current: " << n_current.transpose() << std::endl;
	}
	std::cout << ((n_current - n_prev).transpose() * (n_current - n_prev)) << std::endl;

	boost::unique_lock<boost::shared_mutex> uniqueLock(__access);
	cloud.pts[index].coeffs = new float[4];
	for (int j = 0; j < 4; j++)
	{
		cloud.pts[index].coeffs[j] = n_current(j);
	}
}

void vertexPlaneDistance(
	PointCloud<float> &cloud,
	RadiusResultSet<float, size_t> &resultSet,
	int index
	)
{
	Eigen::Vector3f direction; 
	for (int j = 0; j < 3; j++)
	{
		direction(j) = cloud.pts[index].coeffs[j];
	}
	Eigen::Vector3f origin_vertex(
		cloud.pts[resultSet.m_indices_dists[index].first].x,
		cloud.pts[resultSet.m_indices_dists[index].first].y,
		cloud.pts[resultSet.m_indices_dists[index].first].z
		);
	for (int i = 0; i < resultSet.size(); i++)
	{
		Eigen::Vector3f origin_nieghbour(
			cloud.pts[resultSet.m_indices_dists[i].first].x,
			cloud.pts[resultSet.m_indices_dists[i].first].y,
			cloud.pts[resultSet.m_indices_dists[i].first].z
			);
		//Eigen::ParametrizedLine<float, 3> line = Eigen::ParametrizedLine<float, 3>(origin_nieghbour, direction);
		Eigen::Hyperplane<float, 3> plane = Eigen::Hyperplane<float, 3>(direction, origin_vertex);
		//auto intersection = line.intersection(plane);
		auto intersection = plane.absDistance(origin_nieghbour);
		//std::cout << i << " : " << intersection << std::endl;
	}

}

int thread_solver(
	PointCloud<float> &cloud,
	size_t _leaf_max_size,
	int dim,
	int index,
	const float radius = 1
	)
{
	std::cout << index << " : start." << std::endl;

	std::vector<std::pair<size_t, float> > indices_dists;
	RadiusResultSet<float, size_t> resultSet(radius, indices_dists);

	findNeighborsSphere(cloud, resultSet, _leaf_max_size, dim, index, radius);
	if (resultSet.size() != 0)
	{
		//computeSVD(cloud, resultSet, index);
		computeLMM(cloud, resultSet, index);
		std::cout << index << " result: " << cloud.pts[index].coeffs[0] << " " << cloud.pts[index].coeffs[1] << " " << cloud.pts[index].coeffs[2] << " " << cloud.pts[index].coeffs[3] << " " << std::endl;
		vertexPlaneDistance(cloud, resultSet, index);
	}

	std::cout << index << " : done." << std::endl;
	indices_dists.clear();
	resultSet.clear();
	threadCount.wait();

	return 0;
}

int main(int argc, char* argv[])
{
	std::string filename = argv[1];
	PointCloud<float> cloud;

	timepoint start = now();
	fillPointCloud(filename, cloud);
	timepoint end = now();
	std::cout << "time:     " << difference_micros(start, end) << std::endl;

	float radius = 0.00002;
	/*
	std::vector<std::pair<size_t, float> > indices_dists;
	RadiusResultSet<float, size_t> resultSet(radius, indices_dists);

	start = now();	
	findNeighborsSphere(cloud, resultSet, 10, 3, radius);
	end = now();
	std::cout << "time:     " << difference_micros(start, end) << std::endl;

	start = now();
	computeSVD(cloud, resultSet);
	end = now();
	std::cout << "time:     " << difference_micros(start, end) << std::endl;
	*/

	start = now();	
	int count = cloud.pts.size();
	boost::thread_group *solvers;
	solvers = new boost::thread_group();
	//for (int i = 0; i < count; i++)
	for (int i = 0; i < 1; i++)
	{
		threadCount.post();
		(*solvers).add_thread(new boost::thread(thread_solver, cloud, 10, 3, i, radius));
		if ((*solvers).size() % 10 == 0)
		{
			(*solvers).join_all();
			delete solvers;
			solvers = new boost::thread_group();
		}
	}
	(*solvers).join_all();
	end = now();
	std::cout << "time:     " << difference_micros(start, end) << std::endl;
	std::cout << "count:    " << count << std::endl;
	//std::cout << "count_zero:    " << count_zero << std::endl;

	return 0;
}