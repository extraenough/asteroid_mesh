#include <stdio.h>

#include <string>

#include "PointCloud.h"
#include "nanoflann.hpp"

void PointNeighboursToFile(
	RadiusResultSet<float, size_t> **resultSet,
	int pointsCount,
	std::string &filename
)
{
	// open or create a file for writing
	FILE *file = ;

	// write basic info to the file
	file << pointsCount << endl;
	for (int i=0; i<pointsCount; i++)
	{
		file << resultSet[i].count() << std::endl;;
	}

	// write neighbours (with central points) to the file
	for (int i=0; i<pointsCount; i++)
	{
		for (int j=0; j<resultSet[i].count(); j++)
		{
			file << resultSet[i][j].x << " "
			     << resultSet[i][j].y << " "
			     << resultSet[i][j].z << std::endl;
		}
	}

	// close the file
	file.close();
}
