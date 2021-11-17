#pragma once
#include <vector>
#include <math.h>
#include "MetaData.h"
#include "boost/variant.hpp"
#include "LeafNode.h"
#include "GridNode.h"

using namespace std;


class InnerNode
{
public:
	InnerNode();
	~InnerNode();
	std::vector<double> split_point;
	vector < boost::variant<InnerNode*, LeafNode*, GridNode*, int>> children;
	InnerNode* parent=nullptr;
	bool mergeDelete = false;

	std::vector<double> range_bound;

	int child_index(array<double, 2>& key);
	double getCellArea();

private:

};

