#include "InnerNode.h"

InnerNode::InnerNode()
{
	//split_point = new double[MetaData::dim];
	// boost::variant<LeafNode, InnerNode> node;
	for (int i = 0; i < pow(2, MetaData::dim); i++)
	{
		this->children.push_back(-11);
	}
}

InnerNode::~InnerNode()
{
}

int InnerNode::child_index(array<double, 2> &key)
{
	int index = 0;
	if ((this->children[index]).type() == typeid(GridNode *))
	{
		for (int i = 0; i < split_point.size(); i++)
		{
			if (split_point[i] > key[0])
			{
				index = i - 1;
				break;
			}
		}
	}
	else
	{
		for (int i = 0; i < MetaData::dim; i++)
		{
			if (key[i] > split_point[i])
			{
				index += pow(2, i);
			}
		}
	}
	return index;
}

double InnerNode::getCellArea()
{
	double area = 1.0;
	for (int i = 0; i < MetaData::dim; i++)
	{
		area *= (range_bound[i * 2 + 1] - range_bound[i * 2]);
	}
	return area;
}