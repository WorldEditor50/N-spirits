#ifndef KDTREE_HiPP
#define KDTREE_HiPP
#include <vector>
#include <algorithm>
#include <functional>
#include <stack>
#include <memory>
#include "../basic/tensor.hpp"
#include "../basic/statistics.h"

class KDTree
{
public:
    class Node
    {
    public:
        std::size_t splitIndex;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        Tensor value;
    public:
        Node():splitIndex(0),left(nullptr),right(nullptr){}
    };

public:
    std::size_t featureDim;
    Node root;
protected:

    static std::size_t sort(std::vector<Tensor> &x)
    {
        /* variance of dataset */
        Tensor u = Statistics::sum(x) / x.size();
        Tensor sigma = Statistics::variance(x, u);
        /* sort  */
        std::size_t splitIndex = 0;
        for (std::size_t i = 1; i < sigma.size(); i++) {
            if (sigma[i - 1] < sigma[i]) {
                splitIndex = i;
            }
        }
        std::sort(x.begin(), x.end(), [=](Tensor &x1, Tensor &x2) -> bool {
            return x1[splitIndex] < x2[splitIndex];
        });
        return splitIndex;
    }

    static void generate(std::vector<Tensor> &x, KDTree::Node &node)
    {
        if (x.empty() == true) {
            return;
        }
        /* sort by variance */
        std::size_t splitIndex = sort(x);
        /* split data */
        std::size_t len = x.size();
        std::vector<Tensor> left;
        std::vector<Tensor> right;
        for (std::size_t i = 0; i < x.size(); i++) {
            if (i == len/2) {
                continue;
            }
            if (x[i][splitIndex] <= x[len/2][splitIndex]) {
                left.push_back(x[i]);
            } else {
                right.push_back(x[i]);
            }
        }
        /* assign node */
        node.value = x[len/2];
        node.splitIndex = splitIndex;
        node.left = std::make_shared<Node>();
        node.right = std::make_shared<Node>();
        /* next */
        generate(left, *node.left);
        generate(right, *node.right);
        return;
    }

    static int find(KDTree::Node& root, const Tensor &x, Tensor & nearest, float& dist)
    {
        std::stack<std::shared_ptr<KDTree::Node> > nodes;
        std::shared_ptr<KDTree::Node> node(&root);
        while (node != nullptr) {
            nodes.push(node);
            std::size_t i = node->splitIndex;
            if (x[i] <= node->value[i]) {
                node = node->left;
            } else {
                node = node->right;
            }
        }
        /* first node */
        nearest = nodes.top()->value;
        nodes.pop();
        dist = Statistics::Norm::l2(x, nearest);
        /* find the rest */
        while (nodes.empty()) {
            auto p = nodes.top();
            nodes.pop();
            /* leaf node */
            if (p->left == nullptr && p->right == nullptr) {
                float d = Statistics::Norm::l2(x, nearest);
                if (dist > d) {
                    nearest = p->value;
                    dist = d;
                }
            } else {
                std::size_t i = p->splitIndex;
                float delta = x[i] - p->value[i];
                if (std::fabs(delta) > dist) {
                    float d = Statistics::Norm::l2(x, nearest);
                    if (dist > d) {
                        nearest = p->value;
                        dist = d;
                    }
                }
                std::shared_ptr<KDTree::Node> next = nullptr;
                if (delta > 0) {
                    next = p->left;
                } else {
                    next = p->right;
                }
                if (next != nullptr) {
                    nodes.push(next);
                }
            }
        }
        return 0;
    }
public:
    explicit KDTree(std::vector<Tensor> &x)
    {
        generate(x, root);
    }

    int operator()(const Tensor &x, Tensor & nearest, float& dist)
    {
        find(root, x, nearest, dist);
        return 0;
    }

};

#endif // KDTREE_HiPP
