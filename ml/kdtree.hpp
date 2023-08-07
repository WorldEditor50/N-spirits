#ifndef KDTREE_HiPP
#define KDTREE_HiPP
#include <vector>
#include <algorithm>
#include <functional>
#include <stack>
#include <memory>
#include "../basic/tensor.hpp"
#include "../basic/util.hpp"

class KDTree
{
public:
    class Node
    {
    public:
        using Pointer = std::shared_ptr<Node>;
    public:
        std::size_t compareIndex;
        Pointer left;
        Pointer right;
        Tensor value;
    public:
        Node():compareIndex(0),left(nullptr),right(nullptr){}
        void show() const
        {
            value.printValue();
        }
    };
    class Result
    {
    public:
        float distance;
        Tensor value;
    public:
        Result(){}
        explicit Result(float d, const Tensor &v)
            :distance(d),value(v){}
    };
protected:
    Node::Pointer root;
protected:
    static std::size_t sort(std::vector<Tensor> &x)
    {
        /* variance of dataset */
        Tensor u = util::sum(x) / x.size();
        Tensor sigma = util::variance(x, u);
        /* sort by variance */
        std::size_t compareIndex = 0;
        for (std::size_t i = 1; i < sigma.size(); i++) {
            if (sigma[i - 1] < sigma[i]) {
                compareIndex = i;
            }
        }
        std::sort(x.begin(), x.end(), [=](Tensor &x1, Tensor &x2) -> bool {
            return x1[compareIndex] < x2[compareIndex];
        });
        return compareIndex;
    }

    static Node::Pointer build(std::vector<Tensor> &x)
    {
        if (x.empty() == true) {
            return nullptr;
        }
        std::size_t len = x.size();
        /* sort by variance of specific dimension */
        std::size_t compareIndex = sort(x);
        /* split data */
        std::vector<Tensor> left;
        std::vector<Tensor> right;
        for (std::size_t i = 0; i < x.size(); i++) {
            if (i == len/2) {
                continue;
            }
            if (x[i][compareIndex] <= x[len/2][compareIndex]) {
                left.push_back(x[i]);
            } else {
                right.push_back(x[i]);
            }
        }
        /* create node */
        Node::Pointer node = std::make_shared<Node>();
        node->value = x[len/2];
        node->compareIndex = compareIndex;
        node->left = build(left);
        node->right = build(right);
        return node;
    }

    static void find(Node::Pointer& root,
                     const Tensor &x,
                     std::vector<Result>& nearest)
    {
        std::stack<Node::Pointer> prependingNodes;
        Node::Pointer pNode(root);
        while (pNode != nullptr) {
            prependingNodes.push(pNode);
            std::size_t i = pNode->compareIndex;
            if (x[i] <= pNode->value[i]) {
                pNode = pNode->left;
            } else {
                pNode = pNode->right;
            }
        }
        /* first node */
        Tensor *nearestValue = &prependingNodes.top()->value;
        prependingNodes.pop();
        float nearestDist = util::Norm::l2(x, *nearestValue);
        nearest.push_back(Result(nearestDist, *nearestValue));
        /* find the rest */
        while (prependingNodes.empty() == false) {
            pNode = prependingNodes.top();
            prependingNodes.pop();
            /* leaf node */
            if (pNode->left == nullptr && pNode->right == nullptr) {
                float d = util::Norm::l2(x, pNode->value);
                if (nearestDist >= d) {
                    nearestValue = &pNode->value;
                    nearestDist = d;
                    nearest.push_back(Result(d, pNode->value));
                }
                //std::cout<<"====== d="<<d<<" nearestDist="<<nearestDist<<std::endl;
            } else {

                std::size_t i = pNode->compareIndex;
                float delta = x[i] - pNode->value[i];
                if (std::fabs(delta) > nearestDist) {
                    float d = util::Norm::l2(x, pNode->value);
                    if (nearestDist >= d) {
                        nearestValue = &pNode->value;
                        nearestDist = d;
                        nearest.push_back(Result(d, pNode->value));
                    }
                    //std::cout<<"------- d="<<d<<" nearestDist="<<nearestDist<<std::endl;
                }
                Node::Pointer next = nullptr;
                if (delta > 0) {
                    next = pNode->left;
                } else {
                    next = pNode->right;
                }
                if (next != nullptr) {
                    prependingNodes.push(next);
                }
            }
        }
        return;
    }
public:
    explicit KDTree(std::vector<Tensor> &x)
    {
        root = build(x);
    }
    int find(const Tensor &x, std::size_t k, std::vector<Result> &nearest)
    {
        if (root == nullptr) {
            return -1;
        }
        find(root, x, nearest);
        if (nearest.size() > k) {
            nearest.erase(nearest.begin() + k, nearest.end());
        }
        return 0;
    }

    int find(const Tensor &x, std::vector<Result> &nearest)
    {
        if (root == nullptr) {
            return -1;
        }
        find(root, x, nearest);
        return 0;
    }

    void show(Node::Pointer node) const
    {
        if (node == nullptr) {
            return;
        }
        show(node->left);
        show(node->right);
        node->show();
        return;
    }
    void display()
    {
        show(root);
    }


};

#endif // KDTREE_HiPP
