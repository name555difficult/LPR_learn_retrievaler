# Overall

- 面向任务：Lidar Place Recognition
- 解决问题：现有方法均在某种传统度量（欧式距离、余弦距离、马氏距离）下衡量两个样本映射到Descriptor后的相似度，这限制了learning-based descriptor的可表征空间，会把数据间的relation和descriptor分布限制在一个规则流形（超球面等）上。
- 解决思路：提出learning-based retrieval module，基于参数化可学习的retrieval module去度量两个descriptor之间的相似性。
- 目标：达到甚至超过传统度量的检索性能，并从理论上证明learning-based retrieval可靠性和有效性。
- 理想情况：最好能够通过few-shot or unsupervised or weak-supervised or zero-shot形式完成我们的目标。
