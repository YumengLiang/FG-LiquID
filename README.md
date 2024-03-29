# FG-LiquID
This is code for IMWUT 2021 paper 《FG-LiquID: A Contact-less Fine-grained Liquid Identifier by Pushing the Limits of Millimeter-wave Sensing》
https://dl.acm.org/doi/10.1145/3478075

Just run train.sh or test.sh to start.  
We are not going to make our full dataset public. 
You could put your own data in the folder /data/dataset, and generate the corresponding data list in /data/trainlist.txt and /data/testlist.txt.  The data list is composed of the data path and its category label.

It works well in the environment with:

Python 3.6.9

PyTorch 1.1.0

scipy 1.4.1


The dataset of 30 kinds of liquids is uploaded here:  https://github.com/YumengLiang/dataset-of-mmwave

If this code and dataset helps, please star it and cite our paper as following:

Yumeng Liang, Anfu Zhou, Huanhuan Zhang, Xinzhe Wen, Huadong Ma:
FG-LiquID: A Contact-less Fine-grained Liquid Identifier by Pushing the Limits of Millimeter-wave Sensing. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 5(3): 116:1-116:27 (2021)


An introduce of our paper could be accessed :

Bilibili : https://www.bilibili.com/video/BV1KP4y1W7cD?spm_id_from=333.999.0.0

YouTube: https://www.youtube.com/watch?v=w6XY68spQgs
