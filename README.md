# FG-LiquID
This is code of our deep learning model proposed in IMWUT 2021 paper 《FG-LiquID: A Contact-less Fine-grained Liquid Identifier by Pushing the Limits of Millimeter-wave Sensing》
https://dl.acm.org/doi/10.1145/3478075

Just run train.sh or test.sh to start.  
We are not going to make our full dataset public. 
You could put your own data in the folder /data/dataset, and generate the corresponding data list in /data/trainlist.txt and /data/testlist.txt.  The data list is composed of the data path and its category label.

It works well in the environment with:

Python 3.6.9

PyTorch 1.1.0

scipy 1.4.1




If this code helps, please star it and cite our paper, Thanks!

Yumeng Liang, Anfu Zhou, Huanhuan Zhang, Xinzhe Wen, Huadong Ma:
FG-LiquID: A Contact-less Fine-grained Liquid Identifier by Pushing the Limits of Millimeter-wave Sensing. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 5(3): 116:1-116:27 (2021)
