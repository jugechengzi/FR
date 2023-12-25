# FR: Folded Rationalization with a Unified Encoder  
This repo contains Pytorch implementation of [Folded Rationalization with a Unified Encoder (FR, NeurIPS2022)](https://arxiv.org/abs/2209.08285).    

We would be grateful if you would star this repo before cloning it.

## Tips for building your own method with our code

We have provided some tips for you to better understand our code and build your own method on top of it: [tips](https://github.com/jugechengzi/FR/blob/main/tips.pdf). If you're still having trouble reproducing your code after reading these tips, I'd be happy to set up a video conference to help you.




## Environments  
torch 1.10.2+cu113.   
python 3.7.9.   
tensorboardx 2.4.   
tensorboard 2.6.0    
RTX3090  
We suggest you to create a new environment with: conda create -n FR python=3.7    
Then activate the environment with: conda activate FR  
And then conduct: pip install -r requirements.txt  

## Datasets  
For Beer Reviews, you should first obtain authorization for this dataset from the original author.
 
Beer Reviews: you can get it [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running Example  
### Beer
#### Appearance:  
python run_share_beer.py --lr 0.0001 --batch_size 256 --sparsity_percentage 0.15 --sparsity_lambda 10 --continuity_lambda 10 --epochs 300 --aspect 0  

**_Notes_**: "sparsity_percentage" corresponds to $\alpha$ in Eq.3. "sparsity_lambda" and "continuity_lambda" correspond to $\lambda_1$ and $\lambda_2$, respectively.  Hyperparameters on other datasets are in [hyperparameters](https://github.com/jugechengzi/FR/blob/main/hyperparameters.txt)

## Questions
If you have any questions, just open an issue or send us an e-mail.   
If the repo helps you, please star it.   
Thank you!  

## Result
You will get a result like "best_dev_epoch=205" at last. Then you need to find the result corresponding to the epoch with number "205".  
For Beer-Appearance, you may get a result like:  
traning epoch:205 recall:0.7998 precision:0.8736 f1-score:0.8351 accuracy:0.8421  
Validate  
dev epoch:205 recall:0.8161 precision:0.9337 f1-score:0.8709 accuracy:0.8164
Annotation  
annotation dataset : recall:0.8700 precision:1.0000 f1-score:0.9305 accuracy:0.8718  
The annotation performance: sparsity: 18.4382, precision: 82.9415, recall: 82.6008, f1: 82.7708  
The last line indicates the overlap between the selected tokens and human-annotated rationales. The penultimate line shows the predictive accuracy on the test set. 

## Quick Test 
If you don't have enough GPU resource, we also provide trained models for quick test. Download the models [here](https://drive.google.com/file/d/1jLkLBC5CJxu-M_2yOGi94rGwscofnJVk/view?usp=sharing) and then run test_beer.py and test_hotel.py.


## Citation  
@inproceedings{NEURIPS2022_2e0bd92a,  
 author = {Liu, Wei and Wang, Haozhao and Wang, Jun and Li, Ruixuan and Yue, Chao and Zhang, YuanKai},  
 booktitle = {Advances in Neural Information Processing Systems},  
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},  
 pages = {6954--6966},  
 publisher = {Curran Associates, Inc.},  
 title = {FR: Folded Rationalization with a Unified Encoder},  
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/2e0bd92a1d3600d4288df51ac5e6be5f-Paper-Conference.pdf},  
 volume = {35},  
 year = {2022}  
}  

## Acknowledgements
The code is largely based on [Car](https://github.com/code-terminator/classwise_rationale) and [DMR](https://github.com/kochsnow/distribution-matching-rationality). Most of the hyperparameters (e.g. the '--cls_lambda'=0.9) are also from them. We are grateful for their open source code.




