# FR: Folded Rationalization with a Unified Encoder  
This repo contains Pytorch implementation of [Folded Rationalization with a Unified Encoder (FR, NeurIPS2022)](https://arxiv.org/abs/2209.08285).    
## Star
If the codes help you, please give us a **star**. It plays an important role in encouraging us to open the source codes. Thank you!  



## Environments  
torch 1.10.2+cu113.   
python 3.7.9.   
tensorboardx 2.4.   
tensorboard 2.6.0    
RTX3090  
We suggest you to created a new environment with: conda create -n FR python=3.7  
And then conduct: pip install -r requirements.txt

## Datasets  
For Beer Reviews, you should first obtain authorization for this dataset from the original author.
 
Beer Reviews: you can get it from [here](http://people.csail.mit.edu/taolei/beer/). Then place it in the ./data/beer directory.  
Hotel Reviews: you can get it from [here](https://people.csail.mit.edu/yujia/files/r2a/data.zip). 
Then  find hotel_Location.train, hotel_Location.dev, hotel_Service.train, hotel_Service.dev, hotel_Cleanliness.train, hotel_Cleanliness.dev from data/oracle and put them in the ./data/hotel directory. 
Find hotel_Location.train, hotel_Service.train, hotel_Cleanliness.train from data/target and put them in the ./data/hotel/annotations directory.  
Word embedding: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/). Then put it in the ./data/hotel/embeddings directory.

## Running
### Beer
#### Appearance:  
python run_share_beer.py --lr 0.0001 --batch_size 256 --share 1 --sparsity_percentage 0.15 --sparsity_lambda 10 --continuity_lambda 10 --embedding_dir './data/hotel/embeddings' --embedding_name 'glove.6B.100d.txt' --embedding_dim 100 --epochs 300 --aspect 0  
#### Aroma:   
python run_share_beer.py --lr 0.0001 --batch_size 256 --share 1 --sparsity_percentage 0.123 --sparsity_lambda 12 --continuity_lambda 10 --embedding_dir './data/hotel/embeddings' --embedding_name 'glove.6B.100d.txt' --embedding_dim 100 --epochs 500 --aspect 1  
#### Palate:  
python run_share_beer.py --lr 0.0001 --batch_size 64 --share 1 --sparsity_percentage 0.12 --sparsity_lambda 10 --continuity_lambda 10 --embedding_dir './data/hotel/embeddings' --embedding_name 'glove.6B.100d.txt' --embedding_dim 100 --epochs 500 --aspect 2

### Hotel
#### Location:   
python new_run_share_hotel.py --lr 0.0001 --hidden_dim 200 --batch_size 64 --share 1 --sparsity_percentage 0.09 --sparsity_lambda 8 --continuity_lambda 10 --epochs 200 --aspect 0  
#### Service:   
python new_run_share_hotel.py --lr 0.00005 --hidden_dim 100 --batch_size 256 --share 1 --sparsity_percentage 0.115 --sparsity_lambda 10 --continuity_lambda 10 --epochs 400 --aspect 1  
#### Cleanliness:   
python new_run_share_hotel.py --lr 0.0002 --batch_size 256 --dropout 0.2 --hidden_dim 100 --share 1 --sparsity_percentage 0.1 --sparsity_lambda 10 --continuity_lambda 12 --epochs 100 --aspect 2

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



