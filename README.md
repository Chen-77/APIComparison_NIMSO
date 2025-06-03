# APIComparison_NIMSO
# Introduction
Code and data of the paper [API comparison based on the non-functional information mined from Stack Overflow](https://doi.org/10.1016/j.scico.2024.103228). 
# Example
An example: 
```
python run_classifier.py \
  --task_name=performance \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=./aspect_data \
  --vocab_file=./bert_base/NER_bert/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./bert_base/NER_bert/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./bert_base/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=5 \
  --output_dir=./output
```
Due to the limitation, we make the basic model and trained model public at https://pan.baidu.com/s/1Ge0J9qGP3ls8MCNk3ZlcYA with the extraction code 0824.
# Citation
If you find our work useful in your research, please consider citing:
```
@article{CHEN2025103228,
title = {API comparison based on the non-functional information mined from Stack Overflow},
author = {Zhiqi Chen and Yuzhou Liu and Lei Liu and Huaxiao Liu and Ren Li and Peng Zhang},
journal = {Science of Computer Programming},
volume = {241},
pages = {103228},
year = {2025},
issn = {0167-6423}
}
```
