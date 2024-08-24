# APIComparison_NIMSO
### Code and data of API Comparison Based on the Non-functional Information Mined from Stack Overflow. 
An example: 
```
python run_classifier.py \
  --task_name= performance \
  --do_train= true \
  --do_eval= true \
  --do_predict= true \
  --data_dir= ./aspect_data \
  --vocab_file= ./bert_base/NER_bert/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file= ./bert_base/NER_bert/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint= ./bert_base/BERT_BASE_DIR/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length= 128 \
  --train_batch_size= 16 \
  --learning_rate= 1e-5 \
  --num_train_epochs= 5 \
  --output_dir= ./output
```
### Due to the limitation of uploading large files, we make the trained model and the basic model public at https://pan.baidu.com/s/1Ge0J9qGP3ls8MCNk3ZlcYA. 
### The extraction code is 0824.
