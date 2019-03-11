LSTM=500
DEC_LSTM=700
WRD=300

EP=20
BS=128

RESULTS_HOME=results
DATA_HOME=data

MDL=model_enc500_dec700_w300

train_file_pattern=train.tfrecords
train_num_examples=500000
vocab_size=10000
Nlabels='2'

TRAIN_GPU=0

export CUDA_VISIBLE_DEVICES=$TRAIN_GPU
python src/train.py \
	--file_pattern $train_file_pattern \
	--num_examples $train_num_examples \
	--vocab_size $vocab_size \
	--embeddings_path $embs \
	--checkpoint_dir results/$MDL \
	--lstm_size $LSTM \
	--dec_lstm_size $DEC_LSTM \
	--Nepochs $EP \
	--flip_input \
	--crop_batches \
	--batch_size $BS \
	--embedding_size $WRD \
	--learning_rate 0.0001 \
	--dropout_keep_prob_enc 0.5 \
	--dropout_keep_prob_dec 0.5 \
	--interp_z \
	--interp_ratio 0.5 \
	--cgan \
	--Nlabels $Nlabels \
	--input_size 30 \
	--decode_max_steps 30
