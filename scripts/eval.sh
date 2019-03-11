LSTM=500
DEC_LSTM=700
WRD=300

export CUDA_VISIBLE_DEVICES=3

DATA_DIR=
SAMPLES_DIR=/tmp

MDL=models/model.ckpt-87142_beam
vocab=models/vocab.txt
vocab_size=10000

python src/sample.py \
	--vocab_file $vocab \
	--vocab_size $vocab_size \
	--restore_ckpt_path $MDL \
	--lstm_size $LSTM \
	--dec_lstm_size $DEC_LSTM \
	--flip_input \
	--mdl_name $MDL_NAME \
	--crop_batches \
	--custom_input \
	--flip_label \
	--input_file $DATA_DIR/sentiment.test.0,$DATA_DIR/sentiment.test.1 \
	--beam_search \
	--samples_dir $SAMPLES_DIR
