
# to reproduce the result in Table 3 in paper 
# ppi
python -m graphsage.supervised_train --train_prefix ~/data/ppi/ppi --model_prefix f512512_samp2510_b512_ep100 --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 512 --validate_batch_size 512 --gpu 0 --sigmoid --fast_ver --epochs 100

# reddit
python -m graphsage.supervised_train --train_prefix ~/data/reddit/reddit --model_prefix f512512_samp2510_b32_ep100 --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 32 --validate_batch_size 32 --gpu 0 --sigmoid --fast_ver --epochs 100

# pubmed
python -m graphsage.supervised_train --train_prefix ~/data/pubmed/pubmed --model_prefix f512512_samp2510_b512_ep100 --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 512 --validate_batch_size 512 --gpu 0 --sigmoid --fast_ver --epochs 100


#python -m graphsage.supervised_train --train_prefix ./example_data/toy-ppi --model_prefix 1 --model graphsage_mean --sigmoid
