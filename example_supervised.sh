
# ppi
python -m graphsage.supervised_train --train_prefix ~/data/ppi/ppi --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 32 --validate_batch_size 32 --gpu 0 --sigmoid --epochs 100

# reddit
python -m graphsage.supervised_train --train_prefix ~/data/reddit/reddit --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 32 --validate_batch_size 32 --gpu 0 --epochs 100

# pubmed
python -m graphsage.supervised_train --train_prefix ~/data/pubmed/pubmed --model mean_add --dim_1 512 --dim_2 512 --samples_1 25 --samples_2 10 --batch_size 32 --validate_batch_size 32 --gpu 0 --epochs 100

