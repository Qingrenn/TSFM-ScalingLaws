#!/bin/bash

python -m cli.train_val -cp conf/pretrain -cn default_ddp_val_enc \
model=encoder_10M \
data=lotsa16B_weighted \
val_data=lotsa16B_lsf_monash \
trainer.logger.project=demo_scalinglaws \
run_name=encoder10M_lotsa16B

python -m cli.train_val -cp conf/pretrain -cn default_ddp_val_dec \
model=decoder_10M \
data=lotsa16B_weighted \
val_data=lotsa16B_lsf_monash \
trainer.logger.project=demo_scalinglaws \
run_name=decoder10M_lotsa16B