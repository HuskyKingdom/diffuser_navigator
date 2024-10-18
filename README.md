

<br />
<div align="center" id="readme-top">
  
  <h1 align="center">Diffuser Navigator</h1>

  <p align="center" >





Experimental Repo of Diffuser Navigator.



<br />
<a href="https://yuhang.topsoftint.com">Contact me at: <strong>me@yhscode.com</strong></a>

<a href="https://yhscode.com"><strong>View my full bio.</strong></a>
    <br />
    <br />
  </p>
</div>


## Training

Single GPU:

```
python run.py \
  --exp-config diffuser_baselines/config/diffuser.yaml \
  --run-type train
```

DDP Training:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 run.py \
  --exp-config diffuser_baselines/config/ddp_diffuser.yaml \
  --run-type train
```
