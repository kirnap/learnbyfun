# LLM training in PyTorch and NVIDIA-5090

This is a self-replication of Karpahty's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) course on YouTube.


## Some random training notes

1. There's DDP overhead with a single GPU:
    ```
    (vp1) omer@vp build-nanogpt % python train_gpt2.py 
    DDP is not enabled using device cuda
    -> total desired batch size:  524288
    -> calculated gradient accumulation steps: 32
    loaded 338025 tokens
    1 epoch = 20
    num decayed parameter tensors: 50, with 124,354,560 parameters
    num non-decayed parameter tensors: 98, with 121,344, parameters
    using fused AdamW: True
    step    0| loss: 10.938572 | lr: 6.0000e-05 | norm: 27.0145  | dt: 3663.01ms | tok/sec: 143130.26
    step    1| loss: 9.649529 | lr: 1.2000e-04 | norm: 9.5134  | dt: 2713.86ms | tok/sec: 193189.05
    step    2| loss: 9.224648 | lr: 1.8000e-04 | norm: 5.6872  | dt: 2713.59ms | tok/sec: 193208.23
    step    3| loss: 9.812168 | lr: 2.4000e-04 | norm: 8.2264  | dt: 2713.56ms | tok/sec: 193210.12
    step    4| loss: 9.190122 | lr: 3.0000e-04 | norm: 4.3002  | dt: 2712.87ms | tok/sec: 193259.68
    step    5| loss: 8.676892 | lr: 3.6000e-04 | norm: 3.6255  | dt: 2713.46ms | tok/sec: 193217.59
    step    6| loss: 8.295106 | lr: 4.2000e-04 | norm: 1.9540  | dt: 2714.61ms | tok/sec: 193135.76
    step    7| loss: 8.066936 | lr: 4.8000e-04 | norm: 2.8287  | dt: 2714.53ms | tok/sec: 193141.27
    step    8| loss: 7.713205 | lr: 5.4000e-04 | norm: 1.9201  | dt: 2714.45ms | tok/sec: 193147.21
    step    9| loss: 7.346015 | lr: 6.0000e-04 | norm: 1.8049  | dt: 2714.71ms | tok/sec: 193128.73
    step   10| loss: 7.028761 | lr: 6.0000e-04 | norm: 1.8351  | dt: 2715.57ms | tok/sec: 193067.10
    
    (vp1) omer@vp build-nanogpt % torchrun --standalone --nproc_per_node=1 train_gpt2.py
    -> total desired batch size:  524288
    -> calculated gradient accumulation steps: 32
    loaded 338025 tokens
    1 epoch = 20
    num decayed parameter tensors: 50, with 124,354,560 parameters
    num non-decayed parameter tensors: 98, with 121,344, parameters
    using fused AdamW: True
    step    0| loss: 10.938571 | lr: 6.0000e-05 | norm: 27.0146  | dt: 6437.43ms | tok/sec: 81443.69
    step    1| loss: 9.649517 | lr: 1.2000e-04 | norm: 9.5133  | dt: 2815.50ms | tok/sec: 186214.98
    step    2| loss: 9.224654 | lr: 1.8000e-04 | norm: 5.6877  | dt: 2815.18ms | tok/sec: 186236.17
    step    3| loss: 9.812183 | lr: 2.4000e-04 | norm: 8.2261  | dt: 2814.84ms | tok/sec: 186258.35
    step    4| loss: 9.190122 | lr: 3.0000e-04 | norm: 4.3001  | dt: 2814.42ms | tok/sec: 186286.45
    step    5| loss: 8.676889 | lr: 3.6000e-04 | norm: 3.6255  | dt: 2814.15ms | tok/sec: 186304.14
    step    6| loss: 8.295108 | lr: 4.2000e-04 | norm: 1.9539  | dt: 2814.67ms | tok/sec: 186269.52
    step    7| loss: 8.066954 | lr: 4.8000e-04 | norm: 2.8294  | dt: 2814.24ms | tok/sec: 186298.46
    step    8| loss: 7.713212 | lr: 5.4000e-04 | norm: 1.9197  | dt: 2810.93ms | tok/sec: 186517.52
    step    9| loss: 7.346024 | lr: 6.0000e-04 | norm: 1.8048  | dt: 2811.33ms | tok/sec: 186491.01
    step   10| loss: 7.028761 | lr: 6.0000e-04 | norm: 1.8352  | dt: 2813.80ms | tok/sec: 186327.22
    ```
    
2. Maximum microbatch for 5090 is 40 * 1024 -> which is around ~50K tokens per second
