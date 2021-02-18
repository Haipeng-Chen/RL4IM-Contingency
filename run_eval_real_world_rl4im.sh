#!/bin/bash

# for baseline method
for g in Exhibition Flu Hospital India irvine;
do
    for method in rl;
    do
        for bud in 4;
        do
            echo "start running method: ${method} on graph: ${g} with budget: ${bud}"
            model_path=model_ckpt/Feb17_RL4IM/colge/sacred/3/models
            bash run_interactive.sh 3 python3.7 main.py --config=colge --env-config=basic_env --results-dir=results/Feb18_RL4IM_real_graphs/${g}/rl4im_${method} with method=${method} T=8 budget=${bud} save_every=2 q=0.6 mode='test' realgraph=${g} model_scheme=normal checkpoint_path=${model_path} &
            sleep 3;
        done
    done
done
