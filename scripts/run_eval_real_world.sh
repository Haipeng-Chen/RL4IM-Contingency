#!/bin/bash

# for baseline method
# checkpoint_path=model_ckpt/Feb17_RL4IM/colge/sacred/${bud}/models 
for g in Exhibition Flu Hospital India irvine;
do
    for method in lazy_adaptive_greedy random;
    do
        for bud in 8;
        do
            echo "start running method: ${method} on graph: ${g} with budget: ${bud}"
            bash run_interactive.sh 3 python3.7 main.py --config=colge --env-config=basic_env --results-dir=results/Feb18_RL4IM_real_graphs/${g}/${method}/bud_${bud} with method=${method} T=8 budget=${bud} save_every=2 q=0.6 mode='test' realgraph=${g} model_scheme=normal &
            sleep 3;
        done
    done

    for method in rl;
    do
        for bud in 8;
        do
            echo "start running method: ${method} on graph: ${g} with budget: ${bud}"
            bash run_interactive.sh 3 python3.7 main.py --config=colge --env-config=basic_env --results-dir=results/Feb18_RL4IM_real_graphs/${g}/Base${method}/bud_${bud} with method=${method} T=8 budget=${bud} save_every=2 q=0.6 mode='test' realgraph=${g} model_scheme=normal reward_type=0 &
            sleep 3;
        done
    done
done
