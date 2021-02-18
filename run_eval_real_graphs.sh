#!/bin/bash

# for baseline method
for g in Exhibition Flu Hospital India irvine;
do
    for method in greedy adaptive_greedy lazy_adaptive_greedy;
    do
        for bud in 1 2 4 8;
        do
            echo "start running method: ${method} on graph: ${g} with budget: ${bud}"
            bash run_interactive.sh 3 python3.7 main.py --config=colge \
                                                        --env-config=basic_env --results-dir=results/Feb18_RL4IM_real_graphs/${g}/${method} \
                                                        with T=8 \
                                                        budget=${bud} \
                                                        save_every=2 \
                                                        q=0.6 \
                                                        mode='test' \
                                                        realgraph=${g} \
                                                        model_scheme=normal \
                                                        checkpoint_path=model_ckpt/Feb17_RL4IM/colge/sacred/${bud}/models &
        done
    done
done
