for i in `seq 2 49`;
        do
        for j in `seq 1 50`;
                do
                  python generate.py --stochastic-beam-search --beam $i --nbest $i --no-early-stop --unnormalized --sampling-temperature 0.1 data-bin/test --path ../fairseq/data-bin/wmt14.en-fr.fconv-py/model.pt
                done
        done
