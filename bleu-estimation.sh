for i in `seq 2 25`;
        do
        for j in `seq 1 10`;
                do
                  python generate.py --cps --beam $i --nbest $i --no-early-stop --unnormalized --sampling-temperature 0.1 data-bin/test --path ../fairseq/data-bin/wmt14.en-fr.fconv-py/model.pt
                done
        done
