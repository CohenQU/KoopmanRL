for bottleneck in 4 8 16 32
# for bottleneck in 4
do
    for steps in 2 4 8 16
    #for steps in 2
    do
        for env_id in "ant" "inverted_double_pendulum" "inverted_pendulum"
        do 
            python driver.py --alpha 1 --dataset ${env_id} --noise 0.0 --lr 1e-2 --epochs 100 --batch 64 --folder results/${env_id}_s${steps}b${bottleneck} --lamb 1 --steps ${steps} --bottleneck ${bottleneck} --lr_update 30 200 400 500 --lr_decay 0.2 --pred_steps 1000 --train_size 100000 --test_size 10000 --backward 0 --seed 0 >> log/${env_id}_s${steps}b${bottleneck}.txt
        done
    done
done