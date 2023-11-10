offline_runs="./wandb/offline-run*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
    sleep 5m
done