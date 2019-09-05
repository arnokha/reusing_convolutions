atari_games_list="Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand"

for g in $atari_games_list; do
    game=$g"-v0"
    mkdir -p $g
    n_out_channel_list_2="20 40 80"
    
    ## 2 layers
    for n_out_channels in $n_out_channel_list_2; do
        if [ $n_out_channels == "20" ]; then 
            python test_atari_2_layer_base_gpu.py $game 20 40 
            #python test_atari_2_layer_new_safe.py $game 20 40 
        fi
        if [ $n_out_channels == "40" ]; then 
            python test_atari_2_layer_base_gpu.py $game 40 80 
            #python test_atari_2_layer_new_safe.py $game 40 80 
        fi
        if [ $n_out_channels == "80" ]; then 
            python test_atari_2_layer_base_gpu.py $game 80 160 
            #python test_atari_2_layer_new_safe.py $game 80 160 
        fi
    done
done
