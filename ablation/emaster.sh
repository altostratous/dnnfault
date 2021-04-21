for p in 0.5 0.8; do
  for sigma in 0.0 0.1 0.25; do
    for last_dropout in False True; do
      for first_dropout in True False; do
        for weight_clip in True False; do
          for mc_dropout in True False; do
            for t_mapping in RandomBETMapping None; do
              for e_mapping in None RowHammerUpSprayAttackMapping RowHammerSprayAttackMapping; do
                if [[ $sigma == 0.0 ]] && [[ $first_dropout == False ]]; then continue; fi
                if [[ $t_mapping == RandomBETMapping ]] && [[ $weight_clip == False ]]; then continue; fi
                sbatch ./parameterized.sh --train=False --mc_dropout=$mc_dropout --dropout_p=$p --sigma=$sigma --first_dropout=$first_dropout --last_dropout=$last_dropout --weight_clip=$weight_clip --t_mapping=$t_mapping --e_mapping=$e_mapping
              done
            done
          done
        done
      done
    done
  done
done
