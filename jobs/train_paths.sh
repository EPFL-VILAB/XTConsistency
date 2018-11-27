#!/bin/bash


 ob run --config "finetune_percep_G_g_n_x" "python -m experiments.train_path \"G(g(n(x)))\"";
job run --config "finetune_percep_G_ED_a_x" "python -m experiments.train_path \"G(ED(a(x)))\"";
job run --config "finetune_percep_F_H_g_S_a_x" "python -m experiments.train_path \"F(H(g(S(a(x)))))\"";
job run --config "finetune_percep_F_H_d_x" "python -m experiments.train_path \"F(H(d(x)))\"";
job run --config "finetune_percep_F_f_rn_r_x" "python -m experiments.train_path \"F(f(rn(r(x))))\"";
job run --config "finetune_percep_k3N_Ck3_EC_a_x" "python -m experiments.train_path \"k3N(Ck3(EC(a(x))))\"";
job run --config "finetune_percep_k3N_Ck3_RC_x" "python -m experiments.train_path \"k3N(Ck3(RC(x)))\"";

# job run --config "fine_tune_F_f_S_a_x" "python -m experiments.train_path \"F(f(S(a(x))))\""
# job run --config "fine_tune_rn_nr_S_a_x" "python -m experiments.train_path \"rn(nr(S(a(x))))\"";
# job run --config "fine_tune_rn_Er_a_x" "python -m experiments.train_path \"rn(Er(a(x)))\"";

# job run --config "fine_tune_rn_r_x" "python -m experiments.train_path \"rn(r(x))\"";
# job run --config "fine_tune_rn_Er_a_x" "python -m experiments.train_path \"rn(Er(a(x)))\"";
# job run --config "fine_tune_F_EC_a_x" "python -m experiments.train_path \"F(EC(a(x)))\"";
# job run --config "fine_tune_S_CE_RC_x" "python -m experiments.train_path \"S(CE(RC(x)))\"";
# job run --config "fine_tune_k3N_Ck3_EC_a_x" "python -m experiments.train_path \"k3N(Ck3(EC(a(x))))\"";
# job run --config "fine_tune_F_KC_k_x" "python -m experiments.train_path \"F(KC(k(x)))\"";
# job run --config "fine_tune_rn_Er_CE_RC_x" "python -m experiments.train_path \"rn(Er(CE(RC(x))))\"";
# job run --config "fine_tune_G_d_x" "python -m experiments.train_path \"G(d(x))\"";
# job run --config "fine_tune_F_H_ED_a_x" "python -m experiments.train_path \"F(H(ED(a(x))))\"";
