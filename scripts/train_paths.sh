#!/bin/bash
# [DOWNLOAD NGC] job run --config "fine_tune_rn_r_x" "python -m experiments.train_path \"rn(r(x))\"";
NGC job run --config "fine_tune_k3N_Ck3_RC_x" "python -m experiments.train_path \"k3N(Ck3(RC(x)))\"";
# [DONE] job run --config "fine_tune_F_EC_a_x" "python -m experiments.train_path \"F(EC(a(x)))\"";
# [DONE] job run --config "fine_tune_S_CE_RC_x" "python -m experiments.train_path \"S(CE(RC(x)))\"";
NGC job run --config "fine_tune_k3N_Ck3_EC_a_x" "python -m experiments.train_path \"k3N(Ck3(EC(a(x))))\"";
# [DONE] job run --config "fine_tune_F_KC_k_x" "python -m experiments.train_path \"F(KC(k(x)))\"";
NGC job run --config "fine_tune_rn_Er_CE_RC_x" "python -m experiments.train_path \"rn(Er(CE(RC(x))))\"";
# [DONE] job run --config "fine_tune_G_d_x" "python -m experiments.train_path \"G(d(x))\"";
# [DONE] job run --config "fine_tune_F_H_ED_a_x" "python -m experiments.train_path \"F(H(ED(a(x))))\"";
# [DONE] job run --config "fine_tune_F_H_d_x" "python -m experiments.train_path \"F(H(d(x)))\"";
# [DONE] job run --config "fine_tune_G_ED_a_x" "python -m experiments.train_path \"G(ED(a(x)))\"";
[cloud6] job run --config "fine_tune_F_H_g_S_a_x" "python -m experiments.train_path \"F(H(g(S(a(x)))))\"";
# job run --config "fine_tune_F_RC_x" "python -m experiments.train_path \"F(RC(x))\"";

# [DONE] job run --config "fine_tune_F_f_S_a_x" "python -m experiments.train_path \"F(f(S(a(x))))\""
NGC job run --config "fine_tune_rn_nr_S_a_x" "python -m experiments.train_path \"rn(nr(S(a(x))))\"";
NGC job run --config "fine_tune_rn_Er_a_x" "python -m experiments.train_path \"rn(Er(a(x)))\"";
NGC job run --config "fine_tune_F_f_rn_r_x" "python -m experiments.train_path \"F(f(rn(r(x))))\"";


NGC job run --config "fine_tune_rn_nr_F_RC_x" "python -m experiments.train_path \"rn(nr(F(RC(x))))\"";
NGC job run --config "fine_tune_k3N_Ck3_KC_k_x" "python -m experiments.train_path \"k3N(Ck3(KC(k(x))))\"";

rn(r(x))
k3N(Ck3(RC(x)))
F(EC(a(x)))
S(CE(RC(x)))
k3N(Ck3(EC(a(x))))
F(KC(k(x)))
rn(Er(CE(RC(x))))
G(d(x))
F(H(ED(a(x))))
G(ED(a(x)))
F(H(g(S(a(x)))))
F(RC(x))
F(f(S(a(x))))
rn(nr(S(a(x))))
rn(Er(a(x)))
F(f(rn(r(x))))
rn(nr(F(RC(x))))
k3N(Ck3(KC(k(x))))


F(f(n(x)))
[cloud4] job run --config "fine_tune_F_f_n_x" "python -m experiments.train_path \"F(f(n(x)))\"";
