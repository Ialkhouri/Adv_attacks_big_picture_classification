# Adv_attacks_big_picture_classification
# Efficient Algorithms for Fooling the Big Picture in Classification Tasks - HCC results

Below is a guide for the scripts used to reproduce results of table(s) and figure(s). This is the results for the HCC part of the paper.

## Code, Data, Results

1- Figure 2b: Run script "GLRT_minmax_convex_fixed_simple_ADMM.m". Vary epsilon_c according to the x-axis and obtain the RLA_sup (zeta in the paper) and rho_2 (sigma_2 in the paper). Use fucntion "ADMM_solver" in the script.

2- Table   3: Run script "GLRT_minmax_convex_fixed_simple_ADMM.m". Vary variables ADMM_iterations (Tau in the paper) and rho_aug_lagrang (r in the paper). Use fucntion "ADMM_solver" in the script.

3- Table   6: Use below to obtain the printed results:

    - ID 01 : GLRT_minmax_cvx_fixed_algorithmic_I.m

    - ID 02 : GLRT_minmax_convex_fixed_algorithmic_I_ADMM.m

    - ID 03 : GLRT_minmax_convex_fixed_algorithmic_I_ADMM.m with choosing fucntion ADMM_solver_w_stopping_criteria inside.

    - ID 04 : GLRT_minmax_cvx_fixed_algorithmic_II.m 

    - ID 05 : GLRT_minmax_convex_fixed_algorithmic_II_ADMM.m 

    - ID 06 : GLRT_minmax_convex_fixed_algorithmic_II_ADMM.m with choosing fucntion ADMM_solver_w_stopping_criteria inside.

    - ID 07 : GLRT_minmax_cvx_fixed_ext_constraints.m

    - ID 08 : GLRT_minmax_convex_fixed_ext_ADMM.m

    - ID 09 : GLRT_minmax_convex_fixed_ext_ADMM.m with choosing fucntion ADMM_solver_w_stopping_criteria inside.

    - ID 10 : GLRT_minmax_cvx_fixed_simple.m

    - ID 11 : GLRT_minmax_convex_fixed_simple_ADMM.m

    - ID 12 : GLRT_minmax_convex_fixed_simple_ADMM.m with choosing fucntion ADMM_solver_w_stopping_criteria inside.

4- Figure  5: Use Conf_matrix_plotter.m with enableing display to get the figures.

Note that all Figures are inside folder figs.

Note that, for any script that needs CVX, you need to uncomment 

"
#!/bin/bash

curl -s http://web.cvxr.com/cvx/cvx-rd.tar.gz | tar zx

matlab -nodisplay -r "cd cvx; cvx_setup()"

in the run file 

## Execution 

Since we have many methods here, make sure to change the trials variable to 1000.

## Environment and Dependencies

- MATLAB
- CVX


## Troubleshooting

If you run into any issues or have any questions, contact Ismail @ ialkhouri@knights.ucf.edu


# Efficient Algorithms for Fooling the Big Picture in Classification Tasks - HSC results

Below is a guide for the scripts used to reproduce results of table(s) and figure(s). This is the results for the HCC part of the paper.

## Code, Data, Resultscode ocean get link before verification

1- Figure 2a: Run script "NOC_ADMM_MNIST_model_1.py". Vary epsilon_c according to the x-axis and obtain the RLA_sup (zeta in the paper) and rho_2 (sigma_2 in the paper). Use variable admm_type = "ADMM".

2- Table   2: Run script "NOC_ADMM_MNIST_model_1.py". Vary variables number_of_iterations_tau (Tau in the paper) and r_penalty_factor (r in the paper). Use variable admm_type = "ADMM". 

3- Table   6: Use below to obtain the printed results:

    - ID 01 : Alg1_CVX_MNIST_model_1.py

    - ID 02 : Alg1_ADMM_MNIST_model_1.py

    - ID 03 : Alg1_ADMM_MNIST_model_1.py with variable admm_type = "eADMM". 

    - ID 04 : Alg2_CVX_MNIST_model_1.py 

    - ID 05 : Alg2_ADMM_MNIST_model_1.py 

    - ID 06 : Alg2_ADMM_MNIST_model_1.py with variable admm_type = "eADMM". 

    - ID 07 : REC_CVX_MNIST_model_1.py

    - ID 08 : REC_ADMM_MNIST_model_1.py

    - ID 09 : REC_ADMM_MNIST_model_1.py with variable admm_type = "eADMM". 

    - ID 10 : NOC_CVX_MNIST_model_1.py

    - ID 11 : NOC_ADMM_MNIST_model_1.py

    - ID 12 : NOC_ADMM_MNIST_model_1.py with variable admm_type = "eADMM". 

4- Figure  5: Use Conf_matrix_plotter.m of the other codecoean "Efficient Algorithms for Fooling the Big Picture in Classification Tasks - HCC" with enableing display to get the figures.

Note that all Figures are inside folder figs of "Efficient Algorithms for Fooling the Big Picture in Classification Tasks - HCC".

## Execution 

Since we have many methods here, make sure to change the number of observation variable to 1000.

## Environment and Dependencies

- keras
- tensorflow
- numpy
- scikit-image
- cvxpy


## Troubleshooting

If you run into any issues or have any questions, contact Ismail @ ialkhouri@knights.ucf.edu
