/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */


#ifndef m0609_kinematic_nmpc_COST
#define m0609_kinematic_nmpc_COST

#ifdef __cplusplus
extern "C" {
#endif





// Cost at initial shooting node

int m0609_kinematic_nmpc_cost_y_0_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_0_fun_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_0_fun_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_0_fun_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_0_fun_n_in(void);
int m0609_kinematic_nmpc_cost_y_0_fun_n_out(void);

int m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt_n_in(void);
int m0609_kinematic_nmpc_cost_y_0_fun_jac_ut_xt_n_out(void);



// Cost at path shooting node

int m0609_kinematic_nmpc_cost_y_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_fun_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_fun_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_fun_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_fun_n_in(void);
int m0609_kinematic_nmpc_cost_y_fun_n_out(void);

int m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt_n_in(void);
int m0609_kinematic_nmpc_cost_y_fun_jac_ut_xt_n_out(void);



// Cost at terminal shooting node

int m0609_kinematic_nmpc_cost_y_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_e_fun_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_e_fun_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_e_fun_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_e_fun_n_in(void);
int m0609_kinematic_nmpc_cost_y_e_fun_n_out(void);

int m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt_sparsity_in(int);
const int *m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt_sparsity_out(int);
int m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt_n_in(void);
int m0609_kinematic_nmpc_cost_y_e_fun_jac_ut_xt_n_out(void);




#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // m0609_kinematic_nmpc_COST
