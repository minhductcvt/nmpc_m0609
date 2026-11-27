/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * Licensed under the 2-Clause BSD License.
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_ocp_m0609_kinematic_nmpc_41c0c9b9.h"

#define NX     OCP_M0609_KINEMATIC_NMPC_41C0C9B9_NX
#define NZ     OCP_M0609_KINEMATIC_NMPC_41C0C9B9_NZ
#define NU     OCP_M0609_KINEMATIC_NMPC_41C0C9B9_NU
#define NP     OCP_M0609_KINEMATIC_NMPC_41C0C9B9_NP


int main()
{
    int status = 0;
    ocp_m0609_kinematic_nmpc_41c0c9b9_sim_solver_capsule *capsule = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_solver_create_capsule();
    status = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_get_sim_out(capsule);
    void *acados_sim_dims = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;

  
    x_current[0] = 0;
    x_current[1] = 0;
    x_current[2] = 1.5708;
    x_current[3] = 0;
    x_current[4] = 1.5708;
    x_current[5] = 0;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 0.45;
    p[1] = 0;
    p[2] = 0.15;
    p[3] = 0.05;
    p[4] = 0.25;

    ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_update_params(capsule, p, NP);
  

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_free(capsule);
    if (status) {
        printf("ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_free() returned status %d. \n", status);
    }

    ocp_m0609_kinematic_nmpc_41c0c9b9_acados_sim_solver_free_capsule(capsule);

    return status;
}
