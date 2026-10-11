# Run options of MHD-ALE for the suite, shared by the trial and the production runs (sourced).
# Orders, integrator and step control: those of the fork's examples (README, test_TaylorGreen.sh): velocity degree 2,
# energy and density degree 1, potential degree 3, three-stage SSP Runge-Kutta, CFL 0.5.
MHDALE_ORDERS="-ok 2 -ot 1 -or 1 -oa 3 -s 3 -cfl 0.5 -g1 0.85 -g2 1.02 -cgt 1e-12"
# The periodic box with remap, as the fork's README gives it for periodic x: initial-reference smoother (-mst 1),
# comoving rezone (-crz), the mean-field correction (-pmf), remap every 10 steps, its magnetic audit (-ma).
MHDALE_PERIODIC_ALE="-ale -mst 1 -crz -pmf -ma -fsr -fsri 10 -rmv 1 -rme 1 -rma 1 -rmr 3 -bpt 1 -no-pv -no-vis"
# The planar tests the 2D code can represent (no v_z, no B_z): 13 of the suite's 17.
MHDALE_TESTS="entropy field_loop_novz brio_wu brio_wu_rotated orszag_tang rotor current_sheet blast_2d_b0 blast_2d_b50 blast_2d_b100 kh_uniform kh_localized gem"
