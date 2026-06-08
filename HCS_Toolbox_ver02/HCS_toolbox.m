%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% HCS (Homogeneous Control Systems) Toolbox  (Version 0.2)
%%
%%
%% List of functions (for more info type 'help <name_of_function>')
%%
%%
%%  Homogeneous Euclidean Space:
%%
%%   hnorm             - canonical homogeneous norm (by bisection method)
%%   hphi              - homogeneous transformation of R^n                  
%%   hphi_inv          - inverse homogeneous transformation of R^n       
%%   hadd              - sum of vectors in homogeneous Euclidian space    
%%   hdot              - product by scalar in homogeneous Euclidiean space 
%%   hinner            - inner product in homogeneous Euclidean space      
%%
%%
%%  Homogeneous Geometric Objects:
%%
%%  hproj              - computation of homogeneous projection
%%  hcurve             - generation of points on a homogeneous curve           
%%  hsphere            - generation of points on a homogeneous sphere          
%%  hcone              - generation of points on a unit sphere belonging 
%%                       to linear homogeneous cone   
%% 
%%
%%  Homogeneous Artificial Neural Netorks (ANN):  
%%
%%  ann            - ANN for approximation of a mapping R^n -> R^m         (*)
%%  hann_f         - homogeneous ANN for a function R^n -> R^m             (*)
%%  hann_vf        - homogeneous ANN for a vector field R^n -> R^n         (*)
%%  ann2hann       - data-driven transformation of ANN to homogenenous ANN (*)
%%  
%%
%%  Approximation of Canonical Homogeneous Norm:
%%
%%   hnorm_newton      - canonical homogeneous norm (by Newton method)
%%   hnorm_r           - explicit homogeneous norm for diagonizable dilation
%%   hnorm_ANN         - explicit homogeneous norm by ANN
%%   approx_hnorm_ANN  - approximation of canonical homogeneous norm by ANN
%%   approx_hnorm_r    - approximation of canonical homogeneous norm by 
%%                       explicit homogeneous norm for diagonalizable 
%%                       dilation                                          (needs YALMIP)
%%
%%
%%  Homogenenous Functions and Vector Fields:
%% 
%%  is_hom_f     - numerical test of a homogeneity of a function           (*)
%%  is_hom_vf    - numerical test of a homogeneity of a vector field       (*)
%%  find_deg_f   - data-driven homogeneity degree of a function            (*)
%%  find_deg_vf  - data-driven homogeneity degree of a vector field        (*)
%%  find_hom_f   - data-driven dilation symmetry of a function             (*)
%%  find_hom_vf  - data-driven dilation symmetry of a vector field         (*)
%%
%%
%%  Homogeneous Control:
%%
%%  hpc_design   - Homogeneous Proportional Control (HPC) design  
%%  chpc_design  - constant-time HPC design
%%  hpci_design  - Homogeneous Proportional-Integral Control (HPIC) design   
%%  hsmc_design  - Homogeneous Sliding Mode Controller (HSMC)  design       
%%  hsmci_design - design of HSMC with Integral action (HSMCI)
%%  fhpc_design  - fixed-time HPC design
%%  fhpic_design - fixed-time HPIC design 
%%  lpc2hpc      - upgrading Linear Proportional Control (LPC) to HPC
%%  lpic2hpc     - upgrading Linear PI control (LPIC) to HPIC
%%
%%
%%  (Safety Critical Homogeneous Systems)
%%
%%  hno_design   - homogeneous non-overshooting control design             (*)
%%  h_safety     - homogeneous safety filter                               (*)
%%  is_hcone_inv - numerical test of positive invariance of linear
%%                 homogeneous cone                                        (*)
%%
%%
%% Discretization of Homogeneous Control:
%%
%%  e_hpc       - explicit discretization of HPC
%% si_hpc       - semi-implicit discretization of HPC
%%  c_hpc       - consistent discretization of HPC   
%%  e_hpic      - explicit discretization of HPIC
%%  e_hsmc      - explicit discretization of HSMC
%% si_hsmc      - semi-implicit explicit discretization of HSMC      
%%  e_hsmci     - explicit discretization of HSMC with Integral action
%%  e_fhpc      - explicit discretization of fixed-time HPC
%% si_fhpc      - semi-implicit discretization of Fixed-time HPC
%%  e_fhpic     - explicit discretization of fixed-time HPIC                     
%%
%%
%%  Homogeneous State Estimation:
%%
%%  ho_design     - Homogeneous Observer (HO) design                         
%%  fho_design    - fixed-time HO design                                
%%  lo2ho         - upgrading Linear Observer (LO) to HO                   
%%
%%
%% Discretization of Homogeneous Observer:
%%
%%  e_ho          - explicit discretization of HO                                
%%  e_fho         - explicit discretization of FHO                     
%%  si_ho         - semi-implicit discretization of HO                                
%%  si_fho        - semi-implicit discretization of FHO                     
%%
%%
%%  Block Controlability/Observability Forms:
%%
%%  block_con    - transformation to block controlability form
%%  bloc_obs     - transformation to block bservability form
%%  trans_con    - transformation to partial block controlability form
%%  trans_con    - transformation to partial block observability form
%%  output_form  - transformation to reduced order output control system  
%%
%%
%%  Examples:
%%
%%  To open example type 'edit <name_of_demo>' in Command Line of MATLAB
%%
%%  demo_hnorm     - example of computation of canonical (implicit) homoge-
%%                   neous norm by bisection method    
%%  demo_hnorm_r   - example of approximation of canonical homogeneous norm
%%                   by explicit homogenous norm  
%%  demo_hnorm_ANN - example of approximation of canonical homogeneous norm
%%                   by artificial neural network 
%%  demo_halgebra  - example of algebraic operations on homogeneous 
%%                   Euclidean space                
%%  demo_hsphere   - example of homogeneous balls in 2D         
%%  demo_hcurve    - example of homogeneous curve in 2D                      
%%  demo_hcone     - example of homogeneous cone in 2D                      
%%  demo_hpc       - example of HPC design and simulation
%%  demo_c_hpc     - example of constant-time HPC design and simulation
%%  demo_hpic      - example of HPIC design and simulation
%%  demo_hsmc      - example of HSMC design and simulation
%%  demo_hsmci     - example of HSMCI design and simulation
%%  demo_fhpc      - example of fixed-time HPC design and simulation
%%  demo_fhpic     - example of fixed-time HPIC design and simulation
%%  demo_lpc2hpc   - example of upgrading LPC to HPC/FHPC
%%  demo_lpic2hpic - example of upgrading LPIC to HPIC/FHPIC                 
%%  demo_ho        - example of HO design and simulation                     
%%  demo_fho       - example of fixed-time HO design and simulation  
%%  deme_hfo       - example of filtering HO design and simulation         (*)
%%  demo_lo2ho     - example of upgrading LO to HO                        
%%  demo_block_con - example of block controllability form
%%  demo_block_obs - example of block observability form
%%
%%
%% (*) - function is not realized in this version of HCS toolbox
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('HCS Toolbox for MATLAB (ver 0.2)');
disp('(please type ''help HCS_toolbox'' to print the list of functions)');
