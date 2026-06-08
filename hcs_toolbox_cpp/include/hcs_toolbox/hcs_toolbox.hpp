#pragma once
/**
 * @file hcs_toolbox.hpp
 * @brief Umbrella header for the HCS (Homogeneous Control Systems) Toolbox for C++
 *
 * Port of Andrey Polyakov's MATLAB HCS Toolbox ver 0.2.
 * All functions are in namespace hcs_toolbox.
 *
 * Dependencies: Eigen 3.4+ (including unsupported MatrixFunctions for expm)
 */

#include "hnorm.hpp"
#include "hproj.hpp"
#include "hphi.hpp"
#include "hadd.hpp"
#include "hdot.hpp"
#include "hinner.hpp"
#include "block_con.hpp"
#include "trans_con.hpp"
#include "ZOH.hpp"
#include "lpc2hpc.hpp"
#include "lpic2hpic.hpp"
#include "lqr.hpp"
#include "e_hpc.hpp"
#include "e_hpic.hpp"
#include "e_fhpc.hpp"
#include "e_fhpic.hpp"
