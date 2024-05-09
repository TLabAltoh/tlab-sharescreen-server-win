#pragma once

// If you want to use this code as a .dll file.
// You should not write "cuda_runtime.h" and "device_launch_parameters.h"
// in hedder file. Must be written in the .cu file.

#include <stdio.h>

// This is necessary. when use cuda.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>