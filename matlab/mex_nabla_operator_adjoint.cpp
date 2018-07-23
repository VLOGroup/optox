///@file mex_nabla_operator.cpp
///@brief python wrappers for the nabla operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 21.07.2018

#include <mex.h>

#include <iu/iucore.h>
#include <iu/iumatlab.h>

#include "mex_utils.h"

#include "operators/nabla_operator.h"

typedef float T;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Checking number of arguments
    if (nrhs != 1)
        mexErrMsgIdAndTxt(
            "MATLAB:nablaoperatoradjoint:invalidNumInputs",
            "1 input required");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("MATLAB:nablaoperatoradjoint:maxlhs",
                          "Too many output arguments.");

    // parse input volume
    std::unique_ptr<iu::LinearDeviceMemory<T, 3>> iu_in = getLinearDeviceFromMex<T, 3>(*(prhs[0]), false);

    // output memory
    iu::Size<2> out_size({iu_in->size()[0], iu_in->size()[1]});
    iu::LinearDeviceMemory<T, 2> iu_out(out_size);

    // setup the operator and apply it
    optox::NablaOperator<T, 2> op;

    op.adjoint({&iu_out}, {iu_in.get()});

    // forward the result to matlab
    linearDeviceToMex(iu_out, &plhs[0], false);
}
