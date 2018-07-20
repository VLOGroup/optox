#include <mex.h>
#include <iu/iumatlab.h>

template <typename T, unsigned int N>
std::unique_ptr<iu::LinearDeviceMemory<T, N>> getLinearDeviceFromMex(const mxArray_tag &mex_array,
                                                                     bool flip_memory_layout = true)
{
    iu::LinearHostMemory<T, N> host_mem(mex_array, flip_memory_layout);
    std::unique_ptr<iu::LinearDeviceMemory<T, N>> p(new iu::LinearDeviceMemory<T, N>(host_mem.size()));
    iu::copy(&host_mem, p.get());

    // do not return a copy but rather move its value
    return move(p);
}

template <typename T, unsigned int N>
void linearDeviceToMex(const iu::LinearDeviceMemory<T, N> &device_mem,
                       mxArray_tag **mex_array, bool flip_memory_layout = true)
{
    iu::LinearHostMemory<T, N> host_mem(device_mem.size());
    iu::copy(&device_mem, &host_mem);

    // push the output back to matlab
    iu::matlab::convertCToMatlab(host_mem, mex_array, flip_memory_layout);
}
