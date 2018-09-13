# pragma once

#define TF_CALL_ICG_REAL_NUMBER_TYPES(m)                    \
   TF_CALL_float(m) TF_CALL_double(m)
#define TF_CALL_ICG_COMPLEX_NUMBER_TYPES(m)                 \
   TF_CALL_complex64(m) TF_CALL_complex128(m)
#define TF_CALL_ICG_NUMBER_TYPES(m)                         \
   TF_CALL_ICG_REAL_NUMBER_TYPES(m) TF_CALL_ICG_COMPLEX_NUMBER_TYPES(m)
   