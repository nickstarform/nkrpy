/* Generated by Cython 0.29.14 */

#ifndef __PYX_HAVE__nkrpy__astro__orbit
#define __PYX_HAVE__nkrpy__astro__orbit

#include "Python.h"

#ifndef __PYX_HAVE_API__nkrpy__astro__orbit

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

#ifndef DL_IMPORT
  #define DL_IMPORT(_T) _T
#endif

__PYX_EXTERN_C double eps;

#endif /* !__PYX_HAVE_API__nkrpy__astro__orbit */

/* WARNING: the interface of the module init function changed in CPython 3.5. */
/* It now returns a PyModuleDef instance instead of a PyModule instance. */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initorbit(void);
#else
PyMODINIT_FUNC PyInit_orbit(void);
#endif

#endif /* !__PYX_HAVE__nkrpy__astro__orbit */