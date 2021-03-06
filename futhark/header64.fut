import "/futlib/math"

type real = f64

let i2r     (i : i32 ) : real = r64 i
let ui2r    (i : u16 ) : real = f64.u16 i
let r2i     (a : real) : i32  = t64 a
let r_exp   (a : real) : real = f64.exp  a
let r_sqrt  (a : real) : real = f64.sqrt a
let r_abs   (a : real) : real = f64.abs  a
let r_log   (a : real) : real = f64.log  a
let r_ceil  (a : real) : real = f64.ceil a
let r_round (a : real) : real = f64.round a
let r_isinf (a : real) : bool = f64.isinf a
let r_max   (a : real, b : real) : real = f64.max a b
let r_convert_inf (a : real) : real =
    if (a == f64.inf) then 1.79769e+308
    else if (a == -f64.inf) then 2.22507e-308
    else a
