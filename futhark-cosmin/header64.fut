import "/futlib/math"

type real = f64

let i2r    (i : i32 ) : real = r64 i
let r2i    (a : real ) : i32 = t64 a
let r_exp  (a : real) : real = f64.exp  a
let r_sqrt (a : real) : real = f64.sqrt a
let r_abs  (a : real) : real = f64.abs  a
let r_log  (a : real) : real = f64.log  a
let r_max  (a : real, b : real) : real = f64.max a b
