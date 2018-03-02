import "/futlib/math"

type real = f32

let i2r    (i : i32 ) : real = r32 i
let r2i    (a : real ) : i32 = t32 a
let r_exp  (a : real) : real = f32.exp  a
let r_sqrt (a : real) : real = f32.sqrt a
let r_abs  (a : real) : real = f32.abs  a
let r_log  (a : real) : real = f32.log  a
let r_max  (a : real, b : real) : real = f32.max a b

--let round (x : real) : i32 =
--    let tmp = half + (r_abs x)
--    let sgn = if x >= zero then one else (-one)
--    in  t32 (sgn*tmp)

