import "/futlib/math"

default(f64)

let main [n] (res1 : [n]f64) (res2 : [n]f64) : bool =
  let mapped = map (\(r1, r2) -> f64.abs(r1 - r2) < 0.00001) (zip res1 res2)
  in reduce (&&) true mapped
