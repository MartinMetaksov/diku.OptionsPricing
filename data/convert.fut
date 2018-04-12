import "/futlib/math"

default(f64)

let main [q] (strikes    : [q]f64)
             (maturities : [q]f64) 
             (numofterms : [q]i32) 
             (rrps       : [q]f64) 
             (vols       : [q]f64) : ([q]f64, [q]f64, [q]f64, [q]i32, [q]i32, [q]f64, [q]f64, [q]i32) =
              let strikes100 = map (*100.0) strikes
              let lengths = replicate q 3.0
              let termunits = replicate q 365
              let termsteps = map (/12) numofterms
              let p = replicate q 0
              in (strikes100, maturities, lengths, termunits, termsteps, rrps, vols, p)
