-- Trinomial option pricing - flat version
-- Run as e.g. $cat options.in yield.in | ./trinom-basic
-- Join datasets with yield by running $sh test.sh create
-- ==
-- compiled input @ ../data/fut/0_UNIFORM.in
-- output @ ../data/out/0_UNIFORM.out
-- compiled input @ ../data/fut/1_RAND.in
-- output @ ../data/out/1_RAND.out
-- compiled input @ ../data/fut/2_RANDCONSTHEIGHT.in
-- output @ ../data/out/2_RANDCONSTHEIGHT.out
-- compiled input @ ../data/fut/3_RANDCONSTWIDTH.in
-- output @ ../data/out/3_RANDCONSTWIDTH.out
-- compiled input @ ../data/fut/4_SKEWED.in
-- output @ ../data/out/4_SKEWED.out
-- compiled input @ ../data/fut/5_SKEWEDCONSTHEIGHT.in
-- output @ ../data/out/5_SKEWEDCONSTHEIGHT.out
-- compiled input @ ../data/fut/6_SKEWEDCONSTWIDTH.in
-- output @ ../data/out/6_SKEWEDCONSTWIDTH.out

------ More input files
-- compiled input @ ../data/100000/fut/rand_h_unif_w_100000.in
-- output @ ../data/100000/out/rand_h_unif_w_100000.out
-- compiled input @ ../data/100000/fut/rand_hw_100000.in
-- output @ ../data/100000/out/rand_hw_100000.out
-- compiled input @ ../data/100000/fut/rand_w_unif_h_100000.in
-- output @ ../data/100000/out/rand_w_unif_h_100000.out
-- compiled input @ ../data/100000/fut/skew_h_1_rand_w_100000.in
-- output @ ../data/100000/out/skew_h_1_rand_w_100000.out
-- compiled input @ ../data/100000/fut/skew_h_10_rand_w_100000.in
-- output @ ../data/100000/out/skew_h_10_rand_w_100000.out
-- compiled input @ ../data/100000/fut/skew_hw_1_100000.in
-- output @ ../data/100000/out/skew_hw_1_100000.out
-- compiled input @ ../data/100000/fut/skew_hw_10_100000.in
-- output @ ../data/100000/out/skew_hw_10_100000.out
-- compiled input @ ../data/100000/fut/skew_w_1_rand_h_100000.in
-- output @ ../data/100000/out/skew_w_1_rand_h_100000.out
-- compiled input @ ../data/100000/fut/skew_w_10_rand_h_100000.in
-- output @ ../data/100000/out/skew_w_10_rand_h_100000.out
-- compiled input @ ../data/100000/fut/unif_book_hw_100000.in
-- output @ ../data/100000/out/unif_book_hw_100000.out
-- compiled input @ ../data/100000/fut/unif_hw_100000.in
-- output @ ../data/100000/out/unif_hw_100000.out

import "/futlib/math"

import "/futlib/array"

------------------------------------------------
-- For using double-precision floats select
--    import "header64"
-- For unsing single-precision floats select
--    import "header32"
------------------------------------------------
import "header64"

-- import "header32"

-------------------------------------------------------------
--- Follows code independent of the instantiation of real ---
-------------------------------------------------------------
let zero = i2r 0
let one = i2r 1
let two = i2r 2
let half = one / two
let three = i2r 3
let six = i2r 6
let seven = i2r 7
let year = i2r 365
let hundred = i2r 100

type YieldCurveData = {
    P : real  -- Discount Factor function, P(t) gives the yield (interest rate) for a specific period of time t
  , t : i32   -- Time [days]
}

type TOptionData = {
    StrikePrice                 : real   -- X
  , Maturity                    : real   -- T, [years]
  , Length                      : real   -- t, [years]
  , ReversionRateParameter      : real   -- a, parameter in HW-1F model
  , VolatilityParameter         : real   -- sigma, volatility parameter in HW-1F model
  , TermUnit                    : u16
  , TermStepCount               : u16
  , OptionType                  : i8     -- option type, [0 - Put | 1 - Call]
}

let OptionType_CALL = i8.i32 0
let OptionType_PUT = i8.i32 1

-- | As `map5`@term, but with three more arrays.
let map8 'a 'b 'c 'd 'e 'f 'g 'h [n] 'x (i: a -> b -> c -> d -> e -> f -> g -> h -> x) (as: [n]a) (bs: [n]b) (cs: [n]c) (ds: [n]d) (es: [n]e) (fs: [n]f) (gs: [n]g) (hs: [n]h): *[n]x =
    map (\(a, b, c, d, e, f, g, h) -> i a b c d e f g h) (zip8 as bs cs ds es fs gs hs)

-- | As `unzip8`@term, but with one more array.
let unzip9 [n] 'a 'b 'c 'd 'e 'f 'g 'h 'i (xs: [n](a,b,c,d,e,f,g,h,i)): ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g, [n]h, [n]i) =
    let (as, bs, cs, ds, es, fs, gs, his) = unzip8 (map (\(a,b,c,d,e,f,g,h,i) -> (a,b,c,d,e,f,g,(h,i))) xs)
    let (hs, is) = unzip his
    in (as, bs, cs, ds, es, fs, gs, hs, is)

-- | As `unzip9`@term, but with one more array.
let unzip10 [n] 'a 'b 'c 'd 'e 'f 'g 'h 'i 'j (xs: [n](a,b,c,d,e,f,g,h,i,j)): ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g, [n]h, [n]i, [n]j) =
    let (as, bs, cs, ds, es, fs, gs, hs, ijs) = unzip9 (map (\(a,b,c,d,e,f,g,h,i,j) -> (a,b,c,d,e,f,g,h,(i,j))) xs)
    let (is, js) = unzip ijs
    in (as, bs, cs, ds, es, fs, gs, hs, is, js)

-- | As `unzip10`@term, but with one more array.
let unzip11 [n] 'a 'b 'c 'd 'e 'f 'g 'h 'i 'j 'k (xs: [n](a,b,c,d,e,f,g,h,i,j,k)): ([n]a, [n]b, [n]c, [n]d, [n]e, [n]f, [n]g, [n]h, [n]i, [n]j, [n]k) =
    let (as, bs, cs, ds, es, fs, gs, hs, is, jks) = unzip10 (map (\(a,b,c,d,e,f,g,h,i,j,k) -> (a,b,c,d,e,f,g,h,i,(j,k))) xs)
    let (js, ks) = unzip jks
    in (as, bs, cs, ds, es, fs, gs, hs, is, js, ks)

let sgmScanPlus [n] (flags: [n]i32) (data: [n]i32) : [n]i32 =
    (unzip (scan (\(x_flag,x) (y_flag,y) ->
                        let flag = x_flag | y_flag in
                        if y_flag != 0
                        then (flag, y)
                        else (flag, x + y))
                    (0, 0)
                    (zip flags data))).2

let sgmScanPlusReal [n] (flags: [n]i32) (data: [n]real) : [n]real =
    (unzip (scan (\(x_flag,x) (y_flag,y) ->
                        let flag = x_flag | y_flag in
                        if y_flag != 0
                        then (flag, y)
                        else (flag, x + y))
                    (0, 0.0)
                    (zip flags data))).2

----------------------------------
--- Yield curve interpolation ---
----------------------------------
let getYieldAtYear [ycCount]
                (t : real)
                (termUnit : real)
                (h_YieldCurve : [ycCount]YieldCurveData) : real =
    let tDays = r2i (r_round (t * termUnit) )
    let (p1, t1, p2, t2, done) = loop res = (zero, -1, zero, -1, false) for yield in h_YieldCurve do
        let pi = yield.P
        let ti = yield.t
        let (_, _, p2, t2, done) = res
        in if done then res
            else if (ti >= tDays) then (p2, t2, pi, ti, true)
            else (p2, t2, pi, ti, false)

    in if (t1 == -1 || done == false) then p2   -- result is the first or last yield curve price
        else                                     -- linearly interpolate between two consecutive items
            let coefficient = (i2r (tDays - t1)) / (i2r (t2 - t1))
            in p1 + coefficient * (p2 - p1)


-----------------------------
--- Probability Equations ---
-----------------------------
-- Exhibit 1A (-jmax < j < jmax)
let PU_A (j:i32) (M:real) : real = one/six + ((i2r(j*j))*M*M + (i2r j)*M)*half
let PM_A (j:i32) (M:real) : real = two/three - ((i2r(j*j))*M*M )
let PD_A (j:i32) (M:real) : real = one/six + ((i2r(j*j))*M*M - (i2r j)*M)*half

-- Exhibit 1B (j == -jmax)
let PU_B (j:i32) (M:real) : real = one/six + ((i2r(j*j))*M*M - (i2r j)*M)*half
let PM_B (j:i32) (M:real) : real = -one/three - ((i2r(j*j))*M*M) + two*(i2r j)*M
let PD_B (j:i32) (M:real) : real = seven/six + ((i2r(j*j))*M*M - three*(i2r j)*M)*half

-- Exhibit 1C (j == jmax)
let PU_C (j:i32) (M:real) : real = seven/six + ((i2r(j*j))*M*M + three*(i2r j)*M)*half
let PM_C (j:i32) (M:real) : real = -one/three - ((i2r(j*j))*M*M) - two*(i2r j)*M
let PD_C (j:i32) (M:real) : real = one/six + ((i2r (j*j))*M*M + (i2r j)*M)*half

----------------------------------
--- Wrapper for all equations  ---
----------------------------------
let computeJValue (j : i32) (jmax : i32) (M : real) (expout : i32) : real =
    if (j == -jmax)
    then 
        if (expout == 1) then PU_B j M      -- up
        else if (expout == 2) then PM_B j M -- mid
        else PD_B j M                       -- down
    else if (j == jmax)
    then
        if (expout == 1) then PU_C j M      -- up
        else if (expout == 2) then PM_C j M -- mid
        else PD_C j M                       -- down
    else
        if (expout == 1) then PU_A j M      -- up
        else if (expout == 2) then PM_A j M -- mid
        else PD_A j M                       -- down

----------------------------------
--- forward propagation helper ---
----------------------------------
let fwdHelper (M : real) (dr : real) (dt : real) (alpha : real) (Qs : []real) 
              (i : i32) (jhigh : i32) (jmax : i32) (j : i32) (jind : i32) : real =  

    let expp1 = if (j == jhigh) then zero else Qs[jind + 1] * r_exp (-(alpha + (i2r (j + 1)) * dr) * dt)
    let expm = Qs[jind] * r_exp (-(alpha + (i2r j) * dr) * dt)
    let expm1 = if (j == -jhigh) then zero else Qs[jind - 1] * r_exp(-(alpha + (i2r (j - 1)) * dr) * dt)

    in
    if (i == 1)
    then
        if (j == -jhigh) then (computeJValue (j + 1) jmax M 3) * expp1
        else if (j == jhigh) then (computeJValue (j - 1) jmax M 1) * expm1
        else (computeJValue j jmax M 2) * expm
    else if (i <= jmax)
    then
        if (j == -jhigh) then (computeJValue (j + 1) jmax M 3) * expp1
        else if (j == -jhigh + 1)
            then (computeJValue j jmax M 2) * expm +
                (computeJValue (j + 1) jmax M 3) * expp1
        else if (j == jhigh) then (computeJValue (j - 1) jmax M 1) * expm1
        else if (j == jhigh - 1)
            then (computeJValue (j - 1) jmax M 1) * expm1 +
                (computeJValue j jmax M 2) * expm
        else
            (computeJValue (j - 1) jmax M 1) * expm1 +
            (computeJValue j jmax M 2) * expm +
            (computeJValue (j + 1) jmax M 3) * expp1
    else
        if (j == -jhigh)
            then (computeJValue j jmax M 3) * expm +
                (computeJValue (j + 1) jmax M 3) * expp1
        else if (j == -jhigh + 1)
            then (computeJValue (j - 1) jmax M 2) * expm1 +
                (computeJValue j jmax M 2) * expm +
                (computeJValue (j + 1) jmax M 3) * expp1
                    
        else if (j == jhigh)
            then (computeJValue (j - 1) jmax M 1) * expm1 +
                (computeJValue j jmax M 1) * expm
        else if (j == jhigh - 1)
            then (computeJValue (j - 1) jmax M 1) * expm1 +
                (computeJValue j jmax M 2) * expm +
                (computeJValue (j + 1) jmax M 2) * expp1
                    
        else
            (if (j == -jhigh + 2) then (computeJValue (j - 2) jmax M 1) * Qs[jind - 2] * r_exp (-(alpha + (i2r (j - 2)) * dr) * dt) else zero) +
            (computeJValue (j - 1) jmax M 1) * expm1 +
            (computeJValue j jmax M 2) * expm +
            (computeJValue (j + 1) jmax M 3) * expp1 +
            (if (j == jhigh - 2) then (computeJValue (j + 2) jmax M 3) * Qs[jind + 2] * r_exp (-(alpha + (i2r (j + 2)) * dr) * dt) else zero)


-----------------------------------
--- backward propagation helper ---
-----------------------------------
let bkwdHelper (X : real) (op : i8) (M : real) (dr : real) (dt : real) (alpha : real) 
               (call : []real) (jmax : i32) (j : i32) (jind : i32) (isMaturity : bool) : real = 
    let callExp = r_exp(-(alpha + (i2r j) * dr) * dt)
    let res =
        if (j == jmax) then
            -- Top edge branching
            ((computeJValue j jmax M 1) * call[jind] +
            (computeJValue j jmax M 2) * call[jind - 1] +
            (computeJValue j jmax M 3) * call[jind - 2]) *
                callExp
        else if (j == -jmax) then
            -- Bottom edge branching
            ((computeJValue j jmax M 1) * call[jind + 2] +
            (computeJValue j jmax M 2) * call[jind + 1] +
            (computeJValue j jmax M 3) * call[jind]) *
                callExp
        else
            -- Standard branching
            ((computeJValue j jmax M 1) * call[jind + 1] +
            (computeJValue j jmax M 2) * call[jind] +
            (computeJValue j jmax M 3) * call[jind - 1]) *
                callExp

    let value = if isMaturity
        then
            if (op == OptionType_PUT)
            then r_max(X - res, zero)
            else r_max(res - X, zero)
        else res
    in r_convert_inf value



let trinomialFlat [ycCount] [numAllOptions]
    (h_YieldCurve : [ycCount]YieldCurveData)
    (options : [numAllOptions]TOptionData) 
  : [numAllOptions]real = unsafe
    -- header: get the options
    let (Xs, ops, lens, tus, ns, dts, drs, Ms, jmaxs, widths, heights) = unzip11 (
        map (\{StrikePrice, Maturity, Length, ReversionRateParameter, VolatilityParameter, TermUnit, TermStepCount, OptionType} ->
            let termUnit = ui2r TermUnit
            let termUnitsInYearCount = r2i (r_ceil(year / termUnit))
            let dt = (i2r termUnitsInYearCount) / (ui2r TermStepCount)
            let n = r2i ((ui2r TermStepCount) * (i2r termUnitsInYearCount) * Maturity)
            let a = ReversionRateParameter
            let sigma = VolatilityParameter
            let V  = sigma * sigma * ( one - (r_exp (zero - two * a * dt)) ) / (two * a)
            let dr = r_sqrt( (one + two) * V )
            let M  = (r_exp (zero - a * dt)) - one
            let jmax = r2i (- 0.184 / M) + 1
            let width = 2 * jmax + 1
            in  (StrikePrice, OptionType, Length, termUnit, n, dt, dr, M, jmax, width, n)
    ) options)

    -- make the flag array (probably useful for segmented scan/reduce operations)
    let scanned_lens = scan (+) 0 widths
    let len_valinds = map2 (\m i -> 
        if i == 0 then (0, m) 
        else (scanned_lens[i-1], m)
    ) widths (iota numAllOptions)

    let w = last scanned_lens
    let (len_inds, len_vals) = unzip len_valinds
    let flags = scatter (replicate w 0) len_inds len_vals

    -- make the flat segment-index array
    let sgm_inds = scatter (replicate w 0) len_inds (iota numAllOptions)
    -- sgm_inds can be used to access the index of the current option
    let sgm_inds = sgmScanPlus flags sgm_inds
    
    -- make the segmented (iota (2*m+1)) across all widths
    let q_lens = map (\x->x-1) (sgmScanPlus flags (replicate w 1))

    let Qs = map2 (\i k -> if i == jmaxs[sgm_inds[k]] then one else zero) q_lens (iota w)

    let max_height = reduce (\x y -> i32.max x y) 0 heights
    let seq_len = max_height + 1
    let total_len = numAllOptions * seq_len
    let alphas = replicate total_len zero
    let alphas = scatter alphas
        ( map (\i -> i*seq_len) (iota numAllOptions) )
        ( map (\i -> getYieldAtYear dts[i] tus[i] h_YieldCurve) (iota numAllOptions) )

    -------------------------
    -- FORWARD PROPAGATION --
    -------------------------
    let (_,alphas) = loop (Qs: *[w]real, alphas: *[total_len]real) for i < max_height do
        let jhighs = map (\jmax -> i32.min (i+1) jmax) jmaxs
        let QsCopy = copy Qs

        let Qs = map2 (\jind wind ->
            let opt_ind = sgm_inds[wind]
            let j = jind - jmaxs[opt_ind] in
                if (j < (-jhighs[opt_ind])) || (j > jhighs[opt_ind]) then zero
                else fwdHelper Ms[opt_ind] drs[opt_ind] dts[opt_ind] alphas[opt_ind*seq_len+i] QsCopy (i + 1) jhighs[opt_ind] jmaxs[opt_ind] j wind
        ) (q_lens) (iota w)

        -- compute tmps
        let tmps = map2 (\jind wind ->
            let opt_ind = sgm_inds[wind]
            let j = jind - jmaxs[opt_ind] in
                if (j < (-jhighs[opt_ind])) || (j > jhighs[opt_ind]) then zero
                else Qs[wind] * r_exp (-(i2r j) * drs[opt_ind] * dts[opt_ind])
        ) (q_lens) (iota w)
                
        -- sum up Qs
        let alpha_vals = sgmScanPlusReal flags tmps 
        let alpha_vals = map (\opt_ind -> 
            if (i >= (ns[opt_ind])) then zero
            else 
                let t = (i2r (i+2)) * dts[opt_ind]
                let R = getYieldAtYear t tus[opt_ind] h_YieldCurve
                let P = r_exp(-R * t)
                in (r_log (alpha_vals[scanned_lens[opt_ind]-1] / P) ) / dts[opt_ind]
        ) (iota numAllOptions)

        let alpha_indvals = map (\opt_ind->opt_ind*seq_len + i + 1) (iota numAllOptions)
        let alphas = scatter alphas alpha_indvals alpha_vals
    in  (Qs,alphas)

    --------------------------
    -- BACKWARD PROPAGATION --
    --------------------------
    let call = map (\wind -> if (wind < scanned_lens[numAllOptions - 1]) then hundred else zero) (iota w)

    let call = loop (call: *[w]real) for ii < max_height do
        let i = max_height - 1 - ii 
        let jhighs = map (\jmax -> i32.min i jmax) jmaxs
        
        -- Copy array values to avoid overwriting during update
        let callCopy = copy call

        let call =
            map2 (\jind wind -> 
                let opt_ind = sgm_inds[wind] in
                if (i >= ns[opt_ind]) then callCopy[wind]
                else
                    let j = jind - jmaxs[opt_ind] in
                    if (j < (-jhighs[opt_ind])) || (j > jhighs[opt_ind]) then zero
                    else bkwdHelper Xs[opt_ind] ops[opt_ind] Ms[opt_ind] drs[opt_ind] dts[opt_ind] alphas[opt_ind*seq_len+i] callCopy jmaxs[opt_ind] j wind (i == (r2i (lens[opt_ind] / dts[opt_ind])))
            ) (q_lens) (iota w)
    in call

    -- reshape result
    let (inds, vals) = unzip (
        map (\sgm_ind ->
                let begind = if sgm_ind == 0 then 0 else scanned_lens[sgm_ind-1]
                let m_ind  = begind + jmaxs[sgm_ind]
                in  (sgm_ind, call[m_ind])
            ) (iota numAllOptions)
    )
    let res = scatter (replicate numAllOptions 0.0) inds vals
in res

-----------------
-- Entry point --
-----------------
let main [q] [y] (strikes           : [q]real)
                 (maturities        : [q]real) 
                 (lenghts           : [q]real)
                 (termunits         : [q]u16) 
                 (termstepcounts    : [q]u16) 
                 (rrps              : [q]real) 
                 (vols              : [q]real) 
                 (types             : [q]i8)
                 (yield_p           : [y]real)
                 (yield_t           : [y]i32) : 
                 []real =

  let yield = map2 (\p d -> {P = p, t = d}) yield_p yield_t

  let options = map8 (\s m l u c r v t -> 
    {StrikePrice=s, Maturity=m, Length=l, TermUnit=u, TermStepCount=c, 
    ReversionRateParameter=r, VolatilityParameter=v, OptionType=t }) 
      strikes maturities lenghts termunits termstepcounts rrps vols types
      
  in trinomialFlat yield options