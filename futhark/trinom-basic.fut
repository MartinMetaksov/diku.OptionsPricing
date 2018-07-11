-- Trinomial option pricing - basic version
-- Run as e.g. $cat options.in yield.in | ./trinom-basic
-- Join datasets with yield by running $sh test.sh create
-- ==
-- compiled input @ ../data/fut/options-60000.in
-- output @ ../data/out/options-60000.out
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

-----------------------
--- Data Structures ---
-----------------------

type YieldCurveData = {
    P : real    -- Discount Factor function, P(t) gives the yield (interest rate) for a specific period of time t
  , t : i32     -- Time [days]
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

----------------------------------
--- yield curve interpolation ---
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

-- Wrapper for all the equations
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



let trinomialOptionsHW1FCPU_single [ycCount]
                                   (h_YieldCurve : [ycCount]YieldCurveData)
                                   (optionData : TOptionData) : real = unsafe
    let X  = optionData.StrikePrice
    let T  = optionData.Maturity
    let len  = optionData.Length
    let op = optionData.OptionType
    let termUnit = ui2r optionData.TermUnit
    let termUnitsInYearCount = r2i (r_ceil(year / termUnit))
    let dt = (i2r termUnitsInYearCount) / (ui2r optionData.TermStepCount)
    let n = r2i ((ui2r optionData.TermStepCount) * (i2r termUnitsInYearCount) * T)
    let a = optionData.ReversionRateParameter
    let sigma = optionData.VolatilityParameter
    let V  = sigma*sigma*( one - (r_exp (zero - two*a*dt)) ) / (two*a)
    let dr = r_sqrt( (one+two)*V )
    let M  = (r_exp (zero - a*dt)) - one
    let jmax = r2i (- 0.184 / M) + 1

    ------------------------
    -- Compute Q values
    -------------------------
    -- Define initial tree values
    let Qlen = 2 * jmax + 1
    let Q = replicate Qlen zero
    let Q[jmax] = one

    let alphas = replicate (n + 1) zero
    let alphas[0] = getYieldAtYear dt termUnit h_YieldCurve

    -- time stepping
    let (_,alphas) =
    loop (Q: *[Qlen]real, alphas: *[]real) for i < n do
        let jhigh = i32.min (i+1) jmax
        let alphai = alphas[i]
        -- Reset
        let QCopy = copy Q

        ----------------------------
        -- forward iteration step --
        ----------------------------
        let Q = -- 1. result of size independent of i (hoistable)
                map (\jind -> 
                        let j = jind - jmax in
                        if (j < (-jhigh)) || (j > jhigh) then zero
                        else fwdHelper M dr dt alphai QCopy (i + 1) jhigh jmax j jind
                    ) (iota Qlen)
            
        -- sum up Qs
        let tmps = map (\(q, jind) -> 
                                let j = jind - jmax in
                                if (j < (-jhigh)) || (j > jhigh) then zero
                                else q * r_exp (-(i2r j) * dr * dt)
                    ) (zip Q (iota Qlen))
        let alpha_val = reduce (+) zero tmps

        -- determine new alpha
        let t = (i2r (i+2)) * dt
        let R = getYieldAtYear t termUnit h_YieldCurve
        let P = r_exp(-R * t)
        let alpha_val = r_log (alpha_val / P)
        let alphas[i + 1] = alpha_val / dt

        in  (Q,alphas)

    ------------------------------------------------------------
    -- Compute values at expiration date:
    -- call option value at period end is V(T) = S(T) - X
    -- if S(T) is greater than X, or zero otherwise.
    -- The computation is similar for put options.
    ------------------------------------------------------------
    let Call = replicate Qlen hundred
    
    -- back propagation
    let Call =
    loop (Call: *[Qlen]real) for ii < n do
        let i = n - 1 - ii
        let jhigh = i32.min i jmax
        let alphai = alphas[i]
        let isMaturity = i == (r2i (len / dt))
        
        -- Copy array values to avoid overwriting during update
        let CallCopy = copy Call

        -----------------------------
        -- backward iteration step --
        -----------------------------
        let Call = -- 1. result of size independent of i (hoistable)
                map (\jind -> 
                        let j = jind - jmax in
                        if (j < (-jhigh)) || (j > jhigh) then zero
                        else bkwdHelper X op M dr dt alphai CallCopy jmax j jind isMaturity
                    ) (iota Qlen)

        in  Call

    in Call[jmax]

-- | As `map5`@term, but with three more arrays.
let map8 'a 'b 'c 'd 'e 'f 'g 'h [n] 'x (i: a -> b -> c -> d -> e -> f -> g -> h -> x) (as: [n]a) (bs: [n]b) (cs: [n]c) (ds: [n]d) (es: [n]e) (fs: [n]f) (gs: [n]g) (hs: [n]h): *[n]x =
        map (\(a, b, c, d, e, f, g, h) -> i a b c d e f g h) (zip8 as bs cs ds es fs gs hs)

-----------------
-- Entry point
-----------------
let main [q] [y] (strikes           : [q]real)
                 (maturities        : [q]real) 
                 (lenghts           : [q]real)
                 (termunits         : [q]u16 ) 
                 (termstepcounts    : [q]u16 ) 
                 (rrps              : [q]real) 
                 (vols              : [q]real) 
                 (types             : [q]i8)
                 (yield_p           : [y]real)
                 (yield_t           : [y]i32) : [q]real =
        
  let yield = map2 (\p d -> {P = p, t = d}) yield_p yield_t

  let options = map8 (\s m l u c r v t -> {StrikePrice=s, Maturity=m, Length=l, TermUnit=u, TermStepCount=c,
                                        ReversionRateParameter=r, VolatilityParameter=v, OptionType=t }
                ) strikes maturities lenghts termunits termstepcounts rrps vols types

  in map (trinomialOptionsHW1FCPU_single yield) options
