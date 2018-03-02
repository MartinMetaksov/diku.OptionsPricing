-- Trinomial option pricing
-- ==
-- compiled input @ data/small.in
-- output @ data/small.out
-- compiled input @ data/options-60000.in
-- output @ data/options-60000.out


import "/futlib/math"

import "/futlib/array"

------------------------------------------------
-- For using double-precision floats select
--    default(f64)
--    import "header64"
-- For unsing single-precision floats select
--    default(f32)
--    import "header32"
------------------------------------------------
--default(f64)
--import "header64"

default(f32)
import "header32"

------------------------------------------------------
--- Pushing the compiler in a direction or another ---
------------------------------------------------------
let FORCE_PER_OPTION_THREAD  = true
let WITH_ONLY_STRIKE_VARIANT = true

-------------------------------------------------------------
--- Follows code independent of the instantiation of real ---
-------------------------------------------------------------
let zero=i2r 0
let one= i2r 1
let two= i2r 2
let half= one / two
let three= i2r 3
let six= i2r 6
let seven= i2r 7
let year = i2r 365

-----------------------
--- Data Structures ---
-----------------------

type YieldCurveData = {
    P : real -- Discount Factor function, P(t) gives the yield (interest rate) for a specific period of time t
  , t : real -- Time [days]
}

type TOptionData = {
    StrikePrice   : real  -- X
  , Maturity      : real  -- T, [years]
  , NumberOfTerms : i32   -- n
  , ReversionRateParameter : real -- a, parameter in HW-1F model
  , VolatilityParameter    : real -- sigma, volatility parameter in HW-1F model
}

-----------------------------
--- Probability Equations ---
-----------------------------

-- Exhibit 1A (-jmax < j < jmax)
let PU_A (j:i32, M:real) : real = one/six + ((i2r(j*j))*M*M + (i2r j)*M)*half
let PM_A (j:i32, M:real) : real = two/three - ((i2r(j*j))*M*M )
let PD_A (j:i32, M:real) : real = one/six + ((i2r(j*j))*M*M - (i2r j)*M)*half

-- Exhibit 1B (j == -jmax)
let PU_B (j:i32, M:real) : real = one/six + ((i2r(j*j))*M*M - (i2r j)*M)*half
let PM_B (j:i32, M:real) : real = -one/three - ((i2r(j*j))*M*M) + two*(i2r j)*M
let PD_B (j:i32, M:real) : real = seven/six + ((i2r(j*j))*M*M - three*(i2r j)*M)*half

-- Exhibit 1C (j == jmax)
let PU_C (j:i32, M:real) : real = seven/six + ((i2r(j*j))*M*M + three*(i2r j)*M)*half
let PM_C (j:i32, M:real) : real = -one/three - ((i2r(j*j))*M*M) - two*(i2r j)*M
let PD_C (j:i32, M:real) : real = one/six + ((i2r (j*j))*M*M + (i2r j)*M)*half

----------------------------------
--- forward propagation helper ---
----------------------------------

let fwdHelper (M : real) (dr : real) (dt : real) (alphai : real) (QCopy : []real) 
              (m : i32) (i : i32) (imax : i32) (jmax : i32) (j : i32) : real = 
    let eRdt_u1 = r_exp(-((i2r (j+1))*dr + alphai)*dt)
    let eRdt    = r_exp(-((i2r j    )*dr + alphai)*dt)
    let eRdt_d1 = r_exp(-((i2r (j-1))*dr + alphai)*dt)
    in  if i < jmax
        then let pu = PU_A(j-1, M)
             let pm = PM_A(j  , M)
             let pd = PD_A(j+1, M)
             in  if (i == 0 && j == 0 ) then pm*QCopy[j+m]*eRdt
                 else if (j == -imax+1) then pd*QCopy[j+m+1]*eRdt_u1 + pm*QCopy[j+m]*eRdt
                 else if (j == imax-1 ) then pm*QCopy[j+m]*eRdt + pu*QCopy[j+m-1]*eRdt_d1
                 else if (j == 0-imax ) then pd*QCopy[j+m+1]*eRdt_u1
                 else if (j == imax   ) then pu*QCopy[j+m-1]*eRdt_d1
                 else pd*QCopy[j+m+1]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j+m-1]*eRdt_d1
        else if (j == jmax) 
                then let pm = PU_C(j  , M)
                     let pu = PU_A(j-1, M)
                     in  pm*QCopy[j+m]*eRdt +  pu*QCopy[j-1+m] * eRdt_d1
             else if(j == jmax - 1)
                then let pd = PM_C(j+1, M)
                     let pm = PM_A(j  , M)
                     let pu = PU_A(j-1, M)
                     in  pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j-1+m]*eRdt_d1
             else if (j == jmax - 2)
                then let eRdt_u2 = r_exp(-( (i2r(j+2))*dr + alphai ) * dt)
                     let pd_c = PD_C(j + 2, M)
                     let pd   = PD_A(j + 1, M)
                     let pm   = PM_A(j, M)
                     let pu   = PU_A(j - 1, M)
                     in  pd_c*QCopy[j+2+m]*eRdt_u2 + pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j-1+m]*eRdt_d1
             else if (j == -jmax + 2)
                then let eRdt_d2 = r_exp(-((i2r (j-2))*dr + alphai)*dt)
                     let pd   = PD_A(j + 1, M)
                     let pm   = PM_A(j, M)
                     let pu   = PU_A(j - 1, M)
                     let pu_b = PU_B(j - 2, M)
                     in  pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j-1+m]*eRdt_d1 + pu_b*QCopy[j-2+m]*eRdt_d2
             else if (j == -jmax + 1)
                then let pd = PD_A(j + 1, M)
                     let pm = PM_A(j, M)
                     let pu = PM_B(j - 1, M)
                     in  pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j-1+m]*eRdt_d1
             else if (j == -jmax)
                then let pd = PD_A(j + 1, M)
                     let pm = PD_B(j, M)
                     in  pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt                                            
             else    
                     let pd = PD_A(j + 1, M)
                     let pm = PM_A(j, M)
                     let pu = PU_A(j - 1, M)
                     in  pd*QCopy[j+1+m]*eRdt_u1 + pm*QCopy[j+m]*eRdt + pu*QCopy[j-1+m]*eRdt_d1

-----------------------------------
--- backward propagation helper ---
-----------------------------------
let bkwdHelper (X : real) (M : real) (dr : real) (dt : real) (alphai : real) 
               (CallCopy : []real) (m : i32) (i : i32) (jmax : i32) (j : i32) : real = 
                let eRdt = r_exp(-((i2r j)*dr + alphai)*dt)
                let res =
                  if (i < jmax)
                  then -- central node
                     let pu = PU_A(j, M)
                     let pm = PM_A(j, M)
                     let pd = PD_A(j, M)
                     in  (pu*CallCopy[j+m+1] + pm*CallCopy[j+m] + pd*CallCopy[j+m-1]) * eRdt
                  else if (j == jmax)
                        then -- top node
                             let pu = PU_C(j, M)
                             let pm = PM_C(j, M)
                             let pd = PD_C(j, M)
                             in  (pu*CallCopy[j+m] + pm*CallCopy[j+m-1] + pd*CallCopy[j+m-2]) * eRdt
                       else if (j == -jmax)
                        then -- bottom node
                             let pu = PU_B(j, M)
                             let pm = PM_B(j, M)
                             let pd = PD_B(j, M)
                             in  (pu*CallCopy[j+m+2] + pm*CallCopy[j+m+1] + pd*CallCopy[j+m]) * eRdt
                       else    -- central node
                             let pu = PU_A(j, M)
                             let pm = PM_A(j, M)
                             let pd = PD_A(j, M)
                             in  (pu*CallCopy[j+m+1] + pm*CallCopy[j+m] + pd*CallCopy[j+m-1]) * eRdt

                -- TODO (WMP) This should be parametrized; length of contract, here 3 years
                in if (i == (r2i (three / dt))) 
                    then r_max(X - res, zero)
                    else res



let trinomialOptionsHW1FCPU_single [ycCount]
                                   (h_YieldCurve : [ycCount]YieldCurveData)
                                   (optionData : TOptionData) : real = unsafe
  let X  = optionData.StrikePrice
  let T  = optionData.Maturity
  let n  = optionData.NumberOfTerms
  let dt = T / (i2r n)
  let a  = optionData.ReversionRateParameter
  let sigma = optionData.VolatilityParameter
  let V  = sigma*sigma*( one - (r_exp (zero - two*a*dt)) ) / (two*a)
  let dr = r_sqrt( (one+two)*V )
  let M  = (r_exp (zero - a*dt)) - one
  let jmax = r2i (- 0.184 / M) + 1
  let m  = jmax + 2

  in if FORCE_PER_OPTION_THREAD && X < zero 
     then zero else
  ------------------------
  -- Compute Q values
  -------------------------
  -- Define initial tree values
  let Qlen = 2 * m + 1
  let Q = map (\i -> if i == m then one else zero) (iota Qlen)

  let alphas = replicate (n + 1) zero
  let alphas[0] = (h_YieldCurve[0]).P 
  
  -- time stepping
  let (_,alphas) =
    loop (Q: *[Qlen]real, alphas: *[]real) for i < n do
      let imax = i32.min (i+1) jmax
      -- Reset
      -- let QCopy[m - imax : m + imax + 1] = Q[m - imax : m + imax + 1]
      let QCopy = copy Q

      ----------------------------
      -- forward iteration step --
      ----------------------------
      let Q = -- 1. result of size independent of i (hoistable)
              map (\j -> if (j < (-imax)) || (j > imax)
                         then zero -- Q[j + m]
                         else fwdHelper M dr dt (alphas[i]) QCopy m i imax jmax j
                  ) (map (\a->a-m) (iota Qlen))
            
      -- determine new alphas
      let tmps= map (\jj -> let j = jj - imax in
                            if (j < (-imax)) || (j > imax) then zero 
                            else  Q[j+m] * r_exp(-(i2r j)*dr*dt)
                    ) (iota Qlen)
      let alpha_val = reduce (+) zero tmps

      -- interpolation of yield curve
      let t  = (i2r (i+1))*dt + one -- plus one year
      let t2 = r2i t-- round t
      let t1 = t2 - 1
      let (t2, t1) = if (t2 >= ycCount)
                     then (ycCount - 1, ycCount - 2)
                     else (t2         , t1         )
      let R = 
              ( ((h_YieldCurve[t2]).P ) - ((h_YieldCurve[t1]).P ) ) / 
              ( ((h_YieldCurve[t2]).t ) - ((h_YieldCurve[t1]).t ) ) *
              ( t*year - ((h_YieldCurve[t1]).t ) ) + ((h_YieldCurve[t1]).P )
      let P = r_exp(-R*t)
      let alpha_val = r_log (alpha_val / P)
      let alphas[i + 1] = alpha_val

      in  (Q,alphas)

    ------------------------------------------------------------
    --- Compute values at expiration date:
    --- call option value at period end is V(T) = S(T) - X
    --- if S(T) is greater than X, or zero otherwise.
    --- The computation is similar for put options.
    ------------------------------------------------------------
    let Call = map (\j -> if (j >= -jmax+m) && (j <= jmax + m)
                          then one else zero
                   ) 
                   (iota Qlen)
    
    -- back propagation
    let Call =
    loop (Call: *[Qlen]real) for ii < n do
      let i = n - 1 - ii
      let imax = i32.min (i+1) jmax
      
      -- Copy array values to avoid overwriting during update
      -- let CallCopy[m - imax : m + imax + 1] = Call[m - imax : m + imax + 1]
      let CallCopy = copy Call

      -----------------------------
      -- backward iteration step --
      -----------------------------
      let Call = -- 1. result of size independent of i (hoistable)
             map (\j -> if (j < (-imax)) || (j > imax)
                        then zero -- Call[j + m]
                        else bkwdHelper X M dr dt (alphas[i]) CallCopy m i jmax j
                 ) (map (\a->a-m) (iota Qlen))

      in  Call

    in Call[m] -- r_abs(Call[m] - 0.0000077536006753f64)

-------------------------
-- Static Yield Curve
-------------------------

let h_YieldCurve = [ { P = 0.0501772, t = 3.0    }
                   , { P = 0.0509389, t = 367.0  }
                   , { P = 0.0579733, t = 731.0  }
                   , { P = 0.0630595, t = 1096.0 }
                   , { P = 0.0673464, t = 1461.0 }
                   , { P = 0.0694816, t = 1826.0 }
                   , { P = 0.0708807, t = 2194.0 }
                   , { P = 0.0727527, t = 2558.0 }
                   , { P = 0.0730852, t = 2922.0 }
                   , { P = 0.0739790, t = 3287.0 }
                   , { P = 0.0749015, t = 3653.0 }
                   ]

-----------------
-- Entry point
-----------------
let main [q] (strikes     : [q]real)
             (maturities0 : [q]real) 
             (numofterms0 : [q]i32 ) 
             (rrps0       : [q]real) 
             (vols0       : [q]real) : [q]real =

  let (maturities, numofterms, rrps, vols) =
      if WITH_ONLY_STRIKE_VARIANT
      then ( replicate q (maturities0[0])
           , replicate q (numofterms0[0])
           , replicate q (rrps0[0])
           , vols0 -- replicate q (vols0[0])
           )
      else ( maturities0, numofterms0, rrps0, vols0)

  let options = map (\s m n r v -> { StrikePrice=s, Maturity=m, NumberOfTerms=n,
                                       ReversionRateParameter=r, VolatilityParameter=v }
                    ) strikes maturities numofterms rrps vols

  in  map (trinomialOptionsHW1FCPU_single h_YieldCurve) options

