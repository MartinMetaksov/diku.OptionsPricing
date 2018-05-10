-- Trinomial option pricing
-- ==
-- compiled input @ data/small.in
-- output @ data/small.out
--
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

let round (x : real) : i32 =
    let tmp = half + (r_abs x)
    let sgn = if x >= zero then one else (-one)
    in  r2i (sgn*tmp)

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

let fwdHelper (M : real) (dr : real) (dt : real) (alphai : real) (QCopy : []real, beg_ind: i32) 
              (m : i32) (i : i32) (imax : i32) (jmax : i32) (j : i32) : real = unsafe
    let eRdt_u1 = r_exp(-((i2r (j+1))*dr + alphai)*dt)
    let eRdt    = r_exp(-((i2r j    )*dr + alphai)*dt)
    let eRdt_d1 = r_exp(-((i2r (j-1))*dr + alphai)*dt)
    in  if i < jmax
        then let pu = PU_A(j-1, M)
             let pm = PM_A(j  , M)
             let pd = PD_A(j+1, M)
             in  if (i == 0 && j == 0 ) then pm*QCopy[beg_ind+j+m]*eRdt
                 else if (j == -imax+1) then pd*QCopy[beg_ind+j+m+1]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt
                 else if (j == imax-1 ) then pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j+m-1]*eRdt_d1
                 else if (j == 0-imax ) then pd*QCopy[beg_ind+j+m+1]*eRdt_u1
                 else if (j == imax   ) then pu*QCopy[beg_ind+j+m-1]*eRdt_d1
                 else pd*QCopy[beg_ind+j+m+1]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j+m-1]*eRdt_d1
        else if (j == jmax) 
                then let pm = PU_C(j  , M)
                     let pu = PU_A(j-1, M)
                     in  pm*QCopy[beg_ind+j+m]*eRdt +  pu*QCopy[beg_ind+j-1+m] * eRdt_d1
             else if(j == jmax - 1)
                then let pd = PM_C(j+1, M)
                     let pm = PM_A(j  , M)
                     let pu = PU_A(j-1, M)
                     in  pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m]*eRdt_d1
             else if (j == jmax - 2)
                then let eRdt_u2 = r_exp(-( (i2r(j+2))*dr + alphai ) * dt)
                     let pd_c = PD_C(j + 2, M)
                     let pd   = PD_A(j + 1, M)
                     let pm   = PM_A(j, M)
                     let pu   = PU_A(j - 1, M)
                     in  pd_c*QCopy[beg_ind+j+2+m]*eRdt_u2 + pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m]*eRdt_d1
             else if (j == -jmax + 2)
                then let eRdt_d2 = r_exp(-((i2r (j-2))*dr + alphai)*dt)
                     let pd   = PD_A(j + 1, M)
                     let pm   = PM_A(j, M)
                     let pu   = PU_A(j - 1, M)
                     let pu_b = PU_B(j - 2, M)
                     in  pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m]*eRdt_d1 + pu_b*QCopy[beg_ind+j-2+m]*eRdt_d2
             else if (j == -jmax + 1)
                then let pd = PD_A(j + 1, M)
                     let pm = PM_A(j, M)
                     let pu = PM_B(j - 1, M)
                     in  pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m]*eRdt_d1
             else if (j == -jmax)
                then let pd = PD_A(j + 1, M)
                     let pm = PD_B(j, M)
                     in  pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt
             else    
                     let pd = PD_A(j + 1, M)
                     let pm = PM_A(j, M)
                     let pu = PU_A(j - 1, M)
                     in  pd*QCopy[beg_ind+j+1+m]*eRdt_u1 + pm*QCopy[beg_ind+j+m]*eRdt + pu*QCopy[beg_ind+j-1+m]*eRdt_d1

-----------------------------------
--- backward propagation helper ---
-----------------------------------
let bkwdHelper (X : real) (M : real) (dr : real) (dt : real) (alphai : real) 
               (CallCopy : []real, beg_ind: i32) (m : i32) (i : i32) (jmax : i32) (j : i32) : real = unsafe
                let eRdt = r_exp(-((i2r j)*dr + alphai)*dt)
                let res =
                  if (i < jmax)
                  then -- central node
                     let pu = PU_A(j, M)
                     let pm = PM_A(j, M)
                     let pd = PD_A(j, M)
                     in  (pu*CallCopy[beg_ind+j+m+1] + pm*CallCopy[beg_ind+j+m] + pd*CallCopy[beg_ind+j+m-1]) * eRdt
                  else if (j == jmax)
                        then -- top node
                             let pu = PU_C(j, M)
                             let pm = PM_C(j, M)
                             let pd = PD_C(j, M)
                             in  (pu*CallCopy[beg_ind+j+m] + pm*CallCopy[beg_ind+j+m-1] + pd*CallCopy[beg_ind+j+m-2]) * eRdt
                       else if (j == -jmax)
                        then -- bottom node
                             let pu = PU_B(j, M)
                             let pm = PM_B(j, M)
                             let pd = PD_B(j, M)
                             in  (pu*CallCopy[beg_ind+j+m+2] + pm*CallCopy[beg_ind+j+m+1] + pd*CallCopy[beg_ind+j+m]) * eRdt
                       else    -- central node
                             let pu = PU_A(j, M)
                             let pm = PM_A(j, M)
                             let pd = PD_A(j, M)
                             in  (pu*CallCopy[beg_ind+j+m+1] + pm*CallCopy[beg_ind+j+m] + pd*CallCopy[beg_ind+j+m-1]) * eRdt

                -- TODO (WMP) This should be parametrized; length of contract, here 3 years
                in if (i == (r2i (three / dt))) 
                    then r_max(X - res, zero)
                    else res



let trinomialChunk [ycCount] [numAllOptions] [maxOptionsInChunk]
                   (h_YieldCurve : [ycCount]YieldCurveData)
                   (options : [numAllOptions]TOptionData) 
                   (w: i32)
                   (n_max : i32)
                   (optionsInChunk: i32, optionIndices: [maxOptionsInChunk]i32)
                 : [maxOptionsInChunk]real = unsafe
  if FORCE_PER_OPTION_THREAD && ( (options[optionIndices[0]]).StrikePrice < 0.0) 
     then (replicate maxOptionsInChunk 0.0) else
  -- header: get the options
  let (Xs, ns, dts, drs, Ms, jmaxs, ms) = unzip (
    map (\i -> if (i < 0)
               then (1.0, 0, 1.0, 1.0, 1.0, -1, -1)
               else
                let option = unsafe options[i]
                let X  = option.StrikePrice
                let T  = option.Maturity
                let n  = option.NumberOfTerms
                let dt = T / (i2r n)
                let a  = option.ReversionRateParameter
                let sigma = option.VolatilityParameter
                let V  = sigma*sigma*( one - (r_exp (0.0 - 2.0*a*dt)) ) / (2.0*a)
                let dr = r_sqrt( (1.0+2.0)*V )
                let M  = (r_exp (0.0 - a*dt)) - 1.0
                let jmax = r2i (- 0.184 / M) + 1
                let m  = jmax + 2
                in  (X, n, dt, dr, M, jmax, m)
        ) optionIndices
    )

  -- make the flag array (probably usefull for segmented scan/reduce operations)
  let map_lens   = map2  (\m i -> if i < optionsInChunk then 2*m + 1 else 0) 
                        ms (iota maxOptionsInChunk)
  let scanned_lens = scan (+) 0 map_lens
  let len_valinds= map2  (\i m -> if i == 0 then (0, m) 
                                 else if i < optionsInChunk
                                      then (scanned_lens[i-1], m)
                                      else let last_ind = scanned_lens[optionsInChunk-1]
                                           in  if last_ind < w 
                                               then (last_ind, w - last_ind)
                                               else (-1, 0)
                        ) (iota maxOptionsInChunk) map_lens
  let (len_inds, len_vals) = unzip len_valinds
  let flags = scatter (replicate w 0) len_inds len_vals

  -- make the flat segment-index array
  let sgm_inds = scatter (replicate w 0)
                         len_inds
                         (iota maxOptionsInChunk)
  let sgm_inds = sgmScanPlus flags sgm_inds

  -- make the segmented (iota (2*m+1)) across all ms
  let iota2mp1_p1 = sgmScanPlus flags (replicate w 1)
  let iota2mp1    = map (\x->x-1) iota2mp1_p1

  -- make the replicated segment lengths
  -- let len_flat = sgmScanPlus flags flags

  -- Q = map (\i -> if i == m then one else zero) (iota (2 * m + 1))
  let Qs = map2 (\i k -> if i == ms[sgm_inds[k]] then 1.0 else 0.0 
               ) iota2mp1 (iota w)

  -- alphas = replicate (n + 1) zero
  -- alphas[0] = #P (h_YieldCurve[0])
  -- for simplicity we pad this
  let seq_len = n_max+1
  let alphass = replicate (maxOptionsInChunk * seq_len) 0.0
  let alphass = scatter alphass 
                        ( map (\i->i*seq_len) (iota maxOptionsInChunk) )
                        ( replicate maxOptionsInChunk  (h_YieldCurve[0]).P )
  --let alphass = reshape ((maxOptionsInChunk,seq_len)) alphass

  -- compute n_maxInChunk
  -- let n_maxInChunk = reduce (\x y -> i32.max x y) (-1) ns
  let n_maxInChunk = n_max

  -- time stepping
  let ((Qs,alphass)) = loop ((Qs,alphass)) for i < n_maxInChunk do
      let imaxs = map (\jmax -> i32.min (i+1) jmax) jmaxs

      -- Reset
      -- QCopy[m - imax : m + imax + 1] = Q[m - imax : m + imax + 1]
      let QCopys' = copy Qs

      ----------------------------
      -- forward iteration step --
      ----------------------------

      -- Q = map (\j -> if (j < (-imax)) || (j > imax)
      --                then zero -- Q[j + m]
      --                else fwdHelper M dr dt (alphas[i]) QCopy m i imax jmax j
      --         ) (map (-m) (iota (2*m+1)))
      let Qs'= map2 (\jj w_ind -> let sgm_ind = sgm_inds[w_ind] in
                               if sgm_ind >= optionsInChunk || i >= ns[sgm_ind]
                               then 0.0
                               else let imax = imaxs[sgm_ind]
                                    let m    = ms[sgm_ind] 
                                    let j    = jj - m in
                                    if (j < (-imax)) || (j > imax)
                                    then 0.0
                                    else -- unsafe
                                         let begind = if sgm_ind == 0 then 0 else scanned_lens[sgm_ind-1] in
                                         fwdHelper (Ms[sgm_ind]) (drs[sgm_ind]) (dts[sgm_ind]) 
                                                   (alphass[sgm_ind*seq_len+i]) (QCopys',begind) 
                                                   m i imax (jmaxs[sgm_ind]) j
                   ) iota2mp1 (iota w)
      --------------------------      
      -- determine new alphas --
      --------------------------

      -- tmps= map (\jj -> let j = jj - imax in
      --                   if (j < (-imax)) || (j > imax) then zero 
      --                   else unsafe Q[j+m] * r_exp(-(i2r j)*dr*dt)
      --           ) (iota (2*m+1))--(iota (2*imax+1))
      -- alpha_val = reduce (+) zero tmps
      let tmpss = map2 (\ jj w_ind ->
                            let sgm_ind = sgm_inds[w_ind]
                            let imax    = imaxs[sgm_ind]
                            let j       = jj - imax in
                            if (j < (-imax)) || (j > imax) || 
                               (sgm_ind >= optionsInChunk) || (i >= ns[sgm_ind]) 
                            then 0.0
                            else -- unsafe
                                 let begind = if sgm_ind == 0 then 0 else scanned_lens[sgm_ind-1]
                                 in  Qs'[begind+j+ms[sgm_ind]] * r_exp(-(i2r j)*drs[sgm_ind]*dts[sgm_ind])
                      ) iota2mp1 (iota w)
      let tmpss_scan = sgmScanPlusReal flags tmpss
      let (inds, vals) = unzip (           -- INCORRECT, w-1 might not be the end of a segment
                            map (\w_ind -> if w_ind == w-1 then (sgm_inds[w_ind],tmpss_scan[w_ind])
                                      else if sgm_inds[w_ind] != sgm_inds[w_ind+1]
                                           then -- last element in a segment
                                                (sgm_inds[w_ind], tmpss_scan[w_ind])
                                           else (-1, 0.0) 
                                ) (iota w)
                          )
      let alpha_vals = scatter (replicate maxOptionsInChunk 0.0) inds vals
                                
      ----------------------------------
      -- interpolation of yield curve --
      ----------------------------------
      let Ps = map (\sgm_ind -> 
                        if (sgm_ind >= optionsInChunk) || (i >= ns[sgm_ind]) 
                        then 1.0
                        else --  t = (i2r (i+1))*dt + one -- plus one year
                             let t = (i2r (i+1))*dts[sgm_ind] + 1.0
                             let t2 = round t
                             let t1 = t2 - 1
                             let (t2, t1) = 
                                    if (t2 >= ycCount)
                                    then (ycCount - 1, ycCount - 2)
                                    else (t2         , t1         )
                             let R = -- unsafe
                                ( (h_YieldCurve[t2]).P - (h_YieldCurve[t1]).P ) / 
                                ( (h_YieldCurve[t2]).t - (h_YieldCurve[t1]).t ) *
                                ( t*year - (h_YieldCurve[t1]).t ) + (h_YieldCurve[t1]).P
                             let P = r_exp(-R*t)
                             in  P
                   ) (iota maxOptionsInChunk)      
      
      let alpha_vals = map2 (\alpha_val P -> r_log (alpha_val / P)) alpha_vals Ps

      -------------------------------  
      -- alphas[i + 1] = alpha_val --
      -------------------------------
      let alpha_inds = map (\k -> if (k < optionsInChunk) && (i < ns[k])
                                  then k*seq_len+(i+1)  -- ok, it is in range
                                  else -1 -- either not a valid option or 
                                          -- out of the convergence-loop range
                           ) (iota maxOptionsInChunk)

      let alphass' = scatter alphass alpha_inds alpha_vals

      in  (Qs', alphass')

  ------------------------------------------------------------
  --- Compute values at expiration date:
  --- call option value at period end is V(T) = S(T) - X
  --- if S(T) is greater than X, or zero otherwise.
  --- The computation is similar for put options.
  ------------------------------------------------------------
  -- Call = map (\j -> if (j >= -jmax+m) && (j <= jmax + m)
  --                   then one else zero
  --            ) (iota (2*m+1))
  -- CallCopy = replicate (2 * m + 1) zero

  let Calls = map2 (\ j w_ind -> let sgm_ind = sgm_inds[w_ind]
                                let (jmax,m) = (jmaxs[sgm_ind],ms[sgm_ind]) in
                                if (j >= -jmax+m) && (j <= jmax + m)
                                then 1.0 else 0.0
                  ) iota2mp1 (iota w)

  ----------------------
  -- back propagation --
  ----------------------
  let (Calls) = loop (Calls) for ii < n_maxInChunk do  -- condition is ii < ns[sgm_ind]
      -- i = n - 1 - ii
      let is = map (\sgm_ind -> if sgm_ind >= optionsInChunk then 0
                                else ns[sgm_ind] - 1 - ii 
                   ) (iota maxOptionsInChunk)
      let imaxs = map2 (\jmax i -> i32.min (i+1) jmax) jmaxs is
      
      ----------------------------------------------------------
      -- Copy array values to avoid overwriting during update --
      ----------------------------------------------------------
      -- CallCopy[m - imax : m + imax + 1] = Call[m - imax : m + imax + 1]
      let CallCopys' = copy Calls

      -----------------------------
      -- backward iteration step --
      -----------------------------
      -- Calls' = -- 1. result of size independent of i (hoistable)
      --          map (\j -> if (j < (-imax)) || (j > imax)
      --                     then zero -- Call[j + m]
      --                     else bkwdHelper X M dr dt (alphas[i]) CallCopy m i jmax j
      --              ) (map (-m) (iota (2*m+1)))
      let Calls' = map2  (\jj w_ind -> 
                            let sgm_ind = sgm_inds[w_ind] in
                            let begind = if sgm_ind == 0 then 0 else scanned_lens[sgm_ind-1] in
                            if  sgm_ind >= optionsInChunk || ii >= ns[sgm_ind]
                            then Calls[begind+jj]
                            else let imax = imaxs[sgm_ind]
                                 let i    = is[sgm_ind]
                                 let m    = ms[sgm_ind] 
                                 let j    = jj - m in
                                 if (j < (-imax)) || (j > imax)
                                    then Calls[begind+jj] -- 0.0
                                    else -- unsafe
                                         bkwdHelper (Xs[sgm_ind]) (Ms[sgm_ind]) (drs[sgm_ind]) 
                                                    (dts[sgm_ind]) (alphass[sgm_ind*seq_len+i]) 
                                                    (CallCopys', begind) (ms[sgm_ind]) 
                                                    i (jmaxs[sgm_ind]) j
                        )
                        iota2mp1 (iota w)

      in  (Calls')

------------------
--  in Calls[m] --
------------------
  let (inds, vals) = unzip (
        map (\sgm_ind -> if (sgm_ind >= optionsInChunk)
                   then (-1, 0.0) -- out of range
                   else let begind = if sgm_ind == 0 then 0 else scanned_lens[sgm_ind-1]
                        let m_ind  = begind+ms[sgm_ind]
                        in  (sgm_ind, Calls[m_ind])
            ) (iota maxOptionsInChunk)
    )
  let res = scatter (replicate maxOptionsInChunk 0.0) inds vals
  in  res

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

-- trinomialChunk [ycCount] [numAllOptions] [maxOptionsInChunk]
--                   (h_YieldCurve : [ycCount]YieldCurveData)
--                   (w: i32)
--                   (n_max : i32)
--                   (optionIndices: [maxOptionsInChunk]i32)
--                   (optionsInChunk: i32)
--                   (options : [numAllOptions]TOptionData) 
--                 : [maxOptionsInChunk]real
let formatOptions [numOptions]
                  (w: i32)
                  (options : [numOptions]TOptionData)
                -- (n_max, maxOptionsInChunk, optionsInChunk, optionIndices) 
                : (i32, i32, [](i32, []i32)) = -- unsafe
  let (ns, ms) = unzip (
    map (\option -> 
                let T  = option.Maturity
                let n  = option.NumberOfTerms
                let dt = T / (i2r n)
                let a  = option.ReversionRateParameter
                -- let sigma = option.VolatilityParameter
                let M  = (r_exp (0.0 - a*dt)) - 1.0
                let jmax = r2i (- 0.184 / M) + 1
                let m  = jmax + 2
                in  (n+1, 2*m+1)
        ) options
    )
  let n_max = reduce (\x y -> i32.max x y) 0 ns
  let m_max = reduce (\x y -> i32.max x y) 0 ms

  let maxOptionsInChunk = w / m_max
  let num_chunks = (numOptions + maxOptionsInChunk - 1) / maxOptionsInChunk
  let chunks = map (\ c_ind ->
                              let num = if c_ind == num_chunks - 1 
                                        then numOptions - c_ind*maxOptionsInChunk
                                        else maxOptionsInChunk
                              let arr = 
                                map (\ i -> let opt_ind = c_ind*maxOptionsInChunk + i in  
                                            if opt_ind < numOptions then opt_ind else (-1)
                                    ) (iota maxOptionsInChunk)
                              in (num, arr)
                          )
                          (iota num_chunks)
  in  (n_max, maxOptionsInChunk, chunks)
   

-----------------
-- Entry point
-----------------
let main [q] (strikes    : [q]real)
             (maturities : [q]real) 
             (numofterms : [q]i32 ) 
             (rrps       : [q]real) 
             (vols       : [q]real) 
           : [q]real =

  let options = map5 (\s m n r v -> { StrikePrice=s, Maturity=m, NumberOfTerms=n,
                                       ReversionRateParameter=r, VolatilityParameter=v }
                    ) strikes maturities numofterms rrps vols

  let w = 256
  let (n_max, maxOptionsInChunk, chunks) = formatOptions w options
  let chunkRes = -- : [maxOptionsInChunk]real
        map (trinomialChunk h_YieldCurve options w n_max) chunks
      
  let num_chunks = length chunkRes
  let res_flat = reshape (num_chunks*maxOptionsInChunk) chunkRes
  let (res,_) = split (q) res_flat
  in  res




