/**
Copyright © 2023 Computer Inspirations. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import UInt128

///
/// Groups together algorithms that can be used by all Decimalxx variants
///

// MARK: - Generic Integer Decimal Type

public typealias Sign = FloatingPointSign
public typealias Rounding = FloatingPointRoundingRule

public protocol IntegerDecimal : Codable, Hashable {
  
  associatedtype RawDataFields : UnsignedInteger & FixedWidthInteger
  associatedtype Mantissa : UnsignedInteger & FixedWidthInteger
  
  var data: RawDataFields { get set }
  
  //////////////////////////////////////////////////////////////////
  /// Initializers
  
  /// Initialize with a raw data word
  init(_ word: RawDataFields)
  
  /// Initialize with sign, biased exponent, and unsigned mantissa
  init(sign: Sign, exponent: Int, mantissa: Mantissa, round:Int)
  init(sign: Sign, exponent: Int, mantissa: Mantissa)
  
  //////////////////////////////////////////////////////////////////
  /// Conversions from/to densely packed decimal numbers
  init(dpd: RawDataFields)
  
  var dpd: RawDataFields { get }
  
  //////////////////////////////////////////////////////////////////
  /// Essential data to extract or update from the fields
  
  /// Sign of the number
  var sign: Sign { get set }
  
  /// Unbiased signed exponent of the number
  var exponent: Int { get }
  
  /// Unsigned mantissa integer
  var mantissa: Mantissa { get }
  
  /// Setting requires both the mantissa and exponent so that a
  /// decision can be made on whether the mantissa is small or large.
  mutating func set(exponent: Int, mantissa: Mantissa)
  
  //////////////////////////////////////////////////////////////////
  /// Special number definitions
  static var snan: Self { get }
  
  static func zero(_ sign: Sign) -> Self
  static func nan(_ sign: Sign, _ payload: Int) -> Self
  static func infinite(_ sign: Sign) -> Self
  static func max(_ sign: Sign) -> Self
  
  //////////////////////////////////////////////////////////////////
  /// Decimal number definitions
  static var signBit: Int { get }
  static var specialBits: ClosedRange<Int> { get }
  
  static var exponentBias: Int    { get }
  static var maximumExponent: Int { get } // unbiased & normal
  static var minimumExponent: Int { get } // unbiased & normal
  static var maximumDigits:  Int { get }
  
  static var largestNumber: Mantissa { get }
  
  // For large mantissa
  static var exponentLMBits: ClosedRange<Int> { get }
  static var largeMantissaBits: ClosedRange<Int> { get }
  
  // For small mantissa
  static var exponentSMBits: ClosedRange<Int> { get }
  static var smallMantissaBits: ClosedRange<Int> { get }
}

///
/// Free functionality when complying with IntegerDecimalField
public extension IntegerDecimal {
  
  static var highMantissaBit: Int { 1 << exponentLMBits.lowerBound }
  
  /// These bit fields can be predetermined just from the size of
  /// the number type `RawDataFields` `bitWidth`
  static var maxBit: Int { RawDataFields.bitWidth - 1 }
  static var largestBID : Self {
    Self(sign: .plus, exponent: maximumExponent, mantissa: largestNumber)
  }
  
  static var signBit: Int                    { maxBit }
  static var specialBits: ClosedRange<Int>   { maxBit-2 ... maxBit-1 }
  static var nanBitRange: ClosedRange<Int>   { maxBit-6 ... maxBit-1 }
  static var nanClearRange: ClosedRange<Int> { 0 ... maxBit-7 }
  static var g6tog10Range: ClosedRange<Int>  { maxBit-11 ... maxBit-7 }
  
  // masks for clearing bits
  static var sNanRange: ClosedRange<Int>     { 0 ... maxBit-6 }
  static var sInfinityRange: ClosedRange<Int>{ 0 ... maxBit-5 }
  
  /// bit field definitions for DPD numbers
  static var lowMan: Int    { smallMantissaBits.upperBound }  // 20
  static var upperExp1: Int { exponentSMBits.upperBound }
  static var upperExp2: Int { exponentLMBits.upperBound }
  
  static var expLower: ClosedRange<Int> { lowMan...maxBit-6 }
  static var manLower: ClosedRange<Int> { 0...lowMan-1 }
  static var expUpper: ClosedRange<Int> { lowMan+1...lowMan+6 }
  
  /// Bit patterns prefixes for special numbers
  static var nanPattern: Int      { 0b1_1111_0 }
  static var snanPattern: Int     { 0b1_1111_1 }
  static var infinitePattern: Int { 0b1_1110 }
  static var specialPattern: Int  { 0b11 }
  
  static var trailingPattern: Int { 0x3ff }
  
  @inlinable var sign: Sign {
    get { data.get(bit: Self.signBit) == 0 ? .plus : .minus }
    set { data.set(bit: Self.signBit, with: newValue == .minus ? 1 : 0) }
  }
  
  @inlinable var exponent: Int {
    let range = isSmallMantissa ? Self.exponentSMBits : Self.exponentLMBits
    return data.get(range: range)
  }
  
  @inlinable var mantissa: Mantissa {
    let range = isSmallMantissa ? Self.smallMantissaBits
                                : Self.largeMantissaBits
    if isSmallMantissa {
      return Mantissa(data.get(range:range) + Self.highMantissaBit)
    } else {
      return Mantissa(data.get(range:range))
    }
  }
  
  static func adjustOverflowUnderflow(_ sign: Sign, _ exp: Int,
                      _ mant: Mantissa, _ rmode: Rounding) -> RawDataFields {
    var exp = exp, mant = mant, rmode = rmode
    var raw = RawDataFields(0)
    if mant > largestNumber {
      exp += 1; mant = (largestNumber+1)/10
    }
    
    // check for possible underflow/overflow
    if exp > maximumExponent || exp < minimumExponent {
      if exp < minimumExponent {
        // deal with an underflow situation
        if exp + maximumDigits < 0 {
          // underflow & inexact
          if rmode == .down && sign == .minus {
            raw.set(bit: signBit); raw.set(range: manLower, with: 1)
            return raw
          }
          if rmode == .up && sign == .plus {
            raw.set(range: manLower, with: 1)
            return raw
          }
          raw.set(bit: signBit, with: sign == .minus ? 1 : 0)
          return raw
        }
        
        // swap up & down round modes when negative
        if sign == .minus {
          if rmode == .up { rmode = .down }
          else if rmode == .down { rmode = .up }
        }
        
        // determine the rounding table index
        let roundIndex = roundboundIndex(rmode) >> 2
        
        // get digits to be shifted out
        let extraDigits = -exp
        mant += Mantissa(bid_round_const_table(roundIndex, extraDigits))
        
        let Q = mul64x64to128(UInt64(mant), reciprocals10(extraDigits))
        let amount = shortReciprocalScale[extraDigits]
        var C64 = Q.components.high >> amount
        var remainder_h = UInt128.High(0)
        if rmode == .toNearestOrAwayFromZero {
          if !C64.isMultiple(of: 2) {
            // odd factor so check whether fractional part is exactly 0.5
            let amount2 = 64 - amount
            remainder_h &-= 1  // decrement without overflow check
            remainder_h >>= amount2
            remainder_h &= Q.components.high
            if remainder_h == 0 && Q.components.low < reciprocals10(extraDigits) {
              C64 -= 1
            }
          }
        }
        
        raw = RawDataFields(C64)
        raw.set(bit: signBit, with: sign == .minus ? 1 : 0)
        return raw
      }
      
      if mant == 0 {
        if exp > maximumExponent { exp = maximumExponent }
      }
      while mant < (largestNumber+1)/10 && exp > maximumExponent {
        mant = (mant << 3) + (mant << 1)  // times 10
        exp -= 1
      }
      if exp > maximumExponent {
        raw = infinite(sign).data
        switch rmode {
          case .down:
            if sign == .plus { raw = largestBID.data }
          case .towardZero:
            raw = largestBID.data; raw.set(bit: signBit)
          case .up:
            if sign == .minus {
              raw = largestBID.data
              raw.set(bit: signBit, with: sign == .minus ? 1 : 0)
            }
          default: break
        }
        return raw
      }
    }
    return Self(sign:sign, exponent:exp, mantissa: mant).data
  }
  
  /// Note: `exponent` is assumed to be biased
  mutating func set(exponent: Int, mantissa: Mantissa) {
    if mantissa < Self.highMantissaBit {
      // large mantissa
      data.set(range: Self.exponentLMBits, with: exponent)
      data.set(range: Self.largeMantissaBits, with: Int(mantissa))
    } else {
      // small mantissa
      data.set(range: Self.exponentSMBits, with: exponent)
      data.set(range: Self.smallMantissaBits,
               with:Int(mantissa)-Self.highMantissaBit)
      data.set(range: Self.specialBits, with: Self.specialPattern)
    }
  }

  /// Return `self's` pieces all at once with biased exponent
  func unpack() -> (sign:Sign, exponent:Int, mantissa:Mantissa, valid:Bool) {
    var exponent: Int, mantissa: Mantissa
    if isSpecial {
      if isInfinite {
        mantissa = Mantissa(data); mantissa.clear(range: Self.g6tog10Range)
        if data.get(range: Self.manLower) >= (Self.largestNumber+1)/10 {
          mantissa = Mantissa(data); mantissa.clear(range: Self.sNanRange)
        }
        if isNaNInf {
          mantissa = Mantissa(data); mantissa.clear(range: Self.sInfinityRange)
        }
        return (self.sign, 0, mantissa, false)
      }
      // small mantissa
      exponent = data.get(range: Self.exponentSMBits)
      mantissa = Mantissa(data.get(range: Self.smallMantissaBits) +
                          Self.highMantissaBit)
      if mantissa > Self.largestNumber { mantissa = 0 }
      return (self.sign, exponent, mantissa, mantissa != 0)
    } else {
      // large mantissa
      exponent = data.get(range: Self.exponentLMBits)
      mantissa = Mantissa(data.get(range: Self.largeMantissaBits))
      return (self.sign, exponent, mantissa, mantissa != 0)
    }
  }
  
  /// Return `dpd` pieces all at once
  static func unpack(dpd: RawDataFields) ->
                  (sign: Sign, exponent: Int, high: Int, trailing: Mantissa) {
    let sgn = dpd.get(bit: signBit) == 1 ? Sign.minus : .plus
    var exponent, high: Int, trailing: Mantissa
    let expRange2: ClosedRange<Int>
    
    if dpd.get(range: specialBits) == specialPattern {
      // small mantissa
      expRange2 = (upperExp1-1)...upperExp1
      high = dpd.get(bit: upperExp1-2) + 8
    } else {
      // large mantissa
      expRange2 = (upperExp2-1)...upperExp2
      high = dpd.get(range: upperExp1-2...upperExp1)
    }
    exponent = dpd.get(range: expLower) + dpd.get(range: expRange2) << 6
    trailing = Mantissa(dpd.get(range: 0...lowMan-1))
    return (sgn, exponent, high, trailing)
  }
  
  @inlinable func nanQuiet() -> Mantissa {
    // let quietMask = ~(Mantissa(1) << Self.nanBitRange.lowerBound)
    var data = self.data
    data.clear(bit: Self.nanBitRange.lowerBound)
    return Mantissa(data)
  }
  
  ///////////////////////////////////////////////////////////////////////
  /// Special number definitions
  @inlinable static func infinite(_ s: Sign = .plus) -> Self {
    Self(sign: s, exponent: infinitePattern<<3, mantissa: 0)
  }
  
  @inlinable static func max(_ s: Sign = .plus) -> Self {
    Self(sign:s, exponent:maximumExponent, mantissa:largestNumber)
  }
  
  static func overflow(_ sign: Sign, rndMode: Rounding) -> Self {
    if rndMode == .towardZero || rndMode == (sign != .plus ? .up : .down) {
      return max(sign)
    } else {
      return infinite(sign)
    }
  }
  
  @inlinable static var snan: Self {
    Self(sign: .plus, exponent: snanPattern<<2, mantissa: 0)
  }
  
  @inlinable static func zero(_ sign: Sign) -> Self {
    Self(sign: sign, exponent: exponentBias, mantissa: 0)
  }
  
  @inlinable
  static func nan(_ sign: Sign, _ payload: Int) -> Self {
    let man = payload > largestNumber/10 ? 0 : Mantissa(payload)
    return Self(sign:sign, exponent:nanPattern<<2, mantissa:man)
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Handy routines for testing different aspects of the number
  @inlinable var nanBits: Int { data.get(range: Self.nanBitRange) }
  
  var isSmallMantissa: Bool { isSpecial }
  var isNaN: Bool           { nanBits & Self.nanPattern == Self.nanPattern }
  var isSNaN: Bool          { nanBits & Self.snanPattern == Self.snanPattern }
  
  var isFinite: Bool {
    let infinite = Self.infinitePattern
    let data = data.get(range: Self.signBit-5...Self.signBit-1)
    return (data & infinite != infinite)
  }
  
  var isSpecial: Bool {
    data.get(range: Self.specialBits) == Self.specialPattern
  }
  
  var isNaNInf: Bool {
    nanBits & Self.nanPattern == Self.infinitePattern<<1
  }
  
  var isInfinite: Bool {
    let infinite = Self.infinitePattern
    let data = data.get(range: Self.signBit-5...Self.signBit-1)
    return (data & infinite == infinite) 
  }
  
  var isValid: Bool {
    if isNaN { return false }
    if isSpecial {
      if isInfinite { return false }
      if mantissa > Self.largestNumber || mantissa == 0 { return false }
    } else {
      if mantissa == 0 { return false }
    }
    return true
  }
  
  var isCanonical: Bool {
    if isNaN {
      if (data & 0x01f0 << (Self.maxBit - 16)) != 0 {
        // FIXME: - what is this? Decimal32 had mask of 0x01fc
        return false
      } else if data.get(range:Self.manLower) > Self.largestNumber/10 {
        return false
      } else {
        return true
      }
    } else if isInfinite {
      return data.get(range:0...Self.exponentLMBits.lowerBound+2) == 0
    } else if isSpecial {
      return mantissa <= Self.largestNumber
    } else {
      return true
    }
  }
  
  private func checkNormalScale(_ exp: Int, _ mant: Mantissa) -> Bool {
    // if exponent is less than -95, the number may be subnormal
    // let exp = exp - Self.exponentBias
    if exp < Self.minimumExponent+Self.maximumDigits-1 {
      let tenPower = power(UInt64(10), to: exp)
      let mantPrime = UInt64(mant) * tenPower
      return mantPrime > Self.largestNumber/10 // normal test
    }
    return true // normal
  }
  
  var isNormal: Bool {
    let (_, exp, mant, valid) = self.unpack()
    if !valid { return false }
    return checkNormalScale(exp, mant)
  }
  
  var isSubnormal: Bool {
    let (_, exp, mant, valid) = self.unpack()
    if !valid { return false }
    return !checkNormalScale(exp, mant)
  }
  
  var isZero: Bool {
    if isInfinite { return false }
    if isSpecial {
      return mantissa > Self.largestNumber
    } else {
      return mantissa == 0
    }
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Convert to/from BID/DPD numbers
  
  /// Create a new BID number from the `dpd` DPD number.
  init(dpd: RawDataFields) {
    
    func getNan() -> Int { dpd.get(range: Self.nanBitRange) }
    
    // Convert the dpd number to a bid number
    var (sign, exp, high, trailing) = Self.unpack(dpd: dpd)
    var nan = false
    
    if getNan() & Self.nanPattern == Self.infinitePattern {
      self = Self.infinite(sign); return
    } else if getNan() & Self.nanPattern == Self.nanPattern {
      nan = true; exp = 0
    }
    
    let mask = Self.trailingPattern
    let mils = ((Self.maximumDigits - 1) / 3) - 1
    let shift = mask.bitWidth - mask.leadingZeroBitCount
    var mant = Mantissa(high)
    for i in stride(from: shift*mils, through: 0, by: -shift) {
      mant *= 1000
      mant += Mantissa(Self.intFrom(dpd: Int(trailing >> i) & mask))
    }
    
    if nan { self = Self.nan(sign, Int(mant)) }
    else { self.init(sign: sign, exponent: exp, mantissa: mant) }
  }
  
  /// Convert `self` to a DPD number.
  var dpd: RawDataFields {
    var res : RawDataFields = 0
    var (sign, exp, mantissa, _) = unpack()
    var trailing = mantissa.get(range: Self.manLower) // & 0xfffff
    var nanb = false
    
    if self.isNaNInf {
      return Self.infinite(sign).data
    } else if self.isNaN {
      if trailing > Self.largestNumber/10 {
        trailing = 0
      }
      mantissa = Self.Mantissa(trailing); exp = 0; nanb = true
    } else {
      if mantissa > Self.largestNumber { mantissa = 0 }
    }
    
    let mils = ((Self.maximumDigits - 1) / 3) - 1
    let shift = 10
    var dmant = 0
    for i in stride(from: 0, through: shift*mils, by: shift) {
      dmant |= Self.intToDPD(Int(mantissa) % 1000) << i
      mantissa /= 1000
    }
    
    let signBit = Self.signBit
    let expLower = Self.smallMantissaBits.upperBound...signBit-6
    let manLower = 0...Self.smallMantissaBits.upperBound-1
    
    if mantissa >= 8 {
      let expUpper = signBit-4...signBit-3
      let manUpper = signBit-5...signBit-5
      res.set(range: Self.specialBits, with: Self.specialPattern)
      res.set(range: expUpper, with: exp >> 6)          // upper exponent bits
      res.set(range: manUpper, with: Int(mantissa) & 1) // upper mantisa bits
    } else {
      let expUpper = signBit-2...signBit-1
      let manUpper = signBit-5...signBit-3
      res.set(range: expUpper, with: exp >> 6)      // upper exponent bits
      res.set(range: manUpper, with: Int(mantissa)) // upper mantisa bits
    }
    res.set(bit: signBit, with: sign == .minus ? 1 : 0)
    res.set(range: expLower, with: exp)
    res.set(range: manLower, with: dmant)
    if nanb { res.set(range: Self.nanBitRange, with: Self.nanPattern) }
    return res
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Double/BID conversions
  
  static func bid(from x:Double, _ rndMode:Rounding) -> Self {
    // Unpack the input
    let expMask = (1 << Double.exponentBitCount) - 1
    let sigBits = Double.significandBitCount + 1 // including invisible bit
    let dBits = x.bitPattern.bitWidth
    let shift = 113 - sigBits // from (2^{113-53} * c * r) >> 320
    let elimit = -450         // floor(emin * log_2(10) - 115)
    let minNormExp = -1074    // minimum normal exponent
    
    var s = x.sign, e = Int(x.exponentBitPattern), t = 0
    var c = UInt128(high: 0, low: x.significandBitPattern)
    
    if e == 0 {
      if x.isZero { return zero(s) }
      
      // denormalizd number
      let l = c.leadingZeroBitCount - (dBits - sigBits)
      c <<= 1
      e = -(l - minNormExp)
    } else if e == expMask {
      if x.isInfinite { return infinite(s) }
      if x.isNaN { return nan(s, Int(c._lowWord)) }
    } else {
      c.set(bit: sigBits-1)  // set upper bit
      e += (minNormExp - 1)
      t = c.trailingZeroBitCount
    }
    
    // Now -1126<=e<=971 (971 for max normal, -1074 for min normal,
    // -1126 for min denormal)
    
    // Treat like a quad input for uniformity, so (2^{113-53} * c * r) >> 320,
    // where 320 is the truncation value for the reciprocal multiples, exactly
    // five 64-bit words. So we shift 113-53=60 places
    //
    // Remember to compensate for the fact that exponents are integer for quad
    c = c << shift
    t += shift
    e -= shift // Now e belongs [-1186;911].
    
    // Check for "trivial" overflow, when 2^e * 2^112 > 10^emax * 10^d.
    // We actually check if e >= ceil((emax + d) * log_2(10) - 112)
    // This could be intercepted later, but it's convenient to keep tables
    // smaller
    if e >= 211 {
      // state.formUnion([.overflow, .inexact])
      return overflow(s, rndMode: rndMode)
    }
    
    // Now filter out all the exact cases where we need to specially force
    // the exponent to 0. We can let through inexact cases and those where the
    // main path will do the right thing anyway, e.g. integers outside coeff
    // range.
    //
    // First check that e <= 0, because if e > 0, the input must be >= 2^113,
    // which is too large for the coefficient of any target decimal format.
    // We write a = -(e + t)
    //
    // (1) If e + t >= 0 <=> a <= 0 the input is an integer; treat it specially
    //     iff it fits in the coefficient range. Shift c' = c >> -e, and
    //     compare with the coefficient range; if it's in range then c' is
    //     our coefficient, exponent is 0. Otherwise we pass through.
    //
    // (2) If a > 0 then we have a non-integer input. The special case would
    //     arise as c' / 2^a where c' = c >> t, i.e. 10^-a * (5^a c'). Now
    //     if a > 48 we can immediately forget this, since 5^49 > 10^34.
    //     Otherwise we determine whether we're in range by a table based on
    //     a, and if so get the multiplier also from a table based on a.
    if e <= 0 {
      let a = -(e + t)
      var cint = c
      if a <= 0 {
        cint = cint >> -e
        if cint < largestNumber+1 {
          return Self(sign: s, exponent: exponentBias,
                      mantissa: Mantissa(cint._lowWord))
        }
      } else if a <= 48 {
        var pow5 = Self.coefflimitsBID32(a)
        cint = cint >> t  // srl128(cint.hi, cint.low, t)
        if cint <= pow5 { // le128(cint.hi, cint.low, pow5.hi, pow5.low) {
          var cc = cint
          pow5 = power5(a)
          cc = mul128x128Low(cc, pow5)
          return Self(sign: s, exponent: exponentBias-a,
                      mantissa: Mantissa(cc._lowWord))
        }
      }
    }
    
    // Check for "trivial" underflow, when 2^e * 2^113 <= 10^emin * 1/4,
    // so test e <= floor(emin * log_2(10) - 115)
    // In this case just fix ourselves at that value for uniformity.
    //
    // This is important not only to keep the tables small but to maintain the
    // testing of the round/sticky words as a correct rounding method
    if e <= elimit {
      e = elimit
    }
    
    // Now look up our exponent e, and the breakpoint between e and e+1
    let m_min = Tables.bid_breakpoints_bid32[e-elimit]
    var e_out = Int(Tables.bid_exponents_bid32[e-elimit])
    
    // Choose exponent and reciprocal multiplier based on breakpoint
    var r:UInt256
    if c <= m_min {
      r = Tables.bid_multipliers1_bid32[e-elimit]
    } else {
      r = Tables.bid_multipliers2_bid32[e-elimit]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z=UInt384()
    Self.mul128x256to384(&z, c, r)
    var c_prov = Mantissa(z.w[5])
    
    // Test inexactness and underflow (when testing tininess before rounding)
    //    if ((z.w[4] != 0) || (z.w[3] != 0)) {
    //        // __set_status_flags(pfpsf,BID_INEXACT_EXCEPTION);
    //        state.insert(.inexact)
    //        if (c_prov < 1000000) {
    //            state.insert(.underflow)
    //            // __set_status_flags(pfpsf,BID_UNDERFLOW_EXCEPTION);
    //        }
    //    }
    
    // Round using round-sticky words
    // If we spill over into the next decade, correct
    // Flag underflow where it may be needed even for |result| = SNN
    let ind = roundboundIndex(rndMode, s == .minus, Int(c_prov))
    if bid_roundbound_128[ind] < UInt128(high: z.w[4], low: z.w[3]) {
      c_prov += 1
      let max = largestNumber+1
      if c_prov == max {
        c_prov = max/10
        e_out += 1
        //      } else if c_prov == max/10 && e_out == 0 {
        // let ind = roundboundIndex(rndMode, false, 0) >> 2
        //            if ((((ind & 3) == 0) && (z.w[4] <= 17524406870024074035)) ||
        //                ((ind + (s & 1) == 2) && (z.w[4] <= 16602069666338596454))) {
        //                state.insert(.underflow)
        //            }
      }
    }
    
    // Check for overflow
    if e_out > 90 {
      // state.formUnion([.overflow, .inexact])
      return overflow(s, rndMode: rndMode)
    }
    
    // Set the inexact flag as appropriate and check underflow
    // It's no doubt superfluous to check inexactness, but anyway...
    //    if z.w[4] != 0 || z.w[3] != 0 {
    //        state.insert(.inexact)
    //        if c_prov < 1_000_000 {
    //            state.insert(.underflow)
    //        }
    //    }
    
    // Package up the result
    return Self(sign: s, exponent: e_out+exponentBias, mantissa: c_prov)
  }
  
  func int(_ rmode:Rounding) -> Int64 {
    let x = self
    var res: Int64 = 0
    
    if x.isNaN || x.isInfinite {
      // set invalid flag
      // pfpsc.insert(.invalidOperation)
      
      // return Integer Indefinite
      return Int64.min
    }
    
    // unpack x
    let (x_sign, x_exp, C1, _) = x.unpack()
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 {
      // x is 0
      return 0
    }
    
    // x is not special and is not zero
    // q = nr. of decimal digits in x (1 <= q <= 7)
    //  determine first the nr. of bits in x
    let q : Int = Self.digitsIn(C1)
    let exp = Int(x_exp) - Self.exponentBias // unbiased exponent
    
    if (q + exp) > 19 { // x >= 10^19 ~= 2^63.11... (cannot fit in BID_SINT64)
      // set invalid flag
      // pfpsc.insert(.invalidOperation)
      // return Integer Indefinite
      return Int64.min
    } else if (q + exp) == 19 { // x = c(0)c(1)...c(q-1)00...0 (19 dec. digits)
      // in this case 2^63.11... ~= 10^19 <= x < 10^20 ~= 2^66.43...
      // so x rounded to an integer may or may not fit in a signed 64-bit int
      // the cases that do not fit are identified here; the ones that fit
      // fall through and will be handled with other cases further,
      // under '1 <= q + exp <= 19'
      var C = UInt128()
      if x_sign == .minus { // if n < 0 and q + exp = 19
        // if n <= -2^63 - 1 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[19 dec. digits] >= 2^63+1
        // <=> 0.c(0)c(1)...c(q-1) * 10^20 >= 0x5000000000000000a, 1<=q<=7
        // <=> C * 10^(20-q) >= 0x5000000000000000a, 1<=q<=7
        // 1 <= q <= 7 => 13 <= 20-q <= 19 => 10^(20-q) is 64-bit, and so is C1
        C = Self.mul64x64to128(UInt64(C1), Self.bid_ten2k64(20 - q))
        // Note: C1 * 10^(11-q) has 19 or 20 digits;0x5000000000000000a, has 20
        if (C.components.high > 0x05 ||
            (C.components.high == 0x05 && C.components.low >= 0x0a)) {
          // set invalid flag
          //pfpsc.insert(.invalidOperation)
          // return Integer Indefinite
          return Int64.min
        }
        // else cases that can be rounded to a 64-bit int fall through
        // to '1 <= q + exp <= 19'
      } else { // if n > 0 and q + exp = 19
        // if n >= 2^63 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[19 dec. digits] >= 2^63
        // <=> if 0.c(0)c(1)...c(q-1) * 10^20 >= 0x50000000000000000, 1<=q<=7
        // <=> if C * 10^(20-q) >= 0x5_0000000000000000, 1<=q<=7
        C.components = (5, 0x0000000000000000)
        
        // 1 <= q <= 7 => 13 <= 20-q <= 19 => 10^(20-q) is 64-bit, and so is C1
        C = Self.mul64x64to128(UInt64(C1), Self.bid_ten2k64(20 - q))
        if C.components.high >= 0x05 {
          // actually C.hi == 0x05 && C.lo >= 0x0000000000000000) {
          // set invalid flag
          //pfpsc.insert(.invalidOperation)
          // return Integer Indefinite
          return Int64.min
        }
        // else cases that can be rounded to a 64-bit int fall through
        // to '1 <= q + exp <= 19'
      }    // end else if n > 0 and q + exp = 19
    }    // end else if ((q + exp) == 19)
    
    // n is not too large to be converted to int64: -2^63-1 < n < 2^63
    // Note: some of the cases tested for above fall through to this point
    if (q + exp) <= 0 { // n = +/-0.0...c(0)c(1)...c(q-1)
      // return 0
      return 0
    } else { // if (1 <= q + exp <= 19, 1 <= q <= 7, -6 <= exp <= 18)
      // -2^63-1 < x <= -1 or 1 <= x < 2^63 so x can be rounded
      // to nearest to a 64-bit signed integer
      if exp < 0 { // 2 <= q <= 7, -6 <= exp <= -1, 1 <= q + exp <= 19
        let ind = -exp // 1 <= ind <= 6; ind is a synonym for 'x'
        // chop off ind digits from the lower part of C1
        // C1 fits in 64 bits
        // calculate C* and f*
        // C* is actually floor(C*) in this case
        // C* and f* need shifting and masking, as shown by
        // bid_shiftright128[] and bid_maskhigh128[]
        // 1 <= x <= 6
        // kx = 10^(-x) = bid_ten2mk64[ind - 1]
        // C* = C1 * 10^(-x)
        // the approximation of 10^(-x) was rounded up to 54 bits
        let P128 = Self.mul64x64to128(UInt64(C1), Self.bid_ten2mk64(ind - 1))
        var Cstar = P128.components.high
        // the top Ex bits of 10^(-x) are T* = bid_ten2mk128trunc[ind].lo, e.g.
        // if x=1, T*=bid_ten2mk128trunc[0].lo=0x1999999999999999
        // C* = floor(C*) (logical right shift; C has p decimal digits,
        //     correct by Property 1)
        // n = C* * 10^(e+x)
        
        // shift right C* by Ex-64 = bid_shiftright128[ind]
        let shift = Self.bid_shiftright128[ind - 1] // 0 <= shift <= 39
        Cstar = Cstar >> shift
        
        if x_sign == .minus {
          res = -Int64(Cstar)
        } else {
          res = Int64(Cstar)
        }
      } else if exp == 0 {
        // 1 <= q <= 7
        // res = +/-C (exact)
        if x_sign == .minus {
          res = -Int64(C1)
        } else {
          res = Int64(C1)
        }
      } else {
        // if (exp > 0) => 1 <= exp <= 18, 1 <= q <= 7, 2 <= q + exp <= 20
        // (the upper limit of 20 on q + exp is due to the fact that
        // +/-C * 10^exp is guaranteed to fit in 64 bits)
        // res = +/-C * 10^exp (exact)
        if x_sign == .minus {
          res = -Int64(UInt64(C1) * Self.bid_ten2k64(exp))
        } else {
          res = Int64(UInt64(C1) * Self.bid_ten2k64(exp))
        }
      }
    }
    return res
  }
  
  func uint(_ rmode:Rounding) -> UInt64 {
    let x = self
    
    // check for NaN or Infinity
    if x.isNaN || x.isInfinite {
      // set invalid flag
      // pfpsc.insert(.invalidOperation)
      
      // return Integer Indefinite
      return UInt64(bitPattern: Int64.min)
    }
    
    // unpack x
    let (x_sign, x_exp, C1, _) = x.unpack()
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 {
      // x is 0
      return 0
    }
    
    // x is not special and is not zero
    // q = nr. of decimal digits in x (1 <= q <= 7)
    //  determine first the nr. of bits in x
    let q : Int = Self.digitsIn(C1)
    let exp = Int(x_exp) - Self.exponentBias // unbiased exponent
    
    if (q + exp) > 20 { // x >= 10^20 ~= 2^66.45... (cannot fit in 64 bits)
      // set invalid flag
      // pfpsc.insert(.invalidOperation)
      
      // return Integer Indefinite
      return UInt64(bitPattern: Int64.min)
    } else if (q + exp) == 20 { // x = c(0)c(1)...c(q-1)00...0 (20 dec. digits)
      // in this case 2^63.11... ~= 10^19 <= x < 10^20 ~= 2^66.43...
      // so x rounded to an integer may or may not fit in an unsigned 64-bit int
      // the cases that do not fit are identified here; the ones that fit
      // fall through and will be handled with other cases further,
      // under '1 <= q + exp <= 20'
      if x_sign == .minus {
        // if n < 0 and q + exp = 20 then x is much less than -1
        // set invalid flag
        // pfpsc.insert(.invalidOperation)
        
        // return Integer Indefinite
        return UInt64(bitPattern: Int64.min)
      } else { // if n > 0 and q + exp = 20
        // if n >= 2^64 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[20 dec. digits] >= 2^64
        // <=> 0.c(0)c(1)...c(q-1) * 10^21 >= 5*(2^65)
        // <=> C * 10^(21-q) >= 0xa0000000000000000, 1<=q<=7
        var C = UInt128()
        if q == 1 {
          // C * 10^20 >= 0xa0000000000000000
          C = Self.mul128x64to128(UInt64(C1), Self.bid_ten2k128(0)) //10^20 * C
          if C.components.high >= 0x0a {
            // actually C.w[1] == 0x0a && C.w[0] >= 0x0000000000000000ull) {
            // set invalid flag
            // pfpsc.insert(.invalidOperation)
            
            // return Integer Indefinite
            return UInt64(bitPattern: Int64.min)
          }
          // else cases that can be rounded to a 64-bit int fall through
          // to '1 <= q + exp <= 20'
        } else { // if (2 <= q <= 7) => 14 <= 21 - q <= 19
          // Note: C * 10^(21-q) has 20 or 21 digits; 0xa0000000000000000
          // has 21 digits
          C = Self.mul64x64to128(UInt64(C1), Self.bid_ten2k64(21 - q))
          if C.components.high >= 0x0a {
            // actually C.w[1] == 0x0a && C.w[0] >= 0x0000000000000000ull) {
            // set invalid flag
            // pfpsc.insert(.invalidOperation)
            
            // return Integer Indefinite
            return UInt64(bitPattern: Int64.min)
          }
          // else cases that can be rounded to a 64-bit int fall through
          // to '1 <= q + exp <= 20'
        }
      }
    }
    
    // n is not too large to be converted to int64 if -1 < n < 2^64
    // Note: some of the cases tested for above fall through to this point
    var res = UInt64()
    if (q + exp) <= 0 { // n = +/-0.[0...0]c(0)c(1)...c(q-1)
      // return 0
      return 0
    } else { // if (1 <= q + exp <= 20, 1 <= q <= 7, -6 <= exp <= 19)
      // x <= -1 or 1 <= x < 2^64 so if positive x can be rounded
      // to nearest to a 64-bit unsigned signed integer
      if x_sign == .minus { // x <= -1
        // set invalid flag
        // pfpsc.insert(.invalidOperation)
        
        // return Integer Indefinite
        return UInt64(bitPattern: Int64.min)
      }
      // 1 <= x < 2^64 so x can be rounded
      // to nearest to a 64-bit unsigned integer
      if exp < 0 { // 2 <= q <= 7, -6 <= exp <= -1, 1 <= q + exp <= 6
        let ind = -exp; // 1 <= ind <= 6; ind is a synonym for 'x'
        // chop off ind digits from the lower part of C1
        // C1 fits in 64 bits
        // calculate C* and f*
        // C* is actually floor(C*) in this case
        // C* and f* need shifting and masking, as shown by
        // bid_shiftright128[] and bid_maskhigh128[]
        // 1 <= x <= 6
        // kx = 10^(-x) = bid_ten2mk64[ind - 1]
        // C* = C1 * 10^(-x)
        // the approximation of 10^(-x) was rounded up to 54 bits
        let P128 = Self.mul64x64to128(UInt64(C1), Self.bid_ten2mk64(ind - 1))
        var Cstar = P128.components.high
        
        // the top Ex bits of 10^(-x) are T* = bid_ten2mk128trunc[ind].w[0],
        // e.g. if x=1, T*=bid_ten2mk128trunc[0].w[0]=0x1999999999999999
        // C* = floor(C*) (logical right shift; C has p decimal digits,
        //     correct by Property 1)
        // n = C* * 10^(e+x)
        
        // shift right C* by Ex-64 = bid_shiftright128[ind]
        let shift = Self.bid_shiftright128[ind - 1] // 0 <= shift <= 39
        Cstar = Cstar >> shift
        res = Cstar // the result is positive
      } else if exp == 0 {
        // 1 <= q <= 10
        // res = +C (exact)
        res = UInt64(C1) // the result is positive
      } else { // if (exp > 0) => 1 <= exp <= 9, 1 <= q < 9, 2 <= q + exp <= 10
        // res = +C * 10^exp (exact)
        res = UInt64(C1) * Self.bid_ten2k64(exp) // the result is positive
      }
    }
    return UInt64(res)
  }
  
  static func bid(from x:UInt64, _ rndMode:Rounding) -> Self {
    // Get BID from a 64-bit unsigned integer
    if x <= Self.largestNumber { // x <= 10^7-1 and the result is exact
      return Self(sign: .plus, exponent: exponentBias, mantissa: Mantissa(x))
    } else { // x >= 10^7 and the result may be inexact
      // smallest x is 10^7 which has 8 decimal digits
      // largest x is 0xffffffffffffffff = 18446744073709551615 w/ 20 digits
      var q, ind : Int // number of digits to remove for q
      switch x {
        case 0..<100_000_000:                q =  8; ind =  1
        case  ..<1_000_000_000:              q =  9; ind =  2
        case  ..<10_000_000_000:             q = 10; ind =  3
        case  ..<100_000_000_000:            q = 11; ind =  4
        case  ..<1_000_000_000_000:          q = 12; ind =  5
        case  ..<10_000_000_000_000:         q = 13; ind =  6
        case  ..<100_000_000_000_000:        q = 14; ind =  7
        case  ..<1_000_000_000_000_000:      q = 15; ind =  8
        case  ..<10_000_000_000_000_000:     q = 16; ind =  9
        case  ..<100_000_000_000_000_000:    q = 17; ind = 10
        case  ..<1_000_000_000_000_000_000:  q = 18; ind = 11
        case  ..<10_000_000_000_000_000_000: q = 19; ind = 12
        default:                             q = 20; ind = 13
      }
      
      // overflow and underflow are not possible
      // Note: performance can be improved by inlining this call
      var is_midpoint_lt_even = false, is_midpoint_gt_even = false
      var is_inexact_lt_midpoint = false, is_inexact_gt_midpoint = false
      var res64 = UInt64(), res128 = UInt128(), incr_exp = 0
      var res: Mantissa
      if q <= 19 {
        bid_round64_2_18 ( // would work for 20 digits too if x fits in 64 bits
          q, ind, x, &res64, &incr_exp,
          &is_midpoint_lt_even, &is_midpoint_gt_even,
          &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = Mantissa(res64)
      } else { // q = 20
        let x128 = UInt128(high: 0, low:x)
        bid_round128_19_38 (q, ind, x128, &res128, &incr_exp,
                            &is_midpoint_lt_even, &is_midpoint_gt_even,
                            &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = Mantissa(res128._lowWord) // res.w[1] is 0
      }
      if incr_exp != 0 {
        ind += 1
      }
      // set the inexact flag
      //      if (is_inexact_lt_midpoint || is_inexact_gt_midpoint ||
      //          is_midpoint_lt_even || is_midpoint_gt_even)
      //          *pfpsf |= BID_INEXACT_EXCEPTION;
      // general correction from RN to RA, RM, RP, RZ; result uses ind for exp
      if rndMode != .toNearestOrEven {
        if ((rndMode == .up && is_inexact_lt_midpoint) ||
           ((rndMode == .toNearestOrAwayFromZero || rndMode == .up)
             && is_midpoint_gt_even)) {
          res = res + 1
          if res == largestNumber+1 { // res = 10^7 => rounding overflow
            res = (largestNumber+1)/10 // 10^6
            ind = ind + 1
          }
        } else if (is_midpoint_lt_even || is_inexact_gt_midpoint) &&
                   (rndMode == .down || rndMode == .towardZero) {
          res = res - 1
          // check if we crossed into the lower decade
          if res == largestNumber/10 { // 10^6 - 1
            res = largestNumber // 10^7 - 1
            ind = ind - 1
          }
        } else {
          // exact, the result is already correct
        }
      }
      return Self(sign: .plus, exponent: ind, mantissa: Mantissa(res))
    }
  }
  
  static func handleRounding(_ s:Sign, _ exp:Int, _ c:Int,
                             _ R: Int = 0, _ r:Rounding) -> Self {
    var r = r
    if exp < 0 {
      if exp + maximumDigits < 0 {
        //fpsc.formUnion([.underflow, .inexact])
        if r == .down && s != .plus {
          // 0x8000_0001
          return Self(sign: .minus, exponent: exponentBias, mantissa: 1)
        }
        if r == .up && s == .plus {
          return Self(sign: .plus, exponent: exponentBias, mantissa: 1)
        }
        if exp < 0 { return Self(sign: s, exponent: 0, mantissa: 0) }
        return Self(sign: s, exponent: exponentBias, mantissa: 0)
      }
      
      // swap round modes when negative
      if s != .plus {
        if r == .up { r = .down }
        else if r == .down { r = .up }
      }
      
      // determine the rounding table index
      let roundIndex = roundboundIndex(r) >> 2
      
      // 10*coeff
      var c = (c << 3) + (c << 1)
      if R != 0 {
        c |= 1
      }
      
      // get digits to be shifted out
      let extra_digits = 1-exp
      c += Int(bid_round_const_table(roundIndex, extra_digits))
      
      // get coeff*(2^M[extra_digits])/10^extra_digits
      let Q = mul64x64to128(UInt64(c), reciprocals10(extra_digits))
      
      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      let amount = shortReciprocalScale[extra_digits]
      
      var _C64 = Q.components.high >> amount
      var remainder_h = UInt64(0)
      
      if r == .toNearestOrAwayFromZero {
        if _C64 & 1 != 0 {
          // check whether fractional part of initial_P/10^extra_digits
          // is exactly .5
          
          // get remainder
          let amount2 = 64 - amount
          remainder_h = 0
          remainder_h &-= 1
          remainder_h >>= amount2
          remainder_h = remainder_h & Q.components.high
          
          if remainder_h == 0 &&
              Q.components.low < reciprocals10(extra_digits) {
            _C64 -= 1
          }
        }
      }
      
//      if fpsc.contains(.inexact) {
//        fpsc.insert(.underflow)
//      } else {
//        var status = Status.inexact
//        // get remainder
//        remainder_h = Q.high << (64 - amount)
//
//        switch r {
//          case .toNearestOrAwayFromZero, .toNearestOrEven:
//            // test whether fractional part is 0
//            if (remainder_h == (UInt64(SIGN_MASK) << 32) && (Q.low <
//                                        bid_reciprocals10_64(extra_digits))) {
//              status = Status.clearFlags
//            }
//          case .down, .towardZero:
//            if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
//              status = Status.clearFlags
//            }
//          default:
//            // round up
//            var Stemp = UInt64(0), carry = UInt64(0)
//            __add_carry_out(&Stemp, &carry, Q.low,
//                            bid_reciprocals10_64(extra_digits))
//            if (remainder_h >> (64 - amount)) + carry >= UInt64(1) << amount {
//              status = Status.clearFlags
//            }
//        }
//
//        if !status.isEmpty {
//          status.insert(.underflow)
//          fpsc.formUnion(status)
//        }
//      }
      return Self(sign: s, exponent: exponentBias, mantissa: Mantissa(_C64))
    }
    var exp = exp, c = c
    if c == 0 { if exp > maximumExponent { exp = maximumExponent } }
    while c < (Self.largestNumber+1)/10 && exp > maximumExponent {
      c = (c << 3) + (c << 1)
      exp -= 1
    }
    if UInt32(exp) > maximumExponent {
      // let s = (Word(s) << signBit)
      // fpsc.formUnion([.overflow, .inexact])
      // overflow
      var res = Self.infinite(s)
      switch r {
        case .down:
          if s == .plus {
            res = largestBID
          }
        case .towardZero:
          res = largestBID; res.sign = s
        case .up:
          // round up
          if s != .plus {
            res = largestBID; res.sign = s
          }
        default: break
      }
      return res
    }
    return Self(RawDataFields(c))
  }
  
  
  static func bid_round64_2_18 (
    _ q: Int, _ x:Int, _ C: UInt64, _ ptr_Cstar: inout UInt64,
    _ incr_exp: inout Int, _ ptr_is_midpoint_lt_even: inout Bool,
    _ ptr_is_midpoint_gt_even: inout Bool,
    _ ptr_is_inexact_lt_midpoint: inout Bool,
    _ ptr_is_inexact_gt_midpoint: inout Bool) {
      // Note:
      //    In round128_2_18() positive numbers with 2 <= q <= 18 will be
      //    rounded to nearest only for 1 <= x <= 3:
      //     x = 1 or x = 2 when q = 17
      //     x = 2 or x = 3 when q = 18
      // However, for generality and possible uses outside the IEEE 754 frame,
      // this implementation works for 1 <= x <= q - 1
      
      // assume *ptr_is_midpoint_lt_even, *ptr_is_midpoint_gt_even,
      // *ptr_is_inexact_lt_midpoint, and *ptr_is_inexact_gt_midpoint are
      // initialized to 0 by the caller
      
      // round a number C with q decimal digits, 2 <= q <= 18
      // to q - x digits, 1 <= x <= 17
      // C = C + 1/2 * 10^x where the result C fits in 64 bits
      // (because the largest value is 999999999999999999 + 50000000000000000 =
      // 0x0e92596fd628ffff, which fits in 60 bits)
      var ind = x - 1;    // 0 <= ind <= 16
      let C = C + bid_midpoint64(ind)
      // kx ~= 10^(-x), kx = bid_Kx64[ind] * 2^(-Ex), 0 <= ind <= 16
      // P128 = (C + 1/2 * 10^x) * kx * 2^Ex = (C + 1/2 * 10^x) * Kx
      // the approximation kx of 10^(-x) was rounded up to 64 bits
      var P128 = UInt128().components
      P128 = mul64x64to128(C, bid_Kx64(ind)).components
      // calculate C* = floor (P128) and f*
      // Cstar = P128 >> Ex
      // fstar = low Ex bits of P128
      let shift = bid_Ex64m64(ind)    // in [3, 56]
      var Cstar = P128.high >> shift
      let fstar = UInt128(high: P128.high & bid_mask64(ind), low: P128.low)
      // the top Ex bits of 10^(-x) are T* = bid_ten2mxtrunc64[ind], e.g.
      // if x=1, T*=bid_ten2mxtrunc64[0]=0xcccccccccccccccc
      // if (0 < f* < 10^(-x)) then the result is a midpoint
      //   if floor(C*) is even then C* = floor(C*) - logical right
      //       shift; C* has q - x decimal digits, correct by Prop. 1)
      //   else if floor(C*) is odd C* = floor(C*)-1 (logical right
      //       shift; C* has q - x decimal digits, correct by Pr. 1)
      // else
      //   C* = floor(C*) (logical right shift; C has q - x decimal digits,
      //       correct by Property 1)
      // in the caling function n = C* * 10^(e+x)
      
      // determine inexactness of the rounding of C*
      // if (0 < f* - 1/2 < 10^(-x)) then
      //   the result is exact
      // else // if (f* - 1/2 > T*) then
      //   the result is inexact
      if (fstar.components.high > bid_half(ind) ||
          (fstar.components.high == bid_half(ind) &&
           fstar.components.low != 0)) {
        // f* > 1/2 and the result may be exact
        // Calculate f* - 1/2
        let tmp64 = fstar.components.high - bid_half(ind)
        if (tmp64 != 0 || fstar.components.low > bid_ten2mxtrunc64(ind)) {
          // f* - 1/2 > 10^(-x)
          ptr_is_inexact_lt_midpoint = true
        }    // else the result is exact
      } else {    // the result is inexact; f2* <= 1/2
        ptr_is_inexact_gt_midpoint = true
      }
      // check for midpoints (could do this before determining inexactness)
      if fstar <= bid_ten2mxtrunc64(ind) {
        // the result is a midpoint
        if Cstar & 0x01 != 0 {    // Cstar is odd; MP in [EVEN, ODD]
          // if floor(C*) is odd C = floor(C*) - 1; the result may be 0
          Cstar-=1    // Cstar is now even
          ptr_is_midpoint_gt_even = true
          ptr_is_inexact_lt_midpoint = false
          ptr_is_inexact_gt_midpoint = false
        } else {    // else MP in [ODD, EVEN]
          ptr_is_midpoint_lt_even = true
          ptr_is_inexact_lt_midpoint = false
          ptr_is_inexact_gt_midpoint = false
        }
      }
      // check for rounding overflow, which occurs if Cstar = 10^(q-x)
      ind = q - x;    // 1 <= ind <= q - 1
      if (Cstar == bid_ten2k64(ind)) {    // if  Cstar = 10^(q-x)
        Cstar = bid_ten2k64(ind - 1);    // Cstar = 10^(q-x-1)
        incr_exp = 1;
      } else {    // 10^33 <= Cstar <= 10^34 - 1
        incr_exp = 0;
      }
      ptr_Cstar = Cstar;
    }
  
  static func bid_round128_19_38 (
    _ q:Int, _ x:Int, _ C:UInt128, _ ptr_Cstar:inout UInt128,
    _ incr_exp:inout Int, _ ptr_is_midpoint_lt_even:inout Bool,
    _ ptr_is_midpoint_gt_even:inout Bool,
    _ ptr_is_inexact_lt_midpoint:inout Bool,
    _ ptr_is_inexact_gt_midpoint:inout Bool) {
      var P256 = UInt256(), fstar = UInt256(), Cstar = UInt128(),
          C = C.components
      // Note:
      //    In bid_round128_19_38() positive numbers with 19 <= q <= 38 will be
      //    rounded to nearest only for 1 <= x <= 23:
      //     x = 3 or x = 4 when q = 19
      //     x = 4 or x = 5 when q = 20
      //     ...
      //     x = 18 or x = 19 when q = 34
      //     x = 1 or x = 2 or x = 19 or x = 20 when q = 35
      //     x = 2 or x = 3 or x = 20 or x = 21 when q = 36
      //     x = 3 or x = 4 or x = 21 or x = 22 when q = 37
      //     x = 4 or x = 5 or x = 22 or x = 23 when q = 38
      // However, for generality and possible uses outside the frame of IEEE 754
      // this implementation works for 1 <= x <= q - 1
      
      // assume ptr_is_midpoint_lt_even, ptr_is_midpoint_gt_even,
      // ptr_is_inexact_lt_midpoint, and ptr_is_inexact_gt_midpoint are
      // initialized to 0 by the caller
      
      // round a number C with q decimal digits, 19 <= q <= 38
      // to q - x digits, 1 <= x <= 37
      // C = C + 1/2 * 10^x where the result C fits in 128 bits
      // (because the largest value is 99999999999999999999999999999999999999+
      // 5000000000000000000000000000000000000 =
      // 0x4efe43b0c573e7e68a043d8fffffffff, which fits is 127 bits)
      var ind = x - 1;    // 0 <= ind <= 36
      if (ind <= 18) {    // if 0 <= ind <= 18
        let tmp64 = C.low
        C.low = C.low + bid_midpoint64(ind)
        if (C.low < tmp64) { C.high+=1 }
      } else {    // if 19 <= ind <= 37
        let tmp64 = C.low
        C.low = C.low + bid_midpoint128(ind - 19).components.low
        if (C.low < tmp64) { C.high+=1 }
        C.high = C.high + bid_midpoint128(ind - 19).components.high
      }
      // kx ~= 10^(-x), kx = bid_Kx128[ind] * 2^(-Ex), 0 <= ind <= 36
      // P256 = (C + 1/2 * 10^x) * kx * 2^Ex = (C + 1/2 * 10^x) * Kx
      // the approximation kx of 10^(-x) was rounded up to 128 bits
      mul128x128to256(&P256, UInt128(high: C.high, low: C.low), bid_Kx128(ind))
      // calculate C* = floor (P256) and f*
      // Cstar = P256 >> Ex
      // fstar = low Ex bits of P256
      let shift = bid_Ex128m128[ind] // in [2, 63] but must consider two cases
      if ind <= 18 {    // if 0 <= ind <= 18
        Cstar = UInt128(high:P256.w[3] >> shift,
                        low:(P256.w[2] >> shift) | (P256.w[3] << (64 - shift)))
        
        fstar.w[0] = P256.w[0];
        fstar.w[1] = P256.w[1];
        fstar.w[2] = P256.w[2] & bid_mask128(ind)
        fstar.w[3] = 0x0;
      } else {    // if 19 <= ind <= 37
        Cstar = UInt128(high: 0, low: P256.w[3] >> shift)
        fstar.w[0] = P256.w[0];
        fstar.w[1] = P256.w[1];
        fstar.w[2] = P256.w[2];
        fstar.w[3] = P256.w[3] & bid_mask128(ind)
      }
      // the top Ex bits of 10^(-x) are T* = bid_ten2mxtrunc64[ind], e.g.
      // if x=1, T*=bid_ten2mxtrunc128[0]=0xcccccccccccccccccccccccccccccccc
      // if (0 < f* < 10^(-x)) then the result is a midpoint
      //   if floor(C*) is even then C* = floor(C*) - logical right
      //       shift; C* has q - x decimal digits, correct by Prop. 1)
      //   else if floor(C*) is odd C* = floor(C*)-1 (logical right
      //       shift; C* has q - x decimal digits, correct by Pr. 1)
      // else
      //   C* = floor(C*) (logical right shift; C has q - x decimal digits,
      //       correct by Property 1)
      // in the caling function n = C* * 10^(e+x)
      
      // determine inexactness of the rounding of C*
      // if (0 < f* - 1/2 < 10^(-x)) then
      //   the result is exact
      // else // if (f* - 1/2 > T*) then
      //   the result is inexact
      if ind <= 18 {    // if 0 <= ind <= 18
        if (fstar.w[2] > bid_half(ind) || (fstar.w[2] == bid_half(ind) &&
                                  (fstar.w[1] != 0 || fstar.w[0] != 0))) {
          // f* > 1/2 and the result may be exact
          // Calculate f* - 1/2
          let tmp64 = fstar.w[2] - bid_half(ind)
          let ten2mx = bid_ten2mxtrunc128(ind).components
          if (tmp64 != 0 || fstar.w[1] > ten2mx.high ||
              (fstar.w[1] == ten2mx.high &&
               fstar.w[0] > ten2mx.low)) {    // f* - 1/2 > 10^(-x)
            ptr_is_inexact_lt_midpoint = true
          }    // else the result is exact
        } else {    // the result is inexact; f2* <= 1/2
          ptr_is_inexact_gt_midpoint = true
        }
      } else {    // if 19 <= ind <= 37
        if (fstar.w[3] > bid_half(ind) || (fstar.w[3] == bid_half(ind) &&
                (fstar.w[2] != 0 || fstar.w[1] != 0 || fstar.w[0] != 0))) {
          // f* > 1/2 and the result may be exact
          // Calculate f* - 1/2
          let tmp64 = fstar.w[3] - bid_half(ind)
          let ten2mx = bid_ten2mxtrunc128(ind).components
          if (tmp64 != 0 || fstar.w[2] != 0 ||
              fstar.w[1] > ten2mx.high ||
              (fstar.w[1] == ten2mx.high &&
               fstar.w[0] > ten2mx.low)) {    // f* - 1/2 > 10^(-x)
            ptr_is_inexact_lt_midpoint = true
          }    // else the result is exact
        } else {    // the result is inexact; f2* <= 1/2
          ptr_is_inexact_gt_midpoint = true
        }
      }
      // check for midpoints (could do this before determining inexactness)
      let ten2mx = bid_ten2mxtrunc128(ind).components
      if (fstar.w[3] == 0 && fstar.w[2] == 0 &&
          (fstar.w[1] < ten2mx.high || (fstar.w[1] == ten2mx.high &&
            fstar.w[0] <= ten2mx.low))) {
        // the result is a midpoint
        if (Cstar.components.low & 0x01 != 0) {
          // Cstar is odd; MP in [EVEN, ODD]
          // if floor(C*) is odd C = floor(C*) - 1; the result may be 0
          Cstar-=1    // Cstar is now even
          if Cstar.components.low == 0xffff_ffff_ffff_ffff {
            Cstar -= UInt128(high: 1, low: 0)
          }
          ptr_is_midpoint_gt_even = true
          ptr_is_inexact_lt_midpoint = false
          ptr_is_inexact_gt_midpoint = false
        } else {    // else MP in [ODD, EVEN]
          ptr_is_midpoint_lt_even = true
          ptr_is_inexact_lt_midpoint = false
          ptr_is_inexact_gt_midpoint = false
        }
      }
      // check for rounding overflow, which occurs if Cstar = 10^(q-x)
      ind = q - x    // 1 <= ind <= q - 1
      if ind <= 19 {
        if Cstar == bid_ten2k64(ind) {
          // if  Cstar = 10^(q-x)
          Cstar.components.low = bid_ten2k64(ind - 1)    // Cstar = 10^(q-x-1)
          incr_exp = 1;
        } else {
          incr_exp = 0;
        }
      } else if (ind == 20) {
        // if ind = 20
        if Cstar == bid_ten2k128(0) {
          // if  Cstar = 10^(q-x)
          Cstar = UInt128(high: 0, low: bid_ten2k64(19)) // Cstar = 10^(q-x-1)
          incr_exp = 1
        } else {
          incr_exp = 0
        }
      } else { // if 21 <= ind <= 37
        if Cstar == bid_ten2k128(ind - 20) { // if  Cstar = 10^(q-x)
          Cstar = bid_ten2k128(ind - 21) // Cstar = 10^(q-x-1)
          incr_exp = 1;
        } else {
          incr_exp = 0;
        }
      }
      ptr_Cstar = Cstar
    }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Table-derived functions
  
  /// Returns the number of decimal digits in `2^i` where `i ≥ 0`.
  static func estimateDecDigits(_ i: Int) -> Int { digitsIn(UInt128(1) << i) }
  
  /// Returns ten to the `i`th power or `10^i` where `i ≥ 0`.
  static func power10<T:FixedWidthInteger>(_ i:Int) -> T { power(T(10), to:i) }
  
  /// Returns ten to the `i`th power or `10^i` where `i ≥ 0`.
  static func power5<T:FixedWidthInteger>(_ i: Int) -> T { power(T(5), to: i) }
  
  /// Returns the number of decimal digits in `sig`.
  static func digitsIn<T:BinaryInteger>(_ sig: T) -> Int {
    return digitsIn(sig).digits
  }
  
  /// Returns the number of decimal digits and power of 10 in `sig`.
  static func digitsIn<T:BinaryInteger>(_ sig: T) -> (digits:Int, tenPower:T) {
    // find power of 10 just greater than sig
    var tenPower = T(10), digits = 1
    while sig >= tenPower { tenPower *= 10; digits += 1 }
    return (digits, tenPower)
  }
  
  // bid_ten2mk64 power-of-two scaling
  static var bid_powers : [UInt8] {
    [64, 64, 64, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97, 100, 103, 107]
  }
  
  // bid_ten2k64[i] = 10^i, 0 <= i <= 19
  static func bid_ten2k64<T:FixedWidthInteger>(_ i:Int) -> T { power10(i) }
  
  // Values of 10^(-x) trancated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static func bid_ten2mk64(_ i:Int) -> UInt64 {
     UInt64((UInt128(1) << bid_powers[i]) / power10(i+1))+1
  }
  
  static func bid_midpoint64(_ i:Int) -> UInt64 { 5 * power10(i) }
  
  // bid_midpoint128[i - 20] = 1/2 * 10^i = 5 * 10^(i-1), 20 <= i <= 38
  static func bid_midpoint128(_ i:Int) -> UInt128 { 5 * power10(i) }
  
  /// Returns 10^n such that 2^i < 10^n
  static func bid_power10_index_binexp(_ i:Int) -> UInt64 {
    digitsIn(UInt64(1) << i).tenPower
  }
    
  /// Returns rounding constants for a given rounding mode `rnd` and
  /// power of ten given by `10^(i-1)`.
  static func bid_round_const_table(_ rnd:Int, _ i:Int) -> UInt64 {
    if i == 0 { return 0 }
    switch rnd {
      case 0, 4: return 5 * power10(i-1)
      case 2: return power10(i-1)-1
      default: return 0 // covers rnd = 1, 3
    }
  }
  
  static var coefflimits: [UInt64] {
    [10000000, 2000000, 400000, 80000, 16000, 3200, 640, 128, 25, 5, 1]
  }
  
  static func coefflimitsBID32(_ i: Int) -> UInt128 {
    i > 10 ? 0 : UInt128(coefflimits[i])
  }
  
  static var shortReciprocalScale: [Int8] {
    [1, 1, 5, 7, 11, 12, 17, 21, 24, 27, 31, 34, 37, 41, 44, 47, 51, 54]
  }
  
  // Values of 10^(-x) truncated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static func bid_ten2mxtrunc64(_ i:Int) -> UInt64 {
    UInt64((UInt128(1) << (64+bid_Ex64m64(i))) / power10(i+1))
  }
  
  // Values of 10^(-x) truncated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 37; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 128 bits of the fraction)
  static func bid_ten2mxtrunc128(_ i:Int) -> UInt128 {
    (UInt128(1) << (64+bid_Ex128m128[i])) / power10(i+1)
  }
  
  // Ex-64 from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static func bid_Ex64m64(_ i:Int) -> UInt8 { bid_Ex128m128[i] }
  
  // bid_ten2k128[i - 20] = 10^i, 20 <= i <= 38
  static func bid_ten2k128(_ i:Int) -> UInt128 { power10(i+20) }
  
  // Values of 1/2 in the right position to be compared with the fraction from
  // C * kx, 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the high 64 bits of the fraction)
  static func bid_half<T:UnsignedInteger>(_ i: Int) -> T {
    (T(1) << bid_shiftright128[i+3] - 1)
  }
  
  // Values of mask in the right position to obtain the high Ex - 64 bits
  // of the fraction from C * kx, 1 <= x <= 17; the fraction consists of
  // the low Ex bits in C * kx
  static func bid_mask64(_ i:Int) -> UInt64 {
    (UInt64(1) << bid_shiftright128[i+3]) - 1
  }
  
  // Kx from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static func bid_Kx64(_ i:Int) -> UInt64 { bid_ten2mxtrunc64(i)+1 }
                      
  // Kx from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 128 bits, 1 <= x <= 37
  static func bid_Kx128(_ i:Int) -> UInt128 { bid_ten2mxtrunc128(i)+1 }
  
  /// Returns `2^s[i] / 10^i + 1` where `s` is a table of
  /// reciprocal scaling factors and `i ≥ 0`.
  static func reciprocals10<T:UnsignedInteger>(_ i: Int) -> T {
    if i == 0 { return 1 }
    let twoPower = shortReciprocalScale[i] + 64
    return T(UInt128(1) << twoPower / power10(i)) + 1
  }
  
  static func mul128x128Low(_ A:UInt128, _ B:UInt128) -> UInt128 {
    // var ALBL = UInt128()
    let a = A.components, b = B.components
    let ALBL = mul64x64to128(a.low, b.low)
    let QM64 = b.low*a.high + a.low*b.high
    return UInt128(high: QM64 + ALBL.components.high, low: ALBL.components.low)
  }
  
  static func mul64x64to128(_ CX:UInt64, _ CY:UInt64) -> UInt128 {
    let res = CX.multipliedFullWidth(by: CY)
    return UInt128(high: res.high, low: res.low)
  }
  
  static func mul128x64to128(_ A64:UInt64, _ B128:UInt128) -> UInt128 {
    let ALBH_L = A64 * B128.components.high
    var Q128 = mul64x64to128(A64, B128.components.low)
    Q128.components.high += ALBH_L
    return Q128
  }
  
  static internal func mul64x256to320(_ P:inout UInt384, _ A:UInt64,
                                      _ B:UInt256) {
    var lC = false
    let lP0 = mul64x64to128(A, B.w[0]).components
    let lP1 = mul64x64to128(A, B.w[1]).components
    let lP2 = mul64x64to128(A, B.w[2]).components
    let lP3 = mul64x64to128(A, B.w[3]).components
    P.w[0] = lP0.low
    (P.w[1],lC) = add(lP1.low,lP0.high)
    (P.w[2],lC) = add(lP2.low,lP1.high,lC)
    (P.w[3],lC) = add(lP3.low,lP2.high,lC)
    P.w[4] = lP3.high + (lC ? 1 : 0)
  }
  
  static internal func mul128x128to256(_ P256: inout UInt256, _ A:UInt128,
                                       _ B:UInt128) {
//      var Qll = UInt128(), Qlh = UInt128()
//      var Phl = UInt64(), Phh = UInt64(), CY1 = UInt64(), CY2 = UInt64()
//
    let (hi, lo) = A.multipliedFullWidth(by: B)
//
//      __mul_64x128_full(&Phl, &Qll, A.lo, B)
//      __mul_64x128_full(&Phh, &Qlh, A.hi, B)
    P256.w[0] = lo.components.low
    P256.w[1] = lo.components.high
    P256.w[2] = hi.components.low
    P256.w[3] = hi.components.high
//
//      __add_carry_out(&P256.w[1], &CY1, Qlh.lo, Qll.hi)
//      __add_carry_in_out(&P256.w[2], &CY2, Qlh.hi, Phl, CY1)
//      P256.w[3] = Phh + CY2
  }
  
  // 128x256->384 bit multiplication (missing from existing macros)
  // I derived this by propagating (A).w[2] = 0 in __mul_192x256_to_448
  static internal func mul128x256to384(_  P: inout UInt384, _ A:UInt128,
                                       _ B:UInt256) {
    var P0=UInt384(),P1=UInt384()
    var CY=false
    mul64x256to320(&P0, A.components.low, B)
    mul64x256to320(&P1, A.components.high, B)
    P.w[0] = P0.w[0]
    (P.w[1],CY) = add(P1.w[0],P0.w[1])
    (P.w[2],CY) = add(P1.w[1],P0.w[2],CY)
    (P.w[3],CY) = add(P1.w[2],P0.w[3],CY)
    (P.w[4],CY) = add(P1.w[3],P0.w[4],CY)
    P.w[5] = P1.w[4] + (CY ? 1 : 0)
  }
  
  // bid_shiftright128[] contains the right shift count to obtain C2* from
  // the top 128 bits of the 128x128-bit product C2 * Kx
  static var bid_shiftright128: [UInt8] {
    [ 0, 0, 0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56,
      59, 63, 66, 69, 73, 76, 79, 83, 86, 89, 92, 96, 99, 102
    ]
  }
  
  // Ex-128 from 10^(-x) ~= Kx*2^(-Ex); Kx rounded up to 128 bits, 1 <= x <= 37
  static var bid_Ex128m128: [UInt8] {
    [
     3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56, 59, 63,
     2, 5, 9, 12, 15, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 52, 55, 58
   ]
  }
  
  // bid_maskhigh128[] contains the mask to apply to the top 128 bits of the
  // 128x128-bit product in order to obtain the high bits of f2*
  // the 64-bit word order is L, H
  static func bid_maskhigh128(_ i:Int) -> UInt64 {
    if i < 3 { return 0 }
    return (UInt64(1) << bid_Ex128m128[i-3]) - 1
  }
  
  // Values of mask in the right position to obtain the high Ex - 128 or
  // Ex - 192 bits of the fraction from C * kx, 1 <= x <= 37; the fraction
  // consists of the low Ex bits in C * kx
  static func bid_mask128(_ i: Int) -> UInt64 {
    (UInt64(1) << bid_Ex128m128[i-1]) - 1
  }
  
  @inlinable static func add(_ X:UInt64, _ Y:UInt64) -> (UInt64, Bool) {
    X.addingReportingOverflow(Y)
  }

  @inlinable static func add(_ X:UInt64, _ Y:UInt64, _ CI:Bool) ->
                                                            (UInt64, Bool)  {
    let (x1, over1) = X.addingReportingOverflow(CI ? 1 : 0)
    let (s , over2) = x1.addingReportingOverflow(Y)
    return (s, over1 || over2)
  }
  
  static var bid_roundbound_128: [UInt128] {
    let midPoint = UInt128(high: 1 << 63, low: 0)
    return [
      // .toNearestOrEven
      midPoint,      // positive|even
      midPoint - 1,  // positive|odd
      midPoint,      // negative|even
      midPoint - 1,  // negative|odd
      
      // .down
      UInt128.max,   // positive|even
      UInt128.max,   // positive|odd
      UInt128.min,   // negative|even
      UInt128.min,   // negative|odd
      
      // .up
      UInt128.min,   // positive|even
      UInt128.min,   // positive|odd
      UInt128.max,   // negative|even
      UInt128.max,   // negative|odd
      
      // .towarZero
      UInt128.max,   // positive|even
      UInt128.max,   // positive|odd
      UInt128.max,   // negative|even
      UInt128.max,   // negative|odd
      
      // .toNearestOrAwayFromZero
      midPoint - 1,  // positive|even
      midPoint - 1,  // positive|odd
      midPoint - 1,  // negative|even
      midPoint - 1   // negative|odd
    ]
  }
  
  /// Algorithm to decode a 10-bit portion of a densely-packed decimal
  /// number into a corresponding integer. The strategy here is the break
  /// the decoding process into a number of smaller code spaces used to
  /// calculate the corresponding integral number.
  ///
  /// This algorithm may be sped up by replacement with a table lookup as in
  /// the original code. Tests have verified that this algorithm exactly
  /// reproduces the original table.
  static func intFrom(dpd: Int) -> Int {
    precondition(dpd >= 0 && dpd < 1024, "Illegal dpd decoding input")
    func get(_ range: ClosedRange<Int>) -> Int { dpd.get(range: range) }
    
    // decode the 10-bit dpd number
    let select = (dpd.get(bit:3), get(1...2), get(5...6))
    let bit0 = dpd.get(bit:0), bit4 = dpd.get(bit:4), bit7 = dpd.get(bit:7)
    switch select {
      // this case covers about 50% of the numbers
      case (0, _, _):
        return get(7...9)*100 + get(4...6)*10 + get(0...2)
        
      // following 3 cases cover 37.5% of the numbers
      case (1, 0b00, _):
        return get(7...9)*100 + get(4...6)*10 + bit0 + 8
      case (1, 0b01, _):
        return get(7...9)*100 + (bit4 + 8)*10 + get(5...6)<<1 + bit0
      case (1, 0b10, _):
        return (bit7 + 8)*100 + get(4...6)*10 + get(8...9)<<1 + bit0
        
      // next 3 cases cover another 9.375% of the numbers
      case (1, 0b11, 0b00):
        return (bit7 + 8)*100 + (bit4 + 8)*10 + get(8...9)<<1 + bit0
      case (1, 0b11, 0b01):
        return (bit7 + 8)*100 + (get(8...9)<<1 + bit4)*10 + bit0 + 8
      case (1, 0b11, 0b10):
        return get(7...9)*100 + (bit4 + 8)*10 + bit0 + 8
        
      // final case covers remaining 3.125% of the numbers
      default:
        return (bit7 + 8)*100 + (bit4 + 8)*10 + bit0 + 8
    }
  }
  
  /// Algorithm to encode a 12-bit bcd integer (3 digits) into a
  /// densely-packed decimal. This is the inverse of the `intFrom(dpd:)`.
  ///
  /// This algorithm may be sped up by replacement with a table lookup as in
  /// the original code. Tests have verified that this algorithm exactly
  /// reproduces the original code table.
  static func intToDPD(_ n: Int) -> Int {
    precondition(n >= 0 && n < 1000, "Illegal dpd encoding input")
    
    let hundreds = (n / 100) % 10
    let tens = (n / 10) % 10
    let ones = n % 10
    var res = 0
    
    func setBits4to6() { res.set(range:4...6, with: tens) }
    func setBits7to9() { res.set(range:7...9, with: hundreds) }
    
    func setBit0() { res.set(bit:0, with: ones) }
    func setBit4() { res.set(bit:4, with: tens) }
    func setBit7() { res.set(bit:7, with: hundreds) }
  
    switch (hundreds>7, tens>7, ones>7) {
      case (false, false, false):
        setBits7to9(); setBits4to6(); res.set(range: 0...2, with: ones)
      case (false, false, true):
        res = 0b1000  // base encoding
        setBits7to9(); setBits4to6(); setBit0()
      case (false, true, false):
        res = 0b1010  // base encoding
        setBits7to9(); setBit4(); res.set(range:5...6, with:ones>>1); setBit0()
      case (true, false, false):
        res = 0b1100  // base encoding
        setBits4to6(); res.set(range:8...9, with:ones>>1); setBit7(); setBit0()
      case (true, true, false):
        res = 0b1110  // base encoding
        setBit7(); setBit4(); res.set(range:8...9, with:ones>>1); setBit0()
      case (true, false, true):
        res = 0b010_1110  // base encoding
        setBit7(); res.set(range: 8...9, with: tens>>1); setBit4(); setBit0()
      case (false, true, true):
        res = 0b100_1110  // base encoding
        setBits7to9(); setBit4(); setBit0()
      default:
        res = 0b110_1110 // base encoding
        setBit7(); setBit4(); setBit0()
    }
    return res
  }
}

extension IntegerDecimal {
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - General-purpose comparison functions
  public static func equals (lhs: Self, rhs: Self) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    
    // all data bits equsl case
    if lhs.data == rhs.data { return true }
    
    // infinity cases
    if lhs.isInfinite && rhs.isInfinite { return lhs.sign == rhs.sign }
    if lhs.isInfinite || rhs.isInfinite { return false }
    
    // zero cases
    let xisZero = lhs.isZero, yisZero = rhs.isZero
    if xisZero && yisZero { return true }
    if (xisZero && !yisZero) || (!xisZero && yisZero) { return false }
    
    // normal numbers
    var (xsign, xexp, xman, _) = lhs.unpack()
    var (ysign, yexp, yman, _) = rhs.unpack()
    
    // opposite signs
    if xsign != ysign { return false }
    
    // redundant representations
    if xexp > yexp {
      swap(&xexp, &yexp)
      swap(&xman, &yman)
    }
    if yexp - xexp > Self.maximumDigits-1 { return false }
    for _ in 0..<(yexp - xexp) {
      // recalculate y's significand upwards
      yman *= 10
      if yman > Self.largestNumber { return false }
    }
    return xman == yman
  }
  
  public static func greaterOrEqual (lhs: Self, rhs: Self) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    return !lessThan(lhs: lhs, rhs: rhs)
  }
  
  public static func greaterThan (lhs: Self, rhs: Self) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    return !(lessThan(lhs: lhs, rhs: rhs) || equals(lhs: lhs, rhs: rhs))
  }
  
  public static func lessThan (lhs: Self, rhs: Self) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    
    // all data bits equsl case
    if lhs.data == rhs.data { return false }
    
    // infinity cases
    if lhs.isInfinite {
      if lhs.sign == .minus {
        // lhs is -inf, which is less than y unless y is -inf
        return !rhs.isInfinite || rhs.sign == .plus
      } else {
        // lhs is +inf, which can never be less than y
        return false
      }
    } else if rhs.isInfinite {
      // lhs is finite so:
      //   if rhs is +inf, lhs<rhs
      //   if rhs is -inf, lhs>rhs
      return rhs.sign == .plus
    }
    
    // normal numbers
    let (xsign, xexp, xman, _) = lhs.unpack()
    let (ysign, yexp, yman, _) = rhs.unpack()
    
    // zero cases
    let xisZero = lhs.isZero, yisZero = rhs.isZero
    if xisZero && yisZero { return false }
    else if xisZero { return ysign == .plus }  // x < y if y > 0
    else if yisZero { return xsign == .minus } // x < y if y < 0
    
    // opposite signs
    if xsign != ysign { return ysign == .plus } // x < y if y > 0
    
    // check if both mantissa and exponents and bigger or smaller
    if xman > yman && xexp >= yexp { return xsign == .minus }
    if xman < yman && xexp <= yexp { return xsign == .plus }
    
    // if xexp is `numberOfDigits`-1 greater than yexp, no need to continue
    if xexp - yexp > Self.maximumDigits-1 { return xsign == .minus }
    
    // difference cannot be greater than 10^6
    // if exp_x is 6 less than exp_y, no need for compensation
    if yexp - xexp > Self.maximumDigits-1 { return xsign == .plus }
    
    // need to compensate the mantissa
    var manPrime: Self.Mantissa
    if xexp > yexp {
      manPrime = xman * power(Self.Mantissa(10), to: xexp - yexp)
      if manPrime == yman { return false }
      return (manPrime < yman) != (xsign == .minus)
    }
    
    // adjust y mantissa upwards
    manPrime = yman * power(Self.Mantissa(10), to: yexp - xexp)
    if manPrime == xman { return false }
    
    // if positive, return whichever abs number is smaller
    return (xman < manPrime) != (xsign == .minus)
  }
}

extension IntegerDecimal {
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - General-purpose math functions
  static func add(_ x: Self, _ y: Self, rounding: Rounding) -> Self {
    let xb = x, yb = y
    let (signX, exponentX, mantissaX, validX) = xb.unpack()
    let (signY, exponentY, mantissaY, validY) = yb.unpack()
    
    // Deal with illegal numbers
    if !validX {
      if xb.isNaN {
        if xb.isSNaN || yb.isSNaN { /* invalid Op */ }
        return Self(sign: .plus, exponent: 0, mantissa: xb.nanQuiet())
      }
      if xb.isInfinite {
        if yb.isNaNInf {
          if signX == signY {
            return Self(sign: .plus, exponent: 0, mantissa: mantissaX)
          } else {
            return y // invalid Op
          }
        }
        if yb.isNaN {
          if yb.isSNaN { /* invalid Op */ }
          return Self(sign: .plus, exponent: 0, mantissa: yb.nanQuiet())
        } else {
          // +/- infinity
          return x
        }
      } else {
        // x = 0
        if !yb.isInfinite && mantissaY != 0 {
          if exponentY <= exponentX { return y }
        }
      }
    }
    
    if !validY {
      if yb.isInfinite {
        if yb.isSNaN { /* invalid Op */ }
        return Self(sign: .plus, exponent: 0, mantissa: yb.nanQuiet())
      }
      
      // y = 0
      if mantissaX == 0 {
        // x also 0
        let exp: Int
        var sign = Sign.plus
        if exponentX <= exponentY {
          exp = exponentX
        } else {
          exp = exponentY
        }
        if signX == signY { sign = signX }
        if rounding == .down && signX != signY { sign = .minus }
        return Self(sign: sign, exponent: exp, mantissa: 0)
      } else if exponentY >= exponentX {
        return x
      }
    }
    
    // sort arguments by exponent
    var (signA, exponentA, mantissaA) = (signY, exponentY, mantissaY)
    var (signB, exponentB, mantissaB) = (signX, exponentX, mantissaX)
    if exponentX >= exponentY {
      swap(&signA, &signB)
      swap(&exponentA, &exponentB)
      swap(&mantissaA, &mantissaB)
    }
    
    // exponent difference
    var exponentDiff = exponentA - exponentB
    if exponentDiff > maximumDigits {
      let binExpon = Double(mantissaA).exponent
      let scaleCA = estimateDecDigits(binExpon)
      let d2 = 16 - scaleCA
      if exponentDiff > d2 {
        exponentDiff = d2
        exponentB = exponentA - exponentDiff
      }
    }
    
    let signAB = signA != signB ? Sign.minus : .plus
    let addIn = signAB == .minus ? Int64(1) : 0
    let CB = UInt64(bitPattern: (Int64(mantissaB) + addIn) ^ addIn)
    
    let SU = UInt64(mantissaA) * power10(exponentDiff)
    var S = Int64(bitPattern: SU &+ CB)
    
    if S < 0 {
      signA = signA == .minus ? .plus : .minus // toggle the sign
      S = -S
    }
    var P = UInt64(S)
    var n_digits:Int
    if P == 0 {
      signA = .plus
      if rounding == .down { signA = .minus }
      if mantissaA == 0 { signA = signX }
      n_digits=0
    } else {
      let tempx = Double(P)
      let bin_expon = tempx.exponent
      n_digits = estimateDecDigits(bin_expon)
      if P >= power10(n_digits) {
        n_digits+=1
      }
    }
    
    if n_digits <= maximumDigits {
      return Self(sign: signA, exponent: exponentB, mantissa: Mantissa(P))
    }
    
    let extra_digits = n_digits - maximumDigits
    
    var irmode = roundboundIndex(rounding) >> 2
    if signA == .minus && (UInt(irmode) &- 1) < 2 {
      irmode = 3 - irmode
    }
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += bid_round_const_table(irmode, extra_digits)
    //var Tmp = UInt128()
    let Tmp = P.multipliedFullWidth(by: reciprocals10(extra_digits))
    // __mul_64x64_to_128(&Tmp, P, bid_reciprocals10_64(extra_digits))
    
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = shortReciprocalScale[extra_digits]
    var Q = Tmp.high >> amount
    
    // remainder
    let R = P - Q * power10(extra_digits)
//    if R == ID.bid_round_const_table(irmode, extra_digits) {
//      status = []
//    } else {
//      status.insert(.inexact)
//    }
    
    if rounding == .toNearestOrEven {
      if R == 0 {
        Q &= 0xffff_fffe
      }
    }
    return Self(sign:signA, exponent:exponentB+extra_digits,
                mantissa:Mantissa(Q))
  }
  
  static func round(_ x: Self, _ rmode: Rounding) -> Self {
    var res = Mantissa(0)
    var (x_sign, exp, C1, _) = x.unpack()
    
    // check for NaNs and infinities
    if x.isNaN {    // check for NaN
      if C1.get(range:manLower) > largestNumber/10 {
        C1.clear(range: nanClearRange) // clear G6-G12 and the payload bits
      } else {
        C1.clear(range: g6tog10Range)  // clear G6-G12
      }
      if x.isSNaN {
        // set invalid flag
        // pfpsf.insert(.invalidOperation)
        // return quiet (SNaN)
        C1.clear(bit: nanBitRange.lowerBound)
      } else {    // QNaN
        // return nan(x_sign, Int(C1))
      }
      return Self(RawDataFields(C1))
    } else if x.isInfinite {
      return x
    }
    
    // if x is 0 or non-canonical return 0 preserving the sign bit and
    // the preferred exponent of MAX(Q(x), 0)
    exp = exp - exponentBias
    if C1 == 0 {
      if exp < 0 {
        exp = 0
      }
      return Self(sign:x_sign, exponent:exp+exponentBias, mantissa: 0)
    }
    // x is a finite non-zero number (not 0, non-canonical, or special)
    switch rmode {
      case .toNearestOrEven, .toNearestOrAwayFromZero:
        // return 0 if (exp <= -(p+1))
        if exp <= -(maximumDigits+1) {
          // res = x_sign | zero
          //pfpsf.insert(.inexact)
          return zero(x_sign)
        }
      case .down:
        // return 0 if (exp <= -p)
        if exp <= -maximumDigits {
          if x_sign != .plus {
            return Self(sign: .minus, exponent: exponentBias, mantissa: 1)
            //res = (zero+1) | SIGN_MASK64  // 0xb1c0000000000001
          } else {
            return zero(.plus)
          }
          //pfpsf.insert(.inexact)
        }
      case .up:
        // return 0 if (exp <= -p)
        if exp <= -maximumDigits {
          if x_sign != .plus {
            return zero(.minus) // res = zero | SIGN_MASK64  // 0xb1c0000000000000
          } else {
            return Self(sign: .plus, exponent: exponentBias, mantissa: 1) // res = zero+1
          }
          //pfpsf.insert(.inexact)
        }
      case .towardZero:
        // return 0 if (exp <= -p)
        if exp <= -maximumDigits {
          return zero(x_sign) // x_sign | zero
          //pfpsf.insert(.inexact)
        }
      default: break
    }    // end switch ()
    
    // q = nr. of decimal digits in x (1 <= q <= 54)
    //  determine first the nr. of bits in x
    let q : Int = digitsIn(C1)
    if exp >= 0 {
      // the argument is an integer already
      return x
    }
    
    var ind: Int
    var P128 = UInt128().components, fstar = UInt128().components
    switch rmode {
      case .toNearestOrEven:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient;exp will be 0
          ind = -exp    // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 = C1 + 1/2 * 10^x where the result C1 fits in 64 bits
          // FOR ROUND_TO_NEAREST, WE ADD 1/2 ULP(y) then truncate
          C1 = C1 + Mantissa(bid_midpoint64(ind - 1))
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = mul64x64to128(UInt64(C1), bid_ten2mk64(ind - 1)).components
          
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          //   if floor(C*) is even then C* = floor(C*) - logical right
          //       shift; C* has p decimal digits, correct by Prop. 1)
          //   else if floor(C*) is odd C* = floor(C*)-1 (logical right
          //       shift; C* has p decimal digits, correct by Pr. 1)
          // else
          //   C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 { // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Mantissa((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          // since round_to_even, subtract 1 if current result is odd
          if !res.isMultiple(of: 2) && (fstar.high == 0) &&
              (fstar.low < bid_ten2mk64(ind - 1)) {
            res -= 1
          }
          // determine inexactness of the rounding of C*
          // if (0 < f* - 1/2 < 10^(-x)) then
          //   the result is exact
          // else // if (f* - 1/2 > T*) then
          //   the result is inexact
//          if (ind - 1 <= 2) {
//            if (fstar.low > MASK_SIGN) {
//              // f* > 1/2 and the result may be exact
//              // fstar.low - MASK_SIGN is f* - 1/2
//              if ((fstar.low - MASK_SIGN) > bid_ten2mk64[ind - 1]) {
//                // set the inexact flag
//                //pfpsf.insert(.inexact)
//              }    // else the result is exact
//            } else {    // the result is inexact; f2* <= 1/2
//              // set the inexact flag
//              //pfpsf.insert(.inexact)
//            }
//          } else {    // if 3 <= ind - 1 <= 21
//            if fstar.high > bid_onehalf128[ind - 1] || (fstar.high == bid_onehalf128[ind - 1] && fstar.low != 0) {
//              // f2* > 1/2 and the result may be exact
//              // Calculate f2* - 1/2
//              if fstar.high > bid_onehalf128[ind - 1] || fstar.low > bid_ten2mk64[ind - 1] {
//                // set the inexact flag
//                pfpsf.insert(.inexact)
//              }    // else the result is exact
//            } else {    // the result is inexact; f2* <= 1/2
//              // set the inexact flag
//              pfpsf.insert(.inexact)
//            }
//          }
          // set exponent to zero as it was negative before.
          // res = x_sign | zero | res;
          return Self(sign: x_sign, exponent: exponentBias,
                      mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
         // pfpsf.insert(.inexact)
        }
      case .toNearestOrAwayFromZero:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient;exp will be 0
          ind = -exp   // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 = C1 + 1/2 * 10^x where the result C1 fits in 64 bits
          // FOR ROUND_TO_NEAREST, WE ADD 1/2 ULP(y) then truncate
          C1 = C1 + Mantissa(bid_midpoint64(ind - 1))
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = mul64x64to128(UInt64(C1), bid_ten2mk64(ind - 1)).components
          
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          //   C* = floor(C*) - logical right shift; C* has p decimal digits,
          //       correct by Prop. 1)
          // else
          //   C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Self.Mantissa((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // midpoints are already rounded correctly
          // determine inexactness of the rounding of C*
          // if (0 < f* - 1/2 < 10^(-x)) then
          //   the result is exact
          // else // if (f* - 1/2 > T*) then
          //   the result is inexact
//          if ind - 1 <= 2 {
//            if fstar.low > MASK_SIGN {
//              // f* > 1/2 and the result may be exact
//              // fstar.low - MASK_SIGN is f* - 1/2
//              if (fstar.low - MASK_SIGN) > bid_ten2mk64[ind - 1] {
//                // set the inexact flag
//                pfpsf.insert(.inexact)
//              }    // else the result is exact
//            } else {    // the result is inexact; f2* <= 1/2
//              // set the inexact flag
//              pfpsf.insert(.inexact)
//            }
//          } else {    // if 3 <= ind - 1 <= 21
//            if fstar.high > bid_onehalf128[ind - 1] || (fstar.high == bid_onehalf128[ind - 1] && fstar.low != 0) {
//              // f2* > 1/2 and the result may be exact
//              // Calculate f2* - 1/2
//              if fstar.high > bid_onehalf128[ind - 1] || fstar.low > bid_ten2mk64[ind - 1] {
//                // set the inexact flag
//                pfpsf.insert(.inexact)
//              }    // else the result is exact
//            } else {    // the result is inexact; f2* <= 1/2
//              // set the inexact flag
//              pfpsf.insert(.inexact)
//            }
//          }
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, exponent: exponentBias, mantissa: res)
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
        }
      case .down:
        if (q + exp) > 0 {    // exp < 0 and 1 <= -exp < q
          // need to shift right -exp digits from the coefficient; exp will be 0
          ind = -exp    // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 fits in 64 bits
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = mul64x64to128(UInt64(C1), bid_ten2mk64(ind - 1)).components
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Mantissa(P128.high >> shift)
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (f* > 10^(-x)) then the result is inexact
          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64(ind - 1)) {
            if x_sign != .plus {
              // if negative and not exact, increment magnitude
              res+=1
            }
            // pfpsf.insert(.inexact)
          }
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, exponent: exponentBias,
                      mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp <= 0
          // the result is +0 or -1
          if x_sign != .plus {
            return Self(sign: .minus, exponent: exponentBias, mantissa: 1)
          } else {
            return zero(.plus)
          }
          // pfpsf.insert(.inexact)
        }
      case .up:
        if (q + exp) > 0 {    // exp < 0 and 1 <= -exp < q
          // need to shift right -exp digits from the coefficient;exp will be 0
          ind = -exp    // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 fits in 64 bits
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = mul64x64to128(UInt64(C1), bid_ten2mk64(ind - 1)).components
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Mantissa((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (f* > 10^(-x)) then the result is inexact
          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64(ind - 1)) {
            if x_sign == .plus {
              // if positive and not exact, increment magnitude
              res+=1
            }
            //pfpsf.insert(.inexact)
          }
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, exponent: exponentBias, mantissa: res)
        } else {    // if exp < 0 and q + exp <= 0
          // the result is -0 or +1
          if x_sign != .plus {
            return zero(.minus)
          } else {
            return Self(sign: .plus, exponent: exponentBias, mantissa: 1)
          }
        }
      case .towardZero:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient;exp will be 0
          ind = -exp    // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 fits in 127 bits
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = mul64x64to128(UInt64(C1), bid_ten2mk64(ind - 1)).components
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Mantissa((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (f* > 10^(-x)) then the result is inexact
//          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64[ind - 1]) {
//            pfpsf.insert(.inexact)
//          }
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, exponent: exponentBias,
                      mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
          // pfpsf.insert(.inexact)
        }
      default: break
    }    // end switch ()
    return Self(sign: x_sign, exponent: exp+exponentBias, mantissa: res)
  }
  
  /***************************************************************************
   *  BID32 nextup
   **************************************************************************/
  static func nextup(_ x: Self) -> Self {
    var largestBID = Self(sign: .plus, exponent: maximumExponent,
                          mantissa: largestNumber)
    
    // check for NaNs and infinities
    if x.isNaN { // check for NaN
      var res = x.data
      if res.get(range:manLower) > largestNumber/10 {
        res.clear(range: nanClearRange) // clear G6-G10 and the payload bits
      } else {
        res.clear(range: g6tog10Range)  // x.ma & 0xfe0f_ffff // clear G6-G10
      }
      if x.isSNaN { // SNaN
        // set invalid flag
        // pfpsf.insert(.invalidOperation)
        // pfpsf |= BID_INVALID_EXCEPTION;
        // return quiet (SNaN)
        res.clear(bit: Self.nanBitRange.lowerBound)
      } else {    // QNaN
        // res = x.data
      }
      return Self(res)
    } else if x.isInfinite { // check for Infinity
      if x.sign == .plus { // x is +inf
        return Self.infinite(x.sign)
      } else { // x is -inf
        largestBID.sign = .minus
        return largestBID  // -MAXFP = -9999999 * 10^emax
      }
    }
    // unpack the argument
    // let x_sign = x & SIGN_MASK // 0 for positive, SIGN_MASK for negative
    // var x_exp, C1:UInt32
    // if steering bits are 11 (condition will be 0), then exponent is G[0:7]
    var (x_sign, x_exp, C1, _) = x.unpack() // extractExpSig(x)
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 || (x.isSpecial && C1 > Self.largestNumber) {
      // x is 0: MINFP = 1 * 10^emin
      return Self(sign: .plus, exponent: minimumExponent, mantissa: 1)
    } else { // x is not special and is not zero
      if x == largestBID { // LARGEST_BID {
        // x = +MAXFP = 9999999 * 10^emax
        return Self.infinite(.plus) // INFINITY_MASK // +inf
      } else if x == Self(sign:.minus, exponent:minimumExponent, mantissa:1) {
        // x = -MINFP = 1...99 * 10^emin
        return Self(sign: .minus, exponent: minimumExponent, mantissa: 0)
      } else {
        // -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
        // can add/subtract 1 ulp to the significand
        
        // Note: we could check here if x >= 10^7 to speed up the case q1 = 7
        // q1 = nr. of decimal digits in x (1 <= q1 <= 7)
        //  determine first the nr. of bits in x
        let q1: Int = digitsIn(C1)
        
        // if q1 < P7 then pad the significand with zeros
        if q1 < maximumDigits {
          let ind:Int
          if x_exp > (maximumDigits - q1) {
            ind = maximumDigits - q1 // 1 <= ind <= P7 - 1
            // pad with P7 - q1 zeros, until exponent = emin
            // C1 = C1 * 10^ind
            C1 = C1 * bid_ten2k64(ind)
            x_exp = x_exp - ind
          } else { // pad with zeros until the exponent reaches emin
            ind = x_exp
            C1 = C1 * bid_ten2k64(ind)
            x_exp = minimumExponent // MIN_EXPON
          }
        }
        if x_sign == .plus {    // x > 0
          // add 1 ulp (add 1 to the significand)
          C1 += 1
          if C1 == largestNumber+1 { // 10_000_000 { // if  C1 = 10^7
            C1 = (largestNumber+1)/10 // C1 = 10^6
            x_exp += 1
          }
          // Ok, because MAXFP = 9999999 * 10^emax was caught already
        } else {    // x < 0
          // subtract 1 ulp (subtract 1 from the significand)
          C1 -= 1
          if C1 == largestNumber/10 && x_exp != 0 { // if  C1 = 10^6 - 1
            C1 = largestNumber // C1 = 10^7 - 1
            x_exp -= 1
          }
        }
        // assemble the result
        // if significand has 24 bits
        return Self(sign: x_sign, exponent: x_exp, mantissa: C1)
      } // end -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
    } // end x is not special and is not zero
  //  return res
  }
  
  static func sqrt(_ x: Self, _ rmode:Rounding) -> Self {
    // unpack arguments, check for NaN or Infinity
    var (sign_x, exponent_x, coefficient_x, valid) = x.unpack()
    if !valid {
      // x is Inf. or NaN or 0
      if x.isInfinite {
        var res = coefficient_x; res.clear(range: g6tog10Range)
        if x.isNaNInf && sign_x == .minus {
          return Self.nan(sign_x, 0)
          //status.insert(.invalidOperation)
        }
//        if isSNaN(x) {
//          // status.insert(.invalidOperation)
//        }
        res.clear(bit: Self.nanBitRange.lowerBound)
        return Self(RawDataFields(res))
      }
      // x is 0
      exponent_x = (exponent_x + exponentBias) >> 1
      return Self(sign: sign_x, exponent: exponent_x, mantissa: 0)
    }
    // x<0?
    if sign_x == .minus && coefficient_x != 0 {
      // status.insert(.invalidOperation)
      return Self.nan(.plus, 0)
    }
    
    //--- get number of bits in the coefficient of x ---
    let tempx = Float32(coefficient_x)
    let bin_expon_cx = Int(((tempx.bitPattern >> 23) & 0xff) - 0x7f)
    var digits_x = estimateDecDigits(bin_expon_cx)
    // add test for range
    if coefficient_x >= bid_power10_index_binexp(bin_expon_cx) {
      digits_x+=1
    }
    
    var A10 = coefficient_x
    if exponent_x & 1 == 0 {
      A10 = (A10 << 2) + A10;
      A10 += A10;
    }
    
    let dqe = Double(A10).squareRoot()
    let QE = UInt32(dqe)
    if QE * QE == A10 {
      return Self(sign: .plus, exponent: (exponent_x + exponentBias) >> 1,
                  mantissa: Mantissa(QE))
    }
    // if exponent is odd, scale coefficient by 10
    var scale = Int(13 - digits_x)
    var exponent_q = exponent_x + exponentBias - scale
    scale += (exponent_q & 1)   // exp. bias is even
    
    let CT = UInt128(power10(scale)).components.low
    let CA = UInt64(coefficient_x) * CT
    let dq = Double(CA).squareRoot()
    
    exponent_q = exponent_q >> 1  // square root of 10^x = 10^(x/2)
    
//    status.insert(.inexact)

    let rndMode = roundboundIndex(rmode) >> 2
    var Q:UInt32
    if ((rndMode) & 3) == 0 {
      Q = UInt32(dq+0.5)
    } else {
      Q = UInt32(dq)
      if rmode == .up {
        Q+=1
      }
    }
    return Self(sign: .plus, exponent: exponent_q, mantissa: Mantissa(Q))
  }
}

// MARK: - Extended UInt Definitions
// These are usd primarily for table and extended calculation storage
struct UInt512 { var w = [UInt64](repeating: 0, count: 8) }
struct UInt384 { var w = [UInt64](repeating: 0, count: 6) }
struct UInt256 { var w = [UInt64](repeating: 0, count: 4) }
struct UInt192 { var w = [UInt64](repeating: 0, count: 3) }

// MARK: - Status and Rounding Type Definitions

// Rounding boundaries table, indexed by
// 4 * rounding_mode + 2 * sign + lsb of truncation
// We round up if the round/sticky data is strictly > this boundary
//
// NB: This depends on the particular values of the rounding mode
// numbers, which are supposed to be defined as shown here:
//
// #define .toNearestOrEven         0x00000
// #define .down                    0x00001
// #define .up                      0x00002
// #define .towardZero              0x00003
// #define .toNearestOrAwayFromZero 0x00004
//
// Some of the shortcuts below in "underflow after rounding" also use
// the concrete values.
//
// So we add a directive here to double-check that this is the case
internal func roundboundIndex(_ round:Rounding, _ negative:Bool=false,
                            _ lsb:Int=0) -> Int {
  var index = (lsb & 1) + (negative ? 2 : 0)
  switch round {
    case .toNearestOrEven: index += 0
    case .down: index += 4
    case .up: index += 8
    case .towardZero: index += 12
    default: index += 16
  }
  return index
}

public struct Status: OptionSet, CustomStringConvertible {
  
  public let rawValue: Int
  
  /* IEEE extended flags only */
  private static let DConversion_syntax    = 0x00000001
  private static let DDivision_by_zero     = 0x00000002
  private static let DDivision_impossible  = 0x00000004
  private static let DDivision_undefined   = 0x00000008
  private static let DInsufficient_storage = 0x00000010 /* malloc fails */
  private static let DInexact              = 0x00000020
  private static let DInvalid_context      = 0x00000040
  private static let DInvalid_operation    = 0x00000080
  private static let DLost_digits          = 0x00000100
  private static let DOverflow             = 0x00000200
  private static let DClamped              = 0x00000400
  private static let DRounded              = 0x00000800
  private static let DSubnormal            = 0x00001000
  private static let DUnderflow            = 0x00002000
  
  public static let conversionSyntax    = Status(rawValue:DConversion_syntax)
  public static let divisionByZero      = Status(rawValue:DDivision_by_zero)
  public static let divisionImpossible  = Status(rawValue:DDivision_impossible)
  public static let divisionUndefined   = Status(rawValue:DDivision_undefined)
  public static let insufficientStorage=Status(rawValue:DInsufficient_storage)
  public static let inexact             = Status(rawValue:DInexact)
  public static let invalidContext      = Status(rawValue:DInvalid_context)
  public static let lostDigits          = Status(rawValue:DLost_digits)
  public static let invalidOperation    = Status(rawValue:DInvalid_operation)
  public static let overflow            = Status(rawValue:DOverflow)
  public static let clamped             = Status(rawValue:DClamped)
  public static let rounded             = Status(rawValue:DRounded)
  public static let subnormal           = Status(rawValue:DSubnormal)
  public static let underflow           = Status(rawValue:DUnderflow)
  public static let clearFlags          = Status([])
  
  public static let errorFlags =
  Status(rawValue: Int(DDivision_by_zero | DOverflow |
                       DUnderflow | DConversion_syntax | DDivision_impossible |
                       DDivision_undefined | DInsufficient_storage |
                       DInvalid_context | DInvalid_operation))
  public static let informationFlags =
  Status(rawValue: Int(DClamped | DRounded | DInexact | DLost_digits))
  
  public init(rawValue: Int) { self.rawValue = rawValue }
  
  public var hasError:Bool {!Status.errorFlags.intersection(self).isEmpty}
  public var hasInfo:Bool {!Status.informationFlags.intersection(self).isEmpty}
  
  public var description: String {
    var str = ""
    if self.contains(.conversionSyntax)   { str += "Conversion syntax, "}
    if self.contains(.divisionByZero)     { str += "Division by zero, " }
    if self.contains(.divisionImpossible) { str += "Division impossible, "}
    if self.contains(.divisionUndefined)  { str += "Division undefined, "}
    if self.contains(.insufficientStorage){ str += "Insufficient storage, "}
    if self.contains(.inexact)            { str += "Inexact number, " }
    if self.contains(.invalidContext)     { str += "Invalid context, " }
    if self.contains(.invalidOperation)   { str += "Invalid operation, " }
    if self.contains(.lostDigits)         { str += "Lost digits, " }
    if self.contains(.overflow)           { str += "Overflow, " }
    if self.contains(.clamped)            { str += "Clamped, " }
    if self.contains(.rounded)            { str += "Rounded, " }
    if self.contains(.subnormal)          { str += "Subnormal, " }
    if self.contains(.underflow)          { str += "Underflow, " }
    if str.hasSuffix(", ")                { str.removeLast(2) }
    return str
  }
}

// MARK: - Common Utility Functions

internal func addDecimalPointAndExponent(_ ps:String, _ exponent:Int,
                                         _ maxDigits:Int) -> String {
  var digits = ps.count
  var ps = ps
  var exponent_x = exponent
  if exponent_x == 0 {
    ps.insert(".", at: ps.index(ps.startIndex, offsetBy: exponent_x+1))
  } else if abs(exponent_x) > maxDigits {
    if ps.count > 1 {
      ps.insert(".", at: ps.index(after: ps.startIndex))
    }
    ps += "e"
    if exponent_x < 0 {
      ps += "-"
      exponent_x = -exponent_x
    } else {
      ps += "+"
    }
    ps += String(exponent_x)
  } else if digits <= exponent_x {
    // format the number without an exponent
    while digits <= exponent_x {
      // pad the number with zeros
      ps += "0"; digits += 1
    }
  } else if exponent_x < 0 {
    while exponent_x < -1 {
      // insert leading zeros
      ps = "0" + ps; exponent_x += 1
    }
    ps = "0." + ps
  } else {
    // insert the decimal point
    ps.insert(".", at: ps.index(ps.startIndex, offsetBy: exponent_x+1))
    if ps.hasSuffix(".") { ps.removeLast() }
  }
  return ps
}

// MARK: - Generic String Conversion functions

/// Converts a decimal floating point number `x` into a string
internal func string<T:IntegerDecimal>(from x: T) -> String {
  // unpack arguments, check for NaN or Infinity
  let (sign, exp, coeff, valid) = x.unpack()
  let s = sign == .minus ? "-" : ""
  if valid {
    // x is not special
    let ps = String(coeff)
    let exponent_x = Int(exp) - T.exponentBias + (ps.count - 1)
    return s + addDecimalPointAndExponent(ps, exponent_x, T.maximumDigits)
  } else {
    // x is Inf. or NaN or 0
    var ps = s
    if x.isNaN {
      if x.isSNaN { ps.append("S") }
      ps.append("NaN")
      return ps
    }
    if x.isInfinite {
      ps.append("Inf")
      return ps
    }
    ps.append("0")
    return ps
  }
}

/// Converts a decimal number string of the form:
/// `[+|-] digit {digit} [. digit {digit}] [e [+|-] digit {digit} ]`
/// to a Decimal<n> number
internal func numberFromString<T:IntegerDecimal>(_ s: String,
                                                 round: Rounding) -> T? {
  // keep consistent character case for "infinity", "nan", etc.
  var ps = s.lowercased()
  
  // get first non-whitespace character
  let eos = Character("\0")
  var c = ps.isEmpty ? eos : ps.removeFirst()
  var right_radix_leading_zeros = 0
  
  func handleEmpty() -> T {
    right_radix_leading_zeros = T.exponentBias - right_radix_leading_zeros
    if right_radix_leading_zeros < 0 {
      right_radix_leading_zeros = T.exponentBias
    }
    return T(sign: sign, exponent: right_radix_leading_zeros, mantissa: 0)
  }
  
  // detect special cases (INF or NaN)
  if c == eos || (c != "." && c != "-" && c != "+" && !c.isNumber) {
    // Infinity?
    if c == "i" && (ps.hasPrefix("nfinity") || ps.hasPrefix("nf")) {
      return T.infinite(.plus)
    }
    // return sNaN
    if c == "s" && ps.hasPrefix("nan") {
      // case insensitive check for snan
      return T.snan
    } else {
      // return qNaN & any coefficient
      let coeff = Int(ps.dropFirst(2)) ?? 0 // drop "AN"
      return T.nan(.plus, coeff)
    }
  }
  
  // detect +INF or -INF
  if ps.hasPrefix("infinity") || ps.hasPrefix("inf") {
    if c == "+" {
      return T.infinite()
    } else if c == "-" {
      return T.infinite(.minus)
    } else {
      return T.nan(.plus, 0)
    }
  }
  
  // if +sNaN, +SNaN, -sNaN, or -SNaN
  if ps.hasPrefix("snan") {
    if c == "-" {
      var x = T.snan; x.sign = .minus
      return x
    } else {
      return T.snan
    }
  }
  
  // determine sign
  var sign = Sign.plus
  if c == "-" {
    sign = .minus
  }
  
  // get next character if leading +/- sign
  if c == "-" || c == "+" {
    c = ps.isEmpty ? eos : ps.removeFirst()
  }
  
  // if c isn"t a decimal point or a decimal digit, return NaN
  if c != "." && !c.isNumber {
    // return NaN
    return T.nan(sign, 0)
  }
  
  var rdx_pt_enc = false
  var coefficient_x = T.Mantissa(0)
  
  // detect zero (and eliminate/ignore leading zeros)
  if c == "0" || c == "." {
    if c == "." {
      rdx_pt_enc = true
      c = ps.isEmpty ? eos : ps.removeFirst()
    }
    // if all numbers are zeros (with possibly 1 radix point, the number
    // is zero
    // should catch cases such as: 000.0
    while c == "0" {
      c = ps.isEmpty ? eos : ps.removeFirst()
      // for numbers such as 0.0000000000000000000000000000000000001001,
      // we want to count the leading zeros
      if rdx_pt_enc {
        right_radix_leading_zeros+=1
      }
      // if this character is a radix point, make sure we haven't already
      // encountered one
      if c == "." {
        if !rdx_pt_enc {
          rdx_pt_enc = true
          // if this is the first radix point, and the next character is
          // NULL, we have a zero
          if ps.isEmpty {
            return handleEmpty()
          }
          c = ps.isEmpty ? eos : ps.removeFirst()
        } else {
          // if 2 radix points, return NaN
          return T.nan(sign, 0)
        }
      } else if ps.isEmpty {
        return handleEmpty()
      }
    }
  }
  
  var ndigits = 0
  var dec_expon_scale = 0
  var midpoint = 0
  var rounded_up = 0
  var add_expon = 0

  while c.isNumber || c == "." {
    if c == "." {
      if rdx_pt_enc {
        // return NaN
        return T.nan(sign, 0)
      }
      rdx_pt_enc = true
      c = ps.isEmpty ? eos : ps.removeFirst()
      continue
    }
    if rdx_pt_enc { dec_expon_scale += 1 }
    
    ndigits+=1
    if ndigits <= 7 {
      coefficient_x = (coefficient_x << 1) + (coefficient_x << 3);
      coefficient_x += T.Mantissa(c.wholeNumberValue ?? 0)
    } else if ndigits == 8 {
      // coefficient rounding
      switch round {
        case .toNearestOrEven:
          midpoint = (c == "5" && (coefficient_x & 1 == 0)) ? 1 : 0;
          // if coefficient is even and c is 5, prepare to round up if
          // subsequent digit is nonzero
          // if str[MAXDIG+1] > 5, we MUST round up
          // if str[MAXDIG+1] == 5 and coefficient is ODD, ROUND UP!
          if c > "5" || (c == "5" && (coefficient_x & 1) != 0) {
            coefficient_x += 1
            rounded_up = 1
          }
        case .down:
          if sign == .minus { coefficient_x+=1; rounded_up=1 }
        case .up:
          if sign != .minus { coefficient_x+=1; rounded_up=1 }
        case .toNearestOrAwayFromZero:
          if c >= "5" { coefficient_x+=1; rounded_up=1 }
        default: break
      }
      if coefficient_x == T.largestNumber+1 {
        coefficient_x = (T.largestNumber+1)/10
        add_expon = 1
      }
//      if c > "0" {
//        rounded = 1
//      }
      add_expon += 1
    } else { // ndigits > 8
      add_expon += 1
      if midpoint != 0 && c > "0" {
        coefficient_x += 1
        midpoint = 0
        rounded_up = 1
      }
//      if c > "0" {
//        rounded = 1;
//      }
    }
    c = ps.isEmpty ? eos : ps.removeFirst()
  }
  
  add_expon -= dec_expon_scale + Int(right_radix_leading_zeros)
  
  if c == eos {
//    if rounded != 0 {
//      T.state.insert(.inexact)
//    }
    return T(sign: sign, exponent: add_expon+T.exponentBias,
             mantissa: T.Mantissa(coefficient_x))
  }
  
  if c != "e" {
    // return NaN
    return T.nan(sign, 0)
  }
  c = ps.isEmpty ? eos : ps.removeFirst()
  
  let sgn_expon = c == "-"
  if c == "-" || c == "+" {
    c = ps.isEmpty ? eos : ps.removeFirst()
  }
  if c == eos || !c.isNumber {
    // return NaN
    return T.nan(sign, 0)
  }
  
  var expon_x = 0
  while c.isNumber {
    if expon_x < (1 << T.smallMantissaBits.upperBound) {
      expon_x = (expon_x << 1) + (expon_x << 3)
      expon_x += c.wholeNumberValue ?? 0
    }
    c = ps.isEmpty ? eos : ps.removeFirst()
  }
  
  if c != eos {
    // return NaN
    return T.nan(sign, 0)
  }
  
//  if rounded != 0 {
//    T.state.insert(.inexact)
//  }
  
  if sgn_expon {
    expon_x = -expon_x
  }
  
  expon_x += add_expon + T.exponentBias
  
  if expon_x < 0 {
    if rounded_up != 0 {
      coefficient_x -= 1
    }
    return T.handleRounding(sign, expon_x, Int(coefficient_x),
                            rounded_up, round)
  }
  let mant = T.Mantissa(coefficient_x)
  // return T.handleRounding(sign, expon_x, Int(mant), 0, round)
  let result = T.adjustOverflowUnderflow(sign, expon_x, mant, round)
  return T(result)
}

/// Returns x^exp where x = *num*.
/// - Precondition: x ≥ 0, exp ≥ 0
internal func power<T:FixedWidthInteger>(_ num:T, to exp: Int) -> T {
  // Zero raised to anything except zero is zero (provided exponent is valid)
  guard exp >= 0 else { return T.max }
  if num == 0 { return exp == 0 ? 1 : 0 }
  var z = num
  var y : T = 1
  var n = abs(exp)
  while true {
    if !n.isMultiple(of: 2) { y *= z }
    n >>= 1
    if n == 0 { break }
    z *= z
  }
  return y
}

