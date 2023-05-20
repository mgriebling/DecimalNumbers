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

extension UInt128 {
  public init(w: [UInt64]) {
    let high = UInt128.High(w[1])
    let low  = UInt128.Low(w[0])
    self.init(high: high, low: low)
  }
}

///
/// Groups together algorithms that can be used by all Decimalxx variants
///

// MARK: - Generic Integer Decimal Type

public protocol IntegerDecimal : Codable, Hashable {
  
  associatedtype RawDataFields : UnsignedInteger & FixedWidthInteger
  associatedtype Mantissa : UnsignedInteger & FixedWidthInteger
  
  var data: RawDataFields { get set }
  
  //////////////////////////////////////////////////////////////////
  /// Initializers
  
  /// Initialize with a raw data word
  init(_ word: RawDataFields)
  
  /// Initialize with sign, biased exponent, and unsigned mantissa
  init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa)
  
  //////////////////////////////////////////////////////////////////
  /// Conversions from/to densely packed decimal numbers
  init(dpd: RawDataFields)
  
  var dpd: RawDataFields { get }
  
  //////////////////////////////////////////////////////////////////
  /// Essential data to extract or update from the fields
  
  /// Sign of the number
  var sign: FloatingPointSign { get set }
  
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
  
  static func zero(_ sign: FloatingPointSign) -> Self
  static func nan(_ sign: FloatingPointSign, _ payload: Int) -> Self
  static func infinite(_ sign: FloatingPointSign) -> Self
  static func max(_ sign: FloatingPointSign) -> Self
  
  //////////////////////////////////////////////////////////////////
  /// Decimal number definitions
  static var signBit: ClosedRange<Int> { get }
  static var specialBits: ClosedRange<Int> { get }
  
  static var exponentBias: Int    { get }
  static var maximumExponent: Int { get } // unbiased & normal
  static var minimumExponent: Int { get } // unbiased & normal
  static var numberOfDigits:  Int { get }
  
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
  
  static var highMantissaBit: Int { 1 << (smallMantissaBits.upperBound+3) }
  
  /// These bit fields can be predetermined just from the size of
  /// the number type `RawDataFields` `bitWidth`
  static var maxBit: Int { RawDataFields.bitWidth - 1 }
  
  static var signBit:     ClosedRange<Int> { maxBit   ... maxBit }
  static var specialBits: ClosedRange<Int> { maxBit-2 ... maxBit-1 }
  static var nanBitRange: ClosedRange<Int> { maxBit-6 ... maxBit-1 }
  
  /// bit field definitions for DPD numbers
  static var lowMan: Int   { smallMantissaBits.upperBound }
  static var upperExp1: Int { exponentSMBits.upperBound }
  static var upperExp2: Int { exponentLMBits.upperBound }
  
  static var expLower: ClosedRange<Int> { lowMan...maxBit-6 }
  static var manLower: ClosedRange<Int> { 0...lowMan-1 }
  static var expUpper: ClosedRange<Int> { lowMan+1...lowMan+6 }
  
  /// Bit patterns that designate special numbers
  static var nanPattern: Int      { 0b1_1111_0 }
  static var snanPattern: Int     { 0b1_1111_1 }
  static var infinitePattern: Int { 0b1_1110 }
  static var specialPattern: Int  { 0b11 }
  
  static var trailingPattern: Int { 0x3ff }
  
  @inlinable func getBits(_ range: ClosedRange<Int>) -> Int {
    guard data.bitWidth > range.upperBound else { return 0 }
    return Self.gBits(range, from: data)
  }
  
  @inlinable mutating func setBits(_ range: ClosedRange<Int>, bits: Int) {
    guard data.bitWidth > range.upperBound else { return }
    Self.sBits(range, bits: bits, in: &self.data)
  }
  
  @inlinable var sign: FloatingPointSign {
    get { getBits(Self.signBit) == 0 ? .plus : .minus }
    set { setBits(Self.signBit, bits: newValue == .minus ? 1 : 0) }
  }
  
  @inlinable var exponent: Int {
    let range = isSmallMantissa ? Self.exponentSMBits : Self.exponentLMBits
    return getBits(range)
  }
  
  @inlinable var mantissa: Mantissa {
    if isSmallMantissa {
      return Mantissa(getBits(Self.smallMantissaBits) + Self.highMantissaBit)
    } else {
      return Mantissa(getBits(Self.largeMantissaBits))
    }
  }
  
  /// Note: exponent is assumed to be biased
  mutating func set(exponent: Int, mantissa: Mantissa) {
//    precondition(exponent >= 0 && exponent <= Self.maximumExponent,
//                 "Invalid biased exponent")
    if mantissa < Self.highMantissaBit {
      // large mantissa
      setBits(Self.exponentLMBits, bits: exponent)
      setBits(Self.largeMantissaBits, bits: Int(mantissa))
    } else {
      // small mantissa
      setBits(Self.exponentSMBits, bits: exponent)
      setBits(Self.smallMantissaBits, bits:Int(mantissa)-Self.highMantissaBit)
      setBits(Self.specialBits, bits: Self.specialPattern) // special encoding
    }
  }
  
  /// Return `self's` pieces all at once with unbiased exponent
  func unpack() ->
  (sign: FloatingPointSign, exponent: Int, mantissa: Mantissa, valid: Bool) {
    let exponent: Int, mantissa: Mantissa
    if isSmallMantissa {
      // small mantissa
      exponent = getBits(Self.exponentSMBits)
      mantissa = Mantissa(getBits(Self.smallMantissaBits) +
                          Self.highMantissaBit)
    } else {
      // large mantissa
      exponent = getBits(Self.exponentLMBits)
      mantissa = Mantissa(getBits(Self.largeMantissaBits))
    }
    return (self.sign, exponent, mantissa, self.isValid)
  }
  
  /// Return `dpd` pieces all at once
  static func unpack(dpd: RawDataFields) ->
  (sign: FloatingPointSign, exponent: Int, high: Int, trailing: Mantissa) {
    
    func getBits(_ range: ClosedRange<Int>) -> Int { gBits(range, from:dpd) }
    func getBit(_ bit:Int) -> Int { ((1 << bit) & dpd) == 0 ? 0 : 1 }
    
    let sgn = getBit(signBit.lowerBound)==1 ? FloatingPointSign.minus : .plus
    var exponent, high: Int, trailing: Mantissa
    let expRange2: ClosedRange<Int>
    
    if getBits(Self.specialBits) == 0b11 {
      // small mantissa
      expRange2 = (upperExp1-1)...upperExp1
      high = getBit(lowMan) + 8
    } else {
      // large mantissa
      expRange2 = (upperExp2-1)...upperExp2
      high = getBits(lowMan...lowMan+2)
    }
    exponent = (getBits(expLower) + getBits(expRange2) << 6)
    trailing = Mantissa(getBits(0...lowMan-1))
    return (sgn, exponent, high, trailing)
  }
  
  @inlinable func nanQuiet() -> Mantissa {
    let quietMask = ~(Mantissa(1) << Self.nanBitRange.lowerBound)
    return Mantissa(data) & quietMask
  }
  
  ///////////////////////////////////////////////////////////////////////
  /// Special number definitions
  @inlinable static func infinite(_ s: FloatingPointSign = .plus) -> Self {
    Self(sign: s, exponent: infinitePattern<<3, mantissa: 0)
  }
  
  @inlinable static func max(_ s: FloatingPointSign = .plus) -> Self {
    Self(sign:s, exponent:maximumExponent, mantissa:largestNumber)
  }
  
  static func overflow(_ sign: FloatingPointSign,
                       rnd_mode: FloatingPointRoundingRule) -> Self {
    if rnd_mode == .towardZero || rnd_mode == (sign != .plus ? .up : .down) {
      return max(sign)
    } else {
      return infinite(sign)
    }
  }
  
  @inlinable static var snan: Self {
    Self(sign: .plus, exponent: snanPattern<<2, mantissa: 0)
  }
  
  @inlinable static func zero(_ sign: FloatingPointSign) -> Self {
    Self(sign: sign, exponent: 0, mantissa: 0)
  }
  
  @inlinable
  static func nan(_ sign: FloatingPointSign, _ payload: Int) -> Self {
    let man = payload > largestNumber/10 ? 0 : Mantissa(payload)
    return Self(sign:sign, exponent:nanPattern<<2, mantissa:man)
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Handy routines for testing different aspects of the number
  @inlinable var nanBits: Int { getBits(Self.nanBitRange) }
  
  static func sBits<T:FixedWidthInteger>(_ range: ClosedRange<Int>,
                                         bits: Int, in n: inout T) {
    let width = range.upperBound - range.lowerBound + 1
    let mask = (T(1) << width) - 1
    let mbits = T(bits) & mask    // limit `bits` size
    let smask = ~(mask << range.lowerBound)   // inverted mask
    n = (n & smask) | (mbits << range.lowerBound)
  }
  
  static func sBit<T:FixedWidthInteger>(_ bit:Int, bits:Int, in n:inout T) {
    let sbit = T(1) << bit
    n = bits.isMultiple(of: 2) ? n & ~sbit : n | sbit
  }
  
  static func gBits<T:FixedWidthInteger>(_ range: ClosedRange<Int>,
                                         from data: T) -> Int {
    let width = range.upperBound - range.lowerBound + 1
    return Int((data >> range.lowerBound) & ((T(1) << width) - 1))
  }
  
  var isSmallMantissa: Bool { isSpecial }
  var isSpecial: Bool       { getBits(Self.specialBits)==Self.specialPattern }
  var isNaN: Bool           { nanBits & Self.nanPattern == Self.nanPattern }
  var isSNaN: Bool          { nanBits & Self.snanPattern == Self.snanPattern }
  
  var isNaNInf: Bool {
    nanBits & Self.nanPattern == Self.infinitePattern<<1
  }
  
  var isInfinite: Bool {
    let range = Self.signBit.lowerBound-5...Self.signBit.lowerBound-1
    return getBits(range) & Self.infinitePattern == Self.infinitePattern
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
      if (self.data & 0x01fc << (Self.maxBit - 16)) != 0 {
        // FIXME: - what is this? Decimal32 had mask of 0x01f0
        return false
      } else if mantissa > Self.largestNumber/10 {
        return false
      } else {
        return true
      }
    } else if isInfinite {
      return mantissa == 0
    } else if isSpecial {
      return mantissa <= Self.largestNumber
    } else {
      return true
    }
  }
  
  private func checkNormalScale(_ exp: Int, _ mant: Mantissa) -> Bool {
    // if exponent is less than -95, the number may be subnormal
    let exp = exp + Self.exponentBias
    if exp < Self.minimumExponent+Self.numberOfDigits-1 {
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
    
    func getNan() -> Int { Self.gBits(Self.nanBitRange, from: dpd) }
    
    // Convert the dpd number to a bid number
    var (sign, exp, high, trailing) = Self.unpack(dpd: dpd)
    var nan = false
    
    if getNan() & Self.nanPattern == Self.infinitePattern {
      self = Self.infinite(sign); return
    } else if getNan() & Self.nanPattern == Self.nanPattern {
      nan = true; exp = 0
    }
    
    let mask = Self.trailingPattern
    let mils = ((Self.numberOfDigits - 1) / 3) - 1
    let shift = mask.bitWidth - mask.leadingZeroBitCount
    var mant = Mantissa(high)
    for i in stride(from: shift*mils, through: 0, by: -shift) {
      mant *= 1000
      mant += Mantissa(Self.intFrom(dpd: Int(trailing >> i) & mask))
    }
    
    //    let d1 = Self.intFrom(dpd: (Int(trailing) >> shift) & mask) * 1000
    //    let d2 = Self.intFrom(dpd: Int(trailing) & mask)
    //    let mant = Mantissa(d2 + d1 + high * 1_000_000)
    if nan { self = Self.nan(sign, Int(mant)) }
    else { self.init(sign: sign, exponent: exp, mantissa: mant) }
  }
  
  /// Convert `self` to a DPD number.
  var dpd: RawDataFields {
    var res : RawDataFields = 0
    
    func setBits(_ range: ClosedRange<Int>, bits: Int) {
      Self.sBits(range, bits: bits, in: &res)
    }
    
    var (sign, exp, mantissa, _) = unpack()
    var trailing = mantissa & 0xfffff
    if self.isNaNInf {
      return Self.infinite(sign).data
    } else if self.isNaN {
      if trailing > Self.largestNumber/10 {
        trailing = 0
      }
      return Self.nan(sign, Int(trailing)).data
    }
    
    // FIXME: - b0 - b2 need to be defined for arbitrary length DPDs
    let b0 = Int(mantissa / 1_000_000)
    let b1 = Int(mantissa / 1000) % 1000
    let b2 = Int(mantissa) % 1000
    let dmant = (Self.intToDPD(b1) << 10) | Self.intToDPD(b2)
    let signBit = Self.signBit.lowerBound
    let expLower = Self.smallMantissaBits.upperBound...signBit-6
    let manLower = 0...Self.smallMantissaBits.upperBound-1
    
    exp = exp + Self.exponentBias
    if b0 >= 8 {
      let expUpper = signBit-4...signBit-3
      let manUpper = signBit-5...signBit-5
      setBits(Self.specialBits, bits: Self.specialPattern)
      setBits(expUpper, bits: exp >> 6) // upper exponent bits
      setBits(manUpper, bits: b0 & 1)   // upper mantisa bits
    } else {
      let expUpper = signBit-2...signBit-1
      let manUpper = signBit-5...signBit-3
      setBits(expUpper, bits: exp >> 6) // upper exponent bits
      setBits(manUpper, bits: b0)       // upper mantisa bits
    }
    setBits(Self.signBit, bits: sign == .minus ? 1 : 0)
    setBits(expLower, bits: exp)
    setBits(manLower, bits: dmant)
    return res
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Double/BID conversions
  
  static func bid(from x:Double, _ rnd_mode:Rounding) -> Self {
    // Unpack the input
    let expMask = 1<<11 - 1
    var s = x.sign, e = (Int(x.bitPattern >> 52)) & expMask, t = 0
    var c = UInt128(high: 0, low: x.significandBitPattern)
    
    if e == 0 {
      if x.isZero { return zero(s) }
      
      // denormalizd number
      let l = c.leadingZeroBitCount - (64 - 53)
      c <<= 1
      e = -(l + 1074)
    } else if e == expMask {
      if x.isInfinite { return infinite(s) }
      if x.isNaN { return nan(s, Int(c._lowWord)) }
    } else {
      c |= 1 << 52 // set upper bit
      e -= 1075
      t = c.trailingZeroBitCount
    }
    
    // Now -1126<=e<=971 (971 for max normal, -1074 for min normal,
    // -1126 for min denormal)
    
    // Treat like a quad input for uniformity, so (2^{113-53} * c * r) >> 320,
    // where 320 is the truncation value for the reciprocal multiples, exactly
    // five 64-bit words. So we shift 113-53=60 places
    //
    // Remember to compensate for the fact that exponents are integer for quad
    c = c << 60  //sll128_short(c.hi, c.lo, 60)
    t += (113 - 53)
    e -= (113 - 53) // Now e belongs [-1186;911].
    
    // Check for "trivial" overflow, when 2^e * 2^112 > 10^emax * 10^d.
    // We actually check if e >= ceil((emax + d) * log_2(10) - 112)
    // This could be intercepted later, but it's convenient to keep tables
    // smaller
    if e >= 211 {
      // state.formUnion([.overflow, .inexact])
      return overflow(s, rnd_mode: rnd_mode)
    }
    // Now filter out all the exact cases where we need to specially force
    // the exponent to 0. We can let through inexact cases and those where the
    // main path will do the right thing anyway, e.g. integers outside coeff range.
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
        cint = cint >> -e // srl128(cint.hi, cint.low, -e)
        if cint < largestNumber+1 {  //((cint.components.high == 0) && (cint.components.low < ID.largestNumber+1)) { // MAX_NUMBERP1)) {
          return Self(sign: s, exponent: exponentBias,
                      mantissa: Mantissa(cint._lowWord))  // return_bid32(s, EXPONENT_BIAS, Int(cint.low))
        }
      } else if a <= 48 {
        var pow5 = Self.coefflimitsBID32(a)
        cint = cint >> t  // srl128(cint.hi, cint.low, t)
        if cint <= pow5 { // le128(cint.hi, cint.low, pow5.hi, pow5.low) {
          var cc = cint
          pow5 = power5(a)
          cc = mul128x128Low(cc, pow5)
          return Self(sign: s, exponent: exponentBias-a,
                      mantissa: Mantissa(cc._lowWord)) // return_bid32(s, EXPONENT_BIAS - a, Int(cc.low))
        }
      }
    }
    
    // Check for "trivial" underflow, when 2^e * 2^113 <= 10^emin * 1/4,
    // so test e <= floor(emin * log_2(10) - 115)
    // In this case just fix ourselves at that value for uniformity.
    //
    // This is important not only to keep the tables small but to maintain the
    // testing of the round/sticky words as a correct rounding method
    if e <= -450 {
      e = -450
    }
    
    // Now look up our exponent e, and the breakpoint between e and e+1
    let m_min = Tables.bid_breakpoints_bid32[e+450]
    var e_out = Int(Tables.bid_exponents_bid32[e+450])
    
    // Choose exponent and reciprocal multiplier based on breakpoint
    var r:UInt256
    if c <= m_min { // le128(c.components.high, c.components.low, m_min.high, m_min.low) {
      r = Tables.bid_multipliers1_bid32[e+450]
    } else {
      r = Tables.bid_multipliers2_bid32[e+450]
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
    let ind = roundboundIndex(rnd_mode, s == .minus, Int(c_prov))
    if bid_roundbound_128[ind] < UInt128(high: z.w[4], low: z.w[3]) { // z.w[4], z.w[3])) {
      c_prov += 1
      let max = largestNumber+1
      if c_prov == max {
        c_prov = max/10
        e_out += 1
        //      } else if c_prov == max/10 && e_out == 0 {
        // let ind = roundboundIndex(rnd_mode, false, 0) >> 2
        //            if ((((ind & 3) == 0) && (z.w[4] <= 17524406870024074035)) ||
        //                ((ind + (s & 1) == 2) && (z.w[4] <= 16602069666338596454))) {
        //                state.insert(.underflow)
        //            }
      }
    }
    
    // Check for overflow
    if e_out > 90 {
      // state.formUnion([.overflow, .inexact])
      return overflow(s, rnd_mode: rnd_mode)
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
    return Self(sign: s, exponent: e_out+exponentBias, mantissa: c_prov) // return_bid32 (s, e_out, Int(c_prov))
  }
  
  static func bid (from x:UInt64, _ rnd_mode:Rounding) -> Self {
    // Get BID from a 64-bit unsigned integer
    if x <= Self.largestNumber { // x <= 10^7-1 and the result is exact
      return Self(sign: .plus, exponent: 0, mantissa: Mantissa(x))
    } else { // x >= 10^7 and the result may be inexact
      // the smallest x is 10^7 which has 8 decimal digits
      // the largest x is 0xffffffffffffffff = 18446744073709551615 w/ 20 digits
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
      var res: UInt32
      if (q <= 19) {
        bid_round64_2_18 ( // would work for 20 digits too if x fits in 64 bits
          q, ind, x, &res64, &incr_exp,
          &is_midpoint_lt_even, &is_midpoint_gt_even,
          &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = UInt32(res64)
      } else { // q = 20
        let x128 = UInt128(high: 0, low:x)
        bid_round128_19_38 (q, ind, x128, &res128, &incr_exp,
                            &is_midpoint_lt_even, &is_midpoint_gt_even,
                            &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = UInt32(res128._lowWord) // res.w[1] is 0
      }
      if incr_exp != 0 {
        ind += 1
      }
      // set the inexact flag
      //      if (is_inexact_lt_midpoint || is_inexact_gt_midpoint ||
      //          is_midpoint_lt_even || is_midpoint_gt_even)
      //          *pfpsf |= BID_INEXACT_EXCEPTION;
      // general correction from RN to RA, RM, RP, RZ; result uses ind for exp
      if (rnd_mode != BID_ROUNDING_TO_NEAREST) {
        if ((rnd_mode == BID_ROUNDING_UP && is_inexact_lt_midpoint) ||
           ((rnd_mode == BID_ROUNDING_TIES_AWAY || rnd_mode == BID_ROUNDING_UP)
             && is_midpoint_gt_even)) {
          res = res + 1;
          if res == 10000000 { // res = 10^7 => rounding overflow
            res = 1000000 // 10^6
            ind = ind + 1;
          }
        } else if ((is_midpoint_lt_even || is_inexact_gt_midpoint) &&
                   (rnd_mode == BID_ROUNDING_DOWN ||
                    rnd_mode == BID_ROUNDING_TO_ZERO)) {
          res = res - 1;
          // check if we crossed into the lower decade
          if res == 999999 { // 10^6 - 1
            res = 9999999 // 10^7 - 1
            ind = ind - 1;
          }
        } else {
          // exact, the result is already correct
        }
      }
      return Self(sign: .plus, exponent: ind, mantissa: Mantissa(res))
      //      if (res < 0x00800000) { // res < 2^23
      //        res = ((ind + 101) << 23) | res;
      //      } else { // res >= 2^23
      //        res = 0x60000000 | ((ind + 101) << 21) | (res & 0x001fffff);
      //      }
    }
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
      // However, for generality and possible uses outside the frame of IEEE 754
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
      // (because the largest value is 99999999999999999999999999999999999999 +
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
      let shift = bid_Ex128m128[ind];    // in [2, 63] but have to consider two cases
      if (ind <= 18) {    // if 0 <= ind <= 18
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
        if (Cstar.components.low & 0x01 != 0) {    // Cstar is odd; MP in [EVEN, ODD]
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
    // find power of 10 just greater than sig
    var tenPower = T(10), digits = 1
    while sig >= tenPower { tenPower *= 10; digits += 1 }
    return digits
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
  
  static func bid_midpoint64(_ i:Int) -> UInt64 { 5 * power10(i-1) }
  
  // bid_midpoint128[i - 20] = 1/2 * 10^i = 5 * 10^(i-1), 20 <= i <= 38
  static func bid_midpoint128(_ i:Int) -> UInt128 { 5 * power10(i-1) }
  
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
  static func bid_ten2k128(_ i:Int) -> UInt128 { power10(i) }
  
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
  
  // Values of mask in the right position to obtain the high Ex - 128 or Ex - 192
  // bits of the fraction from C * kx, 1 <= x <= 37; the fraction consists of
  // the low Ex bits in C * kx
  static func bid_mask128(_ i: Int) -> UInt64 {
    return (UInt64(1) << bid_Ex128m128[i-1]) - 1
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
      // BID_ROUNDING_TO_NEAREST
      midPoint,      // positive|even
      midPoint - 1,  // positive|odd
      midPoint,      // negative|even
      midPoint - 1,  // negative|odd
      
      // BID_ROUNDING_DOWN
      UInt128.max,   // positive|even
      UInt128.max,   // positive|odd
      UInt128.min,   // negative|even
      UInt128.min,   // negative|odd
      
      // BID_ROUNDING_UP
      UInt128.min,   // positive|even
      UInt128.min,   // positive|odd
      UInt128.max,   // negative|even
      UInt128.max,   // negative|odd
      
      // BID_ROUNDING_TO_ZERO
      UInt128.max,   // positive|even
      UInt128.max,   // positive|odd
      UInt128.max,   // negative|even
      UInt128.max,   // negative|odd
      
      // BID_ROUNDING_TIES_AWAY
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
    
    func getBits(_ range: ClosedRange<Int>) -> Int { gBits(range, from: dpd) }
    func getBit(_ bit:Int) -> Int { ((1 << bit) & dpd) == 0 ? 0 : 1 }
    
    // decode the 10-bit dpd number
    let select = (getBit(3), getBits(1...2), getBits(5...6))
    let bit0 = getBit(0), bit4 = getBit(4), bit7 = getBit(7)
    switch select {
      // this case covers about 50% of the numbers
      case (0, _, _):
        return getBits(7...9)*100 + getBits(4...6)*10 + getBits(0...2)
        
      // following 3 cases cover 37.5% of the numbers
      case (1, 0b00, _):
        return getBits(7...9)*100 + getBits(4...6)*10 + bit0 + 8
      case (1, 0b01, _):
        return getBits(7...9)*100 + (bit4 + 8)*10 + getBits(5...6)<<1 + bit0
      case (1, 0b10, _):
        return (bit7 + 8)*100 + getBits(4...6)*10 + getBits(8...9)<<1 + bit0
        
      // next 3 cases cover another 9.375% of the numbers
      case (1, 0b11, 0b00):
        return (bit7 + 8)*100 + (bit4 + 8)*10 + getBits(8...9)<<1 + bit0
      case (1, 0b11, 0b01):
        return (bit7 + 8)*100 + (getBits(8...9)<<1 + bit4)*10 + bit0 + 8
      case (1, 0b11, 0b10):
        return getBits(7...9)*100 + (bit4 + 8)*10 + bit0 + 8
        
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
    
    func setBits(_ range: ClosedRange<Int>, bits: Int) {
      sBits(range, bits: bits, in: &res)
    }
    
    func setBit(_ bit: Int, bits: Int) { sBit(bit, bits: bits, in: &res) }
    
    func setBits4to6() { setBits(4...6, bits: tens) }
    func setBits7to9() { setBits(7...9, bits: hundreds) }
    
    func setBit0() { setBit(0, bits: ones) }
    func setBit4() { setBit(4, bits: tens) }
    func setBit7() { setBit(7, bits: hundreds) }
  
    switch (hundreds>7, tens>7, ones>7) {
      case (false, false, false):
        setBits7to9(); setBits4to6(); setBits(0...2, bits: ones)
      case (false, false, true):
        res = 0b1000  // base encoding
        setBits7to9(); setBits4to6(); setBit0()
      case (false, true, false):
        res = 0b1010  // base encoding
        setBits7to9(); setBit4(); setBits(5...6, bits: ones>>1); setBit0()
      case (true, false, false):
        res = 0b1100  // base encoding
        setBits4to6(); setBits(8...9, bits: ones>>1); setBit7(); setBit0()
      case (true, true, false):
        res = 0b1110  // base encoding
        setBit7(); setBit4(); setBits(8...9, bits: ones>>1); setBit0()
      case (true, false, true):
        res = 0b010_1110  // base encoding
        setBit7(); setBits(8...9, bits: tens>>1); setBit4(); setBit0()
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
    if yexp - xexp > Self.numberOfDigits-1 { return false }
    for _ in 0..<(yexp - xexp) {
      // recalculate y's significand upwards
      yman *= 10
      if yman > Self.largestNumber { return false }
    }
    return xman == yman
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
    if xexp - yexp > Self.numberOfDigits-1 { return xsign == .plus }
    
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
        var sign = FloatingPointSign.plus
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
    if exponentDiff > numberOfDigits {
      let binExpon = Double(mantissaA).exponent
      let scaleCA = estimateDecDigits(binExpon)
      let d2 = 16 - scaleCA
      if exponentDiff > d2 {
        exponentDiff = d2
        exponentB = exponentA - exponentDiff
      }
    }
    
    let signAB = signA != signB ? FloatingPointSign.minus : .plus
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
      if rounding == BID_ROUNDING_DOWN { signA = .minus }
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
    
    if n_digits <= numberOfDigits {
      return Self(sign: signA, exponent: exponentB, mantissa: Mantissa(P))
    }
    
    let extra_digits = n_digits - numberOfDigits
    
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
    
    if rounding == BID_ROUNDING_TO_NEAREST {
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
      if C1 > largestNumber/10 { // 999_999_999_999_999 {
        C1 = 0 // x = x & 0xfe00000000000000    // clear G6-G12 and the payload bits
      //} else {
        // nt(C1)) // x = x & 0xfe03ffffffffffff    // clear G6-G12
      }
      if x.isSNaN {
        // set invalid flag
        // pfpsf.insert(.invalidOperation)
        // return quiet (SNaN)
        return Self(sign: x_sign, exponent: -exponentBias, mantissa: x.nanQuiet())
      } else {    // QNaN
        return nan(x_sign, Int(C1))
      }
    } else if x.isInfinite {
      return x
    }
    // unpack x
//    var C1: UInt64
//    if ((x & MASK_STEERING_BITS) == MASK_STEERING_BITS) {
//      // if the steering bits are 11 (condition will be 0), then
//      // the exponent is G[0:w+1]
//      exp = Int((x & MASK_BINARY_EXPONENT2) >> EXPONENT_SHIFT_LARGE64) - EXPONENT_BIAS
//      C1 = (x & MASK_BINARY_SIG2) | MASK_BINARY_OR2
//      if C1 > MAX_NUMBER {    // non-canonical
//        C1 = 0;
//      }
//    } else {    // if ((x & MASK_STEERING_BITS) != MASK_STEERING_BITS)
//      exp = Int((x & MASK_BINARY_EXPONENT1) >> EXPONENT_SHIFT_SMALL64) - EXPONENT_BIAS
//      C1 = (x & MASK_BINARY_SIG1)
//    }
    
    // if x is 0 or non-canonical return 0 preserving the sign bit and
    // the preferred exponent of MAX(Q(x), 0)
    if C1 == 0 {
      if exp < 0 {
        exp = 0
      }
      return Self(sign:x_sign, exponent:exp, mantissa: 0)
    }
    // x is a finite non-zero number (not 0, non-canonical, or special)
    switch rmode {
      case BID_ROUNDING_TO_NEAREST, BID_ROUNDING_TIES_AWAY:
        // return 0 if (exp <= -(p+1))
        if exp <= -17 {
          // res = x_sign | zero
          //pfpsf.insert(.inexact)
          return zero(x_sign)
        }
      case BID_ROUNDING_DOWN:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          if x_sign != .plus {
            return Self(sign: .minus, exponent: 0, mantissa: 1)
            //res = (zero+1) | SIGN_MASK64  // 0xb1c0000000000001
          } else {
            return zero(.plus)
          }
          //pfpsf.insert(.inexact)
        }
      case BID_ROUNDING_UP:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          if x_sign != .plus {
            return zero(.minus) // res = zero | SIGN_MASK64  // 0xb1c0000000000000
          } else {
            return Self(sign: .plus, exponent: 0, mantissa: 1) // res = zero+1
          }
          //pfpsf.insert(.inexact)
        }
      case BID_ROUNDING_TO_ZERO:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          return zero(x_sign) // x_sign | zero
          //pfpsf.insert(.inexact)
        }
      default: break
    }    // end switch ()
    
    // q = nr. of decimal digits in x (1 <= q <= 54)
    //  determine first the nr. of bits in x
    let q = digitsIn(C1)
    if exp >= 0 {
      // the argument is an integer already
      return x
    }
    
    var ind: Int
    var P128 = UInt128().components, fstar = UInt128().components
    switch rmode {
      case BID_ROUNDING_TO_NEAREST:
        if ((q + exp) >= 0) {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient; exp will be 0
          ind = -exp;    // 1 <= ind <= 16; ind is a synonym for 'x'
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
          
          if (ind - 1 <= 2) {    // 0 <= ind - 1 <= 2 => shift = 0
            res = Mantissa(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if (ind - 1 <= 21) {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Mantissa((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          // since round_to_even, subtract 1 if current result is odd
          if (res & 0x1 != 0) && (fstar.high == 0) &&
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
          return Self(sign: x_sign, exponent: 0, mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
         // pfpsf.insert(.inexact)
        }
      case BID_ROUNDING_TIES_AWAY:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient; exp will be 0
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
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
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
          return Self(sign: x_sign, exponent: 0, mantissa: res)
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
        }
      case BID_ROUNDING_DOWN:
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
          return Self(sign: x_sign, exponent: 0, mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp <= 0
          // the result is +0 or -1
          if x_sign != .plus {
            return Self(sign: .minus, exponent: 0, mantissa: 1) // 0xb1c0000000000001
          } else {
            return zero(.plus)
          }
          // pfpsf.insert(.inexact)
        }
      case BID_ROUNDING_UP:
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
          return Self(sign: x_sign, exponent: 0, mantissa: res) // x_sign | zero | res
        } else {    // if exp < 0 and q + exp <= 0
          // the result is -0 or +1
          if x_sign != .plus {
            return zero(.minus)
          } else {
            return Self(sign: .plus, exponent: 0, mantissa: 1)
          }
        }
      case BID_ROUNDING_TO_ZERO:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient; exp will be 0
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
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
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
          return Self(sign: x_sign, exponent: 0, mantissa: Mantissa(res))
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
          // pfpsf.insert(.inexact)
        }
      default: break
    }    // end switch ()
    return Self(sign: x_sign, exponent: exp, mantissa: res)
  }
}

// MARK: - Extended UInt Definitions
// These are usd primarily for table and extended calculation storage
struct UInt512 { var w = [UInt64](repeating: 0, count: 8) }
struct UInt384 { var w = [UInt64](repeating: 0, count: 6) }
struct UInt256 { var w = [UInt64](repeating: 0, count: 4) }
struct UInt192 { var w = [UInt64](repeating: 0, count: 3) }

// MARK: - Status and Rounding Type Definitions

public typealias Rounding = FloatingPointRoundingRule
let BID_ROUNDING_UP = Rounding.up
let BID_ROUNDING_DOWN = Rounding.down
let BID_ROUNDING_TO_ZERO = Rounding.towardZero
let BID_ROUNDING_TO_NEAREST = Rounding.toNearestOrEven
let BID_ROUNDING_TIES_AWAY = Rounding.toNearestOrAwayFromZero

public struct Status: OptionSet, CustomStringConvertible {
    public let rawValue: Int32
    
    /* IEEE extended flags only */
    private static let DEC_Conversion_syntax    = 0x00000001
    private static let DEC_Division_by_zero     = 0x00000002
    private static let DEC_Division_impossible  = 0x00000004
    private static let DEC_Division_undefined   = 0x00000008
    private static let DEC_Insufficient_storage = 0x00000010 /* [when malloc fails]  */
    private static let DEC_Inexact              = 0x00000020
    private static let DEC_Invalid_context      = 0x00000040
    private static let DEC_Invalid_operation    = 0x00000080
    private static let DEC_Lost_digits          = 0x00000100
    private static let DEC_Overflow             = 0x00000200
    private static let DEC_Clamped              = 0x00000400
    private static let DEC_Rounded              = 0x00000800
    private static let DEC_Subnormal            = 0x00001000
    private static let DEC_Underflow            = 0x00002000
    
    public static let conversionSyntax    = Status(rawValue: Int32(DEC_Conversion_syntax))
    public static let divisionByZero      = Status(rawValue: Int32(DEC_Division_by_zero))
    public static let divisionImpossible  = Status(rawValue: Int32(DEC_Division_impossible))
    public static let divisionUndefined   = Status(rawValue: Int32(DEC_Division_undefined))
    public static let insufficientStorage = Status(rawValue: Int32(DEC_Insufficient_storage))
    public static let inexact             = Status(rawValue: Int32(DEC_Inexact))
    public static let invalidContext      = Status(rawValue: Int32(DEC_Invalid_context))
    public static let lostDigits          = Status(rawValue: Int32(DEC_Lost_digits))
    public static let invalidOperation    = Status(rawValue: Int32(DEC_Invalid_operation))
    public static let overflow            = Status(rawValue: Int32(DEC_Overflow))
    public static let clamped             = Status(rawValue: Int32(DEC_Clamped))
    public static let rounded             = Status(rawValue: Int32(DEC_Rounded))
    public static let subnormal           = Status(rawValue: Int32(DEC_Subnormal))
    public static let underflow           = Status(rawValue: Int32(DEC_Underflow))
    public static let clearFlags          = Status([])
    
    public static let errorFlags =
      Status(rawValue: Int32(DEC_Division_by_zero | DEC_Overflow |
        DEC_Underflow | DEC_Conversion_syntax | DEC_Division_impossible |
        DEC_Division_undefined | DEC_Insufficient_storage |
        DEC_Invalid_context | DEC_Invalid_operation))
    public static let informationFlags =
      Status(rawValue: Int32(DEC_Clamped | DEC_Rounded |
        DEC_Inexact | DEC_Lost_digits))
    
    public init(rawValue: Int32) { self.rawValue = rawValue }
    
    public var hasError: Bool { !Status.errorFlags.intersection(self).isEmpty}
    public var hasInfo: Bool {
      !Status.informationFlags.intersection(self).isEmpty
    }
    
    public var description: String {
        var str = ""
        if self.contains(.conversionSyntax)   { str += "Conversion syntax, "}
        if self.contains(.divisionByZero)     { str += "Division by zero, " }
        if self.contains(.divisionImpossible) { str += "Division impossible, "}
        if self.contains(.divisionUndefined)  { str += "Division undefined, "}
        if self.contains(.insufficientStorage){str += "Insufficient storage, "}
        if self.contains(.inexact)            { str += "Inexact number, " }
        if self.contains(.invalidContext)     { str += "Invalid context, " }
        if self.contains(.invalidOperation)   { str += "Invalid operation, " }
        if self.contains(.lostDigits)         { str += "Lost digits, " }
        if self.contains(.overflow)           { str += "Overflow, " }
        if self.contains(.clamped)            { str += "Clamped, " }
        if self.contains(.rounded)            { str += "Rounded, " }
        if self.contains(.subnormal)          { str += "Subnormal, " }
        if self.contains(.underflow)          { str += "Underflow, " }
        if str.hasSuffix(", ") { str.removeLast(2) }
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

// Rounding boundaries table, indexed by
// 4 * rounding_mode + 2 * sign + lsb of truncation
// We round up if the round/sticky data is strictly > this boundary
//
// NB: This depends on the particular values of the rounding mode
// numbers, which are supposed to be defined as here:
//
// #define BID_ROUNDING_TO_NEAREST     0x00000
// #define BID_ROUNDING_DOWN           0x00001
// #define BID_ROUNDING_UP             0x00002
// #define BID_ROUNDING_TO_ZERO        0x00003
// #define BID_ROUNDING_TIES_AWAY      0x00004
//
// Some of the shortcuts below in "underflow after rounding" also use
// the concrete values.
//
// So we add a directive here to double-check that this is the case
internal func roundboundIndex(_ round:Rounding, _ negative:Bool=false,
                            _ lsb:Int=0) -> Int {
  var index = (lsb & 1) + (negative ? 2 : 0)
  switch round {
    case BID_ROUNDING_TO_NEAREST: index += 0
    case BID_ROUNDING_DOWN: index += 4
    case BID_ROUNDING_UP: index += 8
    case BID_ROUNDING_TO_ZERO: index += 12
    default: index += 16
  }
  return index
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
    return s + addDecimalPointAndExponent(ps, exponent_x, T.numberOfDigits)
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
  
//  // remove leading whitespace characters
//  while ps.hasPrefix(" ") { ps.removeFirst() }
  
  // get first non-whitespace character
  var c = ps.isEmpty ? "\0" : ps.removeFirst()
  
  // detect special cases (INF or NaN)
  if c == "\0" || (c != "." && c != "-" && c != "+" && (c < "0" || c > "9")) {
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
  var sign = FloatingPointSign.plus
  if c == "-" {
    sign = .minus
  }
  
  // get next character if leading +/- sign
  if c == "-" || c == "+" {
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  // if c isn"t a decimal point or a decimal digit, return NaN
  if c != "." && (c < "0" || c > "9") {
    // return NaN
    return T.nan(sign, 0)
  }
  
  var rdx_pt_enc = false
  var right_radix_leading_zeros = 0
  var coefficient_x = 0
  
  // detect zero (and eliminate/ignore leading zeros)
  if c == "0" || c == "." {
    if c == "." {
      rdx_pt_enc = true
      c = ps.isEmpty ? "\0" : ps.removeFirst()
    }
    // if all numbers are zeros (with possibly 1 radix point, the number
    // is zero
    // should catch cases such as: 000.0
    while c == "0" {
      c = ps.isEmpty ? "\0" : ps.removeFirst()
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
            right_radix_leading_zeros = T.exponentBias -
            right_radix_leading_zeros
            if right_radix_leading_zeros < 0 {
              right_radix_leading_zeros = 0
            }
            let bits = T.Mantissa(right_radix_leading_zeros)
            return T(sign: sign, exponent: 0, mantissa: bits)
//            return sign == .minus ? -T(bitPattern: bits, bidEncoding: true)
//                                  : T(bitPattern: bits, bidEncoding: true)
          }
          c = ps.isEmpty ? "\0" : ps.removeFirst()
        } else {
          // if 2 radix points, return NaN
          return T.nan(sign, 0)
        }
      } else if ps.isEmpty {
        right_radix_leading_zeros = T.exponentBias - right_radix_leading_zeros
        if right_radix_leading_zeros < 0 {
          right_radix_leading_zeros = 0
        }
        let bits = T.Mantissa(right_radix_leading_zeros)
        return T(sign: sign, exponent: 0, mantissa: bits)
//        return sign == .minus ? -T(bitPattern: bits, bidEncoding: true)
//                              : T(bitPattern: bits, bidEncoding: true)
      }
    }
  }
  
  var ndigits = 0
  var dec_expon_scale = 0
  var midpoint = 0
  var rounded_up = 0
  var add_expon = 0
//Decimal32.exponentBias  var rounded = 0
  while (c >= "0" && c <= "9") || c == "." {
    if c == "." {
      if rdx_pt_enc {
        // return NaN
        return T.nan(sign, 0)
      }
      rdx_pt_enc = true
      c = ps.isEmpty ? "\0" : ps.removeFirst()
      continue
    }
    if rdx_pt_enc { dec_expon_scale += 1 }
    
    ndigits+=1
    if ndigits <= 7 {
      coefficient_x = (coefficient_x << 1) + (coefficient_x << 3);
      coefficient_x += c.wholeNumberValue ?? 0
    } else if ndigits == 8 {
      // coefficient rounding
      switch round {
        case BID_ROUNDING_TO_NEAREST:
          midpoint = (c == "5" && (coefficient_x & 1 == 0)) ? 1 : 0;
          // if coefficient is even and c is 5, prepare to round up if
          // subsequent digit is nonzero
          // if str[MAXDIG+1] > 5, we MUST round up
          // if str[MAXDIG+1] == 5 and coefficient is ODD, ROUND UP!
          if c > "5" || (c == "5" && (coefficient_x & 1) != 0) {
            coefficient_x+=1
            rounded_up = 1
          }
        case BID_ROUNDING_DOWN:
          if sign == .minus { coefficient_x+=1; rounded_up=1 }
        case BID_ROUNDING_UP:
          if sign != .minus { coefficient_x+=1; rounded_up=1 }
        case BID_ROUNDING_TIES_AWAY:
          if c >= "5" { coefficient_x+=1; rounded_up=1 }
        default: break
      }
      if coefficient_x == 10000000 {
        coefficient_x = 1000000
        add_expon = 1
      }
//      if c > "0" {
//        rounded = 1
//      }
      add_expon += 1
    } else { // ndigits > 8
      add_expon+=1
      if midpoint != 0 && c > "0" {
        coefficient_x+=1
        midpoint = 0;
        rounded_up = 1;
      }
//      if c > "0" {
//        rounded = 1;
//      }
    }
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  add_expon -= dec_expon_scale + Int(right_radix_leading_zeros)
  
  if c == "\0" {
//    if rounded != 0 {
//      T.state.insert(.inexact)
//    }
    return T(sign: sign, exponent: add_expon+T.exponentBias,
             mantissa: T.Mantissa(coefficient_x))
//    return T(sign: sign,
//             exponentBitPattern: T.RawExponent(add_expon + T.exponentBias),
//             significantBitPattern: T.BitPattern(coefficient_x))
  }
  
  if c != "e" {
    // return NaN
    return T.nan(sign, 0)
  }
  c = ps.isEmpty ? "\0" : ps.removeFirst()
  let sgn_expon = (c == "-") ? 1 : 0
  var expon_x = 0
  if c == "-" || c == "+" {
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  if c == "\0" || c < "0" || c > "9" {
    // return NaN
    return T.nan(sign, 0)
  }
  
  while (c >= "0") && (c <= "9") {
    if expon_x < (1<<20) {
      expon_x = (expon_x << 1) + (expon_x << 3)
      expon_x += c.wholeNumberValue ?? 0
    }
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  if c != "\0" {
    // return NaN
    return T.nan(sign, 0)
  }
  
//  if rounded != 0 {
//    T.state.insert(.inexact)
//  }
  
  if sgn_expon != 0 {
    expon_x = -expon_x
  }
  
  expon_x += add_expon + T.exponentBias
  
  if expon_x < 0 {
    if rounded_up != 0 {
      coefficient_x-=1
    }
  }
  return T(sign:sign, exponent:expon_x, mantissa:T.Mantissa(coefficient_x)) // round: rounded)
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

