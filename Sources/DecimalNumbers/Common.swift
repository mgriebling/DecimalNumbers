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

import Foundation  // for Locale
import UInt128

///
/// Groups together algorithms that can be used by all Decimalxx variants
///

// MARK: - Generic Integer Decimal Field type

public typealias Sign = FloatingPointSign
public typealias Rounding = FloatingPointRoundingRule
public typealias IntRange = ClosedRange<Int>

public enum DecimalEncoding { case bid, dpd }

protocol IntDecimal : Codable, Hashable {
  
  associatedtype RawData : UnsignedInteger & FixedWidthInteger
  associatedtype RawBitPattern : UnsignedInteger & FixedWidthInteger
  associatedtype RawSignificand : UnsignedInteger & FixedWidthInteger
  
  /// Storage of the Decimal number in a raw binary integer decimal
  /// encoding as per IEEE STD 754-2008
  var data: RawData { get set }
  
  //////////////////////////////////////////////////////////////////
  /// Initializers
  
  /// Initialize with a raw data word
  init(_ word: RawData)
  
  /// Initialize with sign, biased exponent, and unsigned significand
  init(sign: Sign, expBitPattern: Int, sigBitPattern: RawBitPattern)

  
  init(nan payload: RawSignificand, signaling: Bool)
  
  //////////////////////////////////////////////////////////////////
  /// Conversions from/to densely packed decimal numbers
  
  /// Initializes the number from a DPD number
  init(dpd: RawData)
  
  /// Returns a DPD number
  var dpd: RawData { get }
  
  //////////////////////////////////////////////////////////////////
  /// Essential data to extract or update from the fields
  
  /// Sign of the number
  var sign: Sign { get set }
  
  /// Encoded unsigned exponent of the number
  var expBitPattern: Int { get }
  
  /// Encoded unsigned binary integer decimal significand of the number
  var sigBitPattern: RawBitPattern { get }
  
  /// Setting requires both the significand and exponent so that a
  /// decision can be made on whether the significand is small or large.
  mutating func set(exponent: Int, sigBitPattern: RawBitPattern)
  
  //////////////////////////////////////////////////////////////////
  /// Special number definitions
  static var snan: Self { get }
  
  static func zero(_ sign: Sign) -> Self
  static func nan(_ sign: Sign, _ payload: RawSignificand) -> Self
  static func infinite(_ sign: Sign) -> Self
  static func max(_ sign: Sign) -> Self
  
  //////////////////////////////////////////////////////////////////
  /// Decimal number definitions
  static var signBit: Int { get }
  static var specialBits: IntRange { get }
  
  static var exponentBias: Int       { get }
  static var exponentBits: Int       { get }
  static var maxEncodedExponent: Int { get }
  static var minEncodedExponent: Int { get }
  static var maximumDigits:  Int     { get }

  static var largestNumber: RawBitPattern { get }
  
  // For large significand
  static var exponentLMBits: IntRange { get }
  static var largeSignificandBits: IntRange { get }
  
  // For small significand
  static var exponentSMBits: IntRange { get }
  static var smallSignificandBits: IntRange { get }
}

///
/// Free functionality when complying with IntegerDecimalField
extension IntDecimal {
  
  static var highSignificandBit: RawBitPattern {
    RawBitPattern(1) << exponentLMBits.lowerBound
  }
  
  static var largestSignificand: RawBitPattern { (largestNumber+1)/10 }
  static var largestBID : Self { max() }
  
  // Doesn't change for the different types of Decimals
  static var minEncodedExponent: Int { 0 }
  
  /// These bit fields can be predetermined just from the size of
  /// the number type `RawDataFields` `bitWidth`
  static var maxBit: Int              { RawData.bitWidth - 1 }
  static var signBit: Int             { maxBit }
  static var specialBits: IntRange    { maxBit-2 ... maxBit-1 }
  static var nanBitRange: IntRange    { maxBit-6 ... maxBit-1 }
  static var infBitRange: IntRange    { maxBit-5 ... maxBit-1 }
  static var nanClearRange: IntRange  { 0 ... maxBit-7 }
  static var g6tog10Range: IntRange   { maxBit-11 ... maxBit-7 }
  
  static var exponentLMBits: IntRange { maxBit-exponentBits ... maxBit-1 }
  static var exponentSMBits: IntRange { maxBit-exponentBits-2 ... maxBit-3 }
  
  // Two significand sizes must be supported
  static var largeSignificandBits: IntRange { 0...maxBit-exponentBits-1 }
  static var smallSignificandBits: IntRange { 0...maxBit-exponentBits-3 }
  
  // masks for clearing bits
  static var sNanRange: IntRange      { 0 ... maxBit-6 }
  static var sInfinityRange: IntRange { 0 ... maxBit-5 }
  
  static func nanQuiet(_ x:RawBitPattern) -> Self {
    Self(RawData(x.clearing(bit:nanBitRange.lowerBound)))
  }
  
  /// bit field definitions for DPD numbers
  static var lowMan: Int    { smallSignificandBits.upperBound }
  static var upperExp1: Int { exponentSMBits.upperBound }
  static var upperExp2: Int { exponentLMBits.upperBound }
  
  static var expLower: IntRange { lowMan...maxBit-6 }
  static var manLower: IntRange { 0...lowMan-1 }
  static var expUpper: IntRange { lowMan+1...lowMan+6 }
  
  /// Bit patterns prefixes for special numbers
  static var nanPattern: Int      { 0b1_1111_0 }
  static var snanPattern: Int     { 0b1_1111_1 }
  static var infinitePattern: Int { 0b1_1110 }
  static var specialPattern: Int  { 0b11 }
  
  static var trailingPattern: Int { 0x3ff }
  
  public init?(_ s: String, rounding rule: Rounding = .toNearestOrEven) {
    self.init(0)
    if let n:Self = numberFromString(s, round: rule) { data = n.data }
    else { return nil }
  }
  
  public init(nan payload: RawSignificand, signaling: Bool) {
    let pattern = signaling ? Self.snanPattern : Self.nanPattern
    let man = payload > Self.largestNumber/10 ? 0 : RawBitPattern(payload)
    self.init(0)
    set(exponent: pattern<<(Self.exponentBits-6), sigBitPattern: man)
  }
  
  @inlinable var sign: Sign {
    get { Sign(rawValue: data.get(bit: Self.signBit))! }
    set { data.set(bit: Self.signBit, with: newValue.rawValue) }
  }
  
  @inlinable var expBitPattern: Int {
    let range = isSmallSig ? Self.exponentSMBits : Self.exponentLMBits
    return data.getInt(range: range)
  }
  
  @inlinable var sigBitPattern: RawBitPattern {
    let range = isSmallSig ? Self.smallSignificandBits
                           : Self.largeSignificandBits
    if isSmallSig {
      return RawBitPattern(data.get(range:range)) + Self.highSignificandBit
    } else {
      return RawBitPattern(data.get(range:range))
    }
  }
  
  static func adjustOverflowUnderflow(_ sign: Sign, _ exp: Int,
                        _ mant: RawBitPattern, _ rmode: Rounding) -> RawData {
    var exp = exp, mant = mant, rmode = rmode
    var raw = RawData(0)
    if mant > largestNumber {
      exp += 1; mant = largestSignificand
    }
    
    // check for possible underflow/overflow
    if exp > maxEncodedExponent || exp < minEncodedExponent {
      if exp < minEncodedExponent {
        // deal with an underflow situation
        if exp + maximumDigits < 0 {
          // underflow & inexact
          if rmode == .down && sign == .minus {
            return raw.setting(bit:signBit).setting(range: manLower, with: 1)
          }
          if rmode == .up && sign == .plus {
            return raw.setting(range: manLower, with: 1)
          }
          return raw.setting(bit: signBit, with: sign.rawValue)
        }
        
        // swap up & down round modes when negative
        if sign == .minus {
          if rmode == .up { rmode = .down }
          else if rmode == .down { rmode = .up }
        }
        
        // determine the rounding table index
        let roundIndex = rmode.raw
        
        // get digits to be shifted out
        let extraDigits = -exp
        mant += RawBitPattern(roundConstTable(roundIndex, extraDigits))
        
        //var Q = UInt128()
        let Q = UInt64(mant).multipliedFullWidth(by:reciprocals10(extraDigits))
        let amount = shortReciprocalScale[extraDigits]
        var C64 = Q.high >> amount
        var remainder_h = UInt128.High(0)
        if rmode == .toNearestOrAwayFromZero {
          if !C64.isMultiple(of: 2) {
            // odd factor so check whether fractional part is exactly 0.5
            let amount2 = 64 - amount
            remainder_h &-= 1  // decrement without overflow check
            remainder_h >>= amount2
            remainder_h &= Q.high
            if remainder_h==0 && Q.low < reciprocals10(extraDigits) {
              C64 -= 1
            }
          }
        }
        return RawData(C64).setting(bit:signBit, with:sign.rawValue)
      }
      
      if mant == 0 {
        if exp > maxEncodedExponent { exp = maxEncodedExponent }
      }
      while mant < largestSignificand && exp > maxEncodedExponent {
        mant = (mant << 3) + (mant << 1)  // times 10
        exp -= 1
      }
      if exp > maxEncodedExponent {
        raw = infinite(sign).data
        switch rmode {
          case .down:
            if sign == .plus { raw = largestBID.data }
          case .towardZero:
            raw = largestBID.data.setting(bit:signBit)
          case .up:
            if sign == .minus {
              raw = largestBID.data.setting(bit:signBit, with:sign.rawValue)
            }
          default: break
        }
        return raw
      }
    }
    return Self(sign:sign, expBitPattern:exp, sigBitPattern: mant).data
  }
  
  /// Note: `exponent` is assumed to be biased
  mutating func set(exponent: Int, sigBitPattern: RawBitPattern) {
    if sigBitPattern < Self.highSignificandBit {
      // large significand
      data.set(range: Self.exponentLMBits, with: exponent)
      data.set(range: Self.largeSignificandBits, with: sigBitPattern)
    } else {
      // small significand
      data.set(range:Self.exponentSMBits, with: exponent)
      data.set(range:Self.smallSignificandBits,
               with:sigBitPattern-Self.highSignificandBit)
      data.set(range:Self.specialBits, with: Self.specialPattern)
    }
  }

  /// Return `self's` pieces all at once with biased exponent
  func unpack() -> (sign:Sign, exp:Int, sigBits:RawBitPattern, valid:Bool) {
    var exponent: Int, sigBits: RawBitPattern
    if isSpecial {
      if isInfinite {
        sigBits = RawBitPattern(data).clearing(range:Self.g6tog10Range)
        if data.get(range: Self.manLower) >= Self.largestSignificand {
          sigBits = RawBitPattern(data).clearing(range: Self.sNanRange)
        }
        if isNaNInf {
          sigBits = RawBitPattern(data).clearing(range: Self.sInfinityRange)
        }
        return (self.sign, 0, sigBits, false)
      }
      // small significand
      exponent = data.getInt(range: Self.exponentSMBits)
      sigBits = RawBitPattern(data.get(range: Self.smallSignificandBits)) +
                          Self.highSignificandBit
      if sigBits > Self.largestNumber { sigBits = 0 }
      return (self.sign, exponent, sigBits, sigBits != 0)
    } else {
      // large significand
      exponent = data.getInt(range: Self.exponentLMBits)
      sigBits = RawBitPattern(data.get(range: Self.largeSignificandBits))
      return (self.sign, exponent, sigBits, sigBits != 0)
    }
  }
  
  // Unpack decimal floating-point number x into sign, exponent, coefficient.
  // In special cases, return floating-point numbers to be used.
  // Coefficient is normalized in the binary sense with postcorrection k,
  // so that x = 10^e * c / 2^k and the range of c is:
  //
  // 2^23 <= c < 2^24   (decimal32)
  // 2^53 <= c < 2^54   (decimal64)
  // 2^112 <= c < 2^113 (decimal128)
  func unpack<T:BinaryFloatingPoint>() ->
  (s: Sign, e: Int, k: Int, c: RawSignificand, f: T?) {
    let x = self.data, s = self.sign, e = self.expBitPattern-Self.exponentBias
    var k = 0, c = RawSignificand(0)
    if self.isSpecial {
      if self.isInfinite {
        if !self.isNaN {
          return (s, e, k, c, Self.inf(s))
        }
        // if (x & (UInt32(1)<<25)) != 0 { status.insert(.invalidOperation) }
        let payload = x.get(range: Self.manLower)
        let high = payload > Self.largestNumber/10 ? 0 : UInt64(payload)
        return (s, e, k, c, Self.nan(high))
      }
      c = RawSignificand(self.sigBitPattern)
      if c > Self.largestNumber {
        return (s, e, k, c, Self.dzero(s))
      }
      k = 0
    } else {
      c = RawSignificand(self.sigBitPattern)
      if c == 0 { return (s, e, k, c, Self.dzero(s)) }
      k = UInt32(c).leadingZeroBitCount - 8
      c = c << k
    }
    return (s, e, k, c, nil)
  }
  
  @inlinable
  static func overflow<T:BinaryFloatingPoint>(_ s:Sign, _ r:Rounding) -> T {
    if r == .towardZero || r == ((s == .minus) ? Rounding.up : .down) {
      return max(s)
    }
    return inf(s)
  }
  
  @inlinable static func max<T:BinaryFloatingPoint>(_ s:Sign) -> T {
    s == .minus ? -T.greatestFiniteMagnitude : .greatestFiniteMagnitude
  }
  
  @inlinable static func dzero<T:BinaryFloatingPoint>(_ s:Sign) -> T {
    s == .minus ? -T.zero : .zero
  }
  
  @inlinable static func inf<T:BinaryFloatingPoint>(_ s:Sign) -> T {
    s == .minus ? -T.infinity : .infinity
  }
  
  @inlinable static func nan<T:BinaryFloatingPoint>(_ c:UInt64) -> T {
      // nan check is incorrect in Double(nan:signalling:)?
    if T.self == Double.self {
      let t:Double = float(.plus, 2047, UInt64((c<<31)+(1<<51)))
      return T(t)
    } else if T.self == Float.self {
      let t:Float = float(.plus, 255, UInt64((c<<2)+(1<<22)))
      return T(t)
    }
    return T.nan
  }
  
  @inlinable static func float<T:BinaryFloatingPoint>(_ s:Sign, _ e:Int,
                                                      _ c:UInt64) -> T {
    if T.self == Double.self {
      return T(Double(sign: s, exponentBitPattern: UInt(e),
                      significandBitPattern: c))
    } else if T.self == Float.self {
      return T(Float(sign: s, exponentBitPattern: UInt(e),
                     significandBitPattern: UInt32(c)))
    }
    return T(sign: s, exponentBitPattern: T.RawExponent(e),
             significandBitPattern: T.RawSignificand(c))
  }
  
  /// Return `dpd` pieces all at once
  static func unpack(dpd: RawData) ->
              (sign: Sign, exponent: Int, high: Int, trailing: RawBitPattern) {
    let sgn = dpd.get(bit: signBit) == 1 ? Sign.minus : .plus
    var exponent, high: Int, trailing: RawBitPattern
    let expRange2: IntRange
    
    if dpd.get(range: specialBits) == specialPattern {
      // small significand
      expRange2 = (upperExp1-1)...upperExp1
      high = dpd.get(bit: upperExp1-2) + 8
    } else {
      // large significand
      expRange2 = (upperExp2-1)...upperExp2
      high = dpd.getInt(range: upperExp1-2...upperExp1)
    }
    exponent = dpd.getInt(range: expLower) +
                    dpd.getInt(range: expRange2) << (exponentBits-2)
    trailing = RawBitPattern(dpd.get(range: 0...lowMan-1))
    return (sgn, exponent, high, trailing)
  }
  
  @inlinable func nanQuiet() -> RawBitPattern {
    RawBitPattern(data.clearing(bit:Self.nanBitRange.lowerBound))
  }
  
  ///////////////////////////////////////////////////////////////////////
  /// Special number definitions
  @inlinable static func infinite(_ s: Sign = .plus) -> Self {
    Self(sign: s, expBitPattern: infinitePattern<<(exponentBits - 5), sigBitPattern: 0)
  }
  
  @inlinable static func max(_ s: Sign = .plus) -> Self {
    Self(sign:s, expBitPattern:maxEncodedExponent, sigBitPattern:largestNumber)
  }
  
  static func overflow(_ sign: Sign, rndMode: Rounding) -> Self {
    if rndMode == .towardZero || rndMode == (sign != .plus ? .up : .down) {
      return max(sign)
    } else {
      return infinite(sign)
    }
  }
  
  @inlinable static var snan: Self {
    Self(sign: .plus, expBitPattern: snanPattern<<(exponentBits-6),
         sigBitPattern: 0)
  }
  
  @inlinable static func zero(_ sign: Sign = .plus) -> Self {
    Self(sign: sign, expBitPattern: exponentBias, sigBitPattern: 0)
  }
  
  @inlinable
  static func nan(_ sign:Sign = .plus, _ payload:RawSignificand = 0) -> Self {
    let man = payload > largestNumber/10 ? 0 : RawBitPattern(payload)
    return Self(sign:sign, expBitPattern:nanPattern<<(exponentBits-6),
                sigBitPattern:man)
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Handy routines for testing different aspects of the number
  @inlinable var nanBits: Int { data.getInt(range: Self.nanBitRange) }
  
  var isSmallSig: Bool { isSpecial }
  var isNaN: Bool      { nanBits & Self.nanPattern == Self.nanPattern }
  var isSNaN: Bool     { nanBits & Self.snanPattern == Self.snanPattern }
  
  var isFinite: Bool {
    let infinite = Self.infinitePattern
    let data = data.getInt(range: Self.signBit-5...Self.signBit-1)
    return (data & infinite != infinite)
  }
  
  @inlinable
  var isSpecial: Bool {
    data.get(range: Self.specialBits) == Self.specialPattern
  }
  
  @inlinable
  var isNaNInf: Bool {
    nanBits & Self.nanPattern == Self.infinitePattern<<1
  }
  
  @inlinable
  var isInfinite: Bool {
    let data = data.getInt(range: Self.infBitRange)
    return (data & Self.infinitePattern) == Self.infinitePattern
  }
  
  @inlinable
  var isValid: Bool {
    if isNaN { return false }
    if isSpecial {
      if isInfinite { return false }
      if sigBitPattern>Self.largestNumber || sigBitPattern==0 { return false }
    } else {
      if sigBitPattern == 0 { return false }
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
      return sigBitPattern <= Self.largestNumber
    } else {
      return true
    }
  }
  
  /// if exponent < `minEncodedExponent`, the number may be subnormal
  private func checkNormalScale(_ exp: Int, _ mant: RawBitPattern) -> Bool {
    if exp < Self.minEncodedExponent+Self.maximumDigits-1 {
      let tenPower = _power(RawBitPattern(10), to: exp)
      let mantPrime = mant * tenPower
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
      return sigBitPattern > Self.largestNumber
    } else {
      return sigBitPattern == 0
    }
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Convert to/from BID/DPD numbers
  
  /// Create a new BID number from the `dpd` DPD number.
  init(dpd: RawData) {
    
    func getNan() -> Int { dpd.getInt(range: Self.nanBitRange) }
    
    // Convert the dpd number to a bid number
    var (sign, exp, high, trailing) = Self.unpack(dpd: dpd)
    var nan = false
    let nanValue = getNan()
    if (nanValue & Self.nanPattern) == (Self.infinitePattern<<1) {
      self = Self.infinite(sign); return
    } else if (nanValue & Self.nanPattern) == Self.nanPattern {
      nan = true; exp = 0
    }
    
    let mask = Self.trailingPattern
    let mils = ((Self.maximumDigits - 1) / 3) - 1
    let shift = mask.bitWidth - mask.leadingZeroBitCount
    var mant = RawBitPattern(high)
    for i in stride(from: shift*mils, through: 0, by: -shift) {
      mant *= 1000
      mant += RawBitPattern(Self.intFrom(dpd: Int(trailing >> i) & mask))
    }
    
    if nan { self = Self.nan(sign, RawSignificand(mant)) }
    else { self.init(sign: sign, expBitPattern: exp, sigBitPattern: mant) }
  }
  
  var quantum: Self {
    if self.isInfinite { return Self.infinite() }
    if self.isNaN { return Self.nan() }
    let exp = self.expBitPattern
    return Self(sign: .plus, expBitPattern: exp, sigBitPattern: 1)
  }
  
  /// Convert `self` to a DPD number.
  var dpd: RawData {
    var res : RawData = 0
    var (sign, exp, significand, _) = unpack()
    var trailing = significand.get(range: Self.manLower) // & 0xfffff
    var nanb = false
    
    if self.isNaNInf {
      return Self.infinite(sign).data
    } else if self.isNaN {
      if trailing > Self.largestNumber/10 {
        trailing = 0
      }
      significand = Self.RawBitPattern(trailing); exp = 0; nanb = true
    } else {
      if significand > Self.largestNumber { significand = 0 }
    }
    
    let mils = ((Self.maximumDigits - 1) / 3) - 1
    let shift = 10
    var dmant = RawSignificand(0)
    for i in stride(from: 0, through: shift*mils, by: shift) {
      dmant |= RawSignificand(Self.intToDPD(Int(significand % 1000))) << i
      significand /= 1000
    }
    
    let signBit = Self.signBit
    let expLower = Self.smallSignificandBits.upperBound...signBit-6
    let manLower = 0...Self.smallSignificandBits.upperBound-1
    
    if significand >= 8 {
      let expUpper = signBit-4...signBit-3
      let manUpper = signBit-5...signBit-5
      res.set(range: Self.specialBits, with: Self.specialPattern)
      res.set(range: expUpper, with: exp >> (Self.exponentBits-2)) // upper exponent bits
      res.set(range: manUpper, with: Int(significand) & 1) // upper mantisa bits
    } else {
      let expUpper = signBit-2...signBit-1
      let manUpper = signBit-5...signBit-3
      res.set(range: expUpper, with: exp >> (Self.exponentBits-2)) // upper exponent bits
      res.set(range: manUpper, with: Int(significand)) // upper mantisa bits
    }
    res.set(bit: signBit, with: sign.rawValue)
    res.set(range: expLower, with: exp)
    res.set(range: manLower, with: dmant)
    if nanb { res.set(range: Self.nanBitRange, with: Self.nanPattern) }
    return res
  }
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - Double/BID conversions
  
  static func bid(from x:Double, _ rndMode:Rounding) -> Self {
    // Unpack the input
    let s = x.sign, expMask = (1<<11) - 1
    var e = Int(x.exponentBitPattern), t = 0, c = x.significandBitPattern
    if e == 0 {
      if c == 0 { return zero(s) }
      
      // denormalizd number
      let l = c.leadingZeroBitCount - (64 - 53)
      c <<= 1
      e = -(l + 1074)
    } else if e == expMask {
      if c == 0 { return infinite(s) }
      return nan(s, RawSignificand(c))
    } else {
      c.set(bit: 52)  // set upper bit
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
    var cf = UInt128(high: 0, low: c)
    cf <<= (113 - 53)
    t += (113 - 53)
    e -= (113 - 53) // Now e belongs [-1186;911].
    
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
      var cint = cf
      if a <= 0 {
        cint = cint >> -e
        if cint.components.high == 0 && cint.components.low < largestNumber+1 {
          return Self(sign: s, expBitPattern: exponentBias,
                      sigBitPattern: RawBitPattern(cint.components.low))
        }
      } else if a <= 48 {
        var pow5 = Self.coefflimitsBID32(a)
        cint = cint >> t
        if cint <= pow5 {
          var cc = cint
          pow5 = power5(a)
          (cc, _) = cc.multipliedReportingOverflow(by: pow5)
          return Self(sign: s, expBitPattern: exponentBias-a,
                      sigBitPattern: RawBitPattern(cc.components.low))
        }
      }
    }
    
    // Check for "trivial" underflow, when 2^e * 2^113 <= 10^emin * 1/4,
    // so test e <= floor(emin * log_2(10) - 115)
    // In this case just fix ourselves at that value for uniformity.
    //
    // This is important not only to keep the tables small but to maintain the
    // testing of the round/sticky words as a correct rounding method
    if e <= -450 { e = -450 }
    
    // Now look up our exponent e, and the breakpoint between e and e+1
    let m_min = Tables.bid_breakpoints_bid32[e+450]
    var e_out = exponents_bid32(e+450)
    
    // Choose exponent and reciprocal multiplier based on breakpoint
    var r:UInt256
    if cf <= m_min {
      r = Tables.bid_multipliers1_bid32[e+450]
    } else {
      r = Tables.bid_multipliers2_bid32[e+450]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    Self.mul128x256to384(&z, cf, r)
    var c_prov = RawBitPattern(z.w[5])
    
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
    let ind = rndMode.index(negative:s == .minus, lsb:Int(c_prov))
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
    if e_out > 90+exponentBias {
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
    return Self(sign: s, expBitPattern: e_out, sigBitPattern: c_prov)
  }
  
  static func bid(from x:UInt64, _ rndMode:Rounding) -> Self {
    // Get BID from a 64-bit unsigned integer
    if x <= Self.largestNumber { // x <= 10^7-1 and the result is exact
      return Self(sign: .plus, expBitPattern: exponentBias, sigBitPattern: RawBitPattern(x))
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
      var res: RawBitPattern
      if q <= 19 {
        bid_round64_2_18 ( // would work for 20 digits too if x fits in 64 bits
          q, ind, x, &res64, &incr_exp,
          &is_midpoint_lt_even, &is_midpoint_gt_even,
          &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = RawBitPattern(res64)
      } else { // q = 20
        let x128 = UInt128(high: 0, low:x)
        bid_round128_19_38 (q, ind, x128, &res128, &incr_exp,
                            &is_midpoint_lt_even, &is_midpoint_gt_even,
                            &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
        res = RawBitPattern(res128._lowWord) // res.w[1] is 0
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
            res = largestSignificand // 10^6
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
      return Self(sign: .plus, expBitPattern: ind, sigBitPattern: RawBitPattern(res))
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
          return Self(sign: .minus, expBitPattern: exponentBias, sigBitPattern: 1)
        }
        if r == .up && s == .plus {
          return Self(sign: .plus, expBitPattern: exponentBias, sigBitPattern: 1)
        }
        if exp < 0 { return Self(sign: s, expBitPattern: 0, sigBitPattern: 0) }
        return Self(sign: s, expBitPattern: exponentBias, sigBitPattern: 0)
      }
      
      // swap round modes when negative
      if s != .plus {
        if r == .up { r = .down }
        else if r == .down { r = .up }
      }
      
      // determine the rounding table index
      let roundIndex = r.index()
      
      // 10*coeff
      var c = (c << 3) + (c << 1)
      if R != 0 {
        c |= 1
      }
      
      // get digits to be shifted out
      let extra_digits = 1-exp
      c += Int(roundConstTable(roundIndex, extra_digits))
      
      // get coeff*(2^M[extra_digits])/10^extra_digits
      let Q = UInt64(c).multipliedFullWidth(by: reciprocals10(extra_digits))

      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      let amount = shortReciprocalScale[extra_digits]
      
      var _C64 = Q.high >> amount
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
          remainder_h = remainder_h & Q.high
          
          if remainder_h == 0 &&
              Q.low < reciprocals10(extra_digits) {
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
      return Self(sign: s, expBitPattern: 0, sigBitPattern: RawBitPattern(_C64))
    }
    var exp = exp, c = c
    if c == 0 { if exp > maxEncodedExponent { exp = maxEncodedExponent } }
    while c < (Self.largestNumber+1)/10 && exp > maxEncodedExponent {
      c = (c << 3) + (c << 1)
      exp -= 1
    }
    if UInt32(exp) > maxEncodedExponent {
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
    return Self(RawData(c))
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
      // var P128 = UInt128().components
      let P128 = C.multipliedFullWidth(by:  bid_Kx64(ind))

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
      var ind = x - 1    // 0 <= ind <= 36
      if ind <= 18 {    // if 0 <= ind <= 18
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
      let (h, l) = UInt128(high: C.high, low: C.low).multipliedFullWidth(by: bid_Kx128(ind))
      P256 = UInt256(w: [l.components.low, l.components.high,
                         h.components.low, h.components.high])
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
  static func estimateDecDigits(_ i: Int) -> Int { _digitsIn(UInt128(1) << i) }
  
  /// Returns ten to the `i`th power or `10^i` where `i ≥ 0`.
  static func power10<T:FixedWidthInteger>(_ i:Int) -> T { _power(T(10), to:i) }
  
  /// Returns ten to the `i`th power or `10^i` where `i ≥ 0`.
  static func power5<T:FixedWidthInteger>(_ i: Int) -> T { _power(T(5), to: i) }
  
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
    _digitsIn(UInt64(1) << i).tenPower
  }
    
  /// Returns rounding constants for a given rounding mode `rnd` and
  /// power of ten given by `10^(i-1)`.
  static func roundConstTable(_ rnd:Int, _ i:Int) -> UInt64 {
    if i == 0 { return 0 }
    switch rnd {
      case 0, 4: return 5 * power10(i-1)
      case 2: return power10(i)-1
      case 1, 3: return 0
      default: assertionFailure("Illegal rounding mode")
    }
    return 0
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
  
  static var recip_scale32 : [UInt8] { [1, 1, 3, 7, 9, 14, 18, 21, 25] }
  
  static func reciprocals10_32(_ i:Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = recip_scale32[i] + 32
    return UInt64(UInt128(1) << twoPower / power10(i)) + 1
  }
  
  /// Returns trunc((exp10 - 80) × recipLog₁₀2) + 238 for exp10 ≤ 80
  ///         round((exp10 - 80) × recipLog₁₀2) + 238 for exp10 > 80
  static func exponents_binary32(_ exp10: Int) -> Int {
    let actualExp = Double(exp10 - 80)
    let recipLog₁₀2 = 3.321_928_094_887_362_347_870_319_429_489_390_175_864
    if actualExp <= 0 {
      return Int(actualExp * recipLog₁₀2) + 238
    } else {
      return Int(actualExp * recipLog₁₀2) + 239
    }
  }
  
  /// Returns trunc((exp10 - 358) × recipLog₁₀2) + 1134 for exp10 ≤ 358
  ///         trunc((exp10 - 358) × recipLog₁₀2) + 1135 for exp10 > 358
  static func exponents_binary64(_ exp10: Int) -> Int {
    let actualExp = Double(exp10 - 358)
    let recipLog₁₀2 = 3.321_928_094_887_362_347_870_319_429_489_390_175_864
    if actualExp <= 0 {
      return Int(actualExp * recipLog₁₀2) + 1134
    } else {
      return Int(actualExp * recipLog₁₀2) + 1135
    }
  }
  
  /// Returns trunc((exp10 - 450) × recipLog₂10) + 127 for exp10 ≤ 450
  ///         trunc((exp10 - 450) × recipLog₂10) + 128 for exp10 > 450
  // FIXME: - There seem to be errors? in this table, it is inconsistent with
  //          the calculated values. For now, I've stubbed the original values
  //          from the table in the calculated results.
  static func exponents_bid32(_ exp10: Int) -> Int {
    let actualExp = Double(exp10 - 450)
    let recipLog₂10 = 0.301_029_995_663_981_195_213_738_894_724_493_026_768
    
    // Exceptional cases that don't match the calculated values
    switch actualExp {
      case -402: return 7 // calculates to 6
      case -392: return 10 // calculates to 9
      case -299: return 38 // calculates to 37
      case -206: return 66 // calculates to 65
      case -196: return 69 // calculates to 68
      case -103: return 97 // calculates to 96
      case  -10: return 125 // calculates to 124
      case    0: return 128 // calculates to 127
      case   83: return 153 // calculates to 152
      case   93: return 156 // calculates to 155
      case  186: return 184 // calculates to 183
      default: break
    }
    
    if actualExp <= 0 {
      return Swift.max(Int(actualExp * recipLog₂10) + 127, -1)
    } else {
      return Int(actualExp * recipLog₂10) + 128
    }
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
  
  static var reciprocalScale : [Int8] {
    [ 1, 1, 1, 1, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53,
      56, 59, 63,  66, 69, 73, 76, 79, 83, 86, 89, 92, 96, 99, 102, 109 ]
  }
  
  /// Returns `2^s[i] / 10^i + 1` where `s` is a table of
  /// reciprocal scaling factors and `i ≥ 0`.
  static func reciprocals10(_ i: Int) -> UInt128 {
    if i == 0 { return UInt128() }
    let shiftedOne = UInt128(1) << reciprocalScale[i] // upper dividend
    let result = UInt128(power10(i)).dividingFullWidth((shiftedOne, UInt128()))
    return result.quotient + 1
  }
  
  static func reciprocals10(_ i: Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = shortReciprocalScale[i]+64
    return UInt64(UInt128(1) << twoPower / power10(i)) + 1
  }
  
  static internal func mul64x256to320(_ P:inout UInt384, _ A:UInt64,
                                      _ B:UInt256) {
    var lC = false
    let lP0 = A.multipliedFullWidth(by: B.w[0])
    let lP1 = A.multipliedFullWidth(by: B.w[1])
    let lP2 = A.multipliedFullWidth(by: B.w[2])
    let lP3 = A.multipliedFullWidth(by: B.w[3])
    P.w[0] = lP0.low
    (P.w[1],lC) = lP1.low.addingReportingOverflow(lP0.high)
    (P.w[2],lC) = add(lP2.low,lP1.high,lC)
    (P.w[3],lC) = add(lP3.low,lP2.high,lC)
    P.w[4] = lP3.high + (lC ? 1 : 0)
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
    (P.w[1],CY) = P1.w[0].addingReportingOverflow(P0.w[1])
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
  
  /// Table of division factors of 2 for `n+1` when `i` = 0, and factors of 5
  /// for `n+1` when `i` = 1 where 0 ≤ `n` < 1024. When both are factors,
  /// return divisors of both are combined.  For example when `n` = 19, `n+1`
  /// or 20 is a factor of both 2 and 5, so the return for `i=0` is 2 (not 4),
  /// and `i=1` is 1.  This function reproduces the contents of the table
  /// `bid_factors` in the original.
  static func bid_factors(_ n:Int, _ i:Int) -> Int {
    var n = n + 1
    if i == 0 && n.isMultiple(of: 2) {
      var div2 = 1
      n /= 2
      while n.isMultiple(of: 5) { n /= 5 } // remove multiples of 5
      while n.isMultiple(of: 2) { div2 += 1; n /= 2 }
      return div2
    } else if i == 1 && n.isMultiple(of: 5) {
      var div5 = 1
      n /= 5
      while n.isMultiple(of: 2) { n >>= 1 }
      while n.isMultiple(of: 5) { div5 += 1; n /= 5 }
      return div5
    }
    return 0
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

  @inlinable
  static func add(_ X:UInt64, _ Y:UInt64, _ CI:Bool) -> (UInt64, Bool)  {
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
    func get(_ range: IntRange) -> Int { dpd.get(range: range) }
    
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

extension IntDecimal {
  
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
    
    // check if both significand and exponents and bigger or smaller
    if xman > yman && xexp >= yexp { return xsign == .minus }
    if xman < yman && xexp <= yexp { return xsign == .plus }
    
    // if xexp is `numberOfDigits`-1 greater than yexp, no need to continue
    if xexp - yexp > Self.maximumDigits-1 { return xsign == .minus }
    
    // difference cannot be greater than 10^6
    // if exp_x is 6 less than exp_y, no need for compensation
    if yexp - xexp > Self.maximumDigits-1 { return xsign == .plus }
    
    // need to compensate the significand
    var manPrime: Self.RawBitPattern
    if xexp > yexp {
      manPrime = xman * _power(Self.RawBitPattern(10), to: xexp - yexp)
      if manPrime == yman { return false }
      return (manPrime < yman) != (xsign == .minus)
    }
    
    // adjust y significand upwards
    manPrime = yman * _power(Self.RawBitPattern(10), to: yexp - xexp)
    if manPrime == xman { return false }
    
    // if positive, return whichever abs number is smaller
    return (xman < manPrime) != (xsign == .minus)
  }
}

extension IntDecimal {
  
  ///////////////////////////////////////////////////////////////////////
  // MARK: - General-purpose math functions
  static func add(_ x: Self, _ y: Self, rounding: Rounding) -> Self {
    let xb = x, yb = y
    let (signX, exponentX, significandX, validX) = xb.unpack()
    let (signY, exponentY, significandY, validY) = yb.unpack()
    
    // Deal with illegal numbers
    if !validX {
      if xb.isNaN {
        return Self.nanQuiet(significandX)
      }
      if xb.isInfinite {
        if yb.isNaNInf {
          if signX == signY {
            return Self(RawData(significandX))
          } else {
            return Self.nan() // invalid Op
          }
        }
        if yb.isNaN {
          return Self.nanQuiet(significandY)
        } else {
          // +/- infinity
          return Self(RawData(significandX))
        }
      } else {
        // x = 0
        if !yb.isInfinite && significandY != 0 {
          if exponentY <= exponentX { return y }
        }
      }
    }
    
    if !validY {
      if yb.isInfinite {
        return Self.nanQuiet(significandY)
      }
      
      // y = 0
      if significandX == 0 {
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
        return Self(sign: sign, expBitPattern: exp, sigBitPattern: 0)
      } else if exponentY >= exponentX {
        return x
      }
    }
    
    // sort arguments by exponent
    var (signA, exponentA, significandA) = (signY, exponentY, significandY)
    var (signB, exponentB, significandB) = (signX, exponentX, significandX)
    if exponentX >= exponentY {
      swap(&signA, &signB)
      swap(&exponentA, &exponentB)
      swap(&significandA, &significandB)
    }
    
    // exponent difference
    var exponentDiff = exponentA - exponentB
    if exponentDiff > maximumDigits {
      let binExpon = Double(significandA).exponent
      let scaleCA = estimateDecDigits(binExpon)
      let d2 = 16 - scaleCA
      if exponentDiff > d2 {
        exponentDiff = d2
        exponentB = exponentA - exponentDiff
      }
    }
    
    let signAB = signA != signB ? Int64(-1) : 0
    let CB = UInt64(bitPattern: (Int64(significandB) + signAB) ^ signAB)
    
    let SU = UInt64(significandA) * power10(exponentDiff)
    var S = Int64(bitPattern: SU &+ CB)
    
    if S < 0 {
      signA = signA.toggle // == .minus ? .plus : .minus // toggle the sign
      S = -S
    }
    var P = UInt64(S)
    var n_digits:Int
    if P == 0 {
      signA = .plus
      if rounding == .down { signA = .minus }
      if significandA == 0 { signA = signX }
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
      return Self(sign: signA, expBitPattern: exponentB,
                  sigBitPattern: RawBitPattern(P))
    }
    
    let extra_digits = n_digits - maximumDigits
    
    var irmode = rounding.raw
    if signA == .minus && (UInt(irmode) &- 1) < 2 {
      irmode = 3 - irmode
    }
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += roundConstTable(irmode, extra_digits)
    let Tmp = P.multipliedFullWidth(by:  reciprocals10(extra_digits))
    
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = shortReciprocalScale[extra_digits]
    var Q = Tmp.high >> amount
    
    // remainder
    let R = P - Q * power10(extra_digits)
    
    if rounding == .toNearestOrEven {
      if R == 0 {
        Q.clear(bit: 0)  // Q &= 0xffff_fffe
      }
    }
    return Self(Self.adjustOverflowUnderflow(signA, exponentB+extra_digits,
                                        RawBitPattern(Q), rounding))
  }
  
  static func mul (_ x:Self, _ y:Self, _ rmode:Rounding) -> Self  {
    var (signX, exponentX, significandX, validX) = x.unpack()
    let (signY, exponentY, significandY, validY) = y.unpack()
    
    // unpack arguments, check for NaN or Infinity
    let sign = signX != signY ? Sign.minus : .plus
    if !validX {
      // x is Inf. or NaN
      
      // test if x is NaN
      if x.isNaN {
        return Self.nanQuiet(significandX)
      }
      // x is Infinity?
      if x.isInfinite {
        // check if y is 0
        if !y.isInfinite && significandY == 0 {
          // status.insert(.invalidOperation)
          // y==0 , return NaN
          return Self.nan()
        }
        // check if y is NaN
        if y.isNaN {
          // y==NaN , return NaN
          return Self.nanQuiet(significandY)
        }
        // otherwise return +/-Inf
        return Self.infinite(sign) // ((x ^ y) & SIGN_MASK32) | INFINITY_MASK32
      }
      // x is 0
      if !y.isInfinite {
        exponentX += exponentY - exponentBias
        if exponentX > maxEncodedExponent {
          exponentX = maxEncodedExponent
        } else if exponentX < 0 {
          exponentX = 0
        }
        return Self(sign: sign, expBitPattern: exponentX, sigBitPattern: 0) // UInt32(UInt64(signX ^ sign_y) | (UInt64(exponentX) << 23))
      }
    }
    if !validY {
      // y is Inf. or NaN
      // test if y is NaN
      if y.isNaN {
        return Self.nanQuiet(significandY)
      }
      // y is Infinity?
      if y.isInfinite {
        // check if x is 0
        if significandX == 0 {
          //status.insert(.invalidOperation)
          // x==0, return NaN
          return Self.nan()  // (NAN_MASK32);
        }
        // otherwise return +/-Inf
        return Self.infinite(sign)  // ((x ^ y) & SIGN_MASK32) | INFINITY_MASK32
      }
      // y is 0
      exponentX += exponentY - exponentBias //exponentBias
      if exponentX > maxEncodedExponent {
        exponentX = maxEncodedExponent
      } else if exponentX < 0 {
        exponentX = 0
      }
      return Self(sign: sign, expBitPattern: exponentX, sigBitPattern: 0)
    }
    
    // multiplication
    var P = significandX * significandY
    
    //--- get number of bits in C64 ---
    // version 2 (original)
    let tempx = Double(P)
    let bin_expon_p = tempx.exponent 
    var n_digits = Int(estimateDecDigits(bin_expon_p))
    if P >= power10(n_digits) {
      n_digits += 1
    }
    
    exponentX += exponentY - exponentBias
    let extra_digits = (n_digits <= maximumDigits) ? 0 : n_digits-maximumDigits
    exponentX += extra_digits
    
    if extra_digits == 0 {
      return Self(Self.adjustOverflowUnderflow(sign, exponentX, P, rmode))
    }
    
    var rmode1 = rmode.raw
    if sign == .minus && (UInt32(rmode1) &- 1) < 2 {
      rmode1 = 3 - rmode1
    }
    
    if exponentX < 0 { rmode1 = 3 }  // RZ
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += RawBitPattern(roundConstTable(rmode1, extra_digits))
    let Tmp = UInt64(P).multipliedFullWidth(by: reciprocals10(extra_digits))

    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = shortReciprocalScale[extra_digits]
    var Q = RawBitPattern(Tmp.high) >> amount
    
    // remainder
    let R = P - Q * power10(extra_digits)
    if rmode1 == 0 {    //BID_ROUNDING_TO_NEAREST
      if R == 0 {
        Q.clear(bit: 0) // Q &= 0xffff_fffe
      }
    }
    
    if exponentX == -1 && Q == largestNumber && rmode != .towardZero {
      rmode1 = rmode.raw
      if sign == .minus && (UInt32(rmode1) &- 1) < 2 {
        rmode1 = 3 - rmode1
      }
      
      if (R != 0 && rmode1 == 2) ||
         (rmode1&3 == 0 && R+R>=power10(extra_digits)) {
        return Self(sign: sign, expBitPattern:0,
                    sigBitPattern:largestSignificand)
      }
    }
    return Self(adjustOverflowUnderflow(sign, exponentX, Q, rmode))
  }
  
  /**
   *  Division algorithm description:
   * ```
   *  if significandX < significandY
   *    p = number_digits(significandY) - number_digits(significandX)
   *    A = significandX*10^p
   *    B = significandY
   *    CA= A*10^(15+j), j=0 for A>=B, 1 otherwise
   *    Q = 0
   *  else
   *    get Q=(int)(significandX/significandY)
   *        (based on double precision divide)
   *    check for exact divide case
   *    Let R = significandX - Q*significandY
   *    Let m=16-number_digits(Q)
   *    CA=R*10^m, Q=Q*10^m
   *    B = significandY
   *  endif
   *  if CA < 2^64
   *    Q += CA/B  (64-bit unsigned divide)
   *  else
   *    get final Q using double precision divide, followed by 3 integer
   *        iterations
   *  if exact result, eliminate trailing zeros
   *  check for underflow
   *  round coefficient to nearest
   * ```
   */
  static func div(_ x:Self, _ y:Self, _ rmode:Rounding) -> Self {
    var (signX, exponentX, significandX, validX) = x.unpack()
    let (signY, exponentY, significandY, validY) = y.unpack()
    
    // unpack arguments, check for NaN or Infinity
    let sign = signX != signY ? Sign.minus : .plus
    if !validX {
      // test if x is NaN
      if x.isNaN {
        return Self.nanQuiet(significandX)
      }
      
      // x is Infinity?
      if x.isInfinite {
        // check if y is Inf or NaN
        if y.isInfinite {
          // y==Inf, return NaN
          if y.isNaNInf {    // Inf/Inf
            //status.insert(.invalidOperation)
            return nan() // NAN_MASK32
          }
        } else {
          // otherwise return +/-Inf
          return infinite(sign) // ((x ^ y) & SIGN_MASK32) | INFINITY_MASK32
        }
      }
      // x==0
      if !y.isInfinite && significandY == 0 {
        // y==0 , return NaN
        //status.insert(.invalidOperation)
        return nan()
      }
      if !y.isInfinite {
        exponentX = exponentX - exponentY + exponentBias
        if exponentX > maxEncodedExponent {
          exponentX = maxEncodedExponent
        } else if exponentX < 0 {
          exponentX = 0
        }
        return Self(sign: sign, expBitPattern: exponentX, sigBitPattern: 0)
      }
      
    }
    if !validY {
      // y is Inf. or NaN
      // test if y is NaN
      if y.isNaN {
        return Self.nanQuiet(significandY)
      }
      
      // y is Infinity?
      if y.isInfinite {
        // return +/-0
        return Self(sign: sign, expBitPattern: 0, sigBitPattern: 0) //(x ^ y) & SIGN_MASK32
      }
      
      // y is 0
      // status.insert(.divisionByZero)
      return infinite(sign)
    }
    var diff_expon = exponentX - exponentY + exponentBias
    
    var A, B, Q, R: RawBitPattern
    var CA: UInt64
    var ed1, ed2: Int
    if significandX < significandY {
      // get number of decimal digits for c_x, c_y
      //--- get number of bits in the coefficients of x and y ---
      let tempx = Float(significandX)
      let tempy = Float(significandY)
      let bin_index = Int((tempy.bitPattern - tempx.bitPattern) >> 23)
      A = significandX * RawBitPattern(bid_power10_index_binexp(bin_index))
      B = significandY
      
      // compare A, B
      let DU = (A - B) >> 31
      ed1 = 6 + Int(DU)
      ed2 = estimateDecDigits(bin_index) + ed1
      let T : UInt64 = power10(ed1)
      CA = UInt64(A) * T
      
      Q = 0
      diff_expon = diff_expon - ed2
      
    } else {
      // get c_x/c_y
      Q = significandX / significandY
      
      R = significandX - significandY * Q
      
      // will use to get number of dec. digits of Q
      let tempq = Float(Q)
      let bin_expon_cx = tempq.exponent
      
      // exact result ?
      if R == 0 {
        return Self(Self.adjustOverflowUnderflow(sign, diff_expon, Q, rmode))
      }
      // get decimal digits of Q
      var DU = UInt32(bid_power10_index_binexp(bin_expon_cx)) - UInt32(Q) - 1
      DU >>= 31
      
      ed2 = 7 - Int(estimateDecDigits(bin_expon_cx)) - Int(DU)
      
      CA = UInt64(R) * power10(ed2)
      B = significandY
      
      Q *= power10(ed2)
      diff_expon -= ed2
    }
    
    let Q2 = RawBitPattern(CA / UInt64(B))
    let B2 = B + B
    let B4 = B2 + B2
    R = RawBitPattern(UInt32(CA - UInt64(Q2) * UInt64(B)))
    Q += Q2
    
    if R != 0 {
      // set status flags
      // status.insert(.inexact)
    } else {
      // eliminate trailing zeros
      // check whether CX, CY are short
      if significandX <= 1024 && significandY <= 1024 {
        let i = Int(significandY) - 1
        let j = Int(significandX) - 1
        // difference in powers of 2 bid_factors for Y and X
        var nzeros = ed2 - (bid_factors(i,0) + bid_factors(j,0))
        // difference in powers of 5 bid_factors
        let d5 = ed2 - (bid_factors(i,1) + bid_factors(j,1))
        if d5 < nzeros {
          nzeros = d5
        }
        
        if nzeros != 0 {
          var CT = UInt64(Q) * reciprocals10_32(nzeros)
          CT >>= 32
          
          // now get P/10^extra_digits: shift C64 right by M[extra_digits]-128
          let amount = recip_scale32[nzeros]
          Q = RawBitPattern(CT >> amount)
          
          diff_expon += nzeros
        }
      } else {
        var nzeros = 0
        
        // decompose digit
        let PD = UInt64(Q) * 0x068DB8BB
        var digit_h = RawBitPattern(PD >> 40)
        let digit_low = Q - digit_h * 10000
        
        if digit_low == 0 {
          nzeros += 4
        } else {
          digit_h = digit_low
        }
        
        if digit_h.isMultiple(of: 2) {
          if digit_h.isMultiple(of: 1000) { nzeros += 3 }
          else if digit_h.isMultiple(of: 100) { nzeros += 2 }
          else if digit_h.isMultiple(of: 10) { nzeros += 1 }
        }
        
        if nzeros != 0 {
          var CT = UInt64(Q) * reciprocals10_32(nzeros)
          CT >>= 32
          
          // now get P/10^extra_digits: shift C64 right by M[extra_digits]-128
          let amount = recip_scale32[nzeros]
          Q = RawBitPattern(CT >> amount)
        }
        diff_expon += nzeros
      }
      if diff_expon >= 0 {
        return Self(Self.adjustOverflowUnderflow(sign, diff_expon, Q, rmode))
      }
    }
    
    if diff_expon >= 0 {
      var rmode1 = rmode.raw
      if sign == .minus && (UInt32(rmode1) &- 1) < 2 {
        rmode1 = 3 - rmode1
      }
      var R = UInt32(R)
      switch rmode1 {
        case 0, 4:
          // R*10
          R += R
          R = (R << 2) + R
          let B5 = B4 + B
          // compare 10*R to 5*B
          R = UInt32(B5) &- R
          // correction for (R==0 && (Q&1))
          R -= UInt32((Int(Q) | (rmode1 >> 2)) & 1)
          // R<0 ?
          let D = UInt32(R >> 31)
          Q += RawBitPattern(D)
        case 1, 3:
          break
        default:    // rounding up (2)
          Q+=1
      }
      return Self(Self.adjustOverflowUnderflow(sign, diff_expon, Q, rmode))
    } else {
      // UF occurs
      return Self(Self.adjustOverflowUnderflow(sign, diff_expon, Q, rmode))
    }
  }

  static func rem(_ x:Self, _ y:Self) -> Self {
    var (signX, exponentX, significandX, validX) = x.unpack()
    let (_, exponentY, significandY, validY) = y.unpack()
    
    // unpack arguments, check for NaN or Infinity
    if !validX {
      // test if x is NaN
      if x.isNaN {
        return Self.nanQuiet(significandX)
      }
      // x is Infinity?
      if x.isInfinite {
        if !y.isNaN {
          return nan() // 0x7c000000
        }
      }
      // x is 0
      // return x if y != 0
      if !y.isInfinite && significandY != 0 {
        if exponentY < exponentX {
          exponentX = exponentY
        }
        return Self(sign: signX, expBitPattern: exponentX, sigBitPattern: 0)
      }
      
    }
    if !validY {
      // y is Inf. or NaN
      
      // test if y is NaN
      if y.isNaN {
        return Self.nanQuiet(significandY)
      }
      // y is Infinity?
      if y.isInfinite {
        return Self(sign: signX, expBitPattern: exponentX,
                    sigBitPattern: significandX)
      }
      // y is 0, return NaN
      return nan() // 0x7c000000
    }
    
    
    var diff_expon = exponentX - exponentY
    if diff_expon <= 0 {
      diff_expon = -diff_expon
      
      if (diff_expon > 7) {
        // |x|<|y| in this case
        return x
      }
      // set exponent of y to exponentX, scale significandY
      let T : UInt64 = power10(diff_expon)
      let CYL = UInt64(significandY) * T
      if CYL > (UInt64(significandX) << 1) {
        return x
      }
      
      let CY = RawBitPattern(CYL)
      let Q = significandX / CY
      var R = significandX - Q * CY
      
      let R2 = R + R
      if R2 > CY || (R2 == CY && (Q & 1) != 0) {
        R = CY - R
        signX = signX.toggle
      }
      
      return Self(sign: signX, expBitPattern: exponentX, sigBitPattern: R)
    }
    
    var CX = UInt64(significandX)
    var Q64 = UInt64()
    while diff_expon > 0 {
      // get number of digits in coeff_x
      let tempx = Float(CX)
      let bin_expon = tempx.exponent
      let digits_x = estimateDecDigits(bin_expon)
      var e_scale = Int(18 - digits_x)
      if (diff_expon >= e_scale) {
        diff_expon -= e_scale;
      } else {
        e_scale = diff_expon;
        diff_expon = 0;
      }
      
      // scale dividend to 18 or 19 digits
      CX *= power10(e_scale)
      
      // quotient
      Q64 = CX / UInt64(significandY)
      // remainder
      CX -= Q64 * UInt64(significandY)
      
      // check for remainder == 0
      if CX == 0 {
        return Self(sign: signX, expBitPattern: exponentY, sigBitPattern: 0)
      }
    }
    
    significandX = RawBitPattern(CX)
    let R2 = significandX + significandX
    if R2 > significandY || (R2 == significandY && (Q64 & 1) != 0) {
      significandX = significandY - significandX
      signX = signX.toggle
    }
    
    return Self(sign:signX,expBitPattern:exponentY,sigBitPattern:significandX)
  }
  
  /***************************************************************************
   *    BID32 fma
   ***************************************************************************
   *
   *  Algorithm description:
   *
   *  if multiplication is guaranteed exact (short coefficients)
   *     call the unpacked arg. equivalent of bid32_add(x*y, z)
   *  else
   *     get full significandX*significandY product
   *     call subroutine to perform addition of 32-bit argument
   *                                         to 128-bit product
   *
   **************************************************************************/
  static func fma(_ x:Self, _ y:Self, _ z:Self, _ rmode:Rounding) -> Self {
    var (sign_x, exponent_x, significandX, valid_x) = x.unpack()
    var (sign_y, exponent_y, significandY, valid_y) = y.unpack()
    var (sign_z, exponent_z, significandZ, valid_z) = z.unpack()

    // unpack arguments, check for NaN, Infinity, or 0
    let signxy = sign_x != sign_y ? Sign.minus : .plus
    if !valid_x || !valid_y || !valid_z {
      if y.isNaN {
        return Self.nanQuiet(significandY)
      }
      if z.isNaN {
        return Self.nanQuiet(significandZ)
      }
      if x.isNaN {
        return Self.nanQuiet(significandX)
      }
      
      if !valid_x {
        // x is Inf. or 0
        // x is Infinity?
        if x.isInfinite {
          // check if y is 0
          if significandY == 0 {
            // y==0, return NaN
            return nan()
          }
          // test if z is Inf of oposite sign
          if z.isNaNInf && signxy != sign_z {
            // return NaN
            return nan()
          }
          // otherwise return +/-Inf
          return Self.infinite(signxy)
        }
        // x is 0
        if !y.isInfinite && !z.isInfinite {
          if significandZ != 0 {
            exponent_y = exponent_x - exponentBias + exponent_y
   
            if exponent_y >= exponent_z {
              return z
            }
            return addZero(exponent_y, sign_z, exponent_z, significandZ, rmode)
          }
        }
      }
      if !valid_y { // y is Inf. or 0
        // y is Infinity?
        if y.isInfinite {
          // check if x is 0
          if significandX == 0 {
            // y==0, return NaN
  //          pfpsf.insert(.invalidOperation)
            return nan() // NAN_MASK
          }
          // test if z is Inf of oposite sign
          if z.isNaNInf && signxy != sign_z {
  //          pfpsf.insert(.invalidOperation)
            // return NaN
            return nan() // NAN_MASK
          }
          // otherwise return +/-Inf
          return Self.infinite(signxy)
        }
        // y is 0
        if !z.isInfinite {
          if significandZ != 0 {
            exponent_y += exponent_x - exponentBias
            if exponent_y >= exponent_z {
              return z
            }
            return addZero(exponent_y, sign_z, exponent_z, significandZ, rmode)
          }
        }
      }
      
      if !valid_z {
        // y is Inf. or 0
        
        // test if y is NaN/Inf
        if z.isInfinite {
          return Self.nanQuiet(significandZ)
        }
        // z is 0, return x*y
        if significandX == 0 || significandY == 0 {
          //0+/-0
          let exp: Int
          var sign = Sign.plus
          exponent_x += exponent_y - exponentBias
          if exponent_x > maxEncodedExponent {
            exponent_x = maxEncodedExponent
          } else if exponent_x < 0 {
            exponent_x = 0
          }
          if exponent_x <= exponent_z {
            exp = exponent_x
          } else {
            exp = exponent_z
          }
          if signxy == sign_z {
            sign = sign_z
          } else if rmode == .down {
            sign = .minus
          }
          return Self(sign: sign, expBitPattern: exp, sigBitPattern: 0)
        }
        let d2 = exponent_x + exponent_y - exponentBias
        if exponent_z > d2 {
          exponent_z = d2
        }
      }
    }
    
    // multiplication
    let P0 = UInt64(significandX) * UInt64(significandY)
    exponent_x += exponent_y - exponentBias
    
    // sort arguments by exponent
    var sign_a = Sign.plus, exponent_a = 0, coefficient_a = UInt64()
    var sign_b = Sign.plus, exponent_b = 0, coefficient_b = UInt64()
    if exponent_x < exponent_z {
      sign_a = sign_z
      exponent_a = exponent_z
      coefficient_a = UInt64(significandZ)
      sign_b = signxy
      exponent_b = exponent_x
      coefficient_b = P0
    } else {
      sign_a = signxy
      exponent_a = exponent_x
      coefficient_a = P0
      sign_b = sign_z
      exponent_b = exponent_z
      coefficient_b = UInt64(significandZ)
    }
    
    // exponent difference
    var diff_dec_expon = exponent_a - exponent_b
    // var inexact = false
    if diff_dec_expon > 17 {
      let tempx = Double(coefficient_a)
      let bin_expon = tempx.exponent
      let scale_ca = Int(estimateDecDigits(bin_expon))
      
      let d2 = 31 - scale_ca
      if diff_dec_expon > d2 {
        diff_dec_expon = d2
        exponent_b = exponent_a - diff_dec_expon
      }
    }
    
    let sign_ab = Int64(sign_a != sign_b ? -1 : 0)
    let low = UInt64(bitPattern: (Int64(coefficient_b) + sign_ab) ^ sign_ab)
    let high = Int64(bitPattern: low) >> 63
    let CB = UInt128(high: UInt64(bitPattern: high), low: low)
    
    var Tmp = UInt128()
    (Tmp, _) = UInt128(power10(diff_dec_expon))
                              .multipliedReportingOverflow(by: coefficient_a)
    var P = Tmp &+ CB
    if Int64(bitPattern: P.components.high) < 0 {
      sign_a = sign_a.toggle
      var Phigh = 0 &- P.components.high
      if P.components.low != 0 { Phigh &-= 1 }
      P.components = (Phigh, 0 &- P.components.low)
    }
    
    var n_digits = 0
    var bin_expon = 0
    let PC = P.components
    if PC.high != 0 {
      let tempx = Double(PC.high)
      bin_expon = tempx.exponent + 64
      n_digits = estimateDecDigits(bin_expon)
      if P >= power10(n_digits) {
        n_digits += 1
      }
    } else {
      if PC.low != 0 {
        let tempx = Double(PC.low)
        bin_expon = tempx.exponent
        n_digits = estimateDecDigits(bin_expon)
        if PC.low >= Int128(power10(n_digits)).components.low {
          n_digits += 1
        }
      } else { // result = 0
        sign_a = .plus
        if rmode == .down { sign_a = .minus }
        if coefficient_a == 0 { sign_a = sign_x }
        n_digits = 0
      }
    }
    
    if n_digits <= maximumDigits {
      return Self(
      adjustOverflowUnderflow(sign_a,exponent_b,RawBitPattern(PC.low),rmode))
      //bid32Underflow(sign_a, exponent_b, PC.low, 0, rmode)
    }
    
    let extra_digits = n_digits - maximumDigits
    var rmode1 = rmode.raw
    if sign_a == .minus && (UInt32(rmode1) &- 1) < 2 {
      rmode1 = 3 - rmode1
    }
    if exponent_b+extra_digits < 0 { rmode1 = 3 }  // RZ
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    var Stemp = UInt128()
    if extra_digits <= 18 {
      P += UInt128(high:0, low:roundConstTable(rmode1, extra_digits))
    } else {
      Stemp.components = roundConstTable(rmode1, 18)
                            .multipliedFullWidth(by: power10(extra_digits-18))
      P += Stemp
      if rmode == .up {
        P += UInt128(high:0, low:roundConstTable(rmode1, extra_digits-18))
      }
    }
    
    // get P*(2^M[extra_digits])/10^extra_digits
    var Q_high = UInt128(), Q_low = UInt128(), C128 = UInt128()
    (Q_high, Q_low) = P.multipliedFullWidth(by: reciprocals10(extra_digits))

    // now get P/10^extra_digits: shift Q_high right by M(extra_digits)-128
    var amount = Int(reciprocalScale[extra_digits])
    C128 = Q_high >> amount
    
    var C64 = C128.components.low
    var remainder_h, rem_l: UInt64
    if !C64.isMultiple(of: 2) {
      // check whether fractional part of initial_P/10^extra_digits
      // is exactly .5
      // this is the same as fractional part of
      // (initial_P + 0.5*10^extra_digits)/10^extra_digits is exactly zero
      
      // get remainder
      let Qhigh = Q_high.components
      rem_l = Qhigh.low
      if amount < 64 {
        remainder_h = Qhigh.low << (64 - amount); rem_l = 0
      } else {
        remainder_h = Qhigh.high << (128 - amount)
      }
      
      // test whether fractional part is 0
      let Qlow = Q_low.components
      let recip = UInt128(reciprocals10(extra_digits)).components
      if ((remainder_h | rem_l) == 0 && (Qlow.high < recip.high
              || (Qlow.high == recip.high && Qlow.low < recip.low))) {
        C64 -= 1
      }
    }
    
    //var status = Status.inexact
    //var carry = UInt64(), CY = UInt64()
    
    // get remainder
    rem_l = Q_high.components.low
    if amount < 64 {
      remainder_h = Q_high.components.low << (64 - amount); rem_l = 0
    } else {
      remainder_h = Q_high.components.high << (128 - amount)
    }
    
    switch rmode {
      case .awayFromZero, .toNearestOrAwayFromZero:
        break
      case .down, .towardZero:
        break
      default:
        // round up
        var carry = UInt64(), CY:UInt64
        var (high, low) = Stemp.components
        let recip = UInt128(reciprocals10(extra_digits)).components
        let Qlow = Q_low.components
        (low, CY) = add_carry_out(Qlow.low, recip.low)
        (high, carry) = add_carry_in_out(Qlow.high, recip.high, CY)
        Stemp = UInt128(high: high, low: low)
        if amount < 64 {
          // status update
        } else {
          rem_l += carry
          remainder_h >>= (128 - amount)
          if carry != 0 && rem_l == 0 { remainder_h += 1 }
        }
    }
    
    if (UInt32(C64) == largestNumber) && (exponent_b+extra_digits == -1) &&
        (rmode != .towardZero) {
      rmode1 = rmode.raw
      if sign_a == .minus && (UInt32(rmode1) &- 1) < 2 {
        rmode1 = 3 - rmode1
      }
      if extra_digits <= 18 {
        P += UInt128(high:0, low:roundConstTable(rmode1, extra_digits))
      } else {
        Stemp.components = roundConstTable(rmode1, 18)
                           .multipliedFullWidth(by: power10(extra_digits-18))
        P += Stemp
        if rmode == .up {
          P += UInt128(high:0, low:roundConstTable(rmode1, extra_digits-18))
        }
      }
      
      // get P*(2^M[extra_digits])/10^extra_digits
      (Q_high, Q_low) = P.multipliedFullWidth(by: reciprocals10(extra_digits))

      // now get P/10^extra_digits: shift Q_high right by M(extra_digits)-128
      amount = Int(reciprocalScale[extra_digits])
      C128 = Q_high >> amount
      
      C64 = C128.components.low
      if C64 == largestNumber+1 {
        return Self(sign: sign_a, expBitPattern: 0, sigBitPattern: largestSignificand)
      }
    }
    return Self(adjustOverflowUnderflow(sign_a, exponent_b+extra_digits,
                                        RawBitPattern(C64), rmode))
  }
  
  @inlinable static func add_carry_out(
    _ X:UInt64, _ Y:UInt64) -> (S:UInt64, CY:UInt64) {
      let S = X &+ Y
      return (S, S < X ? 1 : 0)  // allow overflow
  }

  @inlinable static func add_carry_in_out(
    _ X:UInt64, _ Y:UInt64, _ CI:UInt64) -> (S:UInt64, CY:UInt64) {
      let X1 = X &+ CI
      let S = X1 &+ Y
      return (S, ((S<X1) || (X1<CI)) ? 1 : 0)
  }
  
  //////////////////////////////////////////////////////////////////////////
  //
  //    0*10^ey + cz*10^ez,   ey<ez
  //
  //////////////////////////////////////////////////////////////////////////
  static func addZero(_ ey:Int, _ sz:Sign, _ ez:Int, _ cz:RawBitPattern,
                      _ r:Rounding) -> Self {
    let diff_expon = ez - ey
    var cz = cz
    
    let tempx = Double(cz)
    let bin_expon = tempx.exponent
    var scale_cz = Int(estimateDecDigits(bin_expon))
    if cz >= power10(scale_cz) {
      scale_cz+=1
    }
    
    var scale_k = 7 - scale_cz
    if diff_expon < scale_k {
      scale_k = diff_expon
    }
    cz *= power10(scale_k)
    
    return Self(adjustOverflowUnderflow(sz, ez - scale_k, cz, r))
  }
  
  func float<T:BinaryFloatingPoint>(_ rmode: Rounding) -> T {
    if T.self == Double.self { return T(float64(rmode)) }
    else if T.self == Float.self { return T(float32(rmode)) }
    return T(float64(rmode))
  }
  
  /// Convert the active decimal floating-point number (any size) to a 64-bit
  /// Double and return this value.
  func float64(_ rmode: Rounding) -> Double {
    var fp:Double?, s:Sign, e:Int, k:Int, high:RawSignificand
    (s, e, k, high, fp) = self.unpack()
    if let x = fp { return x }
    
    let size = Self.signBit+1
    let shift: Int, offset: Int, maxExp: Int, minExp: Int
    switch size {
      case  32: shift = 31; offset = 89; maxExp = .max; minExp = .min
      case  64: shift = 1; offset = 59; maxExp = 309; minExp = -358
      case 128: shift = 6; offset = 0; maxExp = 309; minExp = -358
      default:  shift = 31; offset = 89; maxExp = .max; minExp = .min
    }
    
    // Correct to 2^112 <= c < 2^113 with corresponding exponent adding 113-24=89
    // In fact shift a further 6 places ready for reciprocal multiplication
    // Thus (113-24)+6=95, a shift of 31 given that we've already upacked in c.hi
    let c = UInt128(high: UInt64(high), low: 0) << shift
    k += offset
    
    // check for underflow and overflow
    if e >= maxExp { return Self.overflow(s, rmode) }
    if e <= minExp { e = minExp }
    
    // Check for "trivial" overflow, when 10^e * 1 > 2^{sci_emax+1}, just to
    // keep tables smaller (it would be intercepted later otherwise).
    //
    // (Note that we may have normalized the coefficient, but we have a
    //  corresponding exponent postcorrection to account for; this can
    //  afford to be conservative anyway.)
    //
    // We actually check if e >= ceil((sci_emax + 1) * log_10(2))
    // which in this case is e >= ceil(1024 * log_10(2)) = ceil(308.25) = 309
    
    // Look up the breakpoint and approximate exponent
    let m_min = Tables.bid_breakpoints_binary64[e+358]
    var e_out = Self.exponents_binary64(e+358) - Int(k)
    
    // Choose provisional exponent and reciprocal multiplier based on breakpoint
    var r = UInt256()
    if c <= m_min {
      r = Tables.bid_multipliers1_binary64[e+358]
    } else {
      r = Tables.bid_multipliers2_binary64[e+358]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    Self.mul64x256to320(&z, c.components.high, r)
    z.w[5]=z.w[4]; z.w[4]=z.w[3]; z.w[3]=z.w[2]; z.w[2]=z.w[1]; z.w[1]=z.w[0]
    z.w[0]=0
    
    // Check for exponent underflow and compensate by shifting the product
    // Cut off the process at precision+2, since we can't really shift further
    if e_out < 1 {
      var d = 1 - e_out
      if d > 55 { d = 55 }
      e_out = 1
      let sz = Self.srl256_short(z.w[5], z.w[4], z.w[3], z.w[2], d)
      z.w[2...5] = sz.w[0...3]
    }
    var c_prov = Int64(z.w[5])
    
    // Round using round-sticky words
    // If we spill into the next binade, correct
    let rind = rmode.index(negative:s == .minus, lsb:Int(c_prov))
    if Self.bid_roundbound_128[rind] < UInt128(high: z.w[4], low: z.w[3]) {
      c_prov += 1
      if c_prov == (1 << 53) {
        c_prov = 1 << 52
        e_out += 1
      }
    }

    // Check for overflow
    if e_out >= 2047 { return Self.overflow(s, rmode) }
    
    // Modify the exponent for a tiny result, otherwise chop the implicit bit
    if c_prov < (1 << 52) { e_out = 0 }
    else { c_prov &= (1 << 52) - 1 }
    
    // Set the inexact and underflow flag as appropriate
    //      if (z.w[4] != 0) || (z.w[3] != 0) {
    //          pfpsf.insert(.inexact)
    //      }
    // Package up the result as a binary floating-point number
    return Self.float(s, e_out, UInt64(bitPattern:c_prov))
  }
  
  /// Convert the active decimal floating-point number (any size) to a 64-bit
  /// Double and return this value.
  func float32(_ rmode: Rounding) -> Float {
    var fp:Float?, s:Sign, e:Int, k:Int, high:RawSignificand
    (s, e, k, high, fp) = self.unpack()
    if let x = fp { return x }
    
    let size = Self.signBit+1
    let shift: Int, offset: Int, maxExp: Int, minExp: Int
    switch size {
      case  32: shift = 25; offset = 89; maxExp = 39; minExp = -80
      case  64: shift = 59; offset = 59; maxExp = 39; minExp = -80
      case 128: shift = 0; offset = 0; maxExp = 39; minExp = -80
      default:  shift = 31; offset = 89; maxExp = .max; minExp = .min
    }
    
    // Correct to 2^112 <= c < 2^113 with corresponding exponent adding 113-24=89
    // In fact shift a further 6 places ready for reciprocal multiplication
    // Thus (113-24)+6=95, a shift of 31 given that we've already upacked in c.hi
    let c = UInt128(high: UInt64(high), low: 0) << shift
    k += offset
    
    // check for underflow and overflow
    if e >= maxExp { return Self.overflow(s, rmode) }
    if e <= minExp { e = minExp }
    
    // Check for "trivial" overflow, when 10^e * 1 > 2^{sci_emax+1}, just to
    // keep tables smaller (it would be intercepted later otherwise).
    //
    // (Note that we may have normalized the coefficient, but we have a
    //  corresponding exponent postcorrection to account for; this can
    //  afford to be conservative anyway.)
    //
    // We actually check if e >= ceil((sci_emax + 1) * log_10(2))
    // which in this case is e >= ceil(1024 * log_10(2)) = ceil(308.25) = 309
    
    // Look up the breakpoint and approximate exponent
    let m_min = Tables.bid_breakpoints_binary32[e+80]
    var e_out = Self.exponents_binary32(e+80) - Int(k)
    
    // Choose provisional exponent and reciprocal multiplier based on breakpoint
    var r = UInt256()
    if c <= m_min {
      r = Tables.bid_multipliers1_binary32[e+80]
    } else {
      r = Tables.bid_multipliers2_binary32[e+80]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    Self.mul128x256to384(&z, c, r)
    
    // Check for exponent underflow and compensate by shifting the product
    // Cut off the process at precision+2, since we can't really shift further
    if e_out < 1 {
      var d = 1 - e_out
      if d > 26 { d = 26 }
      e_out = 1
      let sz = Self.srl256_short(z.w[5], z.w[4], z.w[3], z.w[2], d)
      z.w[2...5] = sz.w[0...3]
    }
    var c_prov = Int64(z.w[5])
    
    // Round using round-sticky words
    // If we spill into the next binade, correct
    let rind = rmode.index(negative:s == .minus, lsb:Int(c_prov))
    if Self.bid_roundbound_128[rind] < UInt128(high: z.w[4], low: z.w[3]) {
      c_prov += 1
      if c_prov == (1 << 24) {
        c_prov = 1 << 23
        e_out += 1
      }
    }

    // Check for overflow
    if e_out >= 255 { return Self.overflow(s, rmode) }
    
    // Modify the exponent for a tiny result, otherwise chop the implicit bit
    if c_prov < (1 << 23) { e_out = 0 }
    else { c_prov &= (1 << 23) - 1 }
    
    // Set the inexact and underflow flag as appropriate
    //      if (z.w[4] != 0) || (z.w[3] != 0) {
    //          pfpsf.insert(.inexact)
    //      }
    // Package up the result as a binary floating-point number
    return Self.float(s, e_out, UInt64(bitPattern:c_prov))
  }
  
  // Shift 4-part 2^196 * x3 + 2^128 * x2 + 2^64 * x1 + x0
  // right by "c" bits (must have c < 64)
  static func srl256_short(_ x3:UInt64, _ x2:UInt64,
                           _ x1:UInt64, _ x0:UInt64, _ c:Int) -> UInt256 {
    let _x0 = (x1 << (64 - c)) + (x0 >> c)
    let _x1 = (x2 << (64 - c)) + (x1 >> c)
    let _x2 = (x3 << (64 - c)) + (x2 >> c)
    let _x3 = x3 >> c
    return UInt256(w: [_x0, _x1, _x2, _x3])
  }
  
  static func round(_ x: Self, _ rmode: Rounding) -> Self {
    var res = RawBitPattern(0)
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
      return Self(RawData(C1))
    } else if x.isInfinite {
      return x
    }
    
    // if x is 0 or non-canonical return 0 preserving the sign bit and
    // the preferred exponent of MAX(Q(x), 0)
    exp = exp - exponentBias
    if C1 == 0 {
      if exp < 0 { exp = 0 }
      return Self(sign:x_sign, expBitPattern:exp+exponentBias, sigBitPattern:0)
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
          if x_sign == .minus {
            return Self(sign:.minus,expBitPattern:exponentBias,sigBitPattern:1)
          } else {
            return zero()
          }
        }
      case .up:
        // return 0 if (exp <= -p)
        if exp <= -maximumDigits {
          if x_sign == .minus {
            return zero(.minus)
          } else {
            return Self(sign:.plus,expBitPattern:exponentBias,sigBitPattern:1) // res = zero+1
          }
        }
      case .towardZero:
        // return 0 if (exp <= -p)
        if exp <= -maximumDigits {
          return zero(x_sign)
        }
      default: break
    }    // end switch ()
    
    // q = nr. of decimal digits in x (1 <= q <= 54)
    //  determine first the nr. of bits in x
    let q : Int = _digitsIn(C1)
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
          C1 = C1 + RawBitPattern(bid_midpoint64(ind - 1))
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = UInt64(C1).multipliedFullWidth(by: bid_ten2mk64(ind - 1))
          
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
            res = RawBitPattern(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 { // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = RawBitPattern((P128.high >> shift))
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
          // set exponent to zero as it was negative before.
          // res = x_sign | zero | res;
          return Self(sign: x_sign, expBitPattern: exponentBias,
                      sigBitPattern: RawBitPattern(res))
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
          C1 = C1 + RawBitPattern(bid_midpoint64(ind - 1))
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64[ind - 1]
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          P128 = UInt64(C1).multipliedFullWidth(by: bid_ten2mk64(ind - 1))
          
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          //   C* = floor(C*) - logical right shift; C* has p decimal digits,
          //       correct by Prop. 1)
          // else
          //   C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = RawBitPattern(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = Self.RawBitPattern((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // midpoints are already rounded correctly
          // determine inexactness of the rounding of C*
          // if (0 < f* - 1/2 < 10^(-x)) then
          //   the result is exact
          // else // if (f* - 1/2 > T*) then
          //   the result is inexact
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, expBitPattern: exponentBias, sigBitPattern: res)
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
          P128 = UInt64(C1).multipliedFullWidth(by: bid_ten2mk64(ind - 1))
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = RawBitPattern(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = RawBitPattern(P128.high >> shift)
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
          return Self(sign: x_sign, expBitPattern: exponentBias,
                      sigBitPattern: RawBitPattern(res))
        } else {    // if exp < 0 and q + exp <= 0
          // the result is +0 or -1
          if x_sign != .plus {
            return Self(sign:.minus,expBitPattern:exponentBias,sigBitPattern:1)
          } else {
            return zero()
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
          P128 = UInt64(C1).multipliedFullWidth(by: bid_ten2mk64(ind - 1))

          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = RawBitPattern(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = RawBitPattern((P128.high >> shift))
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
          return Self(sign:x_sign,expBitPattern:exponentBias,sigBitPattern:res)
        } else {    // if exp < 0 and q + exp <= 0
          // the result is -0 or +1
          if x_sign != .plus {
            return zero(.minus)
          } else {
            return Self(sign:.plus,expBitPattern:exponentBias,sigBitPattern:1)
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
          P128 = UInt64(C1).multipliedFullWidth(by: bid_ten2mk64(ind - 1))
 
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = RawBitPattern(P128.high)
            fstar.high = 0
            fstar.low = P128.low
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = RawBitPattern((P128.high >> shift))
            fstar.high = P128.high & bid_maskhigh128(ind - 1)
            fstar.low = P128.low
          }
          // if (f* > 10^(-x)) then the result is inexact
          // set exponent to zero as it was negative before.
          return Self(sign: x_sign, expBitPattern: exponentBias,
                      sigBitPattern: RawBitPattern(res))
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          return zero(x_sign)
          // pfpsf.insert(.inexact)
        }
      default: break
    }    // end switch (
    return Self(sign:x_sign,expBitPattern:exp+exponentBias,sigBitPattern:res)
  }
  
  /***************************************************************************
   *  BID32 nextup
   **************************************************************************/
  static func nextup(_ x: Self) -> Self {
    var largestBID = Self.max()
    
    // check for NaNs and infinities
    if x.isNaN { // check for NaN
      var res = x.data
      if res.get(range:manLower) > largestNumber/10 {
        res.clear(range: nanClearRange) // clear G6-G10 and the payload bits
      } else {
        res.clear(range: g6tog10Range)  // x.ma & 0xfe0f_ffff // clear G6-G10
      }
      if x.isSNaN { // SNaN
        res.clear(bit: Self.nanBitRange.lowerBound)
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
    var (x_sign, x_exp, C1, _) = x.unpack()
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 || (x.isSpecial && C1 > Self.largestNumber) {
      // x is 0: MINFP = 1 * 10^emin
      return Self(sign: .plus, expBitPattern: minEncodedExponent,
                  sigBitPattern: 1)
    } else { // x is not special and is not zero
      if x == largestBID { // LARGEST_BID {
        // x = +MAXFP = 9999999 * 10^emax
        return Self.infinite() // INFINITY_MASK // +inf
      } else if x == Self(sign:.minus, expBitPattern:minEncodedExponent,
                          sigBitPattern:1) {
        // x = -MINFP = 1...99 * 10^emin
        return Self(sign: .minus, expBitPattern: minEncodedExponent,
                    sigBitPattern: 0)
      } else {
        // -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
        // can add/subtract 1 ulp to the significand
        
        // Note: we could check here if x >= 10^7 to speed up the case q1 = 7
        // q1 = nr. of decimal digits in x (1 <= q1 <= 7)
        //  determine first the nr. of bits in x
        let q1: Int = _digitsIn(C1)
        
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
            x_exp = minEncodedExponent // MIN_EXPON
          }
        }
        if x_sign == .plus {    // x > 0
          // add 1 ulp (add 1 to the significand)
          C1 += 1
          if C1 == largestNumber+1 { // 10_000_000 { // if  C1 = 10^7
            C1 = largestSignificand // C1 = 10^6
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
        return Self(sign: x_sign, expBitPattern: x_exp, sigBitPattern: C1)
      } // end -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
    } // end x is not special and is not zero
  //  return res
  }
  
  static func sqrt(_ x: Self, _ rmode:Rounding) -> Self {
    // unpack arguments, check for NaN or Infinity
    var (sign, exponent, significand, valid) = x.unpack()
    if !valid {
      // x is Inf. or NaN or 0
      if x.isInfinite {
        let res = significand.clearing(range: g6tog10Range)
        if x.isNaNInf && sign == .minus { return Self.nan(sign) }
        return Self.nanQuiet(res)
      }
      // x is 0
      exponent = (exponent + exponentBias) >> 1
      return Self(sign: sign, expBitPattern: exponent, sigBitPattern: 0)
    }
    // x<0?
    if sign == .minus && significand != 0 { return Self.nan() }
    
    //--- get number of bits in the coefficient of x ---
    let tempx = Float32(significand)
    let bin_expon_cx = tempx.exponent // Int(((tempx.bitPattern >> 23) & 0xff) - 0x7f)
    var digits = estimateDecDigits(bin_expon_cx)
    // add test for range
    if significand >= bid_power10_index_binexp(bin_expon_cx) {
      digits += 1
    }
    
    var A10 = significand
    if exponent.isMultiple(of: 2) {
      A10 = (A10 << 2) + A10 // A10 * 5
      A10 += A10             // A10 * 2
    }
    
    let dqe = Double(A10).squareRoot()
    let QE = UInt32(dqe)
    if QE * QE == A10 {
      return Self(sign: .plus, expBitPattern: (exponent + exponentBias) >> 1,
                  sigBitPattern: RawBitPattern(QE))
    }
    // if exponent is odd, scale coefficient by 10
    var scale = Int(13 - digits)
    var exponent_q = exponent + exponentBias - scale
    scale += (exponent_q & 1)   // exp. bias is even
    
    let CT = UInt128(power10(scale)).components.low
    let CA = UInt64(significand) * CT
    let dq = Double(CA).squareRoot()
    
    exponent_q = exponent_q >> 1  // square root of 10^x = 10^(x/2)
    
    let rndMode = rmode.raw
    var Q:UInt32
    if rndMode & 3 == 0 {
      Q = UInt32(dq+0.5)
    } else {
      Q = UInt32(dq)
      if rmode == .up {
        Q+=1
      }
    }
    return Self(sign: .plus, expBitPattern: exponent_q,
                sigBitPattern: RawBitPattern(Q))
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
extension Rounding {
  public var raw:Int {
    switch self {
      case .toNearestOrEven        : return 0x00000
      case .down                   : return 0x00001
      case .up                     : return 0x00002
      case .towardZero             : return 0x00003
      case .toNearestOrAwayFromZero: return 0x00004
      case .awayFromZero           : return 0x00000
      @unknown default: fatalError("Unknown rounding mode")
    }
  }
  
  public func index(negative:Bool=false, lsb:Int=0) -> Int {
    return self.raw << 2 + (lsb & 1) + (negative ? 2 : 0)
  }
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

func addDPAndExponent(_ s:String, _ exp:Int, _ maxDigits:Int) -> String {
  let dp = Locale.current.decimalSeparator ?? "."
  let digits = s.count
  var s = s // mutable argument
  
  func addDP(at pos: Int) {
    s.insert(contentsOf: dp, at: s.index(s.startIndex, offsetBy: pos))
  }
  
  if exp == 0 {
    if digits > 1 { addDP(at: exp+1) }
  } else if abs(exp) > maxDigits || exp < -6 {
    let sign = exp < 0 ? "" : "+"
    if digits > 1 { addDP(at: 1) }
    s += "e" + sign + String(exp) // add plus sign since String suppresses it
  } else if digits <= exp {
    // format the number without an exponent
    return s.padding(toLength: exp+1, withPad: "0", startingAt: 0)
  } else if exp < 0 {
    return "0" + dp + "".padding(toLength:-1-exp,withPad:"0",startingAt:0) + s
  } else {
    // insert the decimal point
    addDP(at: exp+1)
    if s.hasSuffix(dp) { s.removeLast() }
  }
  return s
}

// MARK: - Generic String Conversion functions

/// Converts a decimal floating point number `x` into a string
func string<T:IntDecimal>(from x: T) -> String {
  // unpack arguments, check for NaN or Infinity
  let (sign, exp, coeff, valid) = x.unpack()
  let s = sign == .minus ? "-" : ""
  if valid {
    // x is not special
    let ps = String(coeff)
    let exponentX = Int(exp) - T.exponentBias + (ps.count - 1)
    return s + addDPAndExponent(ps, exponentX, T.maximumDigits)
  } else {
    // x is Inf. or NaN or 0
    var ps = s
    if x.isNaN {
      if x.isSNaN { ps += "S" }
      return ps + "NaN"
    }
    if x.isInfinite { return ps + "Inf" }
    return ps + "0"
  }
}

/// Converts a decimal number string of the form:
/// `[+|-] [inf | nan | snan] digit {digit} [. digit {digit}]`
/// `[e [+|-] digit {digit} ]` to a Decimal<n> number `T.maximumDigits`.
func numberFromString<T:IntDecimal>(_ s: String, round: Rounding) -> T? {
  // keep consistent character case for "infinity", "nan", etc.
  var ps = s.lowercased()
  
  let eos = Character("\0")
  var rightRadixLeadingZeros = 0
  
  func handleEmpty() -> T {
    rightRadixLeadingZeros = T.exponentBias - rightRadixLeadingZeros
    if rightRadixLeadingZeros < 0 {
      rightRadixLeadingZeros = T.exponentBias
    }
    return T(sign:sign, expBitPattern:rightRadixLeadingZeros, sigBitPattern:0)
  }
  
  var getChar: Character { ps.isEmpty ? eos : ps.removeFirst() }

  // get first character
  var c = getChar
  
  // determine sign
  var sign = Sign.plus
  if c == "-" { sign = .minus; c = getChar }
  else if c == "+" { c = getChar }

  // detect special cases (INF or NaN)
  // Infinity?
  if c == "i" && (ps.hasPrefix("nfinity") || ps.hasPrefix("nf")) {
    return T.infinite(sign)
  }
  // return sNaN
  if c == "s" && ps.hasPrefix("nan") {
    // case insensitive check for snan
    var x = T.snan; x.sign = sign
    return x
  } else if c == "n" && ps.hasPrefix("an") {
    // return qNaN & any coefficient
    let coeff = T.RawSignificand(ps.dropFirst(2)) // drop "AN"
    return T.nan(sign, coeff ?? 0)
  }

  var dpPresent = false
  var significand = T.RawBitPattern(0)

  // detect zero (and eliminate/ignore leading zeros)
  if c == "0" || c == "." {
    if c == "." {
      dpPresent = true
      c = getChar
    }
    // if all numbers are zeros (with possibly 1 radix point, the number
    // is zero
    // should catch cases such as: 000.0
    while c == "0" {
      c = getChar
      // for numbers such as 0.0000000000000000000000000000000000001001,
      // we want to count the leading zeros
      if dpPresent {
        rightRadixLeadingZeros+=1
      }
      // if this character is a radix point, make sure we haven't already
      // encountered one
      if c == "." {
        if !dpPresent {
          dpPresent = true
          // if this is the first radix point, and the next character is
          // NULL, we have a zero
          if ps.isEmpty {
            return handleEmpty()
          }
          c = getChar
        } else {
          // if 2 radix points, return NaN
          return T.nan(sign)
        }
      } else if ps.isEmpty {
        return handleEmpty()
      }
    }
  }

  var ndigits = 0, exponScale = 0, addExpon = 0
  var midpoint = false, roundedUp = false

  while c.isNumber || c == "." {
    if c == "." {
      if dpPresent { return T.nan(sign) } // two radix points
      dpPresent = true
      c = getChar
      continue
    }
    if dpPresent { exponScale += 1 }

    ndigits+=1
    if ndigits <= T.maximumDigits {
      significand = (significand << 1) + (significand << 3)   // * 10
      significand += T.RawBitPattern(c.wholeNumberValue ?? 0) // + digit
    } else if ndigits == T.maximumDigits+1 {
      // coefficient rounding
      switch round {
        case .toNearestOrEven:
          midpoint = c == "5" && significand.isMultiple(of: 2)
          if c > "5" || (c == "5" && !significand.isMultiple(of: 2)) {
            significand += 1; roundedUp = true
          }
        case .down:
          if sign == .minus { significand += 1; roundedUp = true }
        case .up:
          if sign == .plus  { significand += 1; roundedUp = true }
        case .toNearestOrAwayFromZero:
          if c >= "5" { significand += 1; roundedUp = true }
        default:
          break
      }
      if significand == T.largestNumber+1 {
        significand = (T.largestNumber+1)/10
        addExpon = 1
      }
      addExpon += 1
    } else { // ndigits > T.maximumDigits+1
      addExpon += 1
      if midpoint && c > "0" {
        significand += 1; midpoint = false; roundedUp = true
      }
    }
    c = getChar
  }

  addExpon -= exponScale + Int(rightRadixLeadingZeros)

  if c == eos {
    return T(sign: sign, expBitPattern: addExpon+T.exponentBias,
             sigBitPattern: T.RawBitPattern(significand))
  }

  if c != "e" { return T.nan(sign) }
  c = getChar

  let sgn_expon = c == "-"
  if c == "-" || c == "+" { c = getChar }
  if c == eos || !c.isNumber { return T.nan(sign) }

  var expon_x = 0
  while c.isNumber {
    if expon_x < (1 << 20) {
      expon_x = (expon_x << 1) + (expon_x << 3) // * 10
      expon_x += c.wholeNumberValue ?? 0        // + digit
    }
    c = getChar
  }

  if c != eos { return T.nan(sign) }

  if sgn_expon { expon_x = -expon_x }
  expon_x += addExpon + T.exponentBias

  if expon_x < 0 {
    if roundedUp { significand -= 1 }
    return T.handleRounding(sign, expon_x, Int(significand),
                            roundedUp ? 1 : 0, round)
  }
  let mant = T.RawBitPattern(significand)
  let result = T.adjustOverflowUnderflow(sign, expon_x, mant, round)
  return T(result)
}
