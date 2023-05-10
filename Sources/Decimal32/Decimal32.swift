/**
Copyright © 2023 Computer Inspirations. All rights reserved.
Portions are Copyright (c) 2014 - 2021 Apple Inc. and the
Swift project authors

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

/// Implementation of the 32-bit Decimal32 floating-point operations from
/// IEEE STD 754-2000 for Floating-Point Arithmetic.
///
/// The IEEE Standard 754-2008 for Floating-Point Arithmetic supports two
/// encoding formats: the decimal encoding format, and the binary encoding
/// format. The Intel(R) Decimal Floating-Point Math Library supports primarily
/// the binary encoding format for decimal floating-point values, but the
/// decimal encoding format is supported too in the library, by means of
/// conversion functions between the two encoding formats.

public struct Decimal32 : Codable, Hashable {
  
  /// set to true to monitor variable state (i.e., invalid operations, etc.)
  private static var enableStateOutput = false
  
  /// internal representation of the Decimal32 number
  public typealias Word = UInt32
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Decimal number storage
  /// 32-bit decimal number is stored here
  var x: Word
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Decimal32 number format definition
  
  static let MAX_EXPON     = 191
  static let MIN_EXPON     = 0
  static let EXPONENT_BIAS = 101
  static let MAX_DIGITS    = 7
  static let MAX_NUMBER    = 9_999_999
  static let MAX_NUMBERP1  = MAX_NUMBER+1
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - 32-bit Binary Integer Decimal (BID32) field definitions
  
  static let totalBits        = Word.bitWidth
  static let signBit          = 31
  static let steeringHighBit  = 30 // 11 for Nan, Inf, & mantissa ≥ 2²³
  static let steeringLowBit   = 29
  static let infinityHighBit  = 28 // 11 for infinity, or MSBs of exponent
  static let infinityLowBit   = 27
  static let nanBit           = 26 // 1 for NaN, 0 - infinity
  static let nanSignalBit     = 25 // 0: quiet Nan, 1: signalling Nan
  static let mantissaHighBit1 = 23 // for mantissa < 2²³
  static let mantissaHighBit2 = 21 // for other mantissas
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Decimal32 State variables - this probably needs to change
  //         to better support multi-tasking
  static public var state = Status.clearFlags
  static public var rounding = FloatingPointRoundingRule.toNearestOrEven
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Decimal32 State constants
  public static let zero         = Self(raw: bid32_zero(0))
  public static let radix        = 10
  public static let pi           = Self(floatLiteral: Double.pi)
  public static let nan          = Self(raw: bid32_nan(0, 0x1F<<3, 0) )
  public static let quietNaN     = nan
  public static let signalingNaN = Self(raw: SNAN_MASK)
  public static let infinity     = Self(raw: bid32_inf(0))
  
  public static func nan(with n: Int) -> Self {
    Self(raw: bid32_nan(0, UInt64(n)<<44, 0))
  }
  
  public static let greatestFiniteMagnitude = Self(raw: bid32_max(0))
  public static let leastNormalMagnitude    = Self(raw:
                                              bid32(0,MIN_EXPON,MAX_NUMBER))
  public static let leastNonzeroMagnitude   = Self(raw: bid32(0, 0, 1))
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers
  internal init(raw: Word) { self.init(bitPattern: raw) }
  
  private func showState() {
    if Self.enableStateOutput && !Self.state.isEmpty {
      print("Warning: \(Self.state)")
    }
  }
  
  /// Creates a new instance from either a raw Binary Integer Decimal (BID),
  /// by default or a Densely Packed Decimal (DPD) encoded 32-bit integer
  /// when `bidEncoding` is `true`.
  public init(bitPattern bits: Word, bidEncoding: Bool = true) {
    if bidEncoding { x = bits }
    else { x = Self.dpd_to_bid32(bits) }
  }
  
  public init(integerLiteral value: Int) {
    self = Self.int64_to_BID32(Int64(value), Self.rounding, &Self.state)
  }
  
  public init(_ value: Int = 0) { self.init(integerLiteral: value) }

  public init(sign: FloatingPointSign, exponentBitPattern: UInt32,
              significandDigits: [UInt8]) {
    let mantissa = significandDigits.reduce(into: 0) { $0 = $0 * 10 + Int($1) }
    self.init(sign: sign, exponent: Int(exponentBitPattern),
              significand: Self(mantissa))
  }
  
  public init(sign: FloatingPointSign, exponent: Int, significand: Self) {
    let sgn = sign == .minus ? Self.SIGN_MASK : 0
    let s = Self.unpack(bid32: significand.x)
    self.init()
    if s.valid {
      x = Self.get_BID32_UF(sgn, exponent, UInt64(s.coeff), 0,
                                 Self.rounding, &Self.state)
    }
  }
  
  public init(signOf: Self, magnitudeOf: Self) {
    let sign = signOf.isSignMinus
    self = sign ? -magnitudeOf.magnitude : magnitudeOf.magnitude
  }
}

//////////////////////////////////////////////////////////////////////////
// MARK: - Custom String Convertible compliance
extension Decimal32 : CustomStringConvertible {
  public var description: String { string(from: self) }
}

extension Decimal32 : CustomDebugStringConvertible {
  public var debugDescription: String { description }
}

extension String {
  public init(_ n: Decimal32) {
    self = n.description
  }
}

extension Decimal32: ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    x = Self.double_to_bid32(value, Self.rounding, &Self.state)
  }
}

extension Decimal32: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    let dec : Self = numberFromString(value) ?? Self.zero
    x = dec.x
  }
}

extension Decimal32 {
  public init?<S: StringProtocol>(_ text:S) {
    guard !text.isEmpty else { return nil }
    self.init(stringLiteral: String(text))
  }
}

extension Decimal32 : Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    Self.equal(lhs, rhs, &Self.state)
  }
}

extension Decimal32 : Comparable {
  public static func < (lhs: Self, rhs: Self) -> Bool {
    Self.lessThan(lhs, rhs, &Self.state)
  }
}

extension Decimal32 : AdditiveArithmetic {
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.add(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
  
  public static func - (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.sub(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
}

extension Decimal32 : SignedNumeric {
  public typealias Magnitude = Decimal32
  
  public var magnitude: Self { Self(raw: x & ~Self.SIGN_MASK) }
  
  public mutating func negate() { self.x = x ^ Self.SIGN_MASK }
  
  public init(_ magnitude: Magnitude) {
    self.init(bitPattern: magnitude.x)
  }
  
  public init<T:BinaryInteger>(_ value: T) {
    self.init(Int(value))
  }
  
  public init?<T:BinaryInteger>(exactly source: T) {
    self.init(Int(source))  // FIXME: - Proper init needed
  }
}

extension Decimal32 : Strideable {
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
}

extension Decimal32 : FloatingPoint {
  
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    let dec64 = Self.bid32_to_bid64(x, &Self.state)
    let res = Self.bid64_round_integral_exact(dec64, rule, &Self.state)
    if Self.state == .inexact { Self.state = .clearFlags }
    x = Self.bid64_to_bid32(res, rule, &Self.state)
  }
  
  public mutating func formRemainder(dividingBy other: Self) {
    x = Self.bid32_rem(self.x, other.x, &Self.state)
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Self) {
    let q = (self/other).rounded(.towardZero)
    self -= q * other
  }
  
  public mutating func formSquareRoot() {
    x = Self.sqrt(x, Self.rounding, &Self.state)
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    x = Self.bid32_fma(lhs.x, rhs.x, self.x, Self.rounding, &Self.state)
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Basic arithmetic operations
  
  public func isEqual(to other: Self) -> Bool { self == other }
  public func isLess(than other: Self) -> Bool { self < other }
  public func isLessThanOrEqualTo(_ other: Self) -> Bool {
    self < other || self == other
  }
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.mul(lhs.x, rhs.x, Self.rounding, &Self.state))
  }

  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.div(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
  
  public static func /= (lhs: inout Self, rhs: Self)  { lhs = lhs / rhs }
  public static func *= (lhs: inout Self, rhs: Self)  { lhs = lhs * rhs }
}

//////////////////////////////////////////////////////////////////////////
// MARK: - Numeric State variables

public extension Decimal32 {
  
  var sign: FloatingPointSign { x & Self.SIGN_MASK != 0 ? .minus : .plus }
  
  var dpd32: Word          { Self.bid_to_dpd32(x) }
  var decimal64: UInt64    { Self.bid32_to_bid64(x, &Self.state) }
  var decimal128: UInt128  { Self.bid32_to_bid128(x, &Self.state) }
  var int: Int             { Self.bid32ToInt(x, Self.rounding, &Self.state) }
  var uint: UInt           { Self.bid32ToUInt(x, Self.rounding, &Self.state) }
  var double: Double       { Self.bid32ToDouble(x, Self.rounding, &Self.state)}
  var float: Float         { Float(double) }// FIXME: - Problem in bid32ToFloat
  var isZero: Bool         { _isZero }
  var isSignMinus: Bool    { sign == .minus }
  var isInfinite: Bool     { Self.isInfinite(x) && !isNaN }
  var isNaN: Bool          { (x & Self.NAN_MASK) == Self.NAN_MASK }
  var isSignalingNaN: Bool { (x & Self.SNAN_MASK) == Self.SNAN_MASK }
  var isFinite: Bool       { !Self.isInfinite(x) }
  var isNormal: Bool       { _isNormal }
  var isSubnormal: Bool    { _isSubnormal }
  var isSpecial: Bool      { Self.isSpecial(x) }
  var isCanonical: Bool    { _isCanonical }
  var isBIDFormat: Bool    { true }
  var ulp: Self            { nextUp - self }
  var nextUp: Self         { Self(raw: Self.bid32_nextup(x, &Self.state)) }
  var significand: Self    { Self(raw: Self.frexp(x).res) }
  var exponent: Int        { Self.frexp(x).exp }

  func unpack() -> (negative: Bool, exp: Int, coeff: UInt32, valid: Bool) {
    return Self.unpack(bid32: x)
  }
  
  private var _isZero: Bool {
    if self.isInfinite { return false }
    if self.isSpecial {
      return ((x & Self.SMALL_COEFF_MASK) |
              Self.LARGE_COEFF_HIGH_BIT) > Self.MAX_NUMBER
    } else {
      return (x & Self.LARGE_COEFF_MASK) == 0
    }
  }
  
  private var _isCanonical: Bool {
    if self.isNaN {    // NaN
      if (x & 0x01f00000) != 0 {
        return false
      } else if (x & 0x000fffff) > 999999 {
        return false
      } else {
        return true
      }
    } else if self.isInfinite {
      return (x & 0x03ffffff) == 0
    } else if self.isSpecial { // 24-bit
      return ((x & Self.SMALL_COEFF_MASK) |
              Self.LARGE_COEFF_HIGH_BIT) <= Self.MAX_NUMBER
    } else { // 23-bit coeff.
      return true
    }
  }
  
  static private func validDecode(_ x: Word) -> (exp:Int, sig:Word)? {
    let exp_x:Int
    let sig_x:Word
    if isInfinite(x) { return nil }
    if isSpecial(x) {
      sig_x = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
      // check for zero or non-canonical
      if sig_x > Self.MAX_NUMBER || sig_x == 0 { return nil }
      exp_x = Int((x & MASK_BINARY_EXPONENT2) >> 21)
    } else {
      sig_x = (x & LARGE_COEFF_MASK)
      if sig_x == 0 { return nil } // zero
      exp_x = Int((x & MASK_BINARY_EXPONENT1) >> 23)
    }
    return (exp_x, sig_x)
  }
  
  private var _isNormal: Bool {
    guard let result = Self.validDecode(x) else { return false }
    
    // if exponent is less than -95, the number may be subnormal
    // if (exp_x - 101 = -95) the number may be subnormal
    if result.exp < 6 {
      let sig_x_prime = UInt64(result.sig) *
      UInt64(Self.bid_ten2k64(result.exp))
      return sig_x_prime >= 1000000 // subnormal test
    } else {
      return true // normal
    }
  }
  
  private var _isSubnormal:Bool {
    guard let result = Self.validDecode(x) else { return false }
    
    // if exponent is less than -95, the number may be subnormal
    // if (exp_x - 101 = -95) the number may be subnormal
    if result.exp < 6 {
      let sig_x_prime = UInt64(result.sig) *
      UInt64(Self.bid_ten2k64(result.exp))
      return sig_x_prime < 1000000  // subnormal test
    } else {
      return false // normal
    }
  }
}

extension Decimal32 : DecimalFloatingPoint {

  /// Only for internal use when creating generic Decimal numbers. The chief
  /// reason for this init is to allow suppert utilities to create Decimal
  /// numbers without needing to know details of the Decimal number layout.
  /// Note: No checks are performed on these parameters.
  /// - Parameters:
  ///   - isNegative: Set to `true` if the number is negative.
  ///   - exponent: A signed and biased base 10 exponent for the number.
  ///   - mantissa: An unsigned integer representing the mantissa of the
  ///               number
  ///   - round: If non-zero, perform underflow rounding
  public init(isNegative:Bool, exponent:Int, mantissa:UInt, round:Int = 0) {
    let sign = isNegative ? Self.SIGN_MASK : 0
    if round == 0 {
      x = Self.bid32(sign, exponent, UInt32(mantissa), Self.rounding,
                     &Self.state)
    } else {
      x = Self.get_BID32_UF(sign, exponent, UInt64(mantissa), round,
                            Self.rounding, &Self.state)
    }
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint-required State variables

  public static var exponentMaximum: Int          { MAX_EXPON }
  public static var exponentBias: Int             { EXPONENT_BIAS }
  public static var significandMaxDigitCount: Int { MAX_DIGITS }
  
  public var significandDigitCount: Int {
    let x = unpack()
    if !x.valid { return -1 }
    return Self.digitsIn(x.coeff)
  }
  
  public var exponentBitPattern: Word { Word(unpack().exp) }
  
  public var significandDigits: [UInt8] {
    let x = unpack()
    if !x.valid { return [] }
    return Array(String(x.coeff)).map { UInt8($0.wholeNumberValue!) }
  }
  
  public var decade: Self {
    Self(raw: Self.bid32(0, self.exponent+Self.exponentBias, 1))
  }
  
}

extension Decimal32 {
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - BID32-based masking bits
  
  static let SIGN_MASK             = Word(1) << signBit
  static let SNAN_MASK             = Word(0x3f) << nanSignalBit
  static let SSNAN_MASK            = SNAN_MASK | SIGN_MASK
  static let NAN_MASK              = Word(0x1f) << nanBit
  static let INFINITY_MASK         = Word(0x0f) << infinityLowBit
  static let SINFINITY_MASK        = SIGN_MASK | INFINITY_MASK
  static let STEERING_BITS_MASK    = Word(3) << steeringLowBit
  static let LARGE_COEFF_MASK      = LARGE_COEFF_HIGH_BIT - 1
  static let SMALL_COEFF_MASK      = (Word(1) << mantissaHighBit2) - 1
  static let LARGE_COEFF_HIGH_BIT  = Word(1) << mantissaHighBit1
  static let MASK_BINARY_EXPONENT1 = Word(0xff) << mantissaHighBit1
  static let MASK_BINARY_EXPONENT2 = Word(0xff) << mantissaHighBit2
  static let QUIET_MASK            = Word.max ^ (Word(1) << nanSignalBit)
  static let LARGEST_BID           = Word(0x77f8_967f) // "9.999999e+96"
  static let EXPONENT_MASK         = Word(0xff)
  
  static let COMB_MASK             = Word(0x7ff0_0000)
  static let NAN_MASK12            = NAN_MASK >> 20
  static let INF_MASK12            = INFINITY_MASK >> 20
  static let SPE_MASK12            = STEERING_BITS_MASK >> 20
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - BID64-based masking bits - used for conversions
  
  static let SPECIAL_ENCODING_MASK64 = UInt64(STEERING_BITS_MASK) << 32
  static let INFINITY_MASK64         = UInt64(INFINITY_MASK) << 32
  static let SINFINITY_MASK64        = UInt64(SINFINITY_MASK) << 32
  static let SIGN_MASK64             = UInt64(SIGN_MASK) << 32
  static let NAN_MASK64              = UInt64(NAN_MASK) << 32
  static let SNAN_MASK64             = UInt64(SNAN_MASK) << 32
  static let SSNAN_MASK64            = UInt64(0xfc00_0000_0000_0000)
  static let MASK_BINARY_EXPONENT1_64 = UInt64(0x7fe0_0000_0000_0000)
  static let LARGE_COEFF_MASK64      = UInt64(0x0007_ffff_ffff_ffff)
  static let SMALL_COEFF_MASK64      = UInt64(0x001f_ffff_ffff_ffff)
  static let LARGE_COEFF_HIGH_BIT64  = UInt64(0x0020_0000_0000_0000)
  static let QUIET_MASK64            = UInt64(0xfdff_ffff_ffff_ffff)
  static let MAX_NUMBER64            = UInt64(9_999_999_999_999_999)
  static let EXPONENT_SHIFT_LARGE64  = 51
  static let EXPONENT_MASK64         = 0x3ff
  
  static let DEC64_EXPONENT_BIAS    = 398
  static let EXPONENT_SHIFT_SMALL64 = 53
  
  static let DEC128_EXPONENT_BIAS   = 6176
  
  static let BINARY_EXPONENT_BIAS  = 0x3ff
  static let BINARY_EXPONENT_MASK  = UInt64(COMB_MASK) << 32
  
  /// Creates a 32-bit Binary Integer Decimal from `s`, the negative
  /// sign bit, `e`, the biased exponent, and `c`, the mantissa bits.
  /// There are two possible BID variants: one with a 23-bit mantissa and
  /// a second using 21 mantissa bits.
  static private func bid32(_ sgn:Word, _ expon:Int, _ coeff:Word,
                            _ rmode:Rounding, _ fpsc: inout Status) -> Word {
    assert(sgn == 0 || sgn == SIGN_MASK, "Invalid sign bit = \(sgn)")
    var expon = expon
    var coeff = coeff
    var rmode = rmode
    
    if coeff > MAX_NUMBER {
      expon += 1
      coeff =  1_000_000
    }
    
    // check for possible underflow/overflow
    if UInt32(bitPattern:Int32(expon)) > MAX_EXPON {
      if expon < 0 {
        // underflow
        if expon + MAX_DIGITS < 0 {
          fpsc.formUnion([.underflow, .inexact])
          if rmode == .down && sgn != 0 {
            return 0x80000001
          }
          if (rmode == .up && sgn == 0) {
            return 1
          }
          
          // result is 0
          return sgn
        }
        
        // swap up & down round modes when negative
        if sgn != 0 {
          if rmode == .up { rmode = .down }
          else if rmode == .down { rmode = .up }
        }
        
        // determine the rounding table index
        let roundIndex = roundboundIndex(rmode) >> 2
        
        // get digits to be shifted out
        let extra_digits = -expon
        coeff += UInt32(bid_round_const_table(roundIndex, extra_digits))
        
        // get coeff*(2^M[extra_digits])/10^extra_digits
        var Q = UInt128()
        __mul_64x64_to_128 (&Q, UInt64(coeff), bid_reciprocals10_64(extra_digits));
        
        // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
        let amount = bid_short_recip_scale[extra_digits]
        
        var _C64 = Q.high >> amount
        var remainder_h = UInt64(0)
        
        if rmode == BID_ROUNDING_TO_NEAREST {
          if (_C64 & 1 != 0) {
            // check whether fractional part of initial_P/10^extra_digits is exactly .5
            
            // get remainder
            let amount2 = 64 - amount
            remainder_h = 0
            remainder_h &-= 1
            remainder_h >>= amount2
            remainder_h = remainder_h & Q.high
            
            if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
              _C64 -= 1
            }
          }
        }
        
        if fpsc.contains(.inexact) {
          fpsc.insert(.underflow)
        } else {
          var status = Status.inexact
          // get remainder
          remainder_h = Q.high << (64 - amount)
          
          switch rmode {
            case BID_ROUNDING_TO_NEAREST, BID_ROUNDING_TIES_AWAY:
              // test whether fractional part is 0
              if (remainder_h == SIGN_MASK64 && (Q.low < bid_reciprocals10_64(extra_digits))) {
                status = Status.clearFlags // BID_EXACT_STATUS;
              }
            case BID_ROUNDING_DOWN, BID_ROUNDING_TO_ZERO:
              if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
                status = Status.clearFlags // BID_EXACT_STATUS;
              }
            default:
              // round up
              var Stemp = UInt64(0), carry = UInt64(0)
              __add_carry_out (&Stemp, &carry, Q.low, bid_reciprocals10_64(extra_digits));
              if (remainder_h >> (64 - amount)) + carry >= UInt64(1) << amount {
                status = Status.clearFlags // BID_EXACT_STATUS;
              }
          }
          
          if !status.isEmpty {
            status.insert(.underflow)
            fpsc.formUnion(status)
          }
          
        }
        
        return sgn | UInt32(_C64)
      }
      
      if coeff == 0 { if expon > MAX_EXPON { expon = MAX_EXPON } }
      while coeff < 1000000 && expon > MAX_EXPON {
        coeff = (coeff << 3) + (coeff << 1)
        expon -= 1
      }
      if UInt32(expon) > MAX_EXPON {
        fpsc.formUnion([.overflow, .inexact])
        // overflow
        var r = sgn | INFINITY_MASK
        switch (rmode) {
          case BID_ROUNDING_DOWN:
            if sgn == 0 {
              r = LARGEST_BID
            }
          case BID_ROUNDING_TO_ZERO:
            r = sgn | LARGEST_BID
          case BID_ROUNDING_UP:
            // round up
            if sgn != 0 {
              r = sgn | LARGEST_BID
            }
          default: break
        }
        return r
      }
    }
    
    var mask = UInt32(1) << 23
    
    // check whether coefficient fits in DECIMAL_COEFF_FIT bits
    if coeff < mask {
      var r = UInt32(expon)
      r <<= 23
      r |= (coeff | sgn)
      return r
    }
    
    // special format
    var r = UInt32(expon)
    r <<= 21
    r |= sgn | STEERING_BITS_MASK
    
    // add coeff, without leading bits
    mask = (UInt32(1) << 21) - 1
    r |= coeff & mask
    return r
  }
  
  static private func bid32(_ s:Int, _ e:Int, _ c:Int) -> Word {
    assert(s >= 0 && s < 2, "Invalid sign bit = \(s)")
    if UInt32(c) < UInt32(1) << mantissaHighBit1 {
        return (UInt32(s) << signBit) + (UInt32(e) << mantissaHighBit1) +
                UInt32(c)
    } else {
        return (UInt32(s) << signBit) + UInt32((0x3 << steeringLowBit) -
               (UInt32(1) << mantissaHighBit1)) +
               (UInt32(e) << mantissaHighBit2) + UInt32(c)
    }
  }
  
  static private func isSpecial(_ x: Word) -> Bool {
    (x & STEERING_BITS_MASK) == STEERING_BITS_MASK
  }
  
  static private func isInfinite(_ x: Word) -> Bool {
    (x & INFINITY_MASK) == INFINITY_MASK
  }
  
  static private func isNaN(_ x: Word) -> Bool {
    (x & NAN_MASK) == NAN_MASK
  }
  
  static private func isSNaN(_ x: Word) -> Bool {
    (x & SNAN_MASK) == SNAN_MASK
  }
  
  static private func isNegative(_ x: Word) -> Bool {
    (x & SIGN_MASK) == SIGN_MASK
  }
  
  static private func unpack(bid32 x: Word) ->
  (negative: Bool, exp: Int, coeff: UInt32, valid: Bool) {
    let negative = isNegative(x)
    var coeff: UInt32
    var exp: Int
    if isSpecial(x) {
      // special encodings
      if isInfinite(x) {
        coeff = x & 0xfe0f_ffff
        if (x & 0x000f_ffff) >= 1_000_000 {
          coeff = x & SSNAN_MASK
        }
        if (x & NAN_MASK) == INFINITY_MASK {
          coeff = x & SINFINITY_MASK
        }
        exp = 0
        return (negative, exp, coeff, false) // NaN or Infinity
      }
      // coefficient
      coeff = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
      
      // check for non-canonical value
      if coeff > MAX_NUMBER {
        coeff = 0
      }
      
      // get exponent
      let tmp = x >> 21
      exp = Int(tmp & EXPONENT_MASK)
      return (negative, exp, coeff, coeff != 0)
    }
    
    // exponent
    let tmp = x >> 23
    exp = Int(tmp & EXPONENT_MASK)
    
    // coefficient
    coeff = (x & LARGE_COEFF_MASK)
    return (negative, exp, coeff, coeff != 0)
  }
  
  // Unpack decimal floating-point number x into sign,exponent,coefficient
  // In special cases, call the macros provided
  // Coefficient is normalized in the binary sense with postcorrection k,
  // so that x = 10^e * c / 2^k and the range of c is:
  //
  // 2^23 <= c < 2^24   (decimal32)
  // 2^53 <= c < 2^54   (decimal64)
  // 2^112 <= c < 2^113 (decimal128)
  static private func unpack(bid32 x: Word, _ status: inout Status) ->
    (s: Int, e: Int, k: Int, c: UInt64, valid: Double?) {
    let s = Int(x >> 31)
    var e=0, k=0, c=UInt64()
    if isSpecial(x) {
      if isInfinite(x) {
        if !isNaN(x) {
          return (s, e, k, c, double_inf(s))
        }
        if (x & (UInt32(1)<<25)) != 0 { status.insert(.invalidOperation) }
        let high = ((x & 0xFFFFF) > 999999) ? 0 : UInt64(x) << 44
        return (s, e, k, c, double_nan(s, high, 0))
      }
      e = Int((x >> 21) & ((UInt32(1)<<8)-1)) - EXPONENT_BIAS
      c = UInt64((UInt32(1)<<23) + (x & ((UInt32(1)<<21)-1)))
      if UInt(c) > MAX_NUMBER {
        return (s, e, k, c, double_zero(s))
      }
      k = 0
    } else {
      e = Int((x >> 23) & ((UInt32(1)<<8)-1)) - EXPONENT_BIAS
      c = UInt64(x) & (UInt64(1)<<23 - 1)
      if c == 0 { return (s, e, k, c, double_zero(s)) }
      k = clz(UInt32(c)) - 8
      c = c << k
    }
    return (s, e, k, c, nil)
  }
  
  @inlinable static func double_zero(_ s:Int) -> Double { double(s, 0, 0) }
  @inlinable static func double_inf(_ s:Int) -> Double { double(s, 2047, 0) }
  @inlinable static func double_nan(_ s:Int, _ c_hi:UInt64,
                                    _ c_lo:UInt64) -> Double {
      double(s, 2047, (c_hi>>13)+(UInt64(1)<<51))
  }
  
  @inlinable static func double(_ s:Int, _ e:Int, _ c:UInt64) -> Double {
      Double(bitPattern: (UInt64(s) << 63) + (UInt64(e) << 52) + c)
  }
  
  static func dpd_to_bid32 (_ pda: UInt32) -> UInt32 {
    let in1 = pda
    let sign = in1 & SIGN_MASK
    let comb = (in1 & COMB_MASK) >> 20
    let trailing = Int(in1 & 0x000f_ffff)
    var res : UInt32
    var nanb = UInt32()
    var exp = 0
    var d0 = UInt32()
    
    if (comb & NAN_MASK12) == 0x780 { // G0..G4 = 11110 -> Inf
      return in1 & SINFINITY_MASK
    } else if (comb & NAN_MASK12) == NAN_MASK12 { // G0..G5 = 11111 -> NaN
      nanb = in1 & SSNAN_MASK
      exp = 0
    } else {
      // Normal number
      if (comb & SPE_MASK12) == SPE_MASK12 { // G0..G1 = 11 -> d0 = 8 + G4
        d0 = ((comb >> 6) & 1) | 8;
        exp = Int(((comb & 0x180) >> 1) | (comb & 0x3f))
      } else {
        d0 = (comb >> 6) & 0x7
        exp = Int(((comb & SPE_MASK12) >> 3) | (comb & 0x3f))
      }
    }
    let d1 = bid_d2b2((trailing >> 10) & 0x3ff)
    let d2 = bid_d2b[trailing & 0x3ff]
    
    let bcoeff = UInt32(d2 + d1 + UInt64(1_000_000 * d0))
    if bcoeff < LARGE_COEFF_HIGH_BIT {
      res = UInt32(exp << 23) | bcoeff | sign
    } else {
      res = UInt32(exp << 21) | sign | STEERING_BITS_MASK |
            (bcoeff & SMALL_COEFF_MASK)
    }
    
    res |= nanb
    return res
  }
  
  static func bid_to_dpd32 (_ pba:UInt32) -> UInt32 {
    let ba = pba
    var res : UInt32
    let sign = (ba & SIGN_MASK)
    let comb = (ba & COMB_MASK) >> 20
    var trailing = (ba & 0xfffff)
    var nanb = UInt32(0), exp = 0
    var bcoeff = UInt32(0)
    
    // Detect infinity, and return canonical infinity
    if (comb & NAN_MASK12) == INF_MASK12 {
      return sign | INFINITY_MASK
      // Detect NaN, and canonicalize trailing
    } else if (comb & NAN_MASK12) == NAN_MASK12 {
      if trailing > 999999 {
        trailing = 0
      }
      nanb = ba & 0xfe00_0000
      exp = 0
      bcoeff = trailing
    } else {    // Normal number
      if (comb & SPE_MASK12) == SPE_MASK12 {
        // G0..G1 = 11 -> exp is G2..G11
        exp = Int((comb >> 1) & 0xff)
        bcoeff = ((8 + (comb & 1)) << 20) | trailing
      } else {
        exp = Int((comb >> 3) & 0xff)
        bcoeff = ((comb & 7) << 20) | trailing
      }
      // Zero the coefficient if non-canonical (>= 10^7)
      if bcoeff > MAX_NUMBER {
        bcoeff = 0
      }
    }
    
    let b0 = bcoeff / 1000000
    let b1 = Int(bcoeff / 1000) % 1000
    let b2 = Int(bcoeff % 1000)
    let dcoeff = (bid_b2d[b1] << 10) | bid_b2d[b2]
    
    if b0 >= 8 {   // is b0 8 or 9?
      res = UInt32(UInt64(sign | ((SPE_MASK12 | UInt32((exp >> 6) << 7) |
                  ((b0 & 1) << 6) | UInt32(exp & 0x3f)) << 20)) | dcoeff)
    } else {   // else b0 is 0..7
      res = UInt32(UInt64(sign | ((UInt32((exp >> 6) << 9) | (b0 << 6) |
                                   UInt32(exp & 0x3f)) << 20)) | dcoeff)
    }
    
    res |= nanb
    return res
  }
  
  /*
   * Takes a BID32 as input and converts it to a BID128 and returns it.
   */
  static func bid32_to_bid128(_ x:UInt32, _ pfpsc: inout Status) -> UInt128 {
    let (sign_x, exponent_x, coefficient_x, ok) = unpack(bid32: x)
    if !ok {
      if isInfinite(x){
        if isSNaN(x) {
          pfpsc.insert(.invalidOperation)
        }
        var res = UInt128()
        let low = UInt64(coefficient_x & 0x000fffff)
        __mul_64x128_low(&res, low, bid_power10_table_128(27))
        let high = res.high | ((UInt64(coefficient_x) << 32) &
                               0xfc00000000000000)
        return UInt128(high: high, low: low)
      }
    }
    let tmp = UInt64((exponent_x + DEC128_EXPONENT_BIAS - EXPONENT_BIAS) << 49)
    let sgn = sign_x ? SIGN_MASK : 0
    return UInt128(high: (UInt64(sgn) << 32) | tmp,
                   low: UInt64(coefficient_x))
  }
  
  /*
   * Takes a BID32 as input and converts it to a BID64 and returns it.
   */
  static func bid32_to_bid64 (_ x: UInt32, _ pfpsf: inout Status) -> UInt64 {
    let (sign_x, exponent_x, coefficient_x, ok) = unpack(bid32: x)
    if !ok {
      // Inf, NaN, 0
      if isInfinite(x) {
        if isSNaN(x) {    // sNaN
          pfpsf.insert(.invalidOperation)
        }
        var res = UInt64(coefficient_x & 0x000fffff)
        res *= 1_000_000_000
        res |= (UInt64(coefficient_x) << 32) & SSNAN_MASK64
        return res
      }
    }
    
    // no UF/OF
    let sgn = sign_x ? UInt64(SIGN_MASK) << 32 : 0
    let expon = exponent_x + DEC64_EXPONENT_BIAS - EXPONENT_BIAS
    let coeff = UInt64(coefficient_x)
    var r = UInt64(expon) << EXPONENT_SHIFT_SMALL64
    r |= (coeff | sgn)
    return r
  }
  
  
  /*
   * Takes a BID64 as input and converts it to a BID32 and returns it.
   */
  static func bid64_to_bid32(_ x: UInt64, _ rmode: Rounding,
                             _ pfpsf: inout Status) -> UInt32 {
    // unpack arguments, check for NaN or Infinity, 0
    var (sign_x, exponent_x, coefficient_x, ok) = unpack(bid64: x)
    if !ok {
      if (x & INFINITY_MASK64) == INFINITY_MASK64 {
        let t64 = coefficient_x & 0x0003_ffff_ffff_ffff
        var res = UInt32(t64 / 1_000_000_000)
        res |= UInt32(coefficient_x >> 32) & SSNAN_MASK
        if (x & SNAN_MASK64) == SNAN_MASK64 {    // sNaN
          pfpsf.insert(.invalidOperation)
        }
        return res
      }
      exponent_x = exponent_x - DEC64_EXPONENT_BIAS + EXPONENT_BIAS
      if exponent_x < 0 {
        exponent_x = 0
      }
      if exponent_x > Decimal32.MAX_EXPON {
        exponent_x = Decimal32.MAX_EXPON
      }
      return UInt32(sign_x >> 32) | UInt32(exponent_x << 23)
    }
    
    exponent_x = exponent_x - DEC64_EXPONENT_BIAS + EXPONENT_BIAS
    
    // check number of digits
    if coefficient_x > Decimal32.MAX_NUMBER {
      let tempx = Float(coefficient_x)
      let bin_expon_cx = Int(((tempx.bitPattern >> 23) & 0xff) - 0x7f)
      var extra_digits = Int(bid_estimate_decimal_digits(bin_expon_cx) - 7)
      // add test for range
      if coefficient_x >= bid_power10_index_binexp[bin_expon_cx] {
        extra_digits+=1
      }
      
      var rmode1 = roundboundIndex(rmode) >> 2
      if sign_x != 0 && UInt(rmode1 - 1) < 2 {
        rmode1 = 3 - rmode1
      }
      
      exponent_x += extra_digits
      if (exponent_x < 0) && (exponent_x + MAX_DIGITS >= 0) {
        pfpsf.insert(.underflow)
        if exponent_x == -1 {
          if (coefficient_x + bid_round_const_table(rmode1, extra_digits) >=
              bid_power10_table_128(extra_digits + 7).low) {
            pfpsf = []
          }
          extra_digits -= exponent_x
          exponent_x = 0
        }
      }
      coefficient_x += bid_round_const_table(rmode1, extra_digits)
      var Q = UInt128()
      __mul_64x64_to_128(&Q, coefficient_x, bid_reciprocals10_64(extra_digits))
      
      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      let amount = bid_short_recip_scale[extra_digits]
      
      coefficient_x = Q.high >> amount
      
      if (coefficient_x & 1) != 0 {
        // check whether fractional part of initial_P/10^extra_digits
        // is exactly .5
        
        // get remainder
        let remainder_h = Q.high << (64 - amount)
        
        if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
          coefficient_x-=1
        }
      }
      
      var status = Status.inexact
      // get remainder
      let remainder_h = Q.high << (64 - amount)
      
      switch rmode {
        case BID_ROUNDING_TO_NEAREST, BID_ROUNDING_TIES_AWAY:
          // test whether fractional part is 0
          if (remainder_h == (UInt64(SIGN_MASK) << 32) &&
              (Q.low < bid_reciprocals10_64(extra_digits))) {
            status = []
          }
        case BID_ROUNDING_DOWN, BID_ROUNDING_TO_ZERO:
          if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
            status = []
          }
        default:
          // round up
          var Stemp = UInt64(), carry = UInt64()
          __add_carry_out(&Stemp, &carry, Q.low,
                          bid_reciprocals10_64(extra_digits));
          if (remainder_h >> (64 - amount)) + carry >= (UInt64(1) << amount) {
            status = []
          }
      }
      if !status.isEmpty {
        pfpsf.formUnion(status)
      }
    }
    return bid32(UInt32(sign_x >> 32), exponent_x, Word(coefficient_x),
                 rmode, &pfpsf)
  }
  
  /****************************************************************************
   *  BID64_round_integral_exact
   ***************************************************************************/
  static func bid64_round_integral_exact(_ x: UInt64, _ rmode: Rounding,
                                         _ pfpsf: inout Status) -> UInt64 {
    var res = UInt64(0xbaddbaddbaddbadd)
    var x = x
    let x_sign = x & SIGN_MASK64 // 0 for positive, SIGN_MASK for negative
    var exp = 0
    
    // check for NaNs and infinities
    if (x & NAN_MASK64 ) == NAN_MASK64 {    // check for NaN
      if (x & 0x0003_ffff_ffff_ffff) > 999_999_999_999_999 {
        x = x & 0xfe00000000000000  // clear G6-G12 and the payload bits
      } else {
        x = x & 0xfe03ffffffffffff    // clear G6-G12
      }
      if (x & SNAN_MASK64) == SNAN_MASK64 {    // SNaN
        // set invalid flag
        pfpsf.insert(.invalidOperation)
        // return quiet (SNaN)
        res = x & QUIET_MASK64
      } else {    // QNaN
        res = x
      }
      return res
    } else if (x & INFINITY_MASK64) == INFINITY_MASK64 { // check for Infinity
      return x_sign | INFINITY_MASK64
    }
    // unpack x
    var C1: UInt64
    if (x & SPECIAL_ENCODING_MASK64) == SPECIAL_ENCODING_MASK64 {
      // if the steering bits are 11 (condition will be 0), then
      // the exponent is G[0:w+1]
      exp = Int((x & SMALL_COEFF_MASK64) >> EXPONENT_SHIFT_LARGE64) -
                DEC64_EXPONENT_BIAS
      C1 = (x & LARGE_COEFF_MASK64) | LARGE_COEFF_HIGH_BIT64
      if C1 > MAX_NUMBER64 {    // non-canonical
        C1 = 0;
      }
    } else {    // if ((x & MASK_STEERING_BITS) != MASK_STEERING_BITS)
      exp = Int((x & MASK_BINARY_EXPONENT1_64) >> EXPONENT_SHIFT_SMALL64) -
                DEC64_EXPONENT_BIAS
      C1 = (x & SMALL_COEFF_MASK64)
    }
    
    // if x is 0 or non-canonical return 0 preserving the sign bit and
    // the preferred exponent of MAX(Q(x), 0)
    if C1 == 0 {
      if exp < 0 {
        exp = 0
      }
      return x_sign | ((UInt64(exp) + UInt64(DEC64_EXPONENT_BIAS)) <<
                       EXPONENT_SHIFT_SMALL64)
    }
    // x is a finite non-zero number (not 0, non-canonical, or special)
    let zero = UInt64(0x31c0_0000_0000_0000)
    switch rmode {
      case BID_ROUNDING_TO_NEAREST, BID_ROUNDING_TIES_AWAY:
        // return 0 if (exp <= -(p+1))
        if exp <= -17 {
          res = x_sign | zero
          pfpsf.insert(.inexact)
          return res
        }
      case BID_ROUNDING_DOWN:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          if x_sign != 0 {
            res = (zero+1) | SIGN_MASK64  // 0xb1c0000000000001
          } else {
            res = zero
          }
          pfpsf.insert(.inexact)
          return res
        }
      case BID_ROUNDING_UP:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          if x_sign != 0 {
            res = zero | SIGN_MASK64  // 0xb1c0000000000000
          } else {
            res = zero+1
          }
          pfpsf.insert(.inexact)
          return res
        }
      case BID_ROUNDING_TO_ZERO:
        // return 0 if (exp <= -p)
        if exp <= -16 {
          res = x_sign | zero
          pfpsf.insert(.inexact)
          return res
        }
      default: break
    }    // end switch ()
    
    // q = nr. of decimal digits in x (1 <= q <= 54)
    //  determine first the nr. of bits in x
    let q = digitsIn(UInt32(C1))
    
    if exp >= 0 {    // -exp <= 0
      // the argument is an integer already
      return x
    }
    
    var ind: Int
    var P128 = UInt128(), fstar = UInt128()
    switch rmode {
      case BID_ROUNDING_TO_NEAREST:
        if ((q + exp) >= 0) {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient; exp will be 0
          ind = -exp;    // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 = C1 + 1/2 * 10^x where the result C1 fits in 64 bits
          // FOR ROUND_TO_NEAREST, WE ADD 1/2 ULP(y) then truncate
          C1 = C1 + bid_midpoint64(ind - 1)
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64(ind - 1)
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          __mul_64x64_to_128(&P128, C1, bid_ten2mk64(ind - 1))
          
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
            res = P128.high
            fstar = UInt128(high: 0, low: P128.low)
          } else if (ind - 1 <= 21) { // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = (P128.high >> shift)
            fstar = UInt128(high: P128.high & bid_maskhigh128(ind - 1),
                            low: P128.low)
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
          if (ind - 1 <= 2) {
            if (fstar.low > SIGN_MASK64) {
              // f* > 1/2 and the result may be exact
              // fstar.lo - MASK_SIGN is f* - 1/2
              if ((fstar.low - SIGN_MASK64) > bid_ten2mk64(ind - 1)) {
                // set the inexact flag
                pfpsf.insert(.inexact)
              }    // else the result is exact
            } else {    // the result is inexact; f2* <= 1/2
              // set the inexact flag
              pfpsf.insert(.inexact)
            }
          } else {    // if 3 <= ind - 1 <= 21
            if fstar.high > bid_onehalf128(ind - 1) ||
                (fstar.high == bid_onehalf128(ind - 1) && fstar.low != 0) {
              // f2* > 1/2 and the result may be exact
              // Calculate f2* - 1/2
              if fstar.high > bid_onehalf128(ind - 1) ||
                  fstar.low > bid_ten2mk64(ind - 1) {
                // set the inexact flag
                pfpsf.insert(.inexact)
              }    // else the result is exact
            } else {    // the result is inexact; f2* <= 1/2
              // set the inexact flag
              pfpsf.insert(.inexact)
            }
          }
          // set exponent to zero as it was negative before.
          res = x_sign | zero | res;
          return res
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          res = x_sign | zero;
          pfpsf.insert(.inexact)
          return res
        }
      case BID_ROUNDING_TIES_AWAY:
        if (q + exp) >= 0 {    // exp < 0 and 1 <= -exp <= q
          // need to shift right -exp digits from the coefficient; exp will be 0
          ind = -exp   // 1 <= ind <= 16; ind is a synonym for 'x'
          // chop off ind digits from the lower part of C1
          // C1 = C1 + 1/2 * 10^x where the result C1 fits in 64 bits
          // FOR ROUND_TO_NEAREST, WE ADD 1/2 ULP(y) then truncate
          C1 = C1 + bid_midpoint64(ind - 1)
          // calculate C* and f*
          // C* is actually floor(C*) in this case
          // C* and f* need shifting and masking, as shown by
          // bid_shiftright128[] and bid_maskhigh128[]
          // 1 <= x <= 16
          // kx = 10^(-x) = bid_ten2mk64(ind - 1)
          // C* = (C1 + 1/2 * 10^x) * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          __mul_64x64_to_128(&P128, C1, bid_ten2mk64(ind - 1))
          
          // if (0 < f* < 10^(-x)) then the result is a midpoint
          //   C* = floor(C*) - logical right shift; C* has p decimal digits,
          //       correct by Prop. 1)
          // else
          //   C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = P128.high
            fstar = UInt128(high: 0, low: P128.low)
          } else if ind - 1 <= 21 {  // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = (P128.high >> shift)
            fstar = UInt128(high: P128.high & bid_maskhigh128(ind - 1),
                            low: P128.low)
          }
          // midpoints are already rounded correctly
          // determine inexactness of the rounding of C*
          // if (0 < f* - 1/2 < 10^(-x)) then
          //   the result is exact
          // else // if (f* - 1/2 > T*) then
          //   the result is inexact
          if ind - 1 <= 2 {
            if fstar.low > SIGN_MASK64 {
              // f* > 1/2 and the result may be exact
              // fstar.low - SIGN_MASK64 is f* - 1/2
              if (fstar.low - SIGN_MASK64) > bid_ten2mk64(ind - 1) {
                // set the inexact flag
                pfpsf.insert(.inexact)
              }    // else the result is exact
            } else {    // the result is inexact; f2* <= 1/2
              // set the inexact flag
              pfpsf.insert(.inexact)
            }
          } else {    // if 3 <= ind - 1 <= 21
            if fstar.high > bid_onehalf128(ind - 1) ||
                (fstar.high == bid_onehalf128(ind - 1) && fstar.low != 0) {
              // f2* > 1/2 and the result may be exact
              // Calculate f2* - 1/2
              if fstar.high > bid_onehalf128(ind - 1) ||
                  fstar.low > bid_ten2mk64(ind - 1) {
                // set the inexact flag
                pfpsf.insert(.inexact)
              }    // else the result is exact
            } else {    // the result is inexact; f2* <= 1/2
              // set the inexact flag
              pfpsf.insert(.inexact)
            }
          }
          // set exponent to zero as it was negative before.
          return x_sign | zero | res
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          res = x_sign | zero
          pfpsf.insert(.inexact)
          return res
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
          // kx = 10^(-x) = bid_ten2mk64(ind - 1)
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          __mul_64x64_to_128(&P128, C1, bid_ten2mk64(ind - 1))
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = P128.high
            fstar = UInt128(high: 0, low: P128.low)
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = (P128.high >> shift)
            fstar = UInt128(high: P128.high & bid_maskhigh128(ind - 1),
                            low: P128.low)
          }
          // if (f* > 10^(-x)) then the result is inexact
          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64(ind - 1)) {
            if x_sign != 0 {
              // if negative and not exact, increment magnitude
              res+=1
            }
            pfpsf.insert(.inexact)
          }
          // set exponent to zero as it was negative before.
          return x_sign | zero | res
        } else {    // if exp < 0 and q + exp <= 0
          // the result is +0 or -1
          if x_sign != 0 {
            res = 0xb1c0000000000001
          } else {
            res = zero
          }
          pfpsf.insert(.inexact)
          return res
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
          // kx = 10^(-x) = bid_ten2mk64(ind - 1)
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          __mul_64x64_to_128(&P128, C1, bid_ten2mk64(ind - 1))
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = P128.high
            fstar = UInt128(high: 0, low: P128.low)
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = (P128.high >> shift)
            fstar = UInt128(high: P128.high & bid_maskhigh128(ind - 1),
                            low: P128.low)
          }
          // if (f* > 10^(-x)) then the result is inexact
          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64(ind - 1)) {
            if x_sign == 0 {
              // if positive and not exact, increment magnitude
              res+=1
            }
            pfpsf.insert(.inexact)
          }
          // set exponent to zero as it was negative before.
          return x_sign | zero | res
        } else {    // if exp < 0 and q + exp <= 0
          // the result is -0 or +1
          if x_sign != 0 {
            res = zero | SIGN_MASK64
          } else {
            res = zero+1
          }
          pfpsf.insert(.inexact)
          return res
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
          // kx = 10^(-x) = bid_ten2mk64(ind - 1)
          // C* = C1 * 10^(-x)
          // the approximation of 10^(-x) was rounded up to 64 bits
          __mul_64x64_to_128(&P128, C1, bid_ten2mk64(ind - 1))
          
          // C* = floor(C*) (logical right shift; C has p decimal digits,
          //       correct by Property 1)
          // if (0 < f* < 10^(-x)) then the result is exact
          // n = C* * 10^(e+x)
          
          if ind - 1 <= 2 {    // 0 <= ind - 1 <= 2 => shift = 0
            res = P128.high
            fstar = UInt128(high: 0, low: P128.low)
          } else if ind - 1 <= 21 {    // 3 <= ind - 1 <= 21 => 3 <= shift <= 63
            let shift = bid_shiftright128[ind - 1]    // 3 <= shift <= 63
            res = (P128.high >> shift)
            fstar = UInt128(high: P128.high & bid_maskhigh128(ind - 1),
                            low: P128.low)
          }
          // if (f* > 10^(-x)) then the result is inexact
          if (fstar.high != 0) || (fstar.low >= bid_ten2mk64(ind - 1)) {
            pfpsf.insert(.inexact)
          }
          // set exponent to zero as it was negative before.
          return x_sign | zero | res
        } else {    // if exp < 0 and q + exp < 0
          // the result is +0 or -0
          res = x_sign | zero
          pfpsf.insert(.inexact)
          return res
        }
      default: break
    }    // end switch ()
    return res
  }
  
  static func unpack(bid64 x:UInt64) -> (psign_x:UInt64, pexponent_x:Int,
                                         pcoefficient_x:UInt64, ok:Bool) {
    var tmp, coeff: UInt64
    let psign_x = x & UInt64(SIGN_MASK) << 32
    
    if (x & SPECIAL_ENCODING_MASK64) == SPECIAL_ENCODING_MASK64 {
      // special encodings
      // coefficient
      coeff = (x & LARGE_COEFF_MASK64) | LARGE_COEFF_HIGH_BIT64
      
      if (x & INFINITY_MASK64) == INFINITY_MASK64 {
        let pexponent_x = 0
        var pcoefficient_x = x & 0xfe03ffffffffffff
        if (x & 0x0003ffffffffffff) >= 1_000_000_000_000_000 {
          pcoefficient_x = x & 0xfe00000000000000
        }
        if (x & NAN_MASK64) == INFINITY_MASK64 {
          pcoefficient_x = x & SINFINITY_MASK64
        }
        return (psign_x, pexponent_x, pcoefficient_x, false) // NaN or Infinity
      }
      // check for non-canonical values
      if coeff > UInt64(9_999_999_999_999_999) {
        coeff = 0
      }
      let pcoefficient_x = coeff
      // get exponent
      tmp = x >> EXPONENT_SHIFT_LARGE64
      let pexponent_x = Int(tmp & UInt64(EXPONENT_MASK64))
      return (psign_x, pexponent_x, pcoefficient_x, coeff != 0)
    }
    // exponent
    tmp = x >> EXPONENT_SHIFT_SMALL64;
    let pexponent_x = Int(tmp & UInt64(EXPONENT_MASK64))
    
    // coefficient
    let pcoefficient_x = (x & UInt64(SMALL_COEFF_MASK64))
    return (psign_x, pexponent_x, pcoefficient_x, pcoefficient_x != 0)
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Basic Math Operations
  
  static func add(_ x:UInt32, _ y:UInt32, _ rmode:Rounding,
                  _ status:inout Status) -> UInt32 {
    let (sign_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    let (sign_y, exponent_y, coefficient_y, valid_y) = unpack(bid32: y)
    
    // unpack arguments, check for NaN or Infinity
    if !valid_x {
      // x is Inf. or NaN
      if isNaN(x) {
        if isSNaN(x) || isSNaN(y) {
          status.insert(.invalidOperation)
        }
        return coefficient_x & QUIET_MASK
      }
      // x is Infinity?
      if isInfinite(x) {
        // check if y is Inf
        if (y & NAN_MASK) == INFINITY_MASK {
          if sign_x == sign_y {
            return coefficient_x
          } else {
            // return NaN
            status.insert(.invalidOperation)
            return NAN_MASK
          }
        }
        // check if y is NaN
        if isNaN(y) {
          let res = coefficient_y & QUIET_MASK
          if isSNaN(y) {
            status.insert(.invalidOperation)
          }
          return res
        } else {
          // otherwise return +/-Inf
          return coefficient_x
        }
      } else {
        // x is 0
        if !isInfinite(y) && (coefficient_y != 0) {
          if exponent_y <= exponent_x {
            return y
          }
        }
      }
    }
    if !valid_y {
      // y is Inf. or NaN?
      if isInfinite(y) || isNaN(y) {
        if isSNaN(y) {
          status.insert(.invalidOperation)
        }
        return coefficient_y & QUIET_MASK
      }
      // y is 0
      if coefficient_x == 0 {
        // and x is 0
        var res:UInt32
        if exponent_x <= exponent_y {
          res = UInt32(exponent_x) << 23
        } else {
          res = UInt32(exponent_y) << 23
        }
        if sign_x == sign_y {
          res |= sign_x ? SIGN_MASK : 0
        }
        if rmode == BID_ROUNDING_DOWN && sign_x != sign_y {
          res |= SIGN_MASK
        }
        return res
      } else if exponent_y >= exponent_x {
        return x
      }
    }
    
    // sort arguments by exponent
    var sign_a, sign_b: Bool
    var coefficient_a, coefficient_b: UInt32
    var exponent_a, exponent_b: Int
    if exponent_x < exponent_y {
      sign_a = sign_y
      exponent_a = exponent_y
      coefficient_a = coefficient_y
      sign_b = sign_x
      exponent_b = exponent_x
      coefficient_b = coefficient_x
    } else {
      sign_a = sign_x
      exponent_a = exponent_x
      coefficient_a = coefficient_x
      sign_b = sign_y
      exponent_b = exponent_y
      coefficient_b = coefficient_y
    }
    
    // exponent difference
    var diff_dec_expon = exponent_a - exponent_b
    
    if diff_dec_expon > MAX_DIGITS {
      let tempx = Double(coefficient_a)
      let bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                      BINARY_EXPONENT_BIAS
      let scale_ca = bid_estimate_decimal_digits(bin_expon)
      
      let d2 = 16 - scale_ca
      if diff_dec_expon > d2 {
        diff_dec_expon = Int(d2)
        exponent_b = exponent_a - diff_dec_expon;
      }
    }
    
    var sign_ab = sign_a != sign_b ? Int64(SIGN_MASK)<<32 : 0
    sign_ab = sign_ab >> 63
    let CB = UInt64(bitPattern: (Int64(coefficient_b) + sign_ab) ^ sign_ab)
    
    let SU = UInt64(coefficient_a) * bid_power10_table_128(diff_dec_expon).low
    var S = Int64(bitPattern: SU &+ CB)
    
    if S < 0 {
      sign_a.toggle() // ^= SIGN_MASK
      S = -S
    }
    var P = UInt64(S)
    var n_digits:Int
    if P == 0 {
      sign_a = false
      if rmode == BID_ROUNDING_DOWN { sign_a = true }
      if coefficient_a == 0 { sign_a = sign_x }
      n_digits=0
    } else {
      let tempx = Double(P)
      let bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                      BINARY_EXPONENT_BIAS
      n_digits = Int(bid_estimate_decimal_digits(bin_expon))
      if P >= bid_power10_table_128(n_digits).low {
        n_digits+=1
      }
    }
    
    let sign = sign_a ? SIGN_MASK : 0
    if n_digits <= MAX_DIGITS {
      return bid32(sign, exponent_b, Word(P), rmode, &status)
    }
    
    let extra_digits = n_digits - 7
    
    var irmode = roundboundIndex(rmode) >> 2
    if sign_a && (UInt(irmode) &- 1) < 2 {
      irmode = 3 - irmode
    }
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += bid_round_const_table(irmode, extra_digits)
    var Tmp = UInt128()
    __mul_64x64_to_128(&Tmp, P, bid_reciprocals10_64(extra_digits))
    
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = bid_short_recip_scale[extra_digits]
    var Q = Tmp.high >> amount
    
    // remainder
    let R = P - Q * bid_power10_table_128(extra_digits).low
    if R == bid_round_const_table(irmode, extra_digits) {
      status = []
    } else {
      status.insert(.inexact)
    }
    
    if rmode == BID_ROUNDING_TO_NEAREST {
      if R == 0 {
        Q &= 0xffff_fffe
      }
    }
    return bid32(sign, exponent_b+extra_digits, Word(Q), rmode, &status)
  }
  
  static func sub (_ x:UInt32, _ y:UInt32, _ rmode:Rounding,
                   _ status:inout Status) -> UInt32 {
    var y = y
    if !isNaN(y) { y ^= SIGN_MASK }
    return add(x, y, rmode, &status)
  }
  
  static func mul (_ x:UInt32, _ y:UInt32, _ rmode:Rounding,
                   _ status:inout Status) -> UInt32  {
    var (sign_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    var (sign_y, exponent_y, coefficient_y, valid_y) = unpack(bid32: y)
    
    // unpack arguments, check for NaN or Infinity
    if !valid_x {
      if isSNaN(y) {
        // y is sNaN
        status.insert(.invalidOperation)
      }
      // x is Inf. or NaN
      
      // test if x is NaN
      if isNaN(x) {
        if isSNaN(x) {
          status.insert(.invalidOperation)
        }
        return (coefficient_x & QUIET_MASK)
      }
      // x is Infinity?
      if isInfinite(x) {
        // check if y is 0
        if !isInfinite(y) && (coefficient_y == 0) {
          status.insert(.invalidOperation)
          // y==0 , return NaN
          return NAN_MASK
        }
        // check if y is NaN
        if isNaN(y) {
          // y==NaN , return NaN
          return coefficient_y & QUIET_MASK
        }
        // otherwise return +/-Inf
        return ((x ^ y) & SIGN_MASK) | INFINITY_MASK
      }
      // x is 0
      if !isInfinite(y) {
        if isSpecial(y) {
          exponent_y = Int(UInt32(y >> 21)) & 0xff
        } else {
          exponent_y = Int(UInt32(y >> 23)) & 0xff
        }
        let sign_y = sign_y ? SIGN_MASK : 0
        let sign_x = sign_x ? SIGN_MASK : 0
        
        exponent_x += exponent_y - EXPONENT_BIAS
        if (exponent_x > MAX_EXPON) {
          exponent_x = MAX_EXPON
        } else if (exponent_x < 0) {
          exponent_x = 0
        }
        return UInt32(UInt64(sign_x ^ sign_y) | (UInt64(exponent_x) << 23))
      }
    }
    if !valid_y {
      // y is Inf. or NaN
      // test if y is NaN
      if isNaN(y) {
        if isSNaN(y) {
          // sNaN
          status.insert(.invalidOperation)
        }
        return coefficient_y & QUIET_MASK
      }
      // y is Infinity?
      if isInfinite(y) {
        // check if x is 0
        if coefficient_x == 0 {
          status.insert(.invalidOperation)
          // x==0, return NaN
          return NAN_MASK
        }
        // otherwise return +/-Inf
        return ((x ^ y) & SIGN_MASK) | INFINITY_MASK
      }
      // y is 0
      let sign_y = sign_y ? SIGN_MASK : 0
      let sign_x = sign_x ? SIGN_MASK : 0
      exponent_x += exponent_y - EXPONENT_BIAS
      if exponent_x > MAX_EXPON {
        exponent_x = MAX_EXPON
      } else if exponent_x < 0 {
        exponent_x = 0
      }
      return UInt32(UInt64(sign_x ^ sign_y) | (UInt64(exponent_x) << 23))
    }
    
    // multiply two numbers
    var P = UInt64(coefficient_x) * UInt64(coefficient_y)
    
    //--- get number of bits in C64 ---
    // version 2 (original)
    let tempx = Double(P)
    let bin_expon_p = (Int(tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                      BINARY_EXPONENT_BIAS
    var n_digits = Int(bid_estimate_decimal_digits(bin_expon_p))
    if P >= bid_power10_table_128(n_digits).low {
      n_digits+=1
    }
    
    exponent_x += exponent_y - EXPONENT_BIAS
    
    let extra_digits = Int((n_digits<=7) ? 0 : (n_digits - 7))
    
    exponent_x += extra_digits
    
    if extra_digits == 0 {
      let sign = sign_x != sign_y ? SIGN_MASK : 0
      return bid32(sign, exponent_x, Word(P), rmode, &status)
    }
    
    var rmode1 = roundboundIndex(rmode, sign_x != sign_y, 0)
    if (sign_x != sign_y) && UInt32(rmode1 - 1) < 2 {
      rmode1 = 3 - rmode1
    }
    
    if exponent_x < 0 { rmode1 = 3 }  // RZ
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += bid_round_const_table(rmode1, extra_digits)
    var Tmp = UInt128()
    __mul_64x64_to_128(&Tmp, P, bid_reciprocals10_64(extra_digits))
    
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = bid_short_recip_scale[extra_digits];
    var Q = Tmp.high >> amount
    
    // remainder
    let R = P - Q * bid_power10_table_128(extra_digits).low
    
    if R == bid_round_const_table(rmode1, extra_digits) {
      status = []
    } else {
      status.insert(.inexact)
    }
    
    // __set_status_flags (pfpsf, status);
    if rmode1 == 0 {    //BID_ROUNDING_TO_NEAREST
      if R==0 {
        Q &= 0xffff_fffe
      }
    }
    
    if (exponent_x == -1) && (Q == MAX_NUMBER) &&
        (rmode != BID_ROUNDING_TO_ZERO) {
      rmode1 = roundboundIndex(rmode, sign_x != sign_y, 0)
      if ((sign_x != sign_y) && UInt32(rmode1 - 1) < 2) {
        rmode1 = 3 - rmode1
      }
      
      if ((R != 0) && (rmode == BID_ROUNDING_UP)) || ((rmode1&3 == 0) &&
                          (R+R>=bid_power10_table_128(extra_digits).low)) {
        return bid32(sign_x != sign_y ? 1 : 0, 0, 1000000)
      }
    }
    let sign = sign_x != sign_y ? SIGN_MASK : 0
    return get_BID32_UF(sign, Int(exponent_x), Q, Int(R), rmode, &status)
  }
  
  /*****************************************************************************
   *    BID32 divide
   *****************************************************************************
   *
   *  Algorithm description:
   *
   *  if(coefficient_x<coefficient_y)
   *    p = number_digits(coefficient_y) - number_digits(coefficient_x)
   *    A = coefficient_x*10^p
   *    B = coefficient_y
   *    CA= A*10^(15+j), j=0 for A>=B, 1 otherwise
   *    Q = 0
   *  else
   *    get Q=(int)(coefficient_x/coefficient_y)
   *        (based on double precision divide)
   *    check for exact divide case
   *    Let R = coefficient_x - Q*coefficient_y
   *    Let m=16-number_digits(Q)
   *    CA=R*10^m, Q=Q*10^m
   *    B = coefficient_y
   *  endif
   *    if (CA<2^64)
   *      Q += CA/B  (64-bit unsigned divide)
   *    else
   *      get final Q using double precision divide, followed by 3 integer
   *          iterations
   *    if exact result, eliminate trailing zeros
   *    check for underflow
   *    round coefficient to nearest
   *
   ****************************************************************************/
  
  static func div(_ x:UInt32, _ y:UInt32, _ rmode:Rounding,
                  _ status:inout Status) -> UInt32 {
    var (sign_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    var (sign_y, exponent_y, coefficient_y, valid_y) = unpack(bid32: y)
    
    // unpack arguments, check for NaN or Infinity
    if !valid_x {
      // x is Inf. or NaN
      if isSNaN(y) {   // y is sNaN
        status.insert(.invalidOperation)
      }
      
      // test if x is NaN
      if isNaN(x) {
        if isSNaN(x) {    // sNaN
          status.insert(.invalidOperation)
        }
        return coefficient_x & QUIET_MASK
      }
      // x is Infinity?
      if isInfinite(x) {
        // check if y is Inf or NaN
        if isInfinite(y) {
          // y==Inf, return NaN
          if (y & NAN_MASK) == INFINITY_MASK {    // Inf/Inf
            status.insert(.invalidOperation)
            return NAN_MASK
          }
        } else {
          // otherwise return +/-Inf
          return ((x ^ y) & SIGN_MASK) | INFINITY_MASK
        }
      }
      // x==0
      if !isInfinite(y) && coefficient_y == 0 {
        // y==0 , return NaN
        status.insert(.invalidOperation)
        return NAN_MASK
      }
      if !isInfinite(y) {
        var sign_y = sign_y ? SIGN_MASK : 0
        let sign_x = sign_x ? SIGN_MASK : 0
        if isSpecial(y) {
          exponent_y = Int((UInt32(y >> 21)) & 0xff)
        } else {
          exponent_y = Int((UInt32(y >> 23)) & 0xff)
          sign_y = y & SIGN_MASK
        }
        
        exponent_x = exponent_x - exponent_y + EXPONENT_BIAS
        if exponent_x > MAX_EXPON {
          exponent_x = MAX_EXPON
        } else if exponent_x < 0 {
          exponent_x = 0
        }
        return UInt32(sign_x ^ sign_y) | UInt32(exponent_x) << 23
      }
      
    }
    if !valid_y {
      // y is Inf. or NaN
      // test if y is NaN
      if isNaN(y) {
        if isSNaN(y) {
          // sNaN
          status.insert(.invalidOperation)
        }
        return coefficient_y & QUIET_MASK
      }
      
      // y is Infinity?
      if isInfinite(y) {
        // return +/-0
        return (x ^ y) & SIGN_MASK
      }
      
      // y is 0
      status.insert(.divisionByZero)
      return ((x ^ y) & SIGN_MASK) | INFINITY_MASK
    }
    var diff_expon = exponent_x - exponent_y + EXPONENT_BIAS
    
    var A, B, Q, R: UInt32
    var CA: UInt64
    var ed1, ed2: Int
    if coefficient_x < coefficient_y {
      // get number of decimal digits for c_x, c_y
      //--- get number of bits in the coefficients of x and y ---
      let tempx = Float(coefficient_x)
      let tempy = Float(coefficient_y)
      let bin_index = Int((tempy.bitPattern - tempx.bitPattern) >> 23)
      A = coefficient_x * UInt32(bid_power10_index_binexp[bin_index])
      B = coefficient_y
      
      // compare A, B
      let DU = (A - B) >> 31
      ed1 = 6 + Int(DU)
      ed2 = bid_estimate_decimal_digits(bin_index) + ed1
      let T = bid_power10_table_128(ed1).low
      CA = UInt64(A) * T
      
      Q = 0
      diff_expon = diff_expon - ed2
      
    } else {
      // get c_x/c_y
      Q = coefficient_x/coefficient_y;
      
      R = coefficient_x - coefficient_y * Q;
      
      // will use to get number of dec. digits of Q
      let tempq = Float(Q)
      let bin_expon_cx = Int(tempq.bitPattern >> 23) - 0x7f
      
      // exact result ?
      if R == 0 {
        let sign_x = sign_x ? SIGN_MASK : 0
        let sign_y = sign_y ? SIGN_MASK : 0
        return bid32(sign_x ^ sign_y, diff_expon, Q, rmode, &status)
      }
      // get decimal digits of Q
      var DU = UInt32(bid_power10_index_binexp[bin_expon_cx]) - Q - 1;
      DU >>= 31;
      
      ed2 = 7 - Int(bid_estimate_decimal_digits(bin_expon_cx)) - Int(DU)
      
      let T = bid_power10_table_128(ed2).low
      CA = UInt64(R) * T
      B = coefficient_y
      
      Q *= UInt32(bid_power10_table_128(ed2).low)
      diff_expon -= ed2
    }
    
    let Q2 = UInt32(CA / UInt64(B))
    let B2 = B + B
    let B4 = B2 + B2
    R = UInt32(CA - UInt64(Q2) * UInt64(B))
    Q += Q2
    
    if R != 0 {
      // set status flags
      status.insert(.inexact)
    } else {
      // eliminate trailing zeros
      // check whether CX, CY are short
      if (coefficient_x <= 1024) && (coefficient_y <= 1024) {
        let i = Int(coefficient_y) - 1
        let j = Int(coefficient_x) - 1
        // difference in powers of 2 bid_factors for Y and X
        var nzeros = ed2 - Int(bid_factors[i][0] + bid_factors[j][0])
        // difference in powers of 5 bid_factors
        let d5 = ed2 - Int(bid_factors[i][1] + bid_factors[j][1])
        if d5 < nzeros {
          nzeros = d5
        }
        
        if nzeros != 0 {
          var CT = UInt64(Q) * bid_reciprocals10_32(nzeros)
          CT >>= 32
          
          // now get P/10^extra_digits: shift C64 right by M[extra_digits]-128
          let amount = bid_recip_scale32[nzeros];
          Q = UInt32(CT >> amount)
          
          diff_expon += nzeros
        }
      } else {
        var nzeros = 0
        
        // decompose digit
        let PD = UInt64(Q) * 0x068DB8BB
        var digit_h = UInt32(PD >> 40)
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
//          nzeros += Int(3 & UInt32(
//            bid_packed_10000_zeros[Int(digit_h >> 3)] >> (digit_h & 7)))
        }
        
        if nzeros != 0 {
          var CT = UInt64(Q) * bid_reciprocals10_32(nzeros)
          CT >>= 32
          
          // now get P/10^extra_digits: shift C64 right by M[extra_digits]-128
          let amount = bid_recip_scale32[nzeros]
          Q = UInt32(CT >> amount)
        }
        diff_expon += nzeros
      }
      if diff_expon >= 0 {
        let sign = sign_x != sign_y ? SIGN_MASK : 0
        return bid32(sign, diff_expon, Q, rmode, &status)
      }
    }
    
    let sign = sign_x != sign_y ? SIGN_MASK : 0
    if diff_expon >= 0 {
      var rmode1 = roundboundIndex(rmode) >> 2
      if (sign_x != sign_y) && (UInt32(rmode1) &- 1) < 2 {
        rmode1 = 3 - rmode1
      }
      switch rmode1 {
        case 0, 4:
          // R*10
          R += R
          R = (R << 2) + R
          let B5 = B4 + B
          // compare 10*R to 5*B
          R = B5 &- R
          // correction for (R==0 && (Q&1))
          R -= ((Q | UInt32(rmode1 >> 2)) & 1)
          // R<0 ?
          let D = UInt32(R) >> 31
          Q += D
        case 1, 3:
          break
        default:    // rounding up (2)
          Q+=1
      }
      return bid32(sign, diff_expon, Q, rmode, &status)
    } else {
      // UF occurs
      if diff_expon + 7 < 0 {
        // set status flags
        status.insert(.inexact)
      }
      return get_BID32_UF(sign, diff_expon, UInt64(Q), Int(R), rmode, &status)
    }
  }
  
  static func bid32_rem(_ x:UInt32, _ y:UInt32, _ pfpsf: inout Status) ->
                        UInt32 {
    var (sign_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    var (_,      exponent_y, coefficient_y, valid_y) = unpack(bid32: y)
    
    // unpack arguments, check for NaN or Infinity
    if !valid_x {
      // x is Inf. or NaN or 0
      if isSNaN(y) {   // y is sNaN
        pfpsf.insert(.invalidOperation)
      }
      
      // test if x is NaN
      if isNaN(x) {
        if isSNaN(x) {
          pfpsf.insert(.invalidOperation)
        }
        return coefficient_x & QUIET_MASK
      }
      // x is Infinity?
      if isInfinite(x) {
        if !isNaN(y) {
          pfpsf.insert(.invalidOperation)
          return NAN_MASK  // return NaN
        }
      }
      // x is 0
      // return x if y != 0
      if (y & INFINITY_MASK) < INFINITY_MASK && coefficient_y != 0 {
        if isSpecial(y) {
          exponent_y = Int(y >> 21) & 0xff;
        } else {
          exponent_y = Int(y >> 23) & 0xff;
        }
        
        if exponent_y < exponent_x {
          exponent_x = exponent_y
        }
        
        var x = UInt32(exponent_x)
        x <<= 23
        return x | (sign_x ? SIGN_MASK : 0)
      }
      
    }
    if !valid_y {
      // y is Inf. or NaN
      
      // test if y is NaN
      if isNaN(y) {
        if isSNaN(y) {
          pfpsf.insert(.invalidOperation)
        }
        return coefficient_y & QUIET_MASK
      }
      // y is Infinity?
      if isInfinite(y) {
        let sign = sign_x ? 1 : 0
        return bid32(sign, exponent_x, Int(coefficient_x))
      }
      // y is 0, return NaN
      pfpsf.insert(.invalidOperation)
      return NAN_MASK
    }
    
    
    var diff_expon = exponent_x - exponent_y
    if diff_expon <= 0 {
      diff_expon = -diff_expon
      
      if (diff_expon > 7) {
        // |x|<|y| in this case
        return x
      }
      // set exponent of y to exponent_x, scale coefficient_y
      let T = bid_power10_table_128(diff_expon).low
      let CYL = UInt64(coefficient_y) * T;
      if CYL > (UInt64(coefficient_x) << 1) {
        return x
      }
      
      let CY = UInt32(CYL)
      let Q = coefficient_x / CY
      var R = coefficient_x - Q * CY
      
      let R2 = R + R;
      if R2 > CY || (R2 == CY && (Q & 1) != 0) {
        R = CY - R
        sign_x.toggle()
      }
      
      return bid32(sign_x ? 1 : 0, exponent_x, Int(R))
    }
    
    var CX = UInt64(coefficient_x)
    var Q64 = UInt64()
    while diff_expon > 0 {
      // get number of digits in coeff_x
      let tempx = Float(CX)
      let bin_expon = Int((tempx.bitPattern >> 23) & 0xff) - 0x7f
      let digits_x = Int(bid_estimate_decimal_digits(bin_expon))
      // will not use this test, dividend will have 18 or 19 digits
      //if(CX >= bid_power10_table_128[digits_x].lo)
      //      digits_x++;
      
      var e_scale = Int(18 - digits_x)
      if (diff_expon >= e_scale) {
        diff_expon -= e_scale;
      } else {
        e_scale = diff_expon;
        diff_expon = 0;
      }
      
      // scale dividend to 18 or 19 digits
      CX *= bid_power10_table_128(e_scale).low
      
      // quotient
      Q64 = CX / UInt64(coefficient_y)
      // remainder
      CX -= Q64 * UInt64(coefficient_y)
      
      // check for remainder == 0
      if CX == 0 {
        return bid32(sign_x ? 1 : 0, exponent_y, 0)
      }
    }
    
    coefficient_x = UInt32(CX)
    let R2 = coefficient_x + coefficient_x
    if R2 > coefficient_y || (R2 == coefficient_y && (Q64 & 1) != 0) {
      coefficient_x = coefficient_y - coefficient_x
      sign_x.toggle()
    }
    
    return bid32(sign_x ? 1 : 0, exponent_y, Int(coefficient_x))
  }
  
  static func sqrt(_ x: UInt32, _ rmode:Rounding, _ status:inout Status) ->
                  UInt32 {
    // unpack arguments, check for NaN or Infinity
    var (sign_x, exponent_x, coefficient_x, valid) = unpack(bid32: x)
    if !valid {
      // x is Inf. or NaN or 0
      if isInfinite(x) {
        var res = coefficient_x
        if (coefficient_x & SSNAN_MASK) == SINFINITY_MASK {   // -Infinity
          res = NAN_MASK
          status.insert(.invalidOperation)
        }
        if (x & SNAN_MASK) == SNAN_MASK {   // sNaN
          status.insert(.invalidOperation)
        }
        return res & QUIET_MASK
      }
      // x is 0
      exponent_x = (exponent_x + EXPONENT_BIAS) >> 1
      return (sign_x ? SIGN_MASK : 0) | (UInt32(exponent_x) << 23)
    }
    // x<0?
    if sign_x && coefficient_x != 0 {
      status.insert(.invalidOperation)
      return NAN_MASK
    }
    
    //--- get number of bits in the coefficient of x ---
    let tempx = Float32(coefficient_x)
    let bin_expon_cx = Int(((tempx.bitPattern >> 23) & 0xff) - 0x7f)
    var digits_x = bid_estimate_decimal_digits(bin_expon_cx);
    // add test for range
    if coefficient_x >= bid_power10_index_binexp[bin_expon_cx] {
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
      return bid32(0, (exponent_x + EXPONENT_BIAS) >> 1, Int(QE))
    }
    // if exponent is odd, scale coefficient by 10
    var scale = Int(13 - digits_x)
    var exponent_q = exponent_x + EXPONENT_BIAS - scale
    scale += (exponent_q & 1)   // exp. bias is even
    
    let CT = bid_power10_table_128(scale).low
    let CA = UInt64(coefficient_x) * CT
    let dq = Double(CA).squareRoot()
    
    exponent_q = (exponent_q) >> 1;
    
    status.insert(.inexact)

    let rnd_mode = roundboundIndex(rmode) >> 2
    var Q:UInt32
    if ((rnd_mode) & 3) == 0 {
      Q = UInt32(dq+0.5)
    } else {
      Q = UInt32(dq)
      
      /*// get sign(sqrt(CA)-Q)
       R = CA - Q * Q;
       R = ((BID_SINT32) R) >> 31;
       D = R + R + 1;
       
       C4 = CA;
       Q += D;
       if ((BID_SINT32) (Q * Q - C4) > 0)
       Q--;*/
      if (rmode == BID_ROUNDING_UP) {
        Q+=1
      }
    }
    return bid32(0, exponent_q, Int(Q))
  }
  
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Comparison
  
  static func equal (_ x: Self, _ y: Self, _ status: inout Status) -> Bool {
    // NaN (CASE1)
    // if either number is NAN, the comparison is unordered,
    // rather than equal : return 0
    if x.isNaN || y.isNaN {
      if x.isSignalingNaN || y.isSignalingNaN {
        status.insert(.invalidOperation)  // set exception if sNaN
      }
      return false
    }
    // SIMPLE (CASE2)
    // if all the bits are the same, these numbers are equivalent.
    if x.x == y.x {
      return true
    }
    
    // INFINITY (CASE3)
    if x.isInfinite && y.isInfinite {
      return ((x.x ^ y.x) & SIGN_MASK) != SIGN_MASK
    }
    // ONE INFINITY (CASE3')
    if x.isInfinite || y.isInfinite {
      return false
    }
    
    // if steering bits are 11 (condition will be 0), exponent is G[0:w+1] =>
    //var exp_x, sig_x: UInt32; var non_canon_x: Bool
    var (exp_x, sig_x, non_canon_x) = extractExpSig(x.x)
    
    // if steering bits are 11 (condition will be 0), exponent is G[0:w+1] =>
    var (exp_y, sig_y, non_canon_y) = extractExpSig(y.x)
    
    // ZERO (CASE4)
    // some properties:
    // (+ZERO==-ZERO) => therefore ignore the sign
    //    (ZERO x 10^A == ZERO x 10^B) for any valid A, B =>
    //    therefore ignore the exponent field
    //    (Any non-canonical # is considered 0)
    var x_is_zero = false, y_is_zero = false
    if non_canon_x || sig_x == 0 {
      x_is_zero = true
    }
    if non_canon_y || sig_y == 0 {
      y_is_zero = true
    }
    if x_is_zero && y_is_zero {
      return true
    } else if (x_is_zero && !y_is_zero) || (!x_is_zero && y_is_zero) {
      return false
    }
    // OPPOSITE SIGN (CASE5)
    // now, if the sign bits differ => not equal : return 0
    if ((x.x ^ y.x) & SIGN_MASK) != 0 {
      return false
    }
    // REDUNDANT REPRESENTATIONS (CASE6)
    if exp_x > exp_y {
      // to simplify the loop below,
      swap(&exp_x, &exp_y)  // put the larger exp in y,
      swap(&sig_x, &sig_y)  // and the smaller exp in x
    }
    if exp_y - exp_x > 6 {
      return false    // difference cannot be greater than 10^6
    }
    for _ in 0..<(exp_y - exp_x) {
      // recalculate y's significand upwards
      sig_y = sig_y * 10;
      if (sig_y > MAX_NUMBER) {
        return false
      }
    }
    return sig_y == sig_x
  }
  
  static func lessThan(_ x: Self, _ y: Self, _ pfpsf: inout Status) -> Bool {
    // NaN (CASE1)
    // if either number is NAN, the comparison is unordered : return 0
    var sig_n_prime = UInt128()
    if x.isNaN || y.isNaN {
      if x.isSignalingNaN || y.isSignalingNaN {
        pfpsf.insert(.invalidOperation)    // set exception if sNaN
      }
      return false
    }
    // SIMPLE (CASE2)
    // if all the bits are the same, these numbers are equal.
    if x.x == y.x {
      return false
    }
    // INFINITY (CASE3)
    if x.isInfinite {
      // if x==neg_inf, { res = (y == neg_inf)?0:1; return (res) }
      if x.isSignMinus {
        // x is -inf, so it is less than y unless y is -inf
        return y.isInfinite || !y.isSignMinus
      } else {
        // x is pos_inf, no way for it to be less than y
        return false
      }
    } else if y.isInfinite {
      // x is finite, so:
      //    if y is +inf, x<y
      //    if y is -inf, x>y
      return !y.isSignMinus
    }
    // if steering bits are 11 (condition will be 0), exponent is G[0:w+1] =>
    let (exp_x, sig_x, non_canon_x) = extractExpSig(x.x)
    
    // if steering bits are 11 (condition will be 0), exponent is G[0:w+1] =>
    let (exp_y, sig_y, non_canon_y) = extractExpSig(y.x)
    
    // ZERO (CASE4)
    // some properties:
    // (+ZERO==-ZERO) => therefore ignore the sign, and neither number is greater
    // (ZERO x 10^A == ZERO x 10^B) for any valid A, B =>
    //  therefore ignore the exponent field
    //  (Any non-canonical # is considered 0)
    var x_is_zero = false, y_is_zero = false
    if (non_canon_x || sig_x == 0) {
      x_is_zero = true
    }
    if (non_canon_y || sig_y == 0) {
      y_is_zero = true
    }
    if (x_is_zero && y_is_zero) {
      // if both numbers are zero, they are equal
      return false
    } else if x_is_zero {
      // if x is zero, it is lessthan if Y is positive
      return !y.isSignMinus
    } else if y_is_zero {
      // if y is zero, X is less if it is negative
      return x.isSignMinus
    }
    // OPPOSITE SIGN (CASE5)
    // now, if the sign bits differ, x is less than if y is positive
    if ((x.x ^ y.x) & SIGN_MASK) == SIGN_MASK {
      return !y.isSignMinus
    }
    // REDUNDANT REPRESENTATIONS (CASE6)
    // if both components are either bigger or smaller,
    // it is clear what needs to be done
    if (sig_x > sig_y && exp_x >= exp_y) {
      return x.isSignMinus
    }
    if (sig_x < sig_y && exp_x <= exp_y) {
      return !x.isSignMinus
    }
    // if exp_x is 15 greater than exp_y, no need for compensation
    if (exp_x - exp_y > 15) {
      return x.isSignMinus
      // difference cannot be greater than 10^15
    }
    // if exp_x is 15 less than exp_y, no need for compensation
    if (exp_y - exp_x > 15) {
      return !x.isSignMinus
    }
    // if |exp_x - exp_y| < 15, it comes down to the compensated significand
    if exp_x > exp_y {    // to simplify the loop below,
      // otherwise adjust the x significand upwards
      __mul_64x64_to_128(&sig_n_prime, UInt64(sig_x),
                         bid_ten2k64(exp_x - exp_y))
      // return 0 if values are equal
      if (sig_n_prime.high == 0 && (sig_n_prime.low == sig_y)) {
        return false
      }
      // if postitive, return whichever significand abs is smaller
      // (converse if negative)
      return (((sig_n_prime.high == 0) && sig_n_prime.low < sig_y) !=
              x.isSignMinus)
    }
    // adjust the y significand upwards
    __mul_64x64_to_128(&sig_n_prime, UInt64(sig_y),
                       bid_ten2k64(exp_y - exp_x))
    // return 0 if values are equal
    if (sig_n_prime.high == 0 && (sig_n_prime.low == sig_x)) {
      return false
    }
    // if positive, return whichever significand abs is smaller
    // (converse if negative)
    return (((sig_n_prime.high > 0) ||
             (sig_x < sig_n_prime.low)) != x.isSignMinus)
  }
  
  fileprivate static func extractExpSig(_ x: UInt32) ->
  (exp: Int, sig: UInt32, non_canon: Bool) {
    if (x & STEERING_BITS_MASK) == STEERING_BITS_MASK {
      let exp = Int(x & MASK_BINARY_EXPONENT2) >> 21
      let sig = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
      return (exp, sig, sig > MAX_NUMBER)
    } else {
      let exp = Int(x & MASK_BINARY_EXPONENT1) >> 23
      let sig = x & LARGE_COEFF_MASK
      return (exp, sig, false)
    }
  }
  
  fileprivate static func extractExpSig(_ x: UInt32) ->
  (exp: Int, sig: UInt32) {
    let (exp, sig, nonCanon) = extractExpSig(x)
    if nonCanon { return (0,0) }
    return (exp, sig)
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Math Functions
  /**
   If x is not a floating-point number, the results are unspecified (this
   implementation returns x and *exp = 0). Otherwise, the frexp function
   returns the value res, such that res has a magnitude in the interval
   [1, 10] or zero, and x = res*2^exp. If x is zero, both parts of the
   result are zero. `frexp` does not raise any exceptions
   **/
  static func frexp(_ x: UInt32) -> (res: UInt32, exp: Int) {
    if isInfinite(x) {
      // if NaN or infinity
      let exp = 0
      var res = x
      // the binary frexp quietizes SNaNs, so do the same
      if isSNaN(x) { // x is SNAN
        //   // set invalid flag
        //   *pfpsf |= BID_INVALID_EXCEPTION;
        // return quiet (x)
        res = x & 0xfdffffff
        // } else {
        //   res = x;
      }
      return (res, exp)
    } else {
      // x is 0, non-canonical, normal, or subnormal
      // decode number into exponent and significand
      var exp_x = UInt32(), sig_x = UInt32()
      if isSpecial(x) {
        exp_x = (x & MASK_BINARY_EXPONENT2) >> 21
        sig_x = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
        // check for zero or non-canonical
        if sig_x > MAX_NUMBER || sig_x == 0 {
          // zero of the same sign
          return ((x & SIGN_MASK) | (exp_x << 23), 0)
        }
      } else {
        exp_x = (x & MASK_BINARY_EXPONENT1) >> 23
        sig_x = x & LARGE_COEFF_MASK
        if sig_x == 0 {
          // zero of the same sign
          return ((x & SIGN_MASK) | (exp_x << 23), 0)
        }
      }
      // x is normal or subnormal, with exp_x=biased exponent &
      // sig_x=coefficient
      // determine the number of decimal digits in sig_x, which fits in 24 bits
      // q = nr. of decimal digits in sig_x (1 <= q <= 7)
      //  determine first the nr. of bits in sig_x
      var q = digitsIn(sig_x)
      q-=1  // adjust so result is between 1 and 10
      // Do not add trailing zeros if q < 7; leave sig_x with q digits
      // sig_x = sig_x * bid_mult_factor[7 - q]; // sig_x has now 7 digits
      let exp = Int(exp_x) - EXPONENT_BIAS + q
      let res : UInt32
      // assemble the result
      if sig_x < LARGE_COEFF_HIGH_BIT { // sig_x < 2^23 (fits in 23 bits)
        // replace exponent
        res = UInt32(x & 0x807fffff) | UInt32((-q + EXPONENT_BIAS) << 23)
      } else { // sig_x fits in 24 bits, but not in 23
        // replace exponent
        res = UInt32(x & 0xe01fffff) | UInt32((-q + EXPONENT_BIAS) << 21)
      }
      return (res, exp)
    }
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
   *     get full coefficient_x*coefficient_y product
   *     call subroutine to perform addition of 32-bit argument
   *                                         to 128-bit product
   *
   **************************************************************************/
  static func bid32_fma(_ x:UInt32, _ y:UInt32, _ z:UInt32, _ rmode:Rounding,
                        _ pfpsf:inout Status) -> UInt32 {

    var (sign_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    var (sign_y, exponent_y, coefficient_y, valid_y) = unpack(bid32: y)
    var (sign_z, exponent_z, coefficient_z, valid_z) = unpack(bid32: z)
    
    // unpack arguments, check for NaN, Infinity, or 0
    var res:UInt32
    if !valid_x || !valid_y || !valid_z {
      if isNaN(y) {
        if isInfinite(x) || isInfinite(y) || isInfinite(z) { // sNaN
          pfpsf.insert(.invalidOperation)
        }
        return coefficient_y & QUIET_MASK
      }
      if isNaN(z) {
        if isInfinite(x) || isInfinite(z) {
          pfpsf.insert(.invalidOperation)
        }
        return coefficient_z & QUIET_MASK
      }
      if isNaN(x) {
        if isSNaN(x) {
          pfpsf.insert(.invalidOperation)
        }
        return coefficient_x & QUIET_MASK
      }
      
      
      if !valid_x {
        // x is Inf. or 0
        
        // x is Infinity?
        if isInfinite(x) {
          // check if y is 0
          if coefficient_y == 0 {
            // y==0, return NaN
            if (z & SNAN_MASK) != NAN_MASK {
              pfpsf.insert(.invalidOperation)
            }
            return NAN_MASK
          }
          // test if z is Inf of oposite sign
          if (((z & NAN_MASK) == INFINITY_MASK) && (((x ^ y) ^ z) &
                                                    SIGN_MASK) != 0) {
            // return NaN
            pfpsf.insert(.invalidOperation)
            return NAN_MASK
          }
          // otherwise return +/-Inf
          return ((x ^ y) & SIGN_MASK) | INFINITY_MASK
        }
        // x is 0
        if !isInfinite(y) && !isInfinite(z) {
          
          if coefficient_z != 0 {
            exponent_y = exponent_x - EXPONENT_BIAS + exponent_y
            
            let sign_z = z & SIGN_MASK
            
            if exponent_y >= exponent_z {
              return z
            }
            return UInt32(add_zero32 (exponent_y, sign_z, exponent_z,
                                      coefficient_z, rmode, &pfpsf))
          }
        }
      }
      if !valid_y { // y is Inf. or 0
        // y is Infinity?
        if isInfinite(y) {
          // check if x is 0
          if coefficient_x == 0 {
            // y==0, return NaN
            pfpsf.insert(.invalidOperation)
            return NAN_MASK
          }
          // test if z is Inf of oposite sign
          if (((z & NAN_MASK) == INFINITY_MASK) && (((x ^ y) ^ z)
                                                    & SIGN_MASK) != 0) {
            pfpsf.insert(.invalidOperation)
            // return NaN
            return NAN_MASK
          }
          // otherwise return +/-Inf
          return (((x ^ y) & SIGN_MASK) | INFINITY_MASK)
        }
        // y is 0
        if (z & INFINITY_MASK) != INFINITY_MASK {
          
          if coefficient_z != 0 {
            exponent_y += exponent_x - EXPONENT_BIAS
            
            let sign_z = z & SIGN_MASK
            
            if exponent_y >= exponent_z {
              return z
            }
            return UInt32(add_zero32 (exponent_y, sign_z, exponent_z,
                                      coefficient_z, rmode, &pfpsf))
          }
        }
      }
      
      if !valid_z {
        // y is Inf. or 0
        
        // test if y is NaN/Inf
        if isInfinite(z) {
          return (coefficient_z & QUIET_MASK)
        }
        // z is 0, return x*y
        if (coefficient_x == 0) || (coefficient_y == 0) {
          //0+/-0
          exponent_x += exponent_y - EXPONENT_BIAS;
          if exponent_x > MAX_EXPON {
            exponent_x = MAX_EXPON
          } else if exponent_x < 0 {
            exponent_x = 0
            if exponent_x <= exponent_z {
              res = UInt32(exponent_x) << 23
            } else {
              res = UInt32(exponent_z) << 23
            }
            if (sign_x != sign_y) == sign_z {
              res |= sign_z ? SIGN_MASK : 0
            } else if rmode == BID_ROUNDING_DOWN {
              res |= SIGN_MASK
            }
            return res
          }
          let d2 = exponent_x + exponent_y - EXPONENT_BIAS
          if exponent_z > d2 {
            exponent_z = d2
          }
        }
      }
    }
    
    let P0 = UInt64(coefficient_x) * UInt64(coefficient_y)
    exponent_x += exponent_y - EXPONENT_BIAS;
    
    // sort arguments by exponent
    var sign_a = false, exponent_a = 0, coefficient_a = UInt64()
    var sign_b = false, exponent_b = 0, coefficient_b = UInt64()
    if exponent_x < exponent_z {
      sign_a = sign_z
      exponent_a = exponent_z
      coefficient_a = UInt64(coefficient_z)
      sign_b = sign_x != sign_y
      exponent_b = exponent_x
      coefficient_b = P0
    } else {
      sign_a = sign_x != sign_y
      exponent_a = exponent_x
      coefficient_a = P0
      sign_b = sign_z
      exponent_b = exponent_z
      coefficient_b = UInt64(coefficient_z)
    }
    
    // exponent difference
    var diff_dec_expon = exponent_a - exponent_b
    var inexact = false
    if diff_dec_expon > 17 {
      let tempx = Double(coefficient_a)
      let bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                          BINARY_EXPONENT_BIAS
      let scale_ca = Int(bid_estimate_decimal_digits(bin_expon))
      
      let d2 = 31 - scale_ca
      if diff_dec_expon > d2 {
        diff_dec_expon = d2
        exponent_b = exponent_a - diff_dec_expon
      }
      if coefficient_b != 0 {
        inexact=true
      }
    }
    
    var sign_ab = Int64(sign_a != sign_b ? SIGN_MASK : 0) << 32
    sign_ab = Int64(sign_ab) >> 63
    var CB = UInt128()
    CB = UInt128(high: UInt64(Int64(CB.low) >> 63),
                 low: UInt64((Int64(coefficient_b) + sign_ab) ^ sign_ab))
    
    var Tmp = UInt128(), P = UInt128()
    __mul_64x128_low(&Tmp, coefficient_a, bid_power10_table_128(diff_dec_expon))
    __add_128_128(&P, Tmp, CB)
    if Int64(P.high) < 0 {
      sign_a.toggle()
      var Phigh = 0 - P.high
      if P.low != 0 { Phigh -= 1 }
      P = UInt128(high: Phigh, low:  0 - P.low)
    }
    
    var n_digits = 0
    var bin_expon = 0
    if P.high != 0 {
      let tempx = Double(P.high)
      bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                        BINARY_EXPONENT_BIAS + 64
      n_digits = Int(bid_estimate_decimal_digits(bin_expon))
      if __unsigned_compare_ge_128(P, bid_power10_table_128(n_digits)) {
        n_digits += 1
      }
    } else {
      if P.low != 0 {
        let tempx = Double(P.low)
        bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                          BINARY_EXPONENT_BIAS
        n_digits = Int(bid_estimate_decimal_digits(bin_expon))
        if P.low >= bid_power10_table_128(n_digits).low {
          n_digits += 1
        }
      } else { // result = 0
        sign_a = false
        if rmode == BID_ROUNDING_DOWN { sign_a = true }
        if coefficient_a == 0 { sign_a = sign_x }
        n_digits = 0
      }
    }
    
    if n_digits <= MAX_DIGITS {
      let sign = sign_a ? SIGN_MASK : 0
      return get_BID32_UF(sign, exponent_b, P.low, 0, rmode, &pfpsf)
    }
    
    let extra_digits = n_digits - 7
    
    var rmode1 = roundboundIndex(rmode, sign_a, 0) // rnd_mode;

    if exponent_b+extra_digits < 0 { rmode1=3 }  // RZ
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    var Stemp = UInt128()
    if extra_digits <= 18 {
      __add_128_64(&P, P, bid_round_const_table(rmode1, extra_digits))
    } else {
      __mul_64x64_to_128(&Stemp, bid_round_const_table(rmode1, 18),
                         bid_power10_table_128(extra_digits-18).low)
      __add_128_128 (&P, P, Stemp)
      if rmode == BID_ROUNDING_UP {
        __add_128_64(&P, P, bid_round_const_table(rmode1, extra_digits-18))
      }
    }
    
    // get P*(2^M[extra_digits])/10^extra_digits
    var Q_high = UInt128(), Q_low = UInt128(), C128 = UInt128()
    __mul_128x128_full(&Q_high, &Q_low, P, bid_reciprocals10_128(extra_digits))
    // now get P/10^extra_digits: shift Q_high right by M(extra_digits)-128
    var amount = Int(bid_recip_scale[extra_digits])
    __shr_128_long (&C128, Q_high, amount)
    
    var C64 = C128.low
    var remainder_h, rem_l:UInt64
    if (C64 & 1) != 0 {
      // check whether fractional part of initial_P/10^extra_digits
      // is exactly .5
      // this is the same as fractional part of
      // (initial_P + 0.5*10^extra_digits)/10^extra_digits is exactly zero
      
      // get remainder
      rem_l = Q_high.low
      if amount < 64 {
        remainder_h = Q_high.low << (64 - amount); rem_l = 0
      } else {
        remainder_h = Q_high.high << (128 - amount)
      }
      
      // test whether fractional part is 0
      if ((remainder_h | rem_l) == 0
          && (Q_low.high < bid_reciprocals10_128(extra_digits).high
              || (Q_low.high == bid_reciprocals10_128(extra_digits).high
                  && Q_low.low < bid_reciprocals10_128(extra_digits).low))) {
        C64 -= 1
      }
    }
    
    var status = Status.inexact
    var carry = UInt64(), CY = UInt64()
    
    // get remainder
    rem_l = Q_high.low
    if amount < 64 { remainder_h = Q_high.low << (64 - amount); rem_l = 0 }
    else { remainder_h = Q_high.high << (128 - amount) }
    
    switch rmode {
      case BID_ROUNDING_TO_NEAREST, BID_ROUNDING_TIES_AWAY:
        // test whether fractional part is 0
        if ((remainder_h == 0x8000000000000000 && rem_l == 0)
            && (Q_low.high < bid_reciprocals10_128(extra_digits).high
                || (Q_low.high == bid_reciprocals10_128(extra_digits).high
                    && Q_low.low < bid_reciprocals10_128(extra_digits).low))) {
          status = []
        }
      case BID_ROUNDING_DOWN, BID_ROUNDING_TO_ZERO:
        if ((remainder_h | rem_l) == 0
            && (Q_low.high < bid_reciprocals10_128(extra_digits).high
                || (Q_low.high == bid_reciprocals10_128(extra_digits).high
                    && Q_low.low < bid_reciprocals10_128(extra_digits).low))) {
          status = []
        }
      default:
        // round up
        var low = Stemp.low
        var high = Stemp.high
        __add_carry_out(&low, &CY, Q_low.low,
                        bid_reciprocals10_128(extra_digits).low)
        __add_carry_in_out(&high, &carry, Q_low.high,
                           bid_reciprocals10_128(extra_digits).high, CY)
        Stemp = UInt128(high: high, low: low)
        if amount < 64 {
          if (remainder_h >> (64 - amount)) + carry >= (UInt64(1) << amount) {
            if !inexact {
              status = []
            }
          }
        } else {
          rem_l += carry
          remainder_h >>= (128 - amount)
          if carry != 0 && rem_l == 0 { remainder_h += 1 }
          if remainder_h >= (UInt64(1) << (amount-64)) && !inexact {
            status = []
          }
        }
    }
    
    pfpsf.formUnion(status)
    
    let R = !status.isEmpty ? 1 : 0
    
    if (UInt32(C64) == MAX_NUMBER) && (exponent_b+extra_digits == -1) &&
        (rmode != BID_ROUNDING_TO_ZERO) {
      rmode1 = roundboundIndex(rmode, sign_a, 0)
      //                if (sign_a && (unsigned) (rmode - 1) < 2) {
      //                    rmode = 3 - rmode;
      //                }
      if extra_digits <= 18 {
        __add_128_64 (&P, P, bid_round_const_table(rmode1, extra_digits));
      } else {
        __mul_64x64_to_128(&Stemp, bid_round_const_table(rmode1, 18),
                           bid_power10_table_128(extra_digits-18).low);
        __add_128_128(&P, P, Stemp)
        if rmode == BID_ROUNDING_UP {
          __add_128_64(&P, P, bid_round_const_table(rmode1, extra_digits-18))
        }
      }
      
      // get P*(2^M[extra_digits])/10^extra_digits
      __mul_128x128_full(&Q_high, &Q_low, P,
                         bid_reciprocals10_128(extra_digits))
      // now get P/10^extra_digits: shift Q_high right by M(extra_digits)-128
      amount = Int(bid_recip_scale[extra_digits])
      __shr_128_long(&C128, Q_high, amount);
      
      C64 = C128.low
      if C64 == 10000000 {
        let sign = sign_a ? SIGN_MASK : 0
        return sign | 1000000
      }
    }
    let sign = sign_a ? SIGN_MASK : 0
    return get_BID32_UF(sign, exponent_b+extra_digits, C64, R, rmode, &pfpsf)
  }
  
  //////////////////////////////////////////////////////////////////////////
  //
  //    0*10^ey + cz*10^ez,   ey<ez
  //
  //////////////////////////////////////////////////////////////////////////
  static func add_zero32 (_ exponent_y:Int, _ sign_z:UInt32, _ exponent_z:Int,
                          _ coefficient_z:UInt32, _ rounding_mode:Rounding,
                          _ fpsc:inout Status) -> UInt64 {
      let diff_expon = exponent_z - exponent_y
      var coefficient_z = coefficient_z
      
      let tempx = Double(coefficient_z)
      let bin_expon = Int(((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52)) -
                          BINARY_EXPONENT_BIAS
      var scale_cz = Int(bid_estimate_decimal_digits(bin_expon))
      if coefficient_z >= bid_power10_table_128(scale_cz).low {
          scale_cz+=1
      }
      
      var scale_k = 7 - scale_cz
      if diff_expon < scale_k {
          scale_k = diff_expon
      }
      coefficient_z *= UInt32(bid_power10_table_128(scale_k).low)
      
      return UInt64(bid32(sign_z, exponent_z - scale_k, coefficient_z,
                          rounding_mode, &fpsc))
  }
  
  /***************************************************************************
   *  BID32 nextup
   **************************************************************************/
  static func bid32_nextup (_ x: UInt32, _ pfpsf: inout Status) -> UInt32 {
    var x = x
    var res : UInt32
    
    // check for NaNs and infinities
    if isNaN(x) { // check for NaN
      if (x & 0x000fffff) > 999999 {
        x = x & 0xfe00_0000 // clear G6-G10 and the payload bits
      } else {
        x = x & 0xfe0f_ffff // clear G6-G10
      }
      if isSNaN(x) { // SNaN
        // set invalid flag
        pfpsf.insert(.invalidOperation)
        // pfpsf |= BID_INVALID_EXCEPTION;
        // return quiet (SNaN)
        res = x & 0xfdff_ffff
      } else {    // QNaN
        res = x
      }
      return res
    } else if isInfinite(x) { // check for Infinity
      if (x & SIGN_MASK) == 0 { // x is +inf
        res = INFINITY_MASK
      } else { // x is -inf
        res = 0xf7f8_967f    // -MAXFP = -9999999 * 10^emax
      }
      return res
    }
    // unpack the argument
    let x_sign = x & SIGN_MASK // 0 for positive, SIGN_MASK for negative
    // var x_exp, C1:UInt32
    // if steering bits are 11 (condition will be 0), then exponent is G[0:7]
    var (x_exp, C1) = extractExpSig(x)
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 {
      // x is 0
      res = 0x0000_0001 // MINFP = 1 * 10^emin
    } else { // x is not special and is not zero
      if x == LARGEST_BID {
        // x = +MAXFP = 9999999 * 10^emax
        res = INFINITY_MASK // +inf
      } else if x == 0x8000_0001 {
        // x = -MINFP = 1...99 * 10^emin
        res = SIGN_MASK // -0
      } else {
        // -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
        // can add/subtract 1 ulp to the significand
        
        // Note: we could check here if x >= 10^7 to speed up the case q1 = 7
        // q1 = nr. of decimal digits in x (1 <= q1 <= 7)
        //  determine first the nr. of bits in x
        let q1 = digitsIn(C1)
        
        // if q1 < P7 then pad the significand with zeros
        if q1 < MAX_DIGITS {
          let ind:Int
          if x_exp > (MAX_DIGITS - q1) {
            ind = MAX_DIGITS - q1; // 1 <= ind <= P7 - 1
            // pad with P7 - q1 zeros, until exponent = emin
            // C1 = C1 * 10^ind
            C1 = C1 * UInt32(bid_ten2k64(ind))
            x_exp = x_exp - ind
          } else { // pad with zeros until the exponent reaches emin
            ind = x_exp
            C1 = C1 * UInt32(bid_ten2k64(ind))
            x_exp = MIN_EXPON
          }
        }
        if x_sign == 0 {    // x > 0
          // add 1 ulp (add 1 to the significand)
          C1 += 1
          if C1 == 10_000_000 { // if  C1 = 10^7
            C1 = 1_000_000 // C1 = 10^6
            x_exp += 1
          }
          // Ok, because MAXFP = 9999999 * 10^emax was caught already
        } else {    // x < 0
          // subtract 1 ulp (subtract 1 from the significand)
          C1 -= 1
          if C1 == 999_999 && x_exp != 0 { // if  C1 = 10^6 - 1
            C1 = UInt32(MAX_NUMBER) // C1 = 10^7 - 1
            x_exp -= 1
          }
        }
        // assemble the result
        // if significand has 24 bits
        if (C1 & LARGE_COEFF_HIGH_BIT) != 0 {
          res = x_sign | UInt32(x_exp << 21) | STEERING_BITS_MASK |
          (C1 & SMALL_COEFF_MASK)
        } else {    // significand fits in 23 bits
          res = x_sign | UInt32(x_exp << 23) | C1
        }
      } // end -MAXFP <= x <= -MINFP - 1 ulp OR MINFP <= x <= MAXFP - 1 ulp
    } // end x is not special and is not zero
    return res
  }

  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Conversions
  
  static func int64_to_BID32 (_ value:Int64, _ rnd_mode:Rounding,
                              _ state: inout Status) -> Self {
    // Dealing with 64-bit integer
    let x_sign32 = (value < 0 ? 1 : 0)
    
    // if the integer is negative, use the absolute value
    let C = UInt64(value.magnitude)
    
    var res: UInt32
    if C <= UInt64(MAX_NUMBER) { // |C| <= 10^7-1 and the result is exact
      res = bid32(x_sign32, EXPONENT_BIAS, Int(C))
    } else { // |C| >= 10^7 and the result may be inexact
      // the smallest |C| is 10^7 which has 8 decimal digits
      // the largest |C| is SIGN_MASK64 = 9223372036854775808 w/ 19 digits
      var q, ind : Int
      switch C {
        case 0..<100_000_000:               q =  8; ind = 1
        case  ..<1_000_000_000:             q =  9; ind = 2
        case  ..<10_000_000_000:            q = 10; ind = 3
        case  ..<100_000_000_000:           q = 11; ind = 4
        case  ..<1_000_000_000_000:         q = 12; ind = 5
        case  ..<10_000_000_000_000:        q = 13; ind = 6
        case  ..<100_000_000_000_000:       q = 14; ind = 7
        case  ..<1_000_000_000_000_000:     q = 15; ind = 8
        case  ..<10_000_000_000_000_000:    q = 16; ind = 9
        case  ..<100_000_000_000_000_000:   q = 17; ind = 10
        case  ..<1_000_000_000_000_000_000: q = 18; ind = 11
        default:                            q = 19; ind = 12
      }
      
      // overflow and underflow are not possible
      // Note: performance can be improved by inlining this call
      var is_midpoint_lt_even = false, is_midpoint_gt_even = false
      var is_inexact_lt_midpoint = false
      var is_inexact_gt_midpoint = false, res64 = UInt64(0), incr_exp = 0
      bid_round64_2_18 ( // will work for 19 digits too if C fits in 64 bits
        q, ind, C, &res64, &incr_exp,
        &is_midpoint_lt_even, &is_midpoint_gt_even,
        &is_inexact_lt_midpoint, &is_inexact_gt_midpoint)
      res = UInt32(res64)
      if incr_exp != 0 {
        ind+=1
      }
      // set the inexact flag
      if is_inexact_lt_midpoint || is_inexact_gt_midpoint ||
          is_midpoint_lt_even || is_midpoint_gt_even {
        state.insert(.inexact)
      }
      // general correction from RN to RA, RM, RP, RZ; result uses ind for exp
      if rnd_mode != .toNearestOrAwayFromZero {
        let x_sign = value < 0
        if ((!x_sign && ((rnd_mode == .up && is_inexact_lt_midpoint) ||
                         ((rnd_mode == .toNearestOrEven || rnd_mode == .up) &&
                          is_midpoint_gt_even))) ||
            (x_sign && ((rnd_mode == .down && is_inexact_lt_midpoint) ||
                        ((rnd_mode == .toNearestOrEven || rnd_mode == .down) &&
                         is_midpoint_gt_even)))) {
          res = res + 1
          if res == MAX_NUMBERP1 { // res = 10^7 => rounding overflow
            res = 1_000_000 // 10^6
            ind = ind + 1
          }
        } else if ((is_midpoint_lt_even || is_inexact_gt_midpoint) &&
                ((x_sign && (rnd_mode == .towardZero || rnd_mode == .down)) ||
                (!x_sign && (rnd_mode == .towardZero || rnd_mode == .down)))) {
          res = res - 1
          // check if we crossed into the lower decade
          if res == 999_999 { // 10^6 - 1
            res = UInt32(MAX_NUMBER)  // 10^7 - 1
            ind = ind - 1
          }
        } else {
          // exact, the result is already correct
        }
      }
      res = bid32(x_sign32, ind, Int(res))
    }
    return Self(raw: res)
  }
  
  static func double_to_bid32 (_ x:Double, _ rnd_mode:Rounding,
                               _ state: inout Status) -> UInt32 {
    // Unpack the input
    var s = 0, e = 0, t = 0
    var low = UInt64(), high = UInt64()
    if let res = unpack_binary64(x, &s, &e, &low, &t, &state) {
      return UInt32(res)
    }
    
    // Now -1126<=e<=971 (971 for max normal, -1074 for min normal,
    // -1126 for min denormal)
    
    // Treat like a quad input for uniformity, so (2^{113-53} * c * r) >> 320,
    // where 320 is the truncation value for the reciprocal multiples, exactly
    // five 64-bit words. So we shift 113-53=60 places
    //
    // Remember to compensate for the fact that exponents are integer for quad
    let c = sll128_short(high, low, 60)
    t += (113 - 53)
    e -= (113 - 53) // Now e belongs [-1186;911].
    
    // Check for "trivial" overflow, when 2^e * 2^112 > 10^emax * 10^d.
    // We actually check if e >= ceil((emax + d) * log_2(10) - 112)
    // This could be intercepted later, but it's convenient to keep tables
    // smaller
    if e >= 211 {
      state.formUnion([.overflow, .inexact])
      return bid32_ovf(s)
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
      var cint:UInt128
      let a = -(e + t)
      cint = c
      if a <= 0 {
        cint = srl128(cint.high, cint.low, -e)
        if ((cint.high == 0) && (cint.low < MAX_NUMBERP1)) {
          return bid32(s, EXPONENT_BIAS, Int(cint.low))
        }
      } else if a <= 48 {
        var pow5 = bid_coefflimits_bid32(a)
        cint = srl128(cint.high, cint.low, t)
        if le128(cint.high, cint.low, pow5.high, pow5.low) {
          var cc = cint
          pow5 = bid_power_five(a)
          __mul_128x128_low(&cc, cc, pow5)
          return bid32(s, EXPONENT_BIAS - a, Int(cc.low))
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
    var e_out = Tables.bid_exponents_bid32[e+450]
    
    // Choose exponent and reciprocal multiplier based on breakpoint
    var r:UInt256
    if le128(c.high, c.low, m_min.high, m_min.low) {
      r = Tables.bid_multipliers1_bid32[e+450]
    } else {
      r = Tables.bid_multipliers2_bid32[e+450]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    __mul_128x256_to_384(&z, c, r)
    var c_prov = z.w[5]
    
    // Test inexactness and underflow (when testing tininess before rounding)
    if ((z.w[4] != 0) || (z.w[3] != 0)) {
      state.insert(.inexact)
      if (c_prov < 1000000) {
        state.insert(.underflow)
      }
    }
    
    // Round using round-sticky words
    // If we spill over into the next decade, correct
    // Flag underflow where it may be needed even for |result| = SNN
    let ind = roundboundIndex(rnd_mode, s == 1, Int(c_prov))
    if lt128(bid_roundbound_128[ind].high, bid_roundbound_128[ind].low,
             z.w[4], z.w[3]) {
      c_prov += 1
      if c_prov == MAX_NUMBERP1 {
        c_prov = 1_000_000
        e_out += 1
      } else if c_prov == 1_000_000 && e_out == 0 {
        let ind = roundboundIndex(rnd_mode, false, 0) >> 2
        if ((((ind & 3) == 0) && (z.w[4] <= 17524406870024074035)) ||
            ((ind + (Int(s) & 1) == 2) && (z.w[4] <= 16602069666338596454))) {
          state.insert(.underflow)
        }
      }
    }
    
    // Check for overflow
    if e_out > 90 + EXPONENT_BIAS {
      state.formUnion([.overflow, .inexact])
      return bid32_ovf(s)
    }
    
    // Set the inexact flag as appropriate and check underflow
    // It's no doubt superfluous to check inexactness, but anyway...
    if z.w[4] != 0 || z.w[3] != 0 {
      state.insert(.inexact)
      if c_prov < 1_000_000 {
        state.insert(.underflow)
      }
    }
    
    // Package up the result
    return bid32(s, Int(e_out), Int(c_prov))
  }
  
  // 128x256->384 bit multiplication (missing from existing macros)
  // I derived this by propagating (A).w[2] = 0 in __mul_192x256_to_448
  static func __mul_128x256_to_384(_  P: inout UInt384, _ A:UInt128,
                                   _ B:UInt256) {
    var P0=UInt384(),P1=UInt384()
    var CY:UInt64=0
    __mul_64x256_to_320(&P0, A.low, B);
    __mul_64x256_to_320(&P1, A.high, B);
    P.w[0] = P0.w[0];
    __add_carry_out(&P.w[1],&CY,P1.w[0],P0.w[1]);
    __add_carry_in_out(&P.w[2],&CY,P1.w[1],P0.w[2],CY);
    __add_carry_in_out(&P.w[3],&CY,P1.w[2],P0.w[3],CY);
    __add_carry_in_out(&P.w[4],&CY,P1.w[3],P0.w[4],CY);
    P.w[5] = P1.w[4] + CY;
  }
  
  static func __mul_64x256_to_320(_ P:inout UInt384, _ A:UInt64, _ B:UInt256) {
    var lP0=UInt128(), lP1=UInt128(), lP2=UInt128(), lP3=UInt128()
    var lC:UInt64=0
    __mul_64x64_to_128(&lP0, A, B.w[0])
    __mul_64x64_to_128(&lP1, A, B.w[1])
    __mul_64x64_to_128(&lP2, A, B.w[2])
    __mul_64x64_to_128(&lP3, A, B.w[3])
    P.w[0] = lP0.low
    __add_carry_out(&P.w[1],&lC,lP1.low,lP0.high)
    __add_carry_in_out(&P.w[2],&lC,lP2.low,lP1.high,lC)
    __add_carry_in_out(&P.w[3],&lC,lP3.low,lP2.high,lC)
    P.w[4] = lP3.high + lC
  }
  
  static func __mul_128x128_low(_ Ql: inout UInt128, _ A:UInt128,
                                _ B:UInt128) {
    var ALBL = UInt128()
    __mul_64x64_to_128(&ALBL, A.low, B.low)
    let QM64 = B.low*A.high + A.low*B.high
    Ql = UInt128(high: QM64 + ALBL.high, low: ALBL.low)
  }
  
  static func __mul_128x64_to_128(_ Q128: inout UInt128, _ A64:UInt64,
                                  _ B128:UInt128) {
    let ALBH_L = A64 * B128.high
    __mul_64x64_to_128(&Q128, A64, B128.low)
    Q128 = UInt128(high: Q128.high+ALBH_L, low: Q128.low)
  }
  
  static func __mul_64x128_low(_ Ql:inout UInt128, _ A:UInt64, _ B:UInt128) {
    var ALBL = UInt128(), ALBH = UInt128(), QM2 = UInt128()
    __mul_64x64_to_128(&ALBH, A, B.high)
    __mul_64x64_to_128(&ALBL, A, B.low)
    __add_128_64(&QM2, ALBH, ALBL.high)
    Ql = UInt128(high: QM2.low, low: ALBL.low)
  }
  
  @inlinable static func __add_carry_in_out(_ S: inout UInt64,
                                            _ CY: inout UInt64, _ X:UInt64,
                                            _ Y:UInt64, _ CI: UInt64) {
    let X1 = X + CI
    S = X1 &+ Y
    CY = ((S<X1) || (X1<CI)) ? 1 : 0
  }
  
  // add 128-bit value to 128-bit
  // assume no carry-out
  static func __add_128_128(_ R128:inout UInt128, _ A128:UInt128,
                            _ B128:UInt128) {
    var Q128hi = A128.high + B128.high
    let Q128lo = B128.low &+ A128.low
    if Q128lo < B128.low {
      Q128hi += 1
    }
    R128 = UInt128(high: Q128hi, low: Q128lo)
  }
  
  // add 64-bit value to 128-bit
  static func __add_128_64(_ R128:inout UInt128, _ A128:UInt128,
                           _ B64:UInt64) {
    var R64H = A128.high
    let R128low = B64 &+ A128.low
    if R128low < B64 {
      R64H += 1
    }
    R128 = UInt128(high: R64H, low: R128low)
  }
  
  static func srl128(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
    if c == 0 { return UInt128(w: [lo, hi]) }
    if c >= 64 { return UInt128(w: [hi >> (c - 64), 0]) }
    else { return srl128_short(hi, lo, c) }
  }
  
  // Shift 2-part 2^64 * hi + lo right by "c" bits
  // The "short" form requires a shift 0 < c < 64 and will be faster
  // Note that shifts of 64 can't be relied on as ANSI
  @inlinable static func srl128_short(_ hi:UInt64, _ lo:UInt64, _ c:Int) ->
                                      UInt128 {
    UInt128(high: hi >> c, low: (hi << (64 - c)) + (lo >> c))
  }
  
  @inlinable static func __shr_128_long(_ Q:inout UInt128, _ A:UInt128,
                                        _ k:Int) {
    Q = A >> k
  }
  
  // Shift 2-part 2^64 * hi + lo left by "c" bits
  // The "short" form requires a shift 0 < c < 64 and will be faster
  // Note that shifts of 64 can't be relied on as ANSI
  
  static func sll128_short(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
    UInt128(w: [lo << c, (hi << c) + (lo>>(64-c))])
  }
  
  // Shift 4-part 2^196 * x3 + 2^128 * x2 + 2^64 * x1 + x0
  // right by "c" bits (must have c < 64)
  static func srl256_short(_ x3: inout UInt64, _ x2: inout UInt64,
                           _ x1: inout UInt64, _ x0: inout UInt64,
                           _ c:Int) {
      x0 = (x1 << (64 - c)) + (x0 >> c)
      x1 = (x2 << (64 - c)) + (x1 >> c)
      x2 = (x3 << (64 - c)) + (x2 >> c)
      x3 = x3 >> c
  }
  
  // Compare "<" two 2-part unsigned integers
  @inlinable static func lt128(_ x_hi:UInt64, _ x_lo:UInt64,
                               _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (x_hi < y_hi) || ((x_hi == y_hi) && (x_lo < y_lo))
  }
  
  // Likewise "<="
  @inlinable static func le128(_ x_hi:UInt64, _ x_lo:UInt64,
                               _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (x_hi < y_hi) || ((x_hi == y_hi) && (x_lo <= y_lo))
  }
  
  @inlinable static func __unsigned_compare_ge_128(_ A:UInt128,
                                                   _ B:UInt128) -> Bool {
      A >= B
  }
  
  static func bid32_ovf(_ s:Int) -> UInt32 {
    let rnd_mode = Self.rounding
    if ((rnd_mode == BID_ROUNDING_TO_ZERO) ||
        (rnd_mode==(s != 0 ? BID_ROUNDING_UP : BID_ROUNDING_DOWN))) {
      return bid32_max(s)
    } else {
      return bid32_inf(s)
    }
  }
  
  static func bid32_max(_ s:Int) -> UInt32 { bid32(s,MAX_EXPON,MAX_NUMBER) }
  static func bid32_inf(_ s:Int) -> UInt32 { bid32(s,0xF<<4,0) }
  static func bid32_zero(_ s:Int) -> UInt32 { bid32(s,EXPONENT_BIAS,0) }
  static func bid32_nan(_ s:Int, _ c_hi:UInt64, _ c_lo:UInt64) -> UInt32 {
    bid32(s, 0x1F<<3, c_hi>>44 > 999_999 ? 0 : Int(c_hi>>44))
  }
  
  static func unpack_binary64(
    _ x:Double, _ s: inout Int, _ e: inout Int,_ c: inout UInt64,
    _ t: inout Int, _ status: inout Status) -> UInt32? {
    let expMask = 1<<11 - 1
    e = Int(x.bitPattern >> 52) & expMask
    c = x.significandBitPattern
    s = x.sign == .minus ? 1 : 0
    if e == 0 {
      if c == 0 { return bid32_zero(s) } // number = 0
      
      // denormalized number
      let l = clz(c) - (64 - 53)
      c = c << l
      e = -(l + 1074)
      t = 0
      status.insert(.subnormal)
    } else if e == expMask {
      if c == 0 { return bid32_inf(s) } // number = infinity
      status.insert(.invalidOperation)
      return bid32_nan(s, c << 13, 0)
    } else {
      c |= 1 << 52  // set upper bit
      e -= 1075
      t = ctz64(c)
    }
    return nil
  }
  
  // Counting trailing zeros in an unsigned 64-bit word
  @inlinable static func ctz64(_ n:UInt64) -> Int { n.trailingZeroBitCount }
  
  // Counting leading zeros in an unsigned 64-bit word
  @inlinable static func clz<T:FixedWidthInteger>(_ n:T) -> Int {
    n.leadingZeroBitCount
  }

  static func bid32ToDouble (_ x: UInt32, _ rmode: Rounding,
                               _ pfpsf: inout Status) -> Double {
    let (s, e, coeff, high, value) = unpack(bid32:x, &pfpsf)
    if let value = value { return value }
    
    // Correct to 2^112 <= c < 2^113 with corresponding exponent adding 113-24=89
    // In fact shift a further 6 places ready for reciprocal multiplication
    // Thus (113-24)+6=95, a shift of 31 given that we've already upacked in c.high
    let c = UInt128(high: high << 31, low: 0)
    let k = coeff + 89
    
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
    var e_out = bid_exponents_binary64[e+358] - Int(k)
    
    // Choose provisional exponent and reciprocal multiplier based on breakpoint
    var r = UInt256()
    if c.high < m_min.high {
      r = Tables.bid_multipliers1_binary64[e+358]
    } else {
      r = Tables.bid_multipliers2_binary64[e+358]
      e_out = e_out + 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    __mul_64x256_to_320(&z, c.high, r)
//    z.w[1...5] = z.w[0...4]; z.w[0] = 0
    z.w[5]=z.w[4]; z.w[4]=z.w[3]; z.w[3]=z.w[2]; z.w[2]=z.w[1]
    z.w[1]=z.w[0]; z.w[0]=0
    
    // Check for exponent underflow and compensate by shifting the product
    // Cut off the process at precision+2, since we can't really shift further
    
    var c_prov = Int(z.w[5])
    
    // Round using round-sticky words
    // If we spill into the next binade, correct
    let rind = roundboundIndex(rmode, s != 0, c_prov)
    if (lt128(bid_roundbound_128[rind].high,
              bid_roundbound_128[rind].low, z.w[4], z.w[3])) {
      c_prov = c_prov + 1;
    }
    c_prov = c_prov & ((1 << 52) - 1);
    
    // Set the inexact and underflow flag as appropriate
    if (z.w[4] != 0) || (z.w[3] != 0) {
      pfpsf.insert(.inexact)
    }
    
    // Package up the result as a binary floating-point number
    return double(s, e_out, UInt64(c_prov))
  }
  
  static func binary32_ovf(_ s:Int) -> Float {
    if ((Self.rounding==BID_ROUNDING_TO_ZERO) ||
        (Self.rounding==((s != 0) ? BID_ROUNDING_UP : BID_ROUNDING_DOWN))) {
      return s != 0 ? -Float.greatestFiniteMagnitude :
                      Float.greatestFiniteMagnitude
    } else {
      return s != 0 ? -Float.infinity : Float.infinity
    }
  }
  
  static func bid32ToFloat (_ x: UInt32, _ rmode: Rounding,
                            _ pfpsf: inout Status) -> Float {
    var (s, e, coeff, high, value) = unpack(bid32:x, &pfpsf)
    if let value = value { return Float(value) }
    
    // Correct to 2^112 <= c < 2^113 with corresponding exponent adding 113-24=89
    // Thus a shift of 25 given that we've already upacked in c.high
    let c = UInt128(high: high << 25, low: 0)
    let k = coeff + 89
    
    // Check for "trivial" overflow, when 10^e * 1 > 2^{sci_emax+1}, just to
    // keep tables smaller (it would be intercepted later otherwise).
    //
    // (Note that we may have normalized the coefficient, but we have a
    //  corresponding exponent postcorrection to account for; this can
    //  afford to be conservative anyway.)
    //
    // We actually check if e >= ceil((sci_emax + 1) * log_10(2))
    // which in this case is e >= ceil(128 * log_10(2)) = 39
    if e >= 39 {
      pfpsf.formUnion([.overflow, .inexact])
      return binary32_ovf(s)
    }
    // Also check for "trivial" underflow, when 10^e * 2^113 <= 2^emin * 1/4,
    // so test e <= floor((emin - 115) * log_10(2))
    // In this case just fix ourselves at that value for uniformity.
    //
    // This is important not only to keep the tables small but to maintain the
    // testing of the round/sticky words as a correct rounding method
    if e <= -80 {
      e = -80
    }
    
    // Look up the breakpoint and approximate exponent
    let m_min = Tables.bid_breakpoints_binary32[e+80];
    var e_out = Tables.bid_exponents_binary32[e+80] - k;
    
    // Choose provisional exponent and reciprocal multiplier based on breakpoint
    var r = UInt256()
    let rn: UInt128
    if c.high <= m_min.high {
      rn = Tables.bid_multipliers1_binary32[e+80]
    } else {
      rn = Tables.bid_multipliers2_binary32[e+80]
      e_out = e_out + 1
    }
//    print("Index = \(e+80) -> " + String(rn, radix: 16))
    r.w[0] = rn.low; r.w[1] = rn.high
    
    // Do the reciprocal multiplication
    var z = UInt384()
    __mul_64x256_to_320(&z, c.high, r)
    z.w[5]=z.w[4]; z.w[4]=z.w[3]; z.w[3]=z.w[2]; z.w[2]=z.w[1]; z.w[1]=z.w[0]
    z.w[0]=0
    
    // Check for exponent underflow and compensate by shifting the product
    // Cut off the process at precision+2, since we can't really shift further
    if e_out < 1 {
      var d = 1 - e_out
      if d > 26 {
        d = 26
      }
      e_out = 1
      var zw5 = z.w[5], zw4 = z.w[4], zw3 = z.w[3], zw2 = z.w[2]
      srl256_short(&zw5, &zw4, &zw3, &zw2, d)
      z.w[5] = zw5; z.w[4] = zw4; z.w[3] = zw3; z.w[2] = zw2
    }
    var c_prov = z.w[5]
    
    // Round using round-sticky words
    // If we spill into the next binade, correct
    // Flag underflow where it may be needed even for |result| = SNN
    let ind = roundboundIndex(rmode, s != 0, Int(c_prov))
    if lt128(bid_roundbound_128[ind].high,
             bid_roundbound_128[ind].low, z.w[4], z.w[3]) {
      c_prov = c_prov + 1
      if c_prov == (1 << 24) {
        c_prov = 1 << 23
        e_out = e_out + 1
      }
    }
    
    // Check for overflow
    if e_out >= 255 {
      pfpsf.insert(.overflow)
      return binary32_ovf(s)
    }
    
    // Modify exponent for a tiny result, otherwise lop the implicit bi
    if c_prov < (1 << 23) {
      e_out = 0
    } else {
      c_prov = c_prov & ((1 << 23) - 1)
    }
    
    // Set the inexact and underflow flag as appropriate (tiny after rounding)
    if (z.w[4] != 0) || (z.w[3] != 0) {
      pfpsf.insert(.inexact)
      if e_out == 0 {
        pfpsf.insert(.underflow)
      }
    }
    
    // Package up the result as a binary floating-point number
    return binary32(s, e_out, c_prov)
  }
  
  static func binary32(_ s: Int, _ e:Int, _ c:UInt64) -> Float {
    Float(bitPattern: (UInt32(s) << 31) + (UInt32(e) << 23) + UInt32(c))
  }
  
  /****************************************************************************
   *  BID32_to_uint64_int
   ***************************************************************************/
  static func bid32ToUInt (_ x: UInt32, _ rmode:Rounding,
                           _ pfpsc: inout Status) -> UInt {
    // check for NaN or Infinity
    if isNaN(x) || isInfinite(x) {
      // set invalid flag
      pfpsc.insert(.invalidOperation)
      
      // return Integer Indefinite
      return UInt(bitPattern: Int.min)
    }
    
    // unpack x
    let x_sign = x & SIGN_MASK // 0 for positive, MASK_SIGN32 for negative
    
    // if steering bits are 11 (condition will be 0), exponent is G[0:w+1] =>
    var x_exp, C1: UInt32
    if isSpecial(x) {
      x_exp = (x & MASK_BINARY_EXPONENT2) >> 21 // biased
      C1 = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
      if C1 > MAX_NUMBER {
        // non-canonical
        x_exp = 0
        C1 = 0
      }
    } else {
      x_exp = (x & MASK_BINARY_EXPONENT1) >> 23; // biased
      C1 = x & LARGE_COEFF_MASK
    }
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0x0 {
      // x is 0
      return 0x0
    }
    // x is not special and is not zero
    
    // q = nr. of decimal digits in x (1 <= q <= 7)
    //  determine first the nr. of bits in x
    let q = digitsIn(C1)
    let exp = Int(x_exp) - EXPONENT_BIAS // unbiased exponent
    
    if (q + exp) > 20 { // x >= 10^20 ~= 2^66.45... (cannot fit in 64 bits)
      // set invalid flag
      pfpsc.insert(.invalidOperation)
      
      // return Integer Indefinite
      return UInt(bitPattern: Int.min)
    } else if (q + exp) == 20 { // x = c(0)c(1)...c(q-1)00...0 (20 dec. digits)
      // in this case 2^63.11... ~= 10^19 <= x < 10^20 ~= 2^66.43...
      // so x rounded to an integer may or may not fit in an unsigned 64-bit int
      // the cases that do not fit are identified here; the ones that fit
      // fall through and will be handled with other cases further,
      // under '1 <= q + exp <= 20'
      if x_sign != 0 { // if n < 0 and q + exp = 20 then x is much less than -1
        // set invalid flag
        pfpsc.insert(.invalidOperation)
        
        // return Integer Indefinite
        return UInt(bitPattern: Int.min)
      } else { // if n > 0 and q + exp = 20
        // if n >= 2^64 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[20 dec. digits] >= 2^64
        // <=> 0.c(0)c(1)...c(q-1) * 10^21 >= 5*(2^65)
        // <=> C * 10^(21-q) >= 0xa0000000000000000, 1<=q<=7
        var C = UInt128()
        if q == 1 {
          // C * 10^20 >= 0xa0000000000000000
          let Ten20 = UInt128(high:0x0000000000000005, low:0x6bc75e2d63100000)
          __mul_128x64_to_128(&C, UInt64(C1), Ten20) // 10^20 * C
          if C.high >= 0x0a {
            // actually C.w[1] == 0x0a && C.w[0] >= 0x0000000000000000ull) {
            // set invalid flag
            pfpsc.insert(.invalidOperation)
            
            // return Integer Indefinite
            return UInt(bitPattern: Int.min)
          }
          // else cases that can be rounded to a 64-bit int fall through
          // to '1 <= q + exp <= 20'
        } else { // if (2 <= q <= 7) => 14 <= 21 - q <= 19
          // Note: C * 10^(21-q) has 20 or 21 digits; 0xa0000000000000000
          // has 21 digits
          __mul_64x64_to_128(&C, UInt64(C1), bid_ten2k64(21 - q))
          if C.high >= 0x0a {
            // actually C.w[1] == 0x0a && C.w[0] >= 0x0000000000000000ull) {
            // set invalid flag
            pfpsc.insert(.invalidOperation)
            
            // return Integer Indefinite
            return UInt(bitPattern: Int.min)
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
      return 0x0
    } else { // if (1 <= q + exp <= 20, 1 <= q <= 7, -6 <= exp <= 19)
      // x <= -1 or 1 <= x < 2^64 so if positive x can be rounded
      // to nearest to a 64-bit unsigned signed integer
      if x_sign != 0 { // x <= -1
        // set invalid flag
        pfpsc.insert(.invalidOperation)
        
        // return Integer Indefinite
        return UInt(bitPattern: Int.min)
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
        // kx = 10^(-x) = bid_ten2mk64(ind - 1)
        // C* = C1 * 10^(-x)
        // the approximation of 10^(-x) was rounded up to 54 bits
        var P128 = UInt128()
        __mul_64x64_to_128(&P128, UInt64(C1), bid_ten2mk64(ind - 1))
        var Cstar = P128.high
        
        // the top Ex bits of 10^(-x) are T* = bid_ten2mk128trunc[ind].w[0],
        // e.g.if x=1, T*=bid_ten2mk128trunc[0].w[0]=0x1999999999999999
        // C* = floor(C*) (logical right shift; C has p decimal digits,
        //     correct by Property 1)
        // n = C* * 10^(e+x)
        
        // shift right C* by Ex-64 = bid_shiftright128[ind]
        let shift = bid_shiftright128[ind - 1] // 0 <= shift <= 39
        Cstar = Cstar >> shift
        res = Cstar // the result is positive
      } else if exp == 0 {
        // 1 <= q <= 10
        // res = +C (exact)
        res = UInt64(C1) // the result is positive
      } else { // if (exp > 0) => 1 <= exp <= 9, 1 <= q < 9, 2 <= q + exp <= 10
        // res = +C * 10^exp (exact)
        res = UInt64(C1) * bid_ten2k64(exp) // the result is positive
      }
    }
    return UInt(res)
  }
  
  /****************************************************************************
   *  BID32_to_int64_int
   ***************************************************************************/
  static func bid32ToInt(_ x: Word, _ rmode:Rounding,
                         _ pfpsc: inout Status) -> Int {
    var res: Int = 0
    
    // check for NaN or Infinity and unpack `x`
    let (x_negative, x_exp, C1, valid) = unpack(bid32: x)
    if !valid {
      if C1 == 0 { return 0 }
      pfpsc.insert(.invalidOperation); return Int.min
    }
    
    // check for zeros (possibly from non-canonical values)
    if C1 == 0 {
      // x is 0
      return 0
    }
    // x is not special and is not zero
    
    // q = nr. of decimal digits in x (1 <= q <= 7)
    //  determine first the nr. of bits in x
    let q = digitsIn(C1)
    let exp = Int(x_exp) - EXPONENT_BIAS // unbiased exponent
    
    if (q + exp) > 19 { // x >= 10^19 ~= 2^63.11... (cannot fit in BID_SINT64)
      // set invalid flag
      pfpsc.insert(.invalidOperation)
      // return Integer Indefinite
      return Int.min
    } else if (q + exp) == 19 { // x = c(0)c(1)...c(q-1)00...0 (19 dec. digits)
      // in this case 2^63.11... ~= 10^19 <= x < 10^20 ~= 2^66.43...
      // so x rounded to an integer may or may not fit in a signed 64-bit int
      // the cases that do not fit are identified here; the ones that fit
      // fall through and will be handled with other cases further,
      // under '1 <= q + exp <= 19'
      var C = UInt128(0)
      if x_negative { // if n < 0 and q + exp = 19
        // if n <= -2^63 - 1 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[19 dec. digits] >= 2^63+1
        // <=> 0.c(0)c(1)...c(q-1) * 10^20 >= 0x5000000000000000a, 1<=q<=7
        // <=> C * 10^(20-q) >= 0x5000000000000000a, 1<=q<=7
        // 1 <= q <= 7 => 13 <= 20-q <= 19 => 10^(20-q) is 64-bit, and so is C1
        
        __mul_64x64_to_128(&C, UInt64(C1), bid_ten2k64(20 - q));
        // Note: C1 * 10^(11-q) has 19 or 20 digits; 0x5000000000000000a,has 20
        if (C.high > 0x05 || (C.high == 0x05 && C.low >= 0x0a)) {
          // set invalid flag
          pfpsc.insert(.invalidOperation)
          // return Integer Indefinite
          return Int.min
        }
        // else cases that can be rounded to a 64-bit int fall through
        // to '1 <= q + exp <= 19'
      } else { // if n > 0 and q + exp = 19
        // if n >= 2^63 then n is too large
        // <=> c(0)c(1)...c(q-1)00...0[19 dec. digits] >= 2^63
        // <=> if 0.c(0)c(1)...c(q-1) * 10^20 >= 0x50000000000000000, 1<=q<=7
        // <=> if C * 10^(20-q) >= 0x50000000000000000, 1<=q<=7
        C = UInt128(high: 5, low: 0)
        // 1 <= q <= 7 => 13 <= 20-q <= 19 => 10^(20-q) is 64-bit, and so is C1
        __mul_64x64_to_128(&C, UInt64(C1), bid_ten2k64(20 - q))
        if C.high >= 0x05 {
          // actually C.high == 0x05 && C.low >= 0x0000000000000000) {
          // set invalid flag
          pfpsc.insert(.invalidOperation)
          // return Integer Indefinite
          return Int.min
        }
        // else cases that can be rounded to a 64-bit int fall through
        // to '1 <= q + exp <= 19'
      }    // end else if n > 0 and q + exp = 19
    }    // end else if ((q + exp) == 19)
    
    // n is not too large to be converted to int64: -2^63-1 < n < 2^63
    // Note: some of the cases tested for above fall through to this point
    if (q + exp) <= 0 { // n = +/-0.0...c(0)c(1)...c(q-1)
      // return 0
      return 0x0000000000000000
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
        // kx = 10^(-x) = bid_ten2mk64(ind - 1)
        // C* = C1 * 10^(-x)
        // the approximation of 10^(-x) was rounded up to 54 bits
        var P128 = UInt128()
        __mul_64x64_to_128(&P128, UInt64(C1), bid_ten2mk64(ind - 1))
        var Cstar = P128.high
        // the top Ex bits of 10^(-x) are T* = bid_ten2mk128trunc[ind].lo, e.g.
        // if x=1, T*=bid_ten2mk128trunc[0].lo=0x1999999999999999
        // C* = floor(C*) (logical right shift; C has p decimal digits,
        //     correct by Property 1)
        // n = C* * 10^(e+x)
        
        // shift right C* by Ex-64 = bid_shiftright128[ind]
        let shift = bid_shiftright128[ind - 1] // 0 <= shift <= 39
        Cstar = Cstar >> shift
        
        if x_negative {
          res = -Int(Cstar)
        } else {
          res = Int(Cstar)
        }
      } else if exp == 0 {
        // 1 <= q <= 7
        // res = +/-C (exact)
        if x_negative {
          res = -Int(C1)
        } else {
          res = Int(C1)
        }
      } else {
        // if (exp > 0) => 1 <= exp <= 18, 1 <= q <= 7, 2 <= q + exp <= 20
        // (the upper limit of 20 on q + exp is due to the fact that
        // +/-C * 10^exp is guaranteed to fit in 64 bits)
        // res = +/-C * 10^exp (exact)
        if x_negative {
          res = -Int(UInt64(C1) * bid_ten2k64(exp))
        } else {
          res = Int(UInt64(C1) * bid_ten2k64(exp))
        }
      }
    }
    return res
  }
  
  // 64x64-bit product
  static func __mul_64x64_to_128(_ P128: inout UInt128, _ CX:UInt64,
                                 _ CY:UInt64) {
    let r = CX.multipliedFullWidth(by: CY)
    P128 = UInt128(high: r.high, low: r.low)
  }
  
  static func digitsIn<T:BinaryInteger>(_ sig_x: T) -> Int {
    // find power of 10 just greater than sig_x
    var tenPower = T(10), digits = 1
    while sig_x >= tenPower { tenPower *= 10; digits += 1 }
    return digits
  }
  
  static func __mul_128x128_full(_ Qh:inout UInt128, _ Ql:inout UInt128,
                                 _ A:UInt128, _ B:UInt128) {
    var ALBL = UInt128(), ALBH = UInt128(), AHBL = UInt128(), AHBH = UInt128()
    
    __mul_64x64_to_128(&ALBH, A.low, B.high)
    __mul_64x64_to_128(&AHBL, B.low, A.high)
    __mul_64x64_to_128(&ALBL, A.low, B.low)
    __mul_64x64_to_128(&AHBH, A.high, B.high)
    
    var QM = UInt128(), QM2 = UInt128()
    __add_128_128(&QM, ALBH, AHBL)
    __add_128_64(&QM2, QM, ALBL.high)
    __add_128_64(&Qh, AHBH, QM2.high)
    Ql = UInt128(high: QM2.low, low: ALBL.low)
  }
  
  // bid_shiftright128[] contains the right shift count to obtain C2* from
  // the top 128 bits of the 128x128-bit product C2 * Kx
  static let bid_shiftright128: [UInt8] = [
    0, 0, 0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56,
    59, 63, 66, 69, 73, 76, 79, 83, 86, 89, 92, 96, 99, 102
  ]
  
//  static let bid_power10_index_binexp(_ i:Int) -> UInt64 -> {
//
//  }
  
  static let bid_power10_index_binexp: [UInt64] = [
    0x000000000000000a, 0x000000000000000a, 0x000000000000000a,
    0x000000000000000a, 0x0000000000000064, 0x0000000000000064,
    0x0000000000000064, 0x00000000000003e8, 0x00000000000003e8,
    0x00000000000003e8, 0x0000000000002710, 0x0000000000002710,
    0x0000000000002710, 0x0000000000002710, 0x00000000000186a0,
    0x00000000000186a0, 0x00000000000186a0, 0x00000000000f4240,
    0x00000000000f4240, 0x00000000000f4240, 0x0000000000989680,
    0x0000000000989680, 0x0000000000989680, 0x0000000000989680,
    0x0000000005f5e100, 0x0000000005f5e100, 0x0000000005f5e100,
    0x000000003b9aca00, 0x000000003b9aca00, 0x000000003b9aca00,
    0x00000002540be400, 0x00000002540be400, 0x00000002540be400,
    0x00000002540be400, 0x000000174876e800, 0x000000174876e800,
    0x000000174876e800, 0x000000e8d4a51000, 0x000000e8d4a51000,
    0x000000e8d4a51000, 0x000009184e72a000, 0x000009184e72a000,
    0x000009184e72a000, 0x000009184e72a000, 0x00005af3107a4000,
    0x00005af3107a4000, 0x00005af3107a4000, 0x00038d7ea4c68000,
    0x00038d7ea4c68000, 0x00038d7ea4c68000, 0x002386f26fc10000,
    0x002386f26fc10000, 0x002386f26fc10000, 0x002386f26fc10000,
    0x016345785d8a0000, 0x016345785d8a0000, 0x016345785d8a0000,
    0x0de0b6b3a7640000, 0x0de0b6b3a7640000, 0x0de0b6b3a7640000,
    0x8ac7230489e80000, 0x8ac7230489e80000, 0x8ac7230489e80000,
    0x8ac7230489e80000
  ]
  
  // bid_onehalf128[] contains the high bits of 1/2 positioned correctly for
  // comparison with the high bits of f2*
  // the 64-bit word order is L, H
  static func bid_onehalf128(_ i: Int) -> UInt64 {
    if i < 3 { return 0 }
    return UInt64(1) << (bid_shiftright128[i] - UInt8(i < 22 ? 0 : 64) - 1)
  }
  
  // bid_maskhigh128[] contains the mask to apply to the top 128 bits of the
  // 128x128-bit product in order to obtain the high bits of f2*
  // the 64-bit word order is L, H
  static func bid_maskhigh128(_ i:Int) -> UInt64 {
    if i < 3 { return 0 }
    return (UInt64(1) << (bid_shiftright128[i] - UInt8(i < 22 ? 0 : 64))) - 1
  }
  
  static let bid_recip_scale32 : [UInt8] = [1, 1, 3, 7, 9, 14, 18, 21, 25]
  
  static let bid_recip_scale : [Int8] = [
    1, 1, 1, 1, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53,
    56, 59, 63,  66, 69, 73, 76, 79, 83, 86, 89, 92, 96, 99, 102, 109
  ]
  
  static func bid_reciprocals10_128(_ i: Int) -> UInt128 {
    if i == 0 { return UInt128() }
    let shiftedOne = UInt128(1) << bid_recip_scale[i] // upper dividend
    let result = bid_power10_table_128(i).dividingFullWidth(
      (shiftedOne, UInt128()))
    return result.quotient + 1
  }
  
  // powers of 2/5 bid factors 
  static let bid_factors : [[Int8]] = [
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,2], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [5,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,2],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [6,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,2], [2,0], [0,0], [1,0], [0,0], [4,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [5,0], [0,0], [1,0], [0,0], [2,2],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [4,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,3], [1,0], [0,0], [7,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [4,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,2],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [5,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,2], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [6,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,2],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [5,0], [0,2], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [4,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,3],
    [0,0], [2,0], [0,0], [1,0], [0,1], [8,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [4,0], [0,0], [1,0], [0,2], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [5,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,2],
    [0,0], [1,0], [0,0], [4,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [6,1],
    [0,0], [1,0], [0,0], [2,0], [0,2], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,2],
    [0,0], [5,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,3], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [7,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [4,2],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [5,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,2], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [4,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [6,0], [0,0], [1,2],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [4,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,2], [2,0], [0,0], [1,0], [0,0], [5,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [4,0], [0,0], [1,0], [0,0], [2,3],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [9,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,2], [1,0], [0,0], [4,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [5,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,2],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [4,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,2], [6,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [4,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,2],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [5,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [4,0], [0,4], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [7,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,2],
    [0,0], [2,0], [0,0], [1,0], [0,1], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [5,0], [0,0], [1,0], [0,2], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,2],
    [0,0], [1,0], [0,0], [6,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [4,1],
    [0,0], [1,0], [0,0], [2,0], [0,2], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [5,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,3],
    [0,0], [4,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [8,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,2], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [4,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [5,2],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,2], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [6,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,2],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [5,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [3,0], [0,0], [1,0], [0,3], [2,0], [0,0], [1,0], [0,0], [4,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [7,0], [0,0], [1,0], [0,0], [2,2],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [4,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,1],
    [0,0], [1,0], [0,0], [2,0], [0,2], [1,0], [0,0], [5,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [4,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,2],
    [0,0], [3,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [6,1],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [3,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,2], [4,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [3,0], [0,1], [1,0], [0,0], [2,0], [0,0], [1,1],
    [0,0], [5,0], [0,0], [1,0], [0,1], [2,0], [0,0], [1,0], [0,0], [3,3],
    [0,0], [1,0], [0,0], [2,0], [0,1], [1,0], [0,0], [4,0], [0,0], [1,1],
    [0,0], [2,0], [0,0], [1,0], [0,1], [3,0], [0,0], [1,0], [0,0], [2,1],
    [0,0], [1,0], [0,0], [10,0]
  ]
  
  static func bid_reciprocals10_32(_ i: Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = bid_recip_scale32[i]+32
    return (UInt64(1) << twoPower) / UInt64(bid_power10_table_128(i)) + 1
  }
  
  static let bid_exponents_binary64 = [
     -55, -51, -48, -45, -41, -38, -35, -31, -28, -25, -22, -18, -15, -12, -8,
     -5, -2, 2, 5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55,
     58, 62, 65, 68, 71, 75, 78, 81, 85, 88, 91, 95, 98, 101, 105, 108, 111,
     115, 118, 121, 125, 128, 131, 135, 138, 141, 145, 148, 151, 155, 158, 161,
     164, 168, 171, 174, 178, 181, 184, 188, 191, 194, 198, 201, 204, 208, 211,
     214, 218, 221, 224, 228, 231, 234, 238, 241, 244, 248, 251, 254, 258, 261,
     264, 267, 271, 274, 277, 281, 284, 287, 291, 294, 297, 301, 304, 307, 311,
     314, 317, 321, 324, 327, 331, 334, 337, 341, 344, 347, 351, 354, 357, 360,
     364, 367, 370, 374, 377, 380, 384, 387, 390, 394, 397, 400, 404, 407, 410,
     414, 417, 420, 424, 427, 430, 434, 437, 440, 444, 447, 450, 454, 457, 460,
     463, 467, 470, 473, 477, 480, 483, 487, 490, 493, 497, 500, 503, 507, 510,
     513, 517, 520, 523, 527, 530, 533, 537, 540, 543, 547, 550, 553, 556, 560,
     563, 566, 570, 573, 576, 580, 583, 586, 590, 593, 596, 600, 603, 606, 610,
     613, 616, 620, 623, 626, 630, 633, 636, 640, 643, 646, 649, 653, 656, 659,
     663, 666, 669, 673, 676, 679, 683, 686, 689, 693, 696, 699, 703, 706, 709,
     713, 716, 719, 723, 726, 729, 733, 736, 739, 743, 746, 749, 752, 756, 759,
     762, 766, 769, 772, 776, 779, 782, 786, 789, 792, 796, 799, 802, 806, 809,
     812, 816, 819, 822, 826, 829, 832, 836, 839, 842, 845, 849, 852, 855, 859,
     862, 865, 869, 872, 875, 879, 882, 885, 889, 892, 895, 899, 902, 905, 909,
     912, 915, 919, 922, 925, 929, 932, 935, 939, 942, 945, 948, 952, 955, 958,
     962, 965, 968, 972, 975, 978, 982, 985, 988, 992, 995, 998, 1002, 1005,
     1008, 1012, 1015, 1018, 1022, 1025, 1028, 1032, 1035, 1038, 1041, 1045,
     1048, 1051, 1055, 1058, 1061, 1065, 1068, 1071, 1075, 1078, 1081, 1085,
     1088, 1091, 1095, 1098, 1101, 1105, 1108, 1111, 1115, 1118, 1121, 1125,
     1128, 1131, 1134, 1138, 1141, 1144, 1148, 1151, 1154, 1158, 1161, 1164,
     1168, 1171, 1174, 1178, 1181, 1184, 1188, 1191, 1194, 1198, 1201, 1204,
     1208, 1211, 1214, 1218, 1221, 1224, 1228, 1231, 1234, 1237, 1241, 1244,
     1247, 1251, 1254, 1257, 1261, 1264, 1267, 1271, 1274, 1277, 1281, 1284,
     1287, 1291, 1294, 1297, 1301, 1304, 1307, 1311, 1314, 1317, 1321, 1324,
     1327, 1330, 1334, 1337, 1340, 1344, 1347, 1350, 1354, 1357, 1360, 1364,
     1367, 1370, 1374, 1377, 1380, 1384, 1387, 1390, 1394, 1397, 1400, 1404,
     1407, 1410, 1414, 1417, 1420, 1424, 1427, 1430, 1433, 1437, 1440, 1443,
     1447, 1450, 1453, 1457, 1460, 1463, 1467, 1470, 1473, 1477, 1480, 1483,
     1487, 1490, 1493, 1497, 1500, 1503, 1507, 1510, 1513, 1517, 1520, 1523,
     1526, 1530, 1533, 1536, 1540, 1543, 1546, 1550, 1553, 1556, 1560, 1563,
     1566, 1570, 1573, 1576, 1580, 1583, 1586, 1590, 1593, 1596, 1600, 1603,
     1606, 1610, 1613, 1616, 1620, 1623, 1626, 1629, 1633, 1636, 1639, 1643,
     1646, 1649, 1653, 1656, 1659, 1663, 1666, 1669, 1673, 1676, 1679, 1683,
     1686, 1689, 1693, 1696, 1699, 1703, 1706, 1709, 1713, 1716, 1719, 1722,
     1726, 1729, 1732, 1736, 1739, 1742, 1746, 1749, 1752, 1756, 1759, 1762,
     1766, 1769, 1772, 1776, 1779, 1782, 1786, 1789, 1792, 1796, 1799, 1802,
     1806, 1809, 1812, 1815, 1819, 1822, 1825, 1829, 1832, 1835, 1839, 1842,
     1845, 1849, 1852, 1855, 1859, 1862, 1865, 1869, 1872, 1875, 1879, 1882,
     1885, 1889, 1892, 1895, 1899, 1902, 1905, 1909, 1912, 1915, 1918, 1922,
     1925, 1928, 1932, 1935, 1938, 1942, 1945, 1948, 1952, 1955, 1958, 1962,
     1965, 1968, 1972, 1975, 1978, 1982, 1985, 1988, 1992, 1995, 1998, 2002,
     2005, 2008, 2011, 2015, 2018, 2021, 2025, 2028, 2031, 2035, 2038, 2041,
     2045, 2048, 2051, 2055, 2058, 2061, 2065, 2068, 2071, 2075, 2078, 2081,
     2085, 2088, 2091, 2095, 2098, 2101, 2105, 2108, 2111, 2114, 2118, 2121,
     2124, 2128, 2131, 2134, 2138, 2141, 2144, 2148, 2151, 2154, 2158, 2161
  ]
  
  static func bid_power10_table_128(_ i: Int) -> UInt128 {
    power(UInt128(10), to: i)
  }
  
  /// Returns the number of decimal digits in 2^i.
  static func bid_estimate_decimal_digits(_ i: Int) -> Int {
    let n = UInt128(1) << i
    return digitsIn(n)
  }
  
  static let bid_b2d : [UInt64] = [
    0x000, 0x001, 0x002, 0x003, 0x004, 0x005, 0x006, 0x007, 0x008, 0x009,
    0x010, 0x011, 0x012, 0x013, 0x014, 0x015, 0x016, 0x017, 0x018, 0x019,
    0x020, 0x021, 0x022, 0x023, 0x024, 0x025, 0x026, 0x027, 0x028, 0x029,
    0x030, 0x031, 0x032, 0x033, 0x034, 0x035, 0x036, 0x037, 0x038, 0x039,
    0x040, 0x041, 0x042, 0x043, 0x044, 0x045, 0x046, 0x047, 0x048, 0x049,
    0x050, 0x051, 0x052, 0x053, 0x054, 0x055, 0x056, 0x057, 0x058, 0x059,
    0x060, 0x061, 0x062, 0x063, 0x064, 0x065, 0x066, 0x067, 0x068, 0x069,
    0x070, 0x071, 0x072, 0x073, 0x074, 0x075, 0x076, 0x077, 0x078, 0x079,
    0x00a, 0x00b, 0x02a, 0x02b, 0x04a, 0x04b, 0x06a, 0x06b, 0x04e, 0x04f,
    0x01a, 0x01b, 0x03a, 0x03b, 0x05a, 0x05b, 0x07a, 0x07b, 0x05e, 0x05f,
    0x080, 0x081, 0x082, 0x083, 0x084, 0x085, 0x086, 0x087, 0x088, 0x089,
    0x090, 0x091, 0x092, 0x093, 0x094, 0x095, 0x096, 0x097, 0x098, 0x099,
    0x0a0, 0x0a1, 0x0a2, 0x0a3, 0x0a4, 0x0a5, 0x0a6, 0x0a7, 0x0a8, 0x0a9,
    0x0b0, 0x0b1, 0x0b2, 0x0b3, 0x0b4, 0x0b5, 0x0b6, 0x0b7, 0x0b8, 0x0b9,
    0x0c0, 0x0c1, 0x0c2, 0x0c3, 0x0c4, 0x0c5, 0x0c6, 0x0c7, 0x0c8, 0x0c9,
    0x0d0, 0x0d1, 0x0d2, 0x0d3, 0x0d4, 0x0d5, 0x0d6, 0x0d7, 0x0d8, 0x0d9,
    0x0e0, 0x0e1, 0x0e2, 0x0e3, 0x0e4, 0x0e5, 0x0e6, 0x0e7, 0x0e8, 0x0e9,
    0x0f0, 0x0f1, 0x0f2, 0x0f3, 0x0f4, 0x0f5, 0x0f6, 0x0f7, 0x0f8, 0x0f9,
    0x08a, 0x08b, 0x0aa, 0x0ab, 0x0ca, 0x0cb, 0x0ea, 0x0eb, 0x0ce, 0x0cf,
    0x09a, 0x09b, 0x0ba, 0x0bb, 0x0da, 0x0db, 0x0fa, 0x0fb, 0x0de, 0x0df,
    0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109,
    0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x116, 0x117, 0x118, 0x119,
    0x120, 0x121, 0x122, 0x123, 0x124, 0x125, 0x126, 0x127, 0x128, 0x129,
    0x130, 0x131, 0x132, 0x133, 0x134, 0x135, 0x136, 0x137, 0x138, 0x139,
    0x140, 0x141, 0x142, 0x143, 0x144, 0x145, 0x146, 0x147, 0x148, 0x149,
    0x150, 0x151, 0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x158, 0x159,
    0x160, 0x161, 0x162, 0x163, 0x164, 0x165, 0x166, 0x167, 0x168, 0x169,
    0x170, 0x171, 0x172, 0x173, 0x174, 0x175, 0x176, 0x177, 0x178, 0x179,
    0x10a, 0x10b, 0x12a, 0x12b, 0x14a, 0x14b, 0x16a, 0x16b, 0x14e, 0x14f,
    0x11a, 0x11b, 0x13a, 0x13b, 0x15a, 0x15b, 0x17a, 0x17b, 0x15e, 0x15f,
    0x180, 0x181, 0x182, 0x183, 0x184, 0x185, 0x186, 0x187, 0x188, 0x189,
    0x190, 0x191, 0x192, 0x193, 0x194, 0x195, 0x196, 0x197, 0x198, 0x199,
    0x1a0, 0x1a1, 0x1a2, 0x1a3, 0x1a4, 0x1a5, 0x1a6, 0x1a7, 0x1a8, 0x1a9,
    0x1b0, 0x1b1, 0x1b2, 0x1b3, 0x1b4, 0x1b5, 0x1b6, 0x1b7, 0x1b8, 0x1b9,
    0x1c0, 0x1c1, 0x1c2, 0x1c3, 0x1c4, 0x1c5, 0x1c6, 0x1c7, 0x1c8, 0x1c9,
    0x1d0, 0x1d1, 0x1d2, 0x1d3, 0x1d4, 0x1d5, 0x1d6, 0x1d7, 0x1d8, 0x1d9,
    0x1e0, 0x1e1, 0x1e2, 0x1e3, 0x1e4, 0x1e5, 0x1e6, 0x1e7, 0x1e8, 0x1e9,
    0x1f0, 0x1f1, 0x1f2, 0x1f3, 0x1f4, 0x1f5, 0x1f6, 0x1f7, 0x1f8, 0x1f9,
    0x18a, 0x18b, 0x1aa, 0x1ab, 0x1ca, 0x1cb, 0x1ea, 0x1eb, 0x1ce, 0x1cf,
    0x19a, 0x19b, 0x1ba, 0x1bb, 0x1da, 0x1db, 0x1fa, 0x1fb, 0x1de, 0x1df,
    0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208, 0x209,
    0x210, 0x211, 0x212, 0x213, 0x214, 0x215, 0x216, 0x217, 0x218, 0x219,
    0x220, 0x221, 0x222, 0x223, 0x224, 0x225, 0x226, 0x227, 0x228, 0x229,
    0x230, 0x231, 0x232, 0x233, 0x234, 0x235, 0x236, 0x237, 0x238, 0x239,
    0x240, 0x241, 0x242, 0x243, 0x244, 0x245, 0x246, 0x247, 0x248, 0x249,
    0x250, 0x251, 0x252, 0x253, 0x254, 0x255, 0x256, 0x257, 0x258, 0x259,
    0x260, 0x261, 0x262, 0x263, 0x264, 0x265, 0x266, 0x267, 0x268, 0x269,
    0x270, 0x271, 0x272, 0x273, 0x274, 0x275, 0x276, 0x277, 0x278, 0x279,
    0x20a, 0x20b, 0x22a, 0x22b, 0x24a, 0x24b, 0x26a, 0x26b, 0x24e, 0x24f,
    0x21a, 0x21b, 0x23a, 0x23b, 0x25a, 0x25b, 0x27a, 0x27b, 0x25e, 0x25f,
    0x280, 0x281, 0x282, 0x283, 0x284, 0x285, 0x286, 0x287, 0x288, 0x289,
    0x290, 0x291, 0x292, 0x293, 0x294, 0x295, 0x296, 0x297, 0x298, 0x299,
    0x2a0, 0x2a1, 0x2a2, 0x2a3, 0x2a4, 0x2a5, 0x2a6, 0x2a7, 0x2a8, 0x2a9,
    0x2b0, 0x2b1, 0x2b2, 0x2b3, 0x2b4, 0x2b5, 0x2b6, 0x2b7, 0x2b8, 0x2b9,
    0x2c0, 0x2c1, 0x2c2, 0x2c3, 0x2c4, 0x2c5, 0x2c6, 0x2c7, 0x2c8, 0x2c9,
    0x2d0, 0x2d1, 0x2d2, 0x2d3, 0x2d4, 0x2d5, 0x2d6, 0x2d7, 0x2d8, 0x2d9,
    0x2e0, 0x2e1, 0x2e2, 0x2e3, 0x2e4, 0x2e5, 0x2e6, 0x2e7, 0x2e8, 0x2e9,
    0x2f0, 0x2f1, 0x2f2, 0x2f3, 0x2f4, 0x2f5, 0x2f6, 0x2f7, 0x2f8, 0x2f9,
    0x28a, 0x28b, 0x2aa, 0x2ab, 0x2ca, 0x2cb, 0x2ea, 0x2eb, 0x2ce, 0x2cf,
    0x29a, 0x29b, 0x2ba, 0x2bb, 0x2da, 0x2db, 0x2fa, 0x2fb, 0x2de, 0x2df,
    0x300, 0x301, 0x302, 0x303, 0x304, 0x305, 0x306, 0x307, 0x308, 0x309,
    0x310, 0x311, 0x312, 0x313, 0x314, 0x315, 0x316, 0x317, 0x318, 0x319,
    0x320, 0x321, 0x322, 0x323, 0x324, 0x325, 0x326, 0x327, 0x328, 0x329,
    0x330, 0x331, 0x332, 0x333, 0x334, 0x335, 0x336, 0x337, 0x338, 0x339,
    0x340, 0x341, 0x342, 0x343, 0x344, 0x345, 0x346, 0x347, 0x348, 0x349,
    0x350, 0x351, 0x352, 0x353, 0x354, 0x355, 0x356, 0x357, 0x358, 0x359,
    0x360, 0x361, 0x362, 0x363, 0x364, 0x365, 0x366, 0x367, 0x368, 0x369,
    0x370, 0x371, 0x372, 0x373, 0x374, 0x375, 0x376, 0x377, 0x378, 0x379,
    0x30a, 0x30b, 0x32a, 0x32b, 0x34a, 0x34b, 0x36a, 0x36b, 0x34e, 0x34f,
    0x31a, 0x31b, 0x33a, 0x33b, 0x35a, 0x35b, 0x37a, 0x37b, 0x35e, 0x35f,
    0x380, 0x381, 0x382, 0x383, 0x384, 0x385, 0x386, 0x387, 0x388, 0x389,
    0x390, 0x391, 0x392, 0x393, 0x394, 0x395, 0x396, 0x397, 0x398, 0x399,
    0x3a0, 0x3a1, 0x3a2, 0x3a3, 0x3a4, 0x3a5, 0x3a6, 0x3a7, 0x3a8, 0x3a9,
    0x3b0, 0x3b1, 0x3b2, 0x3b3, 0x3b4, 0x3b5, 0x3b6, 0x3b7, 0x3b8, 0x3b9,
    0x3c0, 0x3c1, 0x3c2, 0x3c3, 0x3c4, 0x3c5, 0x3c6, 0x3c7, 0x3c8, 0x3c9,
    0x3d0, 0x3d1, 0x3d2, 0x3d3, 0x3d4, 0x3d5, 0x3d6, 0x3d7, 0x3d8, 0x3d9,
    0x3e0, 0x3e1, 0x3e2, 0x3e3, 0x3e4, 0x3e5, 0x3e6, 0x3e7, 0x3e8, 0x3e9,
    0x3f0, 0x3f1, 0x3f2, 0x3f3, 0x3f4, 0x3f5, 0x3f6, 0x3f7, 0x3f8, 0x3f9,
    0x38a, 0x38b, 0x3aa, 0x3ab, 0x3ca, 0x3cb, 0x3ea, 0x3eb, 0x3ce, 0x3cf,
    0x39a, 0x39b, 0x3ba, 0x3bb, 0x3da, 0x3db, 0x3fa, 0x3fb, 0x3de, 0x3df,
    0x00c, 0x00d, 0x10c, 0x10d, 0x20c, 0x20d, 0x30c, 0x30d, 0x02e, 0x02f,
    0x01c, 0x01d, 0x11c, 0x11d, 0x21c, 0x21d, 0x31c, 0x31d, 0x03e, 0x03f,
    0x02c, 0x02d, 0x12c, 0x12d, 0x22c, 0x22d, 0x32c, 0x32d, 0x12e, 0x12f,
    0x03c, 0x03d, 0x13c, 0x13d, 0x23c, 0x23d, 0x33c, 0x33d, 0x13e, 0x13f,
    0x04c, 0x04d, 0x14c, 0x14d, 0x24c, 0x24d, 0x34c, 0x34d, 0x22e, 0x22f,
    0x05c, 0x05d, 0x15c, 0x15d, 0x25c, 0x25d, 0x35c, 0x35d, 0x23e, 0x23f,
    0x06c, 0x06d, 0x16c, 0x16d, 0x26c, 0x26d, 0x36c, 0x36d, 0x32e, 0x32f,
    0x07c, 0x07d, 0x17c, 0x17d, 0x27c, 0x27d, 0x37c, 0x37d, 0x33e, 0x33f,
    0x00e, 0x00f, 0x10e, 0x10f, 0x20e, 0x20f, 0x30e, 0x30f, 0x06e, 0x06f,
    0x01e, 0x01f, 0x11e, 0x11f, 0x21e, 0x21f, 0x31e, 0x31f, 0x07e, 0x07f,
    0x08c, 0x08d, 0x18c, 0x18d, 0x28c, 0x28d, 0x38c, 0x38d, 0x0ae, 0x0af,
    0x09c, 0x09d, 0x19c, 0x19d, 0x29c, 0x29d, 0x39c, 0x39d, 0x0be, 0x0bf,
    0x0ac, 0x0ad, 0x1ac, 0x1ad, 0x2ac, 0x2ad, 0x3ac, 0x3ad, 0x1ae, 0x1af,
    0x0bc, 0x0bd, 0x1bc, 0x1bd, 0x2bc, 0x2bd, 0x3bc, 0x3bd, 0x1be, 0x1bf,
    0x0cc, 0x0cd, 0x1cc, 0x1cd, 0x2cc, 0x2cd, 0x3cc, 0x3cd, 0x2ae, 0x2af,
    0x0dc, 0x0dd, 0x1dc, 0x1dd, 0x2dc, 0x2dd, 0x3dc, 0x3dd, 0x2be, 0x2bf,
    0x0ec, 0x0ed, 0x1ec, 0x1ed, 0x2ec, 0x2ed, 0x3ec, 0x3ed, 0x3ae, 0x3af,
    0x0fc, 0x0fd, 0x1fc, 0x1fd, 0x2fc, 0x2fd, 0x3fc, 0x3fd, 0x3be, 0x3bf,
    0x08e, 0x08f, 0x18e, 0x18f, 0x28e, 0x28f, 0x38e, 0x38f, 0x0ee, 0x0ef,
    0x09e, 0x09f, 0x19e, 0x19f, 0x29e, 0x29f, 0x39e, 0x39f, 0x0fe, 0x0ff
  ]
  
  static let bid_d2b : [UInt64] =
  [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 80, 81, 800, 801, 880, 881,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 90, 91, 810, 811, 890, 891,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 82, 83, 820, 821, 808, 809,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 92, 93, 830, 831, 818, 819,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 84, 85, 840, 841, 88, 89,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 94, 95, 850, 851, 98, 99,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 86, 87, 860, 861, 888, 889,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 96, 97, 870, 871, 898, 899,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 180, 181, 900, 901,
    980, 981,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 190, 191, 910, 911,
    990, 991,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 182, 183, 920, 921,
    908, 909,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 192, 193, 930, 931,
    918, 919,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 184, 185, 940, 941,
    188, 189,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 194, 195, 950, 951,
    198, 199,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 186, 187, 960, 961,
    988, 989,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 196, 197, 970, 971,
    998, 999,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 280, 281, 802, 803,
    882, 883,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 290, 291, 812, 813,
    892, 893,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 282, 283, 822, 823,
    828, 829,
    230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 292, 293, 832, 833,
    838, 839,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 284, 285, 842, 843,
    288, 289,
    250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 294, 295, 852, 853,
    298, 299,
    260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 286, 287, 862, 863,
    888, 889,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 296, 297, 872, 873,
    898, 899,
    300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 380, 381, 902, 903,
    982, 983,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 390, 391, 912, 913,
    992, 993,
    320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 382, 383, 922, 923,
    928, 929,
    330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 392, 393, 932, 933,
    938, 939,
    340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 384, 385, 942, 943,
    388, 389,
    350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 394, 395, 952, 953,
    398, 399,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 386, 387, 962, 963,
    988, 989,
    370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 396, 397, 972, 973,
    998, 999,
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 480, 481, 804, 805,
    884, 885,
    410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 490, 491, 814, 815,
    894, 895,
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 482, 483, 824, 825,
    848, 849,
    430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 492, 493, 834, 835,
    858, 859,
    440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 484, 485, 844, 845,
    488, 489,
    450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 494, 495, 854, 855,
    498, 499,
    460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 486, 487, 864, 865,
    888, 889,
    470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 496, 497, 874, 875,
    898, 899,
    500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 580, 581, 904, 905,
    984, 985,
    510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 590, 591, 914, 915,
    994, 995,
    520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 582, 583, 924, 925,
    948, 949,
    530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 592, 593, 934, 935,
    958, 959,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 584, 585, 944, 945,
    588, 589,
    550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 594, 595, 954, 955,
    598, 599,
    560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 586, 587, 964, 965,
    988, 989,
    570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 596, 597, 974, 975,
    998, 999,
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 680, 681, 806, 807,
    886, 887,
    610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 690, 691, 816, 817,
    896, 897,
    620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 682, 683, 826, 827,
    868, 869,
    630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 692, 693, 836, 837,
    878, 879,
    640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 684, 685, 846, 847,
    688, 689,
    650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 694, 695, 856, 857,
    698, 699,
    660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 686, 687, 866, 867,
    888, 889,
    670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 696, 697, 876, 877,
    898, 899,
    700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 780, 781, 906, 907,
    986, 987,
    710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 790, 791, 916, 917,
    996, 997,
    720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 782, 783, 926, 927,
    968, 969,
    730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 792, 793, 936, 937,
    978, 979,
    740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 784, 785, 946, 947,
    788, 789,
    750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 794, 795, 956, 957,
    798, 799,
    760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 786, 787, 966, 967,
    988, 989,
    770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 796, 797, 976, 977,
    998, 999
  ]
  
  static func bid_d2b2(_ i: Int) -> UInt64 { bid_d2b[i] * 1000 }
  static func bid_power_five(_ i: Int) -> UInt128 { power(UInt128(5), to: i) }
  
  static let coefflimits: [UInt64] = [
    10000000, 2000000, 400000, 80000, 16000, 3200, 640, 128, 25, 5, 1
  ]
  
  static func bid_coefflimits_bid32(_ i: Int) -> UInt128 {
    i > 10 ? 0 : UInt128(coefflimits[i])
  }
  
  static let midPoint = UInt128(high: 1 << 63, low: 0)
  
  static let bid_roundbound_128: [UInt128] = [
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
  
  static let bid_short_recip_scale: [Int8] = [
    1, 1, 5, 7, 11, 12, 17, 21, 24, 27, 31, 34, 37, 41, 44, 47, 51, 54
  ]
  
  // bid_ten2k64[i] = 10^i, 0 <= i <= 19
  static func bid_ten2k64(_ i:Int) -> UInt64 {
    UInt64(bid_power10_table_128(i))
  }
  
  static func bid_midpoint64(_ i:Int) -> UInt64 { 5 * bid_ten2k64(i) }
  
  // Ex-64 from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static func bid_Ex64m64(_ i:Int) -> UInt8 {
    UInt8(bid_short_recip_scale[i+3])
  }
  
  // bid_ten2mk64 power-of-two scaling
  static let bid_powers : [UInt8] = [
    64, 64, 64, 67, 70, 73, 77, 80, 83, 87, 90, 93, 97, 100, 103, 107
  ]
  
  // Values of 10^(-x) trancated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static func bid_ten2mk64(_ i:Int) -> UInt64 {
     UInt64((UInt128(1) << bid_powers[i]) / bid_power10_table_128(i+1))+1
  }
  
  // Values of 10^(-x) trancated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static func bid_ten2mxtrunc64(_ i:Int) -> UInt64 {
    UInt64((UInt128(1) << (64+bid_Ex64m64(i))) / bid_power10_table_128(i+1))
  }

  // Kx from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static func bid_Kx64(_ i:Int) -> UInt64 {
    bid_ten2mxtrunc64(i)+1
  }
   
  static func bid_reciprocals10_64(_ i: Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = bid_short_recip_scale[i]+64
    return UInt64(UInt128(1) << twoPower / bid_power10_table_128(i)) + 1
  }
  
  // Values of 1/2 in the right position to be compared with the fraction from
  // C * kx, 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the high 64 bits of the fraction)
  static func bid_half64(_ i: Int) -> UInt64 {
    (UInt64(1) << bid_shiftright128[i+3] - 1)
  }
  
  // Values of mask in the right position to obtain the high Ex - 64 bits
  // of the fraction from C * kx, 1 <= x <= 17; the fraction consists of
  // the low Ex bits in C * kx
  static func bid_mask64(_ i:Int) -> UInt64 {
    (UInt64(1) << bid_shiftright128[i+3]) - 1
  }
  
  static func bid_round_const_table(_ rnd:Int, _ i:Int) -> UInt64 {
    if i == 0 { return 0 }
    switch rnd {
      case 0, 4: return 5 * bid_ten2k64(i-1)
      case 2: return bid_ten2k64(i-1)-1
      default: return 0 // covers rnd = 1, 3
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
    var P128 = UInt128()
    __mul_64x64_to_128(&P128, C, bid_Kx64(ind));
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
    if (fstar.high > bid_half64(ind) || (fstar.high == bid_half64(ind) &&
                                         fstar.low != 0)) {
      // f* > 1/2 and the result may be exact
      // Calculate f* - 1/2
      let tmp64 = fstar.high - bid_half64(ind);
      if (tmp64 != 0 || fstar.low > bid_ten2mxtrunc64(ind)) {
        // f* - 1/2 > 10^(-x)
        ptr_is_inexact_lt_midpoint = true
      }    // else the result is exact
    } else {    // the result is inexact; f2* <= 1/2
      ptr_is_inexact_gt_midpoint = true
    }
    // check for midpoints (could do this before determining inexactness)
    if (fstar.high == 0 && fstar.low <= bid_ten2mxtrunc64(ind)) {
      // the result is a midpoint
      if (Cstar & 0x01 != 0) {    // Cstar is odd; MP in [EVEN, ODD]
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
  
  ///   General pack macro for BID32 with underflow resolution
  static func get_BID32_UF (_ sgn: UInt32, _ expon: Int, _ coeff: UInt64,
                _ R: Int, _ rmode:Rounding, _ fpsc: inout Status) -> UInt32 {
    var expon = expon
    var coeff = coeff
    
    if coeff > MAX_NUMBER {
      expon += 1
      coeff = 1_000_000
    }
    // check for possible underflow/overflow
    if UInt32(bitPattern: Int32(expon)) > MAX_EXPON {
      return handleRounding(sgn, expon, Int(coeff), R, rmode, &fpsc)
    }
    
    var mask = UInt32(1) << 23;
    var r: UInt32
    
    // check whether coefficient fits in DECIMAL_COEFF_FIT bits
    if coeff < mask {
      r = UInt32(expon)
      r <<= 23;
      r |= UInt32(coeff) | sgn
      return r
    }
    // special format
    r = UInt32(expon)
    r <<= 21
    r |= (sgn | STEERING_BITS_MASK)
    // add coeff, without leading bits
    mask = (UInt32(1) << 21) - 1
    r |= (UInt32(coeff) & mask)
    return r
  }
  
  static private func handleRounding(_ s:Word, _ exp:Int, _ c:Int,
                _ R: Int = 0, _ r:Rounding, _ fpsc: inout Status) -> Word {
    let sexp = exp
    var r = r
    if sexp < 0 {
      if sexp + MAX_DIGITS < 0 {
        fpsc.formUnion([.underflow, .inexact])
        if r == .down && s != 0 {
          return 0x8000_0001
        }
        if r == .up && s == 0 {
          return 1
        }
        return Word(s)
      }
      
      // swap round modes when negative
      if s != 0 {
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
      let extra_digits = 1-sexp
      c += Int(bid_round_const_table(roundIndex, extra_digits))
      
      // get coeff*(2^M[extra_digits])/10^extra_digits
      var Q = UInt128()
      __mul_64x64_to_128(&Q, UInt64(c), bid_reciprocals10_64(extra_digits))
      
      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      let amount = bid_short_recip_scale[extra_digits]
      
      var _C64 = Q.high >> amount
      var remainder_h = UInt64(0)
      
      if r == .toNearestOrAwayFromZero {
        if (_C64 & 1 != 0) {
          // check whether fractional part of initial_P/10^extra_digits
          // is exactly .5
          
          // get remainder
          let amount2 = 64 - amount
          remainder_h = 0
          remainder_h &-= 1
          remainder_h >>= amount2
          remainder_h = remainder_h & Q.high
          
          if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
            _C64 -= 1
          }
        }
      }
      
      if fpsc.contains(.inexact) {
        fpsc.insert(.underflow)
      } else {
        var status = Status.inexact
        // get remainder
        remainder_h = Q.high << (64 - amount)
        
        switch r {
          case .toNearestOrAwayFromZero, .toNearestOrEven:
            // test whether fractional part is 0
            if (remainder_h == (UInt64(SIGN_MASK) << 32) && (Q.low <
                                        bid_reciprocals10_64(extra_digits))) {
              status = Status.clearFlags
            }
          case .down, .towardZero:
            if remainder_h == 0 && Q.low < bid_reciprocals10_64(extra_digits) {
              status = Status.clearFlags
            }
          default:
            // round up
            var Stemp = UInt64(0), carry = UInt64(0)
            __add_carry_out(&Stemp, &carry, Q.low,
                            bid_reciprocals10_64(extra_digits))
            if (remainder_h >> (64 - amount)) + carry >= UInt64(1) << amount {
              status = Status.clearFlags
            }
        }
        
        if !status.isEmpty {
          status.insert(.underflow)
          fpsc.formUnion(status)
        }
      }
      return Word(s) | UInt32(_C64)
    }
    var exp = exp
    var c = c
    if c == 0 { if exp > MAX_EXPON { exp = MAX_EXPON } }
    while c < 1000000 && exp > MAX_EXPON {
      c = (c << 3) + (c << 1)
      exp -= 1
    }
    if UInt32(exp) > MAX_EXPON {
      let s = (Word(s) << signBit)
      fpsc.formUnion([.overflow, .inexact])
      // overflow
      var res = s | INFINITY_MASK
      switch r {
        case .down:
          if s == 0 {
            res = LARGEST_BID
          }
        case .towardZero:
          res = s | LARGEST_BID
        case .up:
          // round up
          if s != 0 {
            res = s | LARGEST_BID
          }
        default: break
      }
      return res
    }
    return Word(c)
  }
  
  static func __add_carry_out(_ S: inout UInt64, _ CY: inout UInt64,
                              _ X:UInt64, _ Y:UInt64) {
    S = X &+ Y  // allow overflow
    CY = S < X ? 1 : 0
  }
}

extension UInt128 {
  
  public var high: UInt64 { UInt64(self.words[1]) }
  public var low:  UInt64 { UInt64(self.words[0]) }
  
  init(w: [UInt64]) { self.init(high: w[1], low: w[0]) }
  
}
