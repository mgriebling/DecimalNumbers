//
//  Decimal32.swift
//
//
//  Created by Mike Griebling on 2022-03-07.
//

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

public struct Decimal32 : CustomStringConvertible, ExpressibleByStringLiteral,
                          ExpressibleByIntegerLiteral,
                          ExpressibleByFloatLiteral, Codable, Hashable {
  
  // /set to true to monitor variable state (i.e., invalid operations, etc.)
  private static var enableStateOutput = false
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
  // MARK: - Decimal32 State variables
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
  
  public static let greatestFiniteMagnitude = Self(raw: bid32_max(0))
  public static let leastNormalMagnitude    = Self(raw: bid32(0, 0, 1_000_000))
  public static let leastNonzeroMagnitude   = Self(raw: bid32(0, 0, 1))
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers
  init(raw: UInt32) { x = raw } // only for internal use
  
  private func showState() {
    if Self.enableStateOutput && !Self.state.isEmpty {
      print("Warning: \(Self.state)")
    }
  }
  
  /// Binary Integer Decimal (BID) encoded 32-bit number
  public init(bid32: Word) { x = bid32 }
  
  /// Densely Packed Decimal encoded 32-bit number
  public init(dpd32: Word) { x = Self.dpd_to_bid32(dpd32) }
  
  ///
  public init(integerLiteral value: Int) {
    self = Self.int64_to_BID32(Int64(value), Self.rounding, &Self.state)
  }
  
  public init(_ value: Int = 0) { self.init(integerLiteral: value) }
  public init<T:BinaryInteger>(_ value: T) { self.init(Int(value)) }
  
  public init?<T:BinaryInteger>(exactly source: T) {
    self.init(Int(source))  // FIXME: - Proper init needed
  }
  
  public init(floatLiteral value: Double) {
    x = Self.double_to_bid32(value, Self.rounding, &Self.state)
  }
  
  public init(stringLiteral value: String) {
    if value.hasPrefix("0x") {
      var s = value; s.removeFirst(2)
      let n = Word(s, radix: 16) ?? 0
      x = n
    } else {
      x = Self.bid32_from_string(value, Self.rounding, &Self.state)
    }
  }
  
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
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Custom String Convertible compliance
  public var description: String { Self.bid32_to_string(x) }
  
}

extension Decimal32 : AdditiveArithmetic, Comparable, SignedNumeric,
                      Strideable, FloatingPoint {
  
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    let dec64 = Self.bid32_to_bid64(x, &Self.state)
    let res = Self.bid64_round_integral_exact(dec64, rule, &Self.state)
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
  
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Basic arithmetic operations
  
  public func isEqual(to other: Self) -> Bool { self == other }
  public func isLess(than other: Self) -> Bool { self < other }
  public func isLessThanOrEqualTo(_ other: Self) -> Bool {
    self < other || self == other
  }
  
  public static func == (lhs: Self, rhs: Self) -> Bool {
    Self.equal(lhs, rhs, &Self.state)
  }
  
  public static func < (lhs: Self, rhs: Self) -> Bool {
    Self.lessThan(lhs, rhs, &Self.state)
  }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.add(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.div(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(raw: Self.mul(lhs.x, rhs.x, Self.rounding, &Self.state))
  }
  
  public static func /= (lhs: inout Self, rhs: Self)  { lhs = lhs / rhs }
  public static func *= (lhs: inout Self, rhs: Self)  { lhs = lhs * rhs }
  public static func - (lhs: Self, rhs: Self) -> Self { lhs + (-rhs) }
  
}

public extension Decimal32 {
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Numeric State variables
  
  var sign: FloatingPointSign { x & Self.SIGN_MASK != 0 ? .minus : .plus }
  
  var magnitude: Self { Self(raw: x & ~Self.SIGN_MASK) }
  var dpd32: Word          { Self.bid_to_dpd32(x) }
  var decimal64: UInt64    { Self.bid32_to_bid64(x, &Self.state) }
  var decimal128: UInt128  { UInt128(Self.bid32_to_bid64(x, &Self.state)) << 64 }
  var int: Int             { Self.bid32ToInt(x, Self.rounding, &Self.state) }
  var uint: UInt           { Self.bid32ToUInt(x, Self.rounding, &Self.state) }
  var double: Double       { Self.bid32ToDouble(x, Self.rounding, &Self.state)}
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
  
  mutating func negate()   { self.x = x ^ Self.SIGN_MASK }
  
  private func unpack() -> (negative: Bool, exp: Int, coeff: UInt32)? {
    let s = Self.unpack(bid32: x)
    if !s.valid { return nil }
    return (s.negative, s.exp, s.coeff)
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
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint-required State variables
  
  public static var exponentMaximum: Int          { MAX_EXPON }
  public static var exponentBias: Int             { EXPONENT_BIAS }
  public static var significandMaxDigitCount: Int { MAX_DIGITS }
  
  public var significandDigitCount: Int {
    guard let x = unpack() else { return -1 }
    return Self.digitsIn(x.coeff)
  }
  
  public var exponentBitPattern: Word { Word(unpack()?.exp ?? 0) }
  
  public var significandDigits: [UInt8] {
    guard let x = unpack() else { return [] }
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
        var Q : UInt128 = UInt128(w: [0, 0])
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
        if ((x & (UInt32(1)<<25)) != 0) { status.insert(.invalidOperation) }
        let high = ((x & 0xFFFFF) > 999999) ? 0 : UInt64(x) << 44
        return (s, e, k, c, double_nan(s, high, 0))
      }
      e = Int((x >> 21) & ((UInt32(1)<<8)-1)) - 101
      c = UInt64((UInt32(1)<<23) + (x & ((UInt32(1)<<21)-1)))
      if UInt(c) > MAX_NUMBER {
        return (s, e, k, c, double_zero(s))
      }
      k = 0
    } else {
      e = Int((x >> 23) & ((UInt32(1)<<8)-1)) - 101
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
      var extra_digits = Int(bid_estimate_decimal_digits[bin_expon_cx] - 7)
      // add test for range
      if coefficient_x >= bid_power10_index_binexp[bin_expon_cx] {
        extra_digits+=1
      }
      
      var rmode1 = roundboundIndex(rmode) >> 2
      if sign_x != 0 && UInt(rmode1 - 1) < 2 {
        rmode1 = 3 - rmode1
      }
      
      exponent_x += extra_digits
      if (exponent_x < 0) && (exponent_x + Decimal32.MAX_DIGITS >= 0) {
        pfpsf.insert(.underflow)
        if exponent_x == -1 {
          if (coefficient_x + bid_round_const_table(rmode1, extra_digits) >=
              bid_power10_table_128[extra_digits + 7].low) {
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
      
      var status = Status.inexact //.insert(.inexact)
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
  
  /*****************************************************************************
   *  BID64_round_integral_exact
   ****************************************************************************/
  
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
            fstar = UInt128(high: P128.high & bid_maskhigh128[ind - 1],
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
            if fstar.high > bid_onehalf128[ind - 1] ||
                (fstar.high == bid_onehalf128[ind - 1] && fstar.low != 0) {
              // f2* > 1/2 and the result may be exact
              // Calculate f2* - 1/2
              if fstar.high > bid_onehalf128[ind - 1] ||
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
            fstar = UInt128(high: P128.high & bid_maskhigh128[ind - 1],
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
            if fstar.high > bid_onehalf128[ind - 1] ||
                (fstar.high == bid_onehalf128[ind - 1] && fstar.low != 0) {
              // f2* > 1/2 and the result may be exact
              // Calculate f2* - 1/2
              if fstar.high > bid_onehalf128[ind - 1] ||
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
            fstar = UInt128(high: P128.high & bid_maskhigh128[ind - 1],
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
            fstar = UInt128(high: P128.high & bid_maskhigh128[ind - 1],
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
            fstar = UInt128(high: P128.high & bid_maskhigh128[ind - 1],
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
      let scale_ca = bid_estimate_decimal_digits[bin_expon]
      
      let d2 = 16 - scale_ca
      if diff_dec_expon > d2 {
        diff_dec_expon = Int(d2)
        exponent_b = exponent_a - diff_dec_expon;
      }
    }
    
    let sign_ab = sign_a != sign_b ? Int64(-1) : Int64()
    let CB = UInt64(bitPattern: (Int64(coefficient_b) + sign_ab) ^ sign_ab)
    
    let SU = UInt64(coefficient_a) * bid_power10_table_128[diff_dec_expon].low
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
      n_digits = Int(bid_estimate_decimal_digits[bin_expon])
      if P >= bid_power10_table_128[n_digits].low {
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
    let R = P - Q * bid_power10_table_128[extra_digits].low
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
    var n_digits = Int(bid_estimate_decimal_digits[bin_expon_p])
    if P >= bid_power10_table_128[n_digits].low {
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
    let R = P - Q * bid_power10_table_128[extra_digits].low
    
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
                          (R+R>=bid_power10_table_128[extra_digits].low)) {
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
      ed2 = Int(bid_estimate_decimal_digits[bin_index]) + ed1
      let T = bid_power10_table_128[ed1].low
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
      
      ed2 = 7 - Int(bid_estimate_decimal_digits[bin_expon_cx]) - Int(DU)
      
      let T = bid_power10_table_128[ed2].low
      CA = UInt64(R) * T
      B = coefficient_y
      
      Q *= UInt32(bid_power10_table_128[ed2].low)
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
        
        if (digit_h & 1) == 0 {
          nzeros += Int(3 & UInt32(
            bid_packed_10000_zeros[Int(digit_h >> 3)] >> (digit_h & 7)))
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
      let T = bid_power10_table_128[diff_expon].low
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
      let digits_x = Int(bid_estimate_decimal_digits[bin_expon])
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
      CX *= bid_power10_table_128[e_scale].low
      
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
    var digits_x = bid_estimate_decimal_digits[bin_expon_cx];
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
    
    let CT = bid_power10_table_128[scale].low
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
      if  (x & SNAN_MASK) == SNAN_MASK { // x is SNAN
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
      let scale_ca = Int(bid_estimate_decimal_digits[bin_expon])
      
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
    __mul_64x128_low(&Tmp, coefficient_a, bid_power10_table_128[diff_dec_expon])
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
      n_digits = Int(bid_estimate_decimal_digits[bin_expon])
      if __unsigned_compare_ge_128(P, bid_power10_table_128[n_digits]) {
        n_digits += 1
      }
    } else {
      if P.low != 0 {
        let tempx = Double(P.low)
        bin_expon = Int((tempx.bitPattern & BINARY_EXPONENT_MASK) >> 52) -
                          BINARY_EXPONENT_BIAS
        n_digits = Int(bid_estimate_decimal_digits[bin_expon])
        if P.low >= bid_power10_table_128[n_digits].low {
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
                         bid_power10_table_128[extra_digits-18].low)
      __add_128_128 (&P, P, Stemp)
      if rmode == BID_ROUNDING_UP {
        __add_128_64(&P, P, bid_round_const_table(rmode1, extra_digits-18))
      }
    }
    
    // get P*(2^M[extra_digits])/10^extra_digits
    var Q_high = UInt128(), Q_low = UInt128(), C128 = UInt128()
    __mul_128x128_full (&Q_high, &Q_low, P, bid_reciprocals10_128[extra_digits])
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
    var amount = bid_recip_scale[extra_digits]
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
          && (Q_low.high < bid_reciprocals10_128[extra_digits].high
              || (Q_low.high == bid_reciprocals10_128[extra_digits].high
                  && Q_low.low < bid_reciprocals10_128[extra_digits].low))) {
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
            && (Q_low.high < bid_reciprocals10_128[extra_digits].high
                || (Q_low.high == bid_reciprocals10_128[extra_digits].high
                    && Q_low.low < bid_reciprocals10_128[extra_digits].low))) {
          status = []
        }
      case BID_ROUNDING_DOWN, BID_ROUNDING_TO_ZERO:
        if ((remainder_h | rem_l) == 0
            && (Q_low.high < bid_reciprocals10_128[extra_digits].high
                || (Q_low.high == bid_reciprocals10_128[extra_digits].high
                    && Q_low.low < bid_reciprocals10_128[extra_digits].low))) {
          status = []
        }
      default:
        // round up
        var low = Stemp.low
        var high = Stemp.high
        __add_carry_out(&low, &CY, Q_low.low,
                        bid_reciprocals10_128[extra_digits].low)
        __add_carry_in_out(&high, &carry, Q_low.high,
                           bid_reciprocals10_128[extra_digits].high, CY)
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
                           bid_power10_table_128[extra_digits-18].low);
        __add_128_128(&P, P, Stemp)
        if rmode == BID_ROUNDING_UP {
          __add_128_64(&P, P, bid_round_const_table(rmode1, extra_digits-18))
        }
      }
      
      // get P*(2^M[extra_digits])/10^extra_digits
      __mul_128x128_full(&Q_high, &Q_low, P,
                         bid_reciprocals10_128[extra_digits])
      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      amount = bid_recip_scale[extra_digits]
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
      var scale_cz = Int(bid_estimate_decimal_digits[bin_expon])
      if coefficient_z >= bid_power10_table_128[scale_cz].low {
          scale_cz+=1
      }
      
      var scale_k = 7 - scale_cz
      if diff_expon < scale_k {
          scale_k = diff_expon
      }
      coefficient_z *= UInt32(bid_power10_table_128[scale_k].low)
      
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
          pow5 = bid_power_five[a]
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
    let m_min = bid_breakpoints_bid32[e+450]
    var e_out = bid_exponents_bid32[e+450]
    
    // Choose exponent and reciprocal multiplier based on breakpoint
    var r:UInt256
    if le128(c.high, c.low, m_min.high, m_min.low) {
      r = bid_multipliers1_bid32[e+450]
    } else {
      r = bid_multipliers2_bid32[e+450]
      e_out += 1
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    __mul_128x256_to_384(&z, c, r)
    var c_prov = z.w[5]
    
    // Test inexactness and underflow (when testing tininess before rounding)
    if ((z.w[4] != 0) || (z.w[3] != 0)) {
      // __set_status_flags(pfpsf,BID_INEXACT_EXCEPTION);
      state.insert(.inexact)
      if (c_prov < 1000000) {
        state.insert(.underflow)
        // __set_status_flags(pfpsf,BID_UNDERFLOW_EXCEPTION);
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
          // __set_status_flags(pfpsf,BID_UNDERFLOW_EXCEPTION);
        }
      }
    }
    
    // Check for overflow
    if e_out > 90 + EXPONENT_BIAS {
      // __set_status_flags(pfpsf, BID_OVERFLOW_INEXACT_EXCEPTION);
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
//    if k<64 {
//      Q.low  = A.low >> k;
//      Q.low |= A.high << (64-k);
//      Q.high  = A.high >> k;
//    } else {
//      Q.low = A.high>>(k-64);
//      Q.high = 0;
//    }
  }
  
  // Shift 2-part 2^64 * hi + lo left by "c" bits
  // The "short" form requires a shift 0 < c < 64 and will be faster
  // Note that shifts of 64 can't be relied on as ANSI
  
  static func sll128_short(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
    UInt128(w: [lo << c, (hi << c) + (lo>>(64-c))])
  }
  
  // Compare "<" two 2-part unsigned integers
  @inlinable static func lt128(_ x_hi:UInt64, _ x_lo:UInt64,
                               _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (((x_hi) < (y_hi)) || (((x_hi) == (y_hi)) && ((x_lo) < (y_lo))))
  }
  
  // Likewise "<="
  @inlinable static func le128(_ x_hi:UInt64, _ x_lo:UInt64,
                               _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (((x_hi) < (y_hi)) || (((x_hi) == (y_hi)) && ((x_lo) <= (y_lo))))
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
    let m_min = bid_breakpoints_binary64[e+358]
    var e_out = bid_exponents_binary64[e+358] - Int(k)
    
    // Choose provisional exponent and reciprocal multiplier based on breakpoint
    var r = UInt256()
    if (c.high < m_min.high) {
      r = bid_multipliers1_binary64[e+358]
    } else {
      r = bid_multipliers2_binary64[e+358]
      e_out = e_out + 1;
    }
    
    // Do the reciprocal multiplication
    var z = UInt384()
    __mul_64x256_to_320(&z, c.high, r)
    z.w[5]=z.w[4]; z.w[4]=z.w[3]; z.w[3]=z.w[2]; z.w[2]=z.w[1];
    z.w[1]=z.w[0]; z.w[0]=0
    
    // Check for exponent underflow and compensate by shifting the product
    // Cut off the process at precision+2, since we can't really shift further
    
    var c_prov = Int(z.w[5])
    
    // Round using round-sticky words
    // If we spill into the next binade, correct
    let rind = roundboundIndex(rmode, s != 0, c_prov)
    if (lt128(bid_roundbound_128[rind].high, bid_roundbound_128[rind].low, z.w[4], z.w[3])) {
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

  
  /*****************************************************************************
   *  BID32_to_uint64_int
   ****************************************************************************/
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
      x_exp = (x & MASK_BINARY_EXPONENT2) >> 21; // biased
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
          let Ten20 = UInt128(w: [0x6bc75e2d63100000, 0x0000000000000005])
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
        
        // the top Ex bits of 10^(-x) are T* = bid_ten2mk128trunc[ind].w[0], e.g.
        // if x=1, T*=bid_ten2mk128trunc[0].w[0]=0x1999999999999999
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
    if !valid { pfpsc.insert(.invalidOperation); return Int.min }
    
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
        // Note: C1 * 10^(11-q) has 19 or 20 digits; 0x5000000000000000a, has 20
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
      } else { // if (exp > 0) => 1 <= exp <= 18, 1 <= q <= 7, 2 <= q + exp <= 20
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
  
  static func digitsIn(_ sig_x: UInt32) -> Int {
    // find power of 10 just greater than sig_x
    var tenPower = 10, digits = 1
    while sig_x >= tenPower { tenPower *= 10; digits += 1 }
    return digits
//    let tmp = Float(sig_x) // exact conversion
//    let x_nr_bits = 1 + Int(((UInt(tmp.bitPattern >> 23)) & 0xff) - 0x7f)
//    var q = Int(bid_nr_digits[x_nr_bits - 1].digits)
//    if q == 0 {
//      q = Int(bid_nr_digits[x_nr_bits - 1].digits1)
//      if UInt64(sig_x) >= bid_nr_digits[x_nr_bits - 1].threshold_lo {
//        q+=1
//      }
//    }
//    return q
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
  
  // the first entry of bid_nr_digits[i - 1] (where 1 <= i <= 113), indicates
  // the number of decimal digits needed to represent a binary number with i bits;
  // however, if a binary number of i bits may require either k or k + 1 decimal
  // digits, then the first entry of bid_nr_digits[i - 1] is 0; in this case if the
  // number is less than the value represented by the second and third entries
  // concatenated, then the number of decimal digits k is the fourth entry, else
  // the number of decimal digits is the fourth entry plus 1
//  struct DEC_DIGITS {
//    let digits: UInt
//    let threshold_hi:UInt64
//    let threshold_lo:UInt64
//    let digits1: UInt
//
//    init(_ d: UInt, _ hi: UInt64, _ lo: UInt64, _ d1: UInt) {
//      digits = d; threshold_hi = hi; threshold_lo = lo; digits1 = d1
//    }
//  }
  
  // bid_maskhigh128[] contains the mask to apply to the top 128 bits of the
  // 128x128-bit product in order to obtain the high bits of f2*
  // the 64-bit word order is L, H
  static let bid_maskhigh128: [UInt64] = [
    0x0000000000000000,    //  0 = 128 - 128 bits
    0x0000000000000000,    //  0 = 128 - 128 bits
    0x0000000000000000,    //  0 = 128 - 128 bits
    0x0000000000000007,    //  3 = 131 - 128 bits
    0x000000000000003f,    //  6 = 134 - 128 bits
    0x00000000000001ff,    //  9 = 137 - 128 bits
    0x0000000000001fff,    // 13 = 141 - 128 bits
    0x000000000000ffff,    // 16 = 144 - 128 bits
    0x000000000007ffff,    // 19 = 147 - 128 bits
    0x00000000007fffff,    // 23 = 151 - 128 bits
    0x0000000003ffffff,    // 26 = 154 - 128 bits
    0x000000001fffffff,    // 29 = 157 - 128 bits
    0x00000001ffffffff,    // 33 = 161 - 128 bits
    0x0000000fffffffff,    // 36 = 164 - 128 bits
    0x0000007fffffffff,    // 39 = 167 - 128 bits
    0x000007ffffffffff,    // 43 = 171 - 128 bits
    0x00003fffffffffff,    // 46 = 174 - 128 bits
    0x0001ffffffffffff,    // 49 = 177 - 128 bits
    0x001fffffffffffff,    // 53 = 181 - 128 bits
    0x00ffffffffffffff,    // 56 = 184 - 128 bits
    0x07ffffffffffffff,    // 59 = 187 - 128 bits
    0x7fffffffffffffff,    // 63 = 191 - 128 bits
    0x0000000000000003,    //  2 = 194 - 192 bits
    0x000000000000001f,    //  5 = 197 - 192 bits
    0x00000000000001ff,    //  9 = 201 - 192 bits
    0x0000000000000fff,    // 12 = 204 - 192 bits
    0x0000000000007fff,    // 15 = 207 - 192 bits
    0x000000000007ffff,    // 21 = 211 - 192 bits
    0x00000000003fffff,    // 22 = 214 - 192 bits
    0x0000000001ffffff,    // 25 = 217 - 192 bits
    0x000000000fffffff,    // 28 = 220 - 192 bits
    0x00000000ffffffff,    // 32 = 224 - 192 bits
    0x00000007ffffffff,    // 35 = 227 - 192 bits
    0x0000003fffffffff    // 38 = 230 - 192 bits
  ]
  
  
  // bid_onehalf128[] contains the high bits of 1/2 positioned correctly for
  // comparison with the high bits of f2*
  // the 64-bit word order is L, H
  static let bid_onehalf128: [UInt64] = [
    0x0000000000000000,    //  0 bits
    0x0000000000000000,    //  0 bits
    0x0000000000000000,    //  0 bits
    0x0000000000000004,    //  3 bits
    0x0000000000000020,    //  6 bits
    0x0000000000000100,    //  9 bits
    0x0000000000001000,    // 13 bits
    0x0000000000008000,    // 16 bits
    0x0000000000040000,    // 19 bits
    0x0000000000400000,    // 23 bits
    0x0000000002000000,    // 26 bits
    0x0000000010000000,    // 29 bits
    0x0000000100000000,    // 33 bits
    0x0000000800000000,    // 36 bits
    0x0000004000000000,    // 39 bits
    0x0000040000000000,    // 43 bits
    0x0000200000000000,    // 46 bits
    0x0001000000000000,    // 49 bits
    0x0010000000000000,    // 53 bits
    0x0080000000000000,    // 56 bits
    0x0400000000000000,    // 59 bits
    0x4000000000000000,    // 63 bits
    0x0000000000000002,    // 66 bits
    0x0000000000000010,    // 69 bits
    0x0000000000000100,    // 73 bits
    0x0000000000000800,    // 76 bits
    0x0000000000004000,    // 79 bits
    0x0000000000040000,    // 83 bits
    0x0000000000200000,    // 86 bits
    0x0000000001000000,    // 89 bits
    0x0000000008000000,    // 92 bits
    0x0000000080000000,    // 96 bits
    0x0000000400000000,    // 99 bits
    0x0000002000000000    // 102 bits
  ]
  
  static let bid_recip_scale : [Int] = [
    129 - 128,    // 1
    129 - 128,    // 1/10
    129 - 128,    // 1/10^2
    129 - 128,    // 1/10^3
    3,    // 131 - 128
    6,    // 134 - 128
    9,    // 137 - 128
    13,    // 141 - 128
    16,    // 144 - 128
    19,    // 147 - 128
    23,    // 151 - 128
    26,    // 154 - 128
    29,    // 157 - 128
    33,    // 161 - 128
    36,    // 164 - 128
    39,    // 167 - 128
    43,    // 171 - 128
    46,    // 174 - 128
    49,    // 177 - 128
    53,    // 181 - 128
    56,    // 184 - 128
    59,    // 187 - 128
    63,    // 191 - 128
    
    66,    // 194 - 128
    69,    // 197 - 128
    73,    // 201 - 128
    76,    // 204 - 128
    79,    // 207 - 128
    83,    // 211 - 128
    86,    // 214 - 128
    89,    // 217 - 128
    92,    // 220 - 128
    96,    // 224 - 128
    99,    // 227 - 128
    102,    // 230 - 128
    109,    // 237 - 128, 1/10^35
  ]
  
  
//  static func bid_reciprocals10_128(_ i: Int) -> UInt128 {
//    if i == 0 { return 1 }
//    let twoPower = bid_recip_scale[i]+128
//    return (UInt128(1) << twoPower) / bid_power10_table_128[i] + 1
//  }
  
  static let bid_reciprocals10_128: [UInt128] = [
    UInt128(w: [0, 0]),   // 0 extra digits
    UInt128(w: [0x3333333333333334, 0x3333333333333333]), // 1 extra digit
    UInt128(w: [0x51eb851eb851eb86, 0x051eb851eb851eb8]), // 2 extra digits
    UInt128(w: [0x3b645a1cac083127, 0x0083126e978d4fdf]), // 3 extra digits
    UInt128(w: [0x4af4f0d844d013aa, 0x00346dc5d6388659]), //  10^(-4) * 2^131
    UInt128(w: [0x08c3f3e0370cdc88, 0x0029f16b11c6d1e1]), //  10^(-5) * 2^134
    UInt128(w: [0x6d698fe69270b06d, 0x00218def416bdb1a]), //  10^(-6) * 2^137
    UInt128(w: [0xaf0f4ca41d811a47, 0x0035afe535795e90]), //  10^(-7) * 2^141
    UInt128(w: [0xbf3f70834acdaea0, 0x002af31dc4611873]), //  10^(-8) * 2^144
    UInt128(w: [0x65cc5a02a23e254d, 0x00225c17d04dad29]), //  10^(-9) * 2^147
    UInt128(w: [0x6fad5cd10396a214, 0x0036f9bfb3af7b75]), // 10^(-10) * 2^151
    UInt128(w: [0xbfbde3da69454e76, 0x002bfaffc2f2c92a]), // 10^(-11) * 2^154
    UInt128(w: [0x32fe4fe1edd10b92, 0x00232f33025bd422]), // 10^(-12) * 2^157
    UInt128(w: [0x84ca19697c81ac1c, 0x00384b84d092ed03]), // 10^(-13) * 2^161
    UInt128(w: [0x03d4e1213067bce4, 0x002d09370d425736]), // 10^(-14) * 2^164
    UInt128(w: [0x3643e74dc052fd83, 0x0024075f3dceac2b]), // 10^(-15) * 2^167
    UInt128(w: [0x56d30baf9a1e626b, 0x0039a5652fb11378]), // 10^(-16) * 2^171
    UInt128(w: [0x12426fbfae7eb522, 0x002e1dea8c8da92d]), // 10^(-17) * 2^174
    UInt128(w: [0x41cebfcc8b9890e8, 0x0024e4bba3a48757]), // 10^(-18) * 2^177
    UInt128(w: [0x694acc7a78f41b0d, 0x003b07929f6da558]), // 10^(-19) * 2^181
    UInt128(w: [0xbaa23d2ec729af3e, 0x002f394219248446]), // 10^(-20) * 2^184
    UInt128(w: [0xfbb4fdbf05baf298, 0x0025c768141d369e]), // 10^(-21) * 2^187
    UInt128(w: [0x2c54c931a2c4b759, 0x003c7240202ebdcb]), // 10^(-22) * 2^191
    UInt128(w: [0x89dd6dc14f03c5e1, 0x00305b66802564a2]), // 10^(-23) * 2^194
    UInt128(w: [0xd4b1249aa59c9e4e, 0x0026af8533511d4e]), // 10^(-24) * 2^197
    UInt128(w: [0x544ea0f76f60fd49, 0x003de5a1ebb4fbb1]), // 10^(-25) * 2^201
    UInt128(w: [0x76a54d92bf80caa1, 0x00318481895d9627]), // 10^(-26) * 2^204
    UInt128(w: [0x921dd7a89933d54e, 0x00279d346de4781f]), // 10^(-27) * 2^207
    UInt128(w: [0x8362f2a75b862215, 0x003f61ed7ca0c032]), // 10^(-28) * 2^211
    UInt128(w: [0xcf825bb91604e811, 0x0032b4bdfd4d668e]), // 10^(-29) * 2^214
    UInt128(w: [0x0c684960de6a5341, 0x00289097fdd7853f]), // 10^(-30) * 2^217
    UInt128(w: [0x3d203ab3e521dc34, 0x002073accb12d0ff]), // 10^(-31) * 2^220
    UInt128(w: [0x2e99f7863b696053, 0x0033ec47ab514e65]), // 10^(-32) * 2^224
    UInt128(w: [0x587b2c6b62bab376, 0x002989d2ef743eb7]), // 10^(-33) * 2^227
    UInt128(w: [0xad2f56bc4efbc2c5, 0x00213b0f25f69892]), // 10^(-34) * 2^230
    UInt128(w: [0x0f2abc9d8c9689d1, 0x01a95a5b7f87a0ef]), // 35 extra digits
  ]
  
  static let bid_packed_10000_zeros: [UInt8] = [
    0x3, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x3, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20,
    0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4,
    0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1,
    0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0,
    0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40,
    0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x2, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4, 0x10,
    0x40, 0x0, 0x1, 0x4, 0x20, 0x40, 0x0, 0x1, 0x4, 0x10, 0x40, 0x0, 0x1, 0x4,
    0x10, 0x40, 0x0
  ]
  
  
  static let bid_factors : [[Int8]] = [
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 2],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [5, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 2],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [6, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 2],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [5, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 2],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 3],  [1, 0],  [0, 0],  [7, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 2],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [5, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 2],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [6, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 2],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [5, 0],  [0, 2],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 3],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [8, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 2],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [5, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 2],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [6, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 2],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 2],
    [0, 0],  [5, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 3],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [7, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 2],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [5, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 2],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [6, 0],  [0, 0],  [1, 2],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 2],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [5, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 3],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [9, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 2],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [5, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 2],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 2],  [6, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 2],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [5, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 4],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [7, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 2],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [5, 0],  [0, 0],  [1, 0],  [0, 2],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 2],
    [0, 0],  [1, 0],  [0, 0],  [6, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 2],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [5, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 3],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [8, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 2],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [5, 2],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 2],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [6, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 2],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [5, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 3],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [4, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [7, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 2],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [4, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 2],  [1, 0],  [0, 0],  [5, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [4, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 2],
    [0, 0],  [3, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [6, 1],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [3, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 2],  [4, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [3, 0],  [0, 1],  [1, 0],  [0, 0],  [2, 0],  [0, 0],  [1, 1],
    [0, 0],  [5, 0],  [0, 0],  [1, 0],  [0, 1],  [2, 0],  [0, 0],  [1, 0],  [0, 0],  [3, 3],
    [0, 0],  [1, 0],  [0, 0],  [2, 0],  [0, 1],  [1, 0],  [0, 0],  [4, 0],  [0, 0],  [1, 1],
    [0, 0],  [2, 0],  [0, 0],  [1, 0],  [0, 1],  [3, 0],  [0, 0],  [1, 0],  [0, 0],  [2, 1],
    [0, 0],  [1, 0],  [0, 0],  [10, 0]
  ]
  
  
  static let bid_recip_scale32 : [UInt8] = [
    1, 1, 3, 7, 9, 14, 18, 21, 25
  ]
  
  
  static func bid_reciprocals10_32(_ i: Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = bid_recip_scale32[i]+32
    return (UInt64(1) << twoPower) / UInt64(bid_power10_table_128[i]) + 1
  }
  
//  static let bid_bid_reciprocals10_32: [UInt64] = [
//    1, //dummy,
//    0x33333334,
//    0x147AE148,
//    0x20C49BA6,
//    0x346DC5D7,
//    0x29F16B12,
//    0x431BDE83,
//    0x35AFE536,
//    0x55E63B89
//  ]
  
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
  
  
  static let bid_multipliers1_binary64: [UInt256] = [
    UInt256(w: [1837554224478941466, 10276842184138466546, 11651621577776737258, 7754513766366540701]),
    UInt256(w: [5760157408726726321, 11034712383513929495, 9588106495324154738, 4846571103979087938]),
    UInt256(w: [2588510742481019997, 4570018442537636061, 2761761082300417615, 6058213879973859923]),
    UInt256(w: [7847324446528662900, 1100837034744657172, 17287259408157685731, 7572767349967324903]),
    UInt256(w: [14127949815935190120, 16828924211211268396, 17722066157739635437, 4732979593729578064]),
    UInt256(w: [17659937269918987650, 7201097208731921783, 3705838623464992681, 5916224492161972581]),
    UInt256(w: [17463235568971346659, 13613057529342290133, 9243984297758628755, 7395280615202465726]),
    UInt256(w: [13220365239820785614, 6202317946625237381, 1165804167671755068, 4622050384501541079]),
    UInt256(w: [2690398494493818305, 7752897433281546727, 15292313264871857547, 5777562980626926348]),
    UInt256(w: [17198056173399436594, 5079435773174545504, 668647507380270318, 7221953725783657936]),
    UInt256(w: [3050826143039744126, 15572666753322957689, 835809384225337897, 9027442157229572420]),
    UInt256(w: [13435981385468309839, 2815387693185766699, 9745752901995611994, 5642151348268482762]),
    UInt256(w: [12183290713407999394, 12742606653336984182, 2958819090639739184, 7052689185335603453]),
    UInt256(w: [6005741354905223435, 15928258316671230228, 8310209881727061884, 8815861481669504316]),
    UInt256(w: [12976960383670540455, 731789411064743084, 14417253212934189486, 5509913426043440197]),
    UInt256(w: [16221200479588175569, 10138108800685704663, 4186508460885573145, 6887391782554300247]),
    UInt256(w: [15664814581057831557, 17284322019284518733, 621449557679578527, 8609239728192875309]),
    UInt256(w: [12096352122374838675, 17720230289693906064, 2694248982763430531, 5380774830120547068]),
    UInt256(w: [15120440152968548344, 17538601843689994676, 3367811228454288164, 6725968537650683835]),
    UInt256(w: [453806117501133814, 3476508230902941730, 18044822090850023918, 8407460672063354793]),
    UInt256(w: [4895314841865596538, 16007875699596502293, 4360484779140183092, 5254662920039596746]),
    UInt256(w: [10730829570759383576, 1563100550786076250, 14673978010780004674, 6568328650049495932]),
    UInt256(w: [4190164926594453662, 11177247725337371121, 18342472513475005842, 8210410812561869915]),
    UInt256(w: [14148068125190003299, 11597465846763244854, 9158202311708184699, 5131506757851168697]),
    UInt256(w: [8461713119632728315, 9885146290026668164, 16059438908062618778, 6414383447313960871]),
    UInt256(w: [10577141399540910394, 3133060825678559397, 15462612616650885569, 8017979309142451089]),
    UInt256(w: [8916556383926762949, 13487378062117569383, 2746603857765721624, 5011237068214031931]),
    UInt256(w: [6534009461481065782, 16859222577646961729, 17268312877489315742, 6264046335267539913]),
    UInt256(w: [12779197845278720131, 11850656185203926353, 7750333041579480966, 7830057919084424892]),
    UInt256(w: [1069469625658118226, 2794974097325066067, 14067330187841951412, 4893786199427765557]),
    UInt256(w: [15171895087354811494, 3493717621656332583, 3749104679520275553, 6117232749284706947]),
    UInt256(w: [14353182840766126464, 8978833045497803633, 74694830972956537, 7646540936605883684]),
    UInt256(w: [2053210247837747184, 17140985699504597031, 9270056306212873643, 4779088085378677302]),
    UInt256(w: [16401570865079347692, 16814546105953358384, 2364198345911316246, 5973860106723346628]),
    UInt256(w: [2055219507639632999, 11794810595586922173, 2955247932389145308, 7467325133404183285]),
    UInt256(w: [3590355201488464576, 16595128659096602166, 4152872966956909769, 4667078208377614553]),
    UInt256(w: [13711316038715356528, 6908852768588588995, 9802777227123525116, 5833847760472018191]),
    UInt256(w: [12527459029966807756, 8636065960735736244, 7641785515477018491, 7292309700590022739]),
    UInt256(w: [15659323787458509695, 6183396432492282401, 4940545875918885210, 9115387125737528424]),
    UInt256(w: [2869548339520486704, 8476308788735064405, 3087841172449303256, 5697116953585955265]),
    UInt256(w: [8198621442827996284, 10595385985918830506, 8471487483989016974, 7121396191982444081]),
    UInt256(w: [1024904766680219546, 4020860445543762325, 15201045373413659122, 8901745239978055101]),
    UInt256(w: [2946408488388831169, 7124723796892239357, 11806496367597230903, 5563590774986284438]),
    UInt256(w: [8294696628913426865, 4294218727687911292, 5534748422641762821, 6954488468732855548]),
    UInt256(w: [10368370786141783581, 9979459428037277019, 6918435528302203526, 8693110585916069435]),
    UInt256(w: [4174388732124920786, 1625476124095910233, 2018179195975183252, 5433194116197543397]),
    UInt256(w: [9829671933583538887, 2031845155119887791, 7134410013396366969, 6791492645246929246]),
    UInt256(w: [7675403898552035704, 7151492462327247643, 18141384553600234519, 8489365806558661557]),
    UInt256(w: [2491284427381328363, 11387211816595611633, 13644208355213840526, 5305853629099163473]),
    UInt256(w: [7725791552654048358, 5010642733889738733, 3220202388735136946, 6632317036373954342]),
    UInt256(w: [14268925459244948351, 15486675454216949224, 13248625022773696990, 8290396295467442927]),
    UInt256(w: [8918078412028092720, 5067486140458205361, 15197919666874642475, 5181497684667151829]),
    UInt256(w: [15759284033462503804, 1722671657145368797, 5162341528311139382, 6476872105833939787]),
    UInt256(w: [5864046986545966042, 11376711608286486805, 1841240891961536323, 8096090132292424734]),
    UInt256(w: [5970872375804922729, 4804601745965360301, 14985833612758123914, 5060056332682765458]),
    UInt256(w: [12075276488183541315, 15229124219311476184, 9508919979092879084, 6325070415853456823]),
    UInt256(w: [15094095610229426643, 589661200429793614, 7274463955438710952, 7906338019816821029]),
    UInt256(w: [4822123737966003748, 368538250268621009, 6852382981362888297, 4941461262385513143]),
    UInt256(w: [10639340690884892589, 5072358831263164165, 3953792708276222467, 6176826577981891429]),
    UInt256(w: [17910861882033503640, 1728762520651567302, 9553926903772665988, 7721033222477364286]),
    UInt256(w: [6582602657843551871, 10303848612262005372, 1359518296430528338, 4825645764048352679]),
    UInt256(w: [8228253322304439839, 3656438728472730907, 15534455925820324135, 6032057205060440848]),
    UInt256(w: [5673630634453161895, 18405606465873077346, 971325833565853552, 7540071506325551061]),
    UInt256(w: [8157705164960614088, 11503504041170673341, 2912921655192352422, 4712544691453469413]),
    UInt256(w: [14808817474628155514, 5156008014608565868, 8252838087417828432, 5890680864316836766]),
    UInt256(w: [64277769575642777, 6445010018260707336, 1092675572417509732, 7363351080396045958]),
    UInt256(w: [80347211969553471, 8056262522825884170, 10589216502376662973, 9204188850495057447]),
    UInt256(w: [4661903025908358824, 7341007085979871558, 13535789341626496214, 5752618031559410904]),
    UInt256(w: [15050750819240224337, 18399630894329615255, 16919736677033120267, 7190772539449263630]),
    UInt256(w: [14201752505622892517, 18387852599484631165, 11926298809436624526, 8988465674311579538]),
    UInt256(w: [11181938325228001776, 6880721856250506574, 12065622774325278233, 5617791046444737211]),
    UInt256(w: [4754050869680226411, 13212588338740521122, 10470342449479209887, 7022238808055921514]),
    UInt256(w: [15165935623955058822, 11904049404998263498, 3864556024994236551, 8777798510069901893]),
    UInt256(w: [14090395783399299668, 14357559905764996542, 4721190524835091796, 5486124068793688683]),
    UInt256(w: [8389622692394348777, 17946949882206245678, 1289802137616476841, 6857655085992110854]),
    UInt256(w: [1263656328638160163, 8598629297475643386, 10835624708875371860, 8572068857490138567]),
    UInt256(w: [5401471223826238006, 14597515347777052924, 13689794470688189268, 5357543035931336604]),
    UInt256(w: [6751839029782797507, 18246894184721316155, 17112243088360236585, 6696928794914170755]),
    UInt256(w: [3828112768801108980, 8973559675619481482, 16778617842022907828, 8371160993642713444]),
    UInt256(w: [7004256498928081017, 14831846834116951734, 1263264114409541584, 5231975621026695903]),
    UInt256(w: [17978692660514877079, 93064468936638051, 15414138198294090693, 6539969526283369878]),
    UInt256(w: [17861679807216208444, 4728016604598185468, 10044300711012837558, 8174961907854212348]),
    UInt256(w: [1940177842655354470, 16790068433156029630, 15501059981237799281, 5109351192408882717]),
    UInt256(w: [11648594340173968895, 7152527486162873325, 5541266921265085390, 6386688990511103397]),
    UInt256(w: [725684869935297407, 18164031394558367465, 11538269670008744641, 7983361238138879246]),
    UInt256(w: [11982768089778030640, 4434990593957897809, 2599732525328077497, 4989600773836799529]),
    UInt256(w: [1143402056940374587, 10155424260874760166, 7861351675087484775, 6237000967295999411]),
    UInt256(w: [10652624608030244042, 8082594307666062303, 5215003575431968065, 7796251209119999264]),
    UInt256(w: [13575419407659984382, 16580836488359758699, 3259377234644980040, 4872657005699999540]),
    UInt256(w: [12357588241147592574, 2279301536740146758, 4074221543306225051, 6090821257124999425]),
    UInt256(w: [6223613264579714909, 16684184976207347160, 9704462947560169217, 7613526571406249281]),
    UInt256(w: [3889758290362321819, 3510086582488510119, 17594504388293575521, 4758454107128905800]),
    UInt256(w: [250511844525514369, 8999294246538025553, 3546386411657417785, 5948067633911132251]),
    UInt256(w: [4924825824084280865, 15860803826599919845, 18268041069853935943, 7435084542388915313]),
    UInt256(w: [5383859149266369493, 16830531419266031759, 4499996641017628108, 4646927838993072071]),
    UInt256(w: [2118137918155573962, 2591420200372988083, 1013309782844647232, 5808659798741340089]),
    UInt256(w: [16482730452976631164, 3239275250466235103, 5878323246983196944, 7260824748426675111]),
    UInt256(w: [15991727047793401051, 4049094063082793879, 2736218040301608276, 9076030935533343889]),
    UInt256(w: [16912358432511957513, 11754055826281521982, 13239351321256974932, 5672519334708339930]),
    UInt256(w: [11917076003785171083, 14692569782851902478, 7325817114716442857, 7090649168385424913]),
    UInt256(w: [5672972967876688046, 4530654173282714386, 13768957411822941476, 8863311460481781141]),
    UInt256(w: [8157294123350317933, 12055030895156472299, 10911441391603032374, 5539569662801113213]),
    UInt256(w: [5584931635760509512, 5845416582090814566, 18250987757931178372, 6924462078501391516]),
    UInt256(w: [16204536581555412698, 7306770727613518207, 4366990623704421349, 8655577598126739396]),
    UInt256(w: [17045364391113214793, 6872574713972142831, 11952741176670039151, 5409735998829212122]),
    UInt256(w: [16695019470464130587, 3979032374037790635, 5717554433982773131, 6762169998536515153]),
    UInt256(w: [16257088319652775329, 362104449119850390, 11758629060905854318, 8452712498170643941]),
    UInt256(w: [5548994181355596677, 14061373335982070206, 9654986172279852900, 5282945311356652463]),
    UInt256(w: [16159614763549271654, 17576716669977587757, 7457046696922428221, 6603681639195815579]),
    UInt256(w: [6364460399154425855, 8135837782189820985, 4709622352725647373, 8254602048994769474]),
    UInt256(w: [15507002795539985920, 7390741623082332067, 7555199988880917512, 5159126280621730921]),
    UInt256(w: [14772067475997594496, 9238427028852915084, 14055686004528534794, 6448907850777163651]),
    UInt256(w: [18340271287441503, 2324661749211368048, 12957921487233280589, 8061134813471454564]),
    UInt256(w: [11462669554650940, 3758756602470798982, 17322072966375576176, 5038209258419659102]),
    UInt256(w: [9237700373798089483, 4698445753088498727, 12429219171114694412, 6297761573024573878]),
    UInt256(w: [6935439448820223949, 5873057191360623409, 6313151927038592207, 7872201966280717348]),
    UInt256(w: [15863864701581109728, 10588189772241471486, 13169091991253895937, 4920126228925448342]),
    UInt256(w: [10606458840121611352, 17846923233729227262, 7237992952212594113, 6150157786156810428]),
    UInt256(w: [4034701513297238382, 8473595986879370366, 9047491190265742642, 7687697232696013035]),
    UInt256(w: [16356746501092937701, 9907683510226994382, 3348838984702395199, 4804810770435008147]),
    UInt256(w: [11222561089511396318, 7772918369356355074, 18021106786160157711, 6006013463043760183]),
    UInt256(w: [4804829325034469590, 5104461943268055939, 17914697464272809235, 7507516828804700229]),
    UInt256(w: [697175318932849542, 884445705328841010, 13502528924384199724, 4692198018002937643]),
    UInt256(w: [10094841185520837735, 1105557131661051262, 12266475137052861751, 5865247522503672054]),
    UInt256(w: [3395179445046271361, 15217004469858477790, 6109721884461301380, 7331559403129590068]),
    UInt256(w: [13467346343162615009, 574511513613545621, 7637152355576626726, 9164449253911987585]),
    UInt256(w: [10722934473690328333, 14194127751290629725, 16302435268303861463, 5727780783694992240]),
    UInt256(w: [18015354110540298320, 13130973670685899252, 1931300011670275213, 7159725979618740301]),
    UInt256(w: [4072448564465821284, 2578659033075210354, 7025811033015231921, 8949657474523425376]),
    UInt256(w: [7156966371218526206, 13140876941740476231, 4391131895634519950, 5593535921577140860]),
    UInt256(w: [4334521945595769854, 7202724140320819481, 5488914869543149938, 6991919901971426075]),
    UInt256(w: [10029838450422100221, 18226777212255800159, 2249457568501549518, 8739899877464282594]),
    UInt256(w: [13186178059154894494, 6780049739232487195, 6017596998740856353, 5462437423415176621]),
    UInt256(w: [11871036555516230214, 13086748192467996898, 12133682266853458345, 6828046779268970776]),
    UInt256(w: [5615423657540511959, 2523377185302832411, 15167102833566822932, 8535058474086213470]),
    UInt256(w: [1203796776749126023, 10800482777669046065, 4867753252551876428, 5334411546303883419]),
    UInt256(w: [6116431989363795432, 13500603472086307581, 1473005547262457631, 6668014432879854274]),
    UInt256(w: [12257226005132132194, 12264068321680496572, 11064628970932847847, 8335018041099817842]),
    UInt256(w: [16884138290062358430, 14582571728691392213, 11527079125260417808, 5209386275687386151]),
    UInt256(w: [7270114807295784325, 18228214660864240267, 9797162888148134356, 6511732844609232689]),
    UInt256(w: [4475957490692342502, 4338524252370748718, 16858139628612555850, 8139666055761540861]),
    UInt256(w: [16632531486964877776, 7323263676159105852, 12842180277096541358, 5087291284850963038]),
    UInt256(w: [2343920284996545604, 18377451632053658124, 6829353309515900889, 6359114106063703798]),
    UInt256(w: [2929900356245682005, 9136756484784908943, 17760063673749651920, 7948892632579629747]),
    UInt256(w: [8748716750294633109, 5710472802990568089, 8794196786879838498, 4968057895362268592]),
    UInt256(w: [15547581956295679290, 16361463040592985919, 10992745983599798122, 6210072369202835740]),
    UInt256(w: [14822791426942211209, 11228456763886456591, 13740932479499747653, 7762590461503544675]),
    UInt256(w: [16181773669479963862, 9323628486642729321, 6282239790473648331, 4851619038439715422]),
    UInt256(w: [6392159031567791115, 7042849589876023748, 17076171774946836222, 6064523798049644277]),
    UInt256(w: [7990198789459738893, 18026934024199805493, 7510156663401381565, 7580654747562055347]),
    UInt256(w: [7299717252626030761, 13572676774338572385, 2388004905412169526, 4737909217226284592]),
    UInt256(w: [13736332584209926355, 7742473931068439673, 2985006131765211908, 5922386521532855740]),
    UInt256(w: [3335357674980244231, 9678092413835549592, 3731257664706514885, 7402983151916069675]),
    UInt256(w: [2084598546862652645, 8354650767860912447, 26193031227877851, 4626864469947543547]),
    UInt256(w: [16440806238860479518, 5831627441398752654, 13867799344317011026, 5783580587434429433]),
    UInt256(w: [11327635761720823589, 16512906338603216626, 3499691125114100070, 7229475734293036792]),
    UInt256(w: [4936172665296253678, 11417760886399244975, 4374613906392625088, 9036844667866295990]),
    UInt256(w: [10002636943451240405, 7136100553999528109, 16569191746777554392, 5648027917416434993]),
    UInt256(w: [17114982197741438410, 8920125692499410136, 6876431628189779278, 7060034896770543742]),
    UInt256(w: [2946983673467246397, 1926785078769486863, 17818911572091999906, 8825043620963179677]),
    UInt256(w: [8759393823558110854, 5815926692658317193, 13442662741771193893, 5515652263101987298]),
    UInt256(w: [15560928297875026471, 11881594384250284395, 7579956390359216558, 6894565328877484123]),
    UInt256(w: [14839474353916395185, 5628620943458079686, 4863259469521632794, 8618206661096855154]),
    UInt256(w: [4662985452770359087, 8129574108088687708, 7651223186878408400, 5386379163185534471]),
    UInt256(w: [5828731815962948858, 10161967635110859635, 4952342965170622596, 6732973953981918089]),
    UInt256(w: [2674228751526298169, 12702459543888574544, 10802114724890666149, 8416217442477397611]),
    UInt256(w: [1671392969703936356, 10244880224144053042, 4445478693842972391, 5260135901548373507]),
    UInt256(w: [11312613248984696253, 8194414261752678398, 945162348876327585, 6575169876935466884]),
    UInt256(w: [4917394524376094508, 14854703845618235902, 1181452936095409481, 8218962346169333605]),
    UInt256(w: [16908429633017222779, 2366660875870315582, 3044251094273324878, 5136851466355833503]),
    UInt256(w: [11912165004416752666, 12181698131692670286, 17640371923123819809, 6421064332944791878]),
    UInt256(w: [5666834218666165025, 1392064609333674146, 12827092867049998954, 8026330416180989848]),
    UInt256(w: [8153457405093741045, 5481726399260934245, 8016933041906249346, 5016456510113118655]),
    UInt256(w: [14803507774794564210, 16075530035930943614, 5409480283955423778, 6270570637641398319]),
    UInt256(w: [9281012681638429454, 10871040508058903710, 2150164336516891819, 7838213297051747899]),
    UInt256(w: [1188946907596630505, 4488557308323120867, 17484753774818915051, 4898883310657342436]),
    UInt256(w: [15321241689777951843, 999010616976513179, 3409198144814092198, 6123604138321678046]),
    UInt256(w: [14539866093795051900, 10472135308075417282, 13484869717872391055, 7654505172902097557]),
    UInt256(w: [13699102327049295341, 13462613595188217657, 10733886582883938361, 4784065733063810973]),
    UInt256(w: [3288819853529455465, 2993208938703108360, 18029044247032310856, 5980082166329763716]),
    UInt256(w: [4111024816911819331, 3741511173378885450, 4089561235080836954, 7475102707912204646]),
    UInt256(w: [7181076528997274986, 6950130501789191310, 16391033827207686808, 4671939192445127903]),
    UInt256(w: [18199717698101369540, 8687663127236489137, 15877106265582220606, 5839923990556409879]),
    UInt256(w: [8914589067344548213, 1636206872190835614, 15234696813550387854, 7299904988195512349]),
    UInt256(w: [1919864297325909458, 11268630627093320326, 5208312961655821105, 9124881235244390437]),
    UInt256(w: [15034973241110857124, 125365114292243347, 5561038610248582143, 5703050772027744023]),
    UInt256(w: [14182030532961183500, 13991764448147467896, 2339612244383339774, 7128813465034680029]),
    UInt256(w: [17727538166201479375, 8266333523329559062, 7536201323906562622, 8911016831293350036]),
    UInt256(w: [6468025335448536706, 554772433653586510, 13933497864296377447, 5569385519558343772]),
    UInt256(w: [17308403706165446690, 14528523597349146849, 17416872330370471808, 6961731899447929715]),
    UInt256(w: [7800446577424644651, 18160654496686433562, 17159404394535701856, 8702164874309912144]),
    UInt256(w: [9486965129317790811, 11350409060429020976, 10724627746584813660, 5438853046443695090]),
    UInt256(w: [11858706411647238513, 14188011325536276220, 4182412646376241267, 6798566308054618863]),
    UInt256(w: [14823383014559048142, 13123328138492957371, 616329789542913680, 8498207885068273579]),
    UInt256(w: [6958771374885711137, 8202080086558098357, 16526107182960178714, 5311379928167670986]),
    UInt256(w: [13310150237034526825, 1029228071342847138, 11434261941845447585, 6639224910209588733]),
    UInt256(w: [7414315759438382723, 5898221107605946827, 457769372024645769, 8299031137761985917]),
    UInt256(w: [2328104340435295250, 15215603238322186527, 2591948866729097557, 5186894461101241198]),
    UInt256(w: [16745188480826282774, 5184445992620569446, 12463308120266147755, 6483618076376551497]),
    UInt256(w: [11708113564178077660, 1868871472348323904, 1744077095050520982, 8104522595470689372]),
    UInt256(w: [7317570977611298537, 15003102725499866152, 10313420221261351421, 5065326622169180857]),
    UInt256(w: [9146963722014123172, 4918820351592668978, 17503461295004077181, 6331658277711476071]),
    UInt256(w: [2210332615662878157, 10760211457918224127, 17267640600327708572, 7914572847139345089]),
    UInt256(w: [8298986912430380704, 15948504198053665887, 3874746347563736001, 4946608029462090681]),
    UInt256(w: [5762047622110587976, 6100572192284918647, 9455118952882057906, 6183260036827613351]),
    UInt256(w: [2590873509210847066, 16849087277210924117, 7207212672675184478, 7729075046034516689]),
    UInt256(w: [3925138952470473368, 5918993529829439669, 16033722966490460059, 4830671903771572930]),
    UInt256(w: [9518109709015479614, 2787055893859411682, 10818781671258299266, 6038339879714466163]),
    UInt256(w: [2674265099414573710, 12707191904179040411, 8911791070645486178, 7547924849643082704]),
    UInt256(w: [17812316751629966233, 12553680958539288160, 5569869419153428861, 4717453031026926690]),
    UInt256(w: [3818651865827906175, 1857043142891946489, 16185708810796561885, 5896816288783658362]),
    UInt256(w: [9385000850712270622, 6932989947042321015, 11008763976640926548, 7371020360979572953]),
    UInt256(w: [7119565044962950374, 8666237433802901269, 18372640989228546089, 9213775451224466191]),
    UInt256(w: [6755571162315537936, 16945613442195283053, 18400429645908923161, 5758609657015291369]),
    UInt256(w: [13056149971321810324, 7346958747461940104, 9165479002103990240, 7198262071269114212]),
    UInt256(w: [16320187464152262904, 9183698434327425130, 11456848752629987800, 8997827589086392765]),
    UInt256(w: [14811803183522552219, 5739811521454640706, 9466373479607436327, 5623642243178995478]),
    UInt256(w: [9291381942548414466, 2563078383390912979, 2609594812654519601, 7029552803973744348]),
    UInt256(w: [7002541409758130179, 7815533997666029128, 3261993515818149501, 8786941004967180435]),
    UInt256(w: [4376588381098831362, 7190551757754962157, 18179647011882201102, 5491838128104487771]),
    UInt256(w: [10082421494800927106, 18211561734048478504, 18112872746425363473, 6864797660130609714]),
    UInt256(w: [12603026868501158883, 8929394112278434418, 13417718896176928534, 8580997075163262143]),
    UInt256(w: [12488577811240612206, 969185301746633607, 15303603337751662190, 5363123171977038839]),
    UInt256(w: [10999036245623377353, 10434853664038067817, 14517818153762189833, 6703903964971298549]),
    UInt256(w: [18360481325456609595, 17655253098474972675, 4312214636920573579, 8379879956214123187]),
    UInt256(w: [9169457819196687045, 8728690177333163970, 389291138861664535, 5237424972633826992]),
    UInt256(w: [2238450237141082998, 6299176703239067059, 486613923577080669, 6546781215792283740]),
    UInt256(w: [16633120851708517460, 12485656897476221727, 608267404471350836, 8183476519740354675]),
    UInt256(w: [17313229559958905269, 17026907597777414387, 16521068192290451936, 5114672824837721671]),
    UInt256(w: [17029850931521243682, 2836890423512216368, 16039649221935677017, 6393341031047152089]),
    UInt256(w: [2840569590692002986, 8157799047817658365, 6214503472137432559, 7991676288808940112]),
    UInt256(w: [4081199003396195818, 12016153432527118334, 3884064670085895349, 4994797680505587570]),
    UInt256(w: [14324870791100020581, 1185133735376734205, 14078452874462144995, 6243497100631984462]),
    UInt256(w: [4071030433592862014, 15316475224503081469, 8374694056222905435, 7804371375789980578]),
    UInt256(w: [4850237030209232711, 7266954006100731966, 9845869803566703801, 4877732109868737861]),
    UInt256(w: [15286168324616316697, 13695378526053302861, 16919023272885767655, 6097165137335922326]),
    UInt256(w: [5272652350488232159, 12507537139139240673, 11925407054252433761, 7621456421669902908]),
    UInt256(w: [14824622765123614859, 899681684320943564, 16676751445762546909, 4763410263543689317]),
    UInt256(w: [84034382694966958, 5736288123828567360, 7010881251921019924, 5954262829429611647]),
    UInt256(w: [105042978368708697, 7170360154785709200, 4151915546473887001, 7442828536787014559]),
    UInt256(w: [65651861480442936, 16010690142809538010, 9512476244187261231, 4651767835491884099]),
    UInt256(w: [9305436863705329478, 15401676660084534608, 7278909286806688635, 5814709794364855124]),
    UInt256(w: [11631796079631661847, 14640409806678280356, 9098636608508360794, 7268387242956068905]),
    UInt256(w: [14539745099539577309, 9077140221493074637, 15984981779062838897, 9085484053695086131]),
    UInt256(w: [11393183696425929770, 17202427684501641408, 7684770602700580358, 5678427533559428832]),
    UInt256(w: [14241479620532412213, 12279662568772275952, 9605963253375725448, 7098034416949286040]),
    UInt256(w: [17801849525665515266, 15349578210965344940, 12007454066719656810, 8872543021186607550]),
    UInt256(w: [1902783916686171233, 14205172400280728492, 2892972773272397602, 5545339388241629719]),
    UInt256(w: [2378479895857714042, 8533093463496134807, 17451274021872660715, 6931674235302037148]),
    UInt256(w: [16808157925104306264, 6054680810942780604, 3367348453631274278, 8664592794127546436]),
    UInt256(w: [1281726666335415607, 17619233562121401590, 11327964820374322231, 5415370496329716522]),
    UInt256(w: [10825530369774045317, 17412355934224364083, 4936583988613126981, 6769213120412145653]),
    UInt256(w: [8920226943790168742, 7930386862498291392, 10782416004193796631, 8461516400515182066]),
    UInt256(w: [5575141839868855464, 11874020816702513976, 11350696021048510798, 5288447750321988791]),
    UInt256(w: [6968927299836069330, 5619153984023366662, 9576684007883250594, 6610559687902485989]),
    UInt256(w: [17934531161649862470, 16247314516883984135, 16582541028281451146, 8263199609878107486]),
    UInt256(w: [18126611003672245900, 14766257591479877988, 5752402124248519062, 5164499756173817179]),
    UInt256(w: [4211519680880755759, 9234449952495071678, 2578816636883260924, 6455624695217271474]),
    UInt256(w: [14487771637955720506, 11543062440618839597, 12446892832958851963, 8069530869021589342]),
    UInt256(w: [11360700282936019269, 4908571016173080796, 3167622002171894573, 5043456793138493339]),
    UInt256(w: [14200875353670024086, 10747399788643738899, 17794585557997031928, 6304320991423116673]),
    UInt256(w: [13139408173660142203, 13434249735804673624, 8408173892214126198, 7880401239278895842]),
    UInt256(w: [8212130108537588877, 3784720066450533111, 9866794701061216778, 4925250774549309901]),
    UInt256(w: [5653476617244598192, 13954272119917942197, 16945179394753908876, 6156563468186637376]),
    UInt256(w: [11678531789983135644, 17442840149897427746, 2734730169732834479, 7695704335233296721]),
    UInt256(w: [11910768387166847682, 17819304121326974197, 13238421402151491309, 4809815209520810450]),
    UInt256(w: [1053402428676395890, 8439072096376554035, 7324654715834588329, 6012269011901013063]),
    UInt256(w: [15151811091127658574, 15160526138898080447, 4544132376365847507, 7515336264876266329]),
    UInt256(w: [16387410959595868465, 7169485827597606327, 14369297781297124452, 4697085165547666455]),
    UInt256(w: [15872577681067447677, 8961857284497007909, 13349936208194017661, 5871356456934583069]),
    UInt256(w: [6005664046052145885, 15814007624048647791, 2852362204960358364, 7339195571168228837]),
    UInt256(w: [2895394039137794452, 1320765456351258123, 8177138774627835860, 9173994463960286046]),
    UInt256(w: [17950522338956979196, 10048850447074312134, 499025715715009508, 5733746539975178779]),
    UInt256(w: [13214780886841448187, 12561063058842890168, 14458840199925925597, 7167183174968973473]),
    UInt256(w: [16518476108551810234, 1866270768271448998, 4238492194625243285, 8958978968711216842]),
    UInt256(w: [5712361549417493492, 3472262239383349576, 7260743640068164957, 5599361855444510526]),
    UInt256(w: [7140451936771866865, 8952013817656574874, 18299301586939982004, 6999202319305638157]),
    UInt256(w: [18148936957819609390, 11190017272070718592, 9039068928392813793, 8749002899132047697]),
    UInt256(w: [11343085598637255869, 76231767403117264, 17178633126313978381, 5468126811957529810]),
    UInt256(w: [14178856998296569836, 4706975727681284484, 12249919371037697168, 6835158514946912263]),
    UInt256(w: [17723571247870712295, 5883719659601605605, 10700713195369733556, 8543948143683640329]),
    UInt256(w: [13383075039132889136, 12900696824105779311, 18217160793174553232, 5339967589802275205]),
    UInt256(w: [12117157780488723516, 16125871030132224139, 8936392936186027828, 6674959487252844007]),
    UInt256(w: [10534761207183516491, 1710594713955728558, 6558805151805146882, 8343699359066055009]),
    UInt256(w: [1972539736062309903, 5680807714649718253, 15628468265946686561, 5214812099416284380]),
    UInt256(w: [7077360688505275283, 11712695661739535720, 1088841258723806585, 6518515124270355476]),
    UInt256(w: [8846700860631594104, 805811521892255938, 1361051573404758232, 8148143905337944345]),
    UInt256(w: [10140874056322134219, 503632201182659961, 12379872279446443655, 5092589940836215215]),
    UInt256(w: [17287778588830055677, 14464598306760488663, 10863154330880666664, 6365737426045269019]),
    UInt256(w: [16998037217610181693, 18080747883450610829, 8967256895173445426, 7957171782556586274]),
    UInt256(w: [12929616270220057510, 15912153445584019672, 10216221577910791295, 4973232364097866421]),
    UInt256(w: [16162020337775071888, 15278505788552636686, 17381962990815877023, 6216540455122333026]),
    UInt256(w: [10979153385364064051, 14486446217263407954, 12504081701665070471, 7770675568902916283]),
    UInt256(w: [11473656884279927936, 15971557913430711827, 5509208054326975092, 4856672230564322677]),
    UInt256(w: [9730385086922522016, 1517703318078838168, 11498196086336106770, 6070840288205403346]),
    UInt256(w: [12162981358653152520, 11120501184453323518, 5149373071065357654, 7588550360256754183]),
    UInt256(w: [2990177330730832421, 2338627221855939295, 10135887197056930390, 4742843975160471364]),
    UInt256(w: [17572779718695704238, 12146656064174699926, 12669858996321162987, 5928554968950589205]),
    UInt256(w: [12742602611514854490, 10571634061790987004, 2002265690119290022, 7410693711188236507]),
    UInt256(w: [17187498669051559864, 1995585270191978973, 17392317120820413928, 4631683569492647816]),
    UInt256(w: [7649315281032286118, 2494481587739973717, 3293652327315965794, 5789604461865809771]),
    UInt256(w: [14173330119717745552, 12341474021529742954, 17952123464427120954, 7237005577332262213]),
    UInt256(w: [8493290612792406132, 6203470490057402885, 8605096275251737481, 9046256971665327767]),
    UInt256(w: [7614149642208947785, 15406384102354346563, 12295714199673417781, 5653910607290829854]),
    UInt256(w: [4906001034333796827, 5422922072660769492, 6146270712736996419, 7067388259113537318]),
    UInt256(w: [6132501292917246033, 2166966572398573961, 16906210427776021332, 8834235323891921647]),
    UInt256(w: [15362028354141748531, 10577726144603884533, 17483910545001095188, 5521397077432451029]),
    UInt256(w: [5367477387395021951, 13222157680754855667, 8019830125969205273, 6901746346790563787]),
    UInt256(w: [2097660715816389535, 2692639045661405872, 5413101639034118688, 8627182933488204734]),
    UInt256(w: [1311037947385243460, 1682899403538378670, 17218246579678487892, 5391989333430127958]),
    UInt256(w: [10862169471086330132, 2103624254422973337, 12299436187743334057, 6739986666787659948]),
    UInt256(w: [18189397857285300569, 7241216336456104575, 15374295234679167571, 8424983333484574935]),
    UInt256(w: [18285902688444394712, 2219917201071371407, 16526463549315561588, 5265614583427859334]),
    UInt256(w: [18245692342128105486, 2774896501339214259, 11434707399789676177, 6582018229284824168]),
    UInt256(w: [18195429409232743953, 8080306645101405728, 14293384249737095221, 8227522786606030210]),
    UInt256(w: [11372143380770464971, 7356034662402072532, 13545051174513072417, 5142201741628768881]),
    UInt256(w: [14215179225963081214, 13806729346429978569, 3096255912859176809, 6427752177035961102]),
    UInt256(w: [3933915977171687805, 3423353627755309500, 13093691927928746820, 8034690221294951377]),
    UInt256(w: [11682069522587080686, 11362968054201844245, 1266028427314384906, 5021681388309344611]),
    UInt256(w: [767528847951687146, 4980338030897529499, 15417593589425144845, 6277101735386680763]),
    UInt256(w: [14794469115221772644, 10837108557049299777, 14660305968354043152, 7846377169233350954]),
    UInt256(w: [2329014169372526047, 6773192848155812361, 13774377248648664874, 4903985730770844346]),
    UInt256(w: [7522953730143045462, 17689863097049541259, 7994599523956055284, 6129982163463555433]),
    UInt256(w: [4792006144251418924, 3665584797602374958, 14604935423372457010, 7662477704329444291]),
    UInt256(w: [16830061895439300539, 6902676516928872252, 6822241630394091679, 4789048565205902682]),
    UInt256(w: [2590833295589574058, 4016659627733702412, 17751174074847390407, 5986310706507378352]),
    UInt256(w: [3238541619486967573, 409138516239740111, 3742223519849686393, 7482888383134222941]),
    UInt256(w: [8941617539820436589, 11784926618718307329, 4644732709119747947, 4676805239458889338]),
    UInt256(w: [15788707943202933640, 10119472254970496257, 15029287923254460742, 5846006549323611672]),
    UInt256(w: [5900826873721503338, 3425968281858344514, 339865830358524312, 7307508186654514591]),
    UInt256(w: [16599405629006654981, 4282460352322930642, 14259890343230319102, 9134385233318143238]),
    UInt256(w: [14986314536556547267, 16511595775483995363, 4300745446091561534, 5708990770823839524]),
    UInt256(w: [14121207152268296180, 11416122682500218396, 5375931807614451918, 7136238463529799405]),
    UInt256(w: [17651508940335370225, 5046781316270497187, 11331600777945452802, 8920298079412249256]),
    UInt256(w: [8726350078495912439, 7765924341096448646, 7082250486215908001, 5575186299632655785]),
    UInt256(w: [1684565561265114740, 14319091444797948712, 13464499126197272905, 6968982874540819731]),
    UInt256(w: [2105706951581393425, 4063806250715272178, 12218937889319203228, 8711228593176024664]),
    UInt256(w: [5927752863165758795, 11763250943551820919, 7636836180824502017, 5444517870735015415]),
    UInt256(w: [2798005060529810589, 869005624157612437, 4934359207603239618, 6805647338418769269]),
    UInt256(w: [8109192344089651141, 10309629067051791354, 10779635027931437426, 8507059173023461586]),
    UInt256(w: [9679931233483419867, 11055204185334757500, 11348957910884536295, 5316911983139663491]),
    UInt256(w: [12099914041854274834, 9207319213241058971, 9574511370178282465, 6646139978924579364]),
    UInt256(w: [10513206533890455638, 16120835034978711618, 11968139212722853081, 8307674973655724205]),
    UInt256(w: [11182440102108922678, 3157992869220612905, 9785930017165477128, 5192296858534827628]),
    UInt256(w: [142992072353989635, 3947491086525766132, 12232412521456846410, 6490371073168534535]),
    UInt256(w: [178740090442487044, 14157735895011983473, 10678829633393670108, 8112963841460668169]),
    UInt256(w: [11640927602595024163, 18071956971237265478, 18203483566939513577, 5070602400912917605]),
    UInt256(w: [5327787466389004395, 8754888158764418136, 8919296403392228260, 6338253001141147007]),
    UInt256(w: [6659734332986255494, 10943610198455522670, 6537434485812897421, 7922816251426433759]),
    UInt256(w: [17997392013398573396, 9145599383248395620, 11003425581274142744, 4951760157141521099]),
    UInt256(w: [4049995943038665129, 11431999229060494526, 9142595958165290526, 6189700196426901374]),
    UInt256(w: [14285866965653107219, 5066626999470842349, 2204872910851837350, 7737125245533626718]),
    UInt256(w: [11234509862746885964, 17001699929951440180, 15213103624564562055, 4835703278458516698]),
    UInt256(w: [14043137328433607455, 16640438894011912321, 9793007493850926761, 6044629098073145873]),
    UInt256(w: [3718863605259845606, 6965490562232726690, 16852945385741046356, 7555786372591432341]),
    UInt256(w: [6935975771714791408, 13576803638250229989, 12838933875301847924, 4722366482869645213]),
    UInt256(w: [13281655733070877164, 16971004547812787486, 2213609288845146193, 5902958103587056517]),
    UInt256(w: [7378697629483820647, 7378697629483820646, 7378697629483820646, 7378697629483820646]),
    UInt256(w: [0, 0, 0, 9223372036854775808]),
    UInt256(w: [0, 0, 0, 5764607523034234880]),
    UInt256(w: [0, 0, 0, 7205759403792793600]),
    UInt256(w: [0, 0, 0, 9007199254740992000]),
    UInt256(w: [0, 0, 0, 5629499534213120000]),
    UInt256(w: [0, 0, 0, 7036874417766400000]),
    UInt256(w: [0, 0, 0, 8796093022208000000]),
    UInt256(w: [0, 0, 0, 5497558138880000000]),
    UInt256(w: [0, 0, 0, 6871947673600000000]),
    UInt256(w: [0, 0, 0, 8589934592000000000]),
    UInt256(w: [0, 0, 0, 5368709120000000000]),
    UInt256(w: [0, 0, 0, 6710886400000000000]),
    UInt256(w: [0, 0, 0, 8388608000000000000]),
    UInt256(w: [0, 0, 0, 5242880000000000000]),
    UInt256(w: [0, 0, 0, 6553600000000000000]),
    UInt256(w: [0, 0, 0, 8192000000000000000]),
    UInt256(w: [0, 0, 0, 5120000000000000000]),
    UInt256(w: [0, 0, 0, 6400000000000000000]),
    UInt256(w: [0, 0, 0, 8000000000000000000]),
    UInt256(w: [0, 0, 0, 5000000000000000000]),
    UInt256(w: [0, 0, 0, 6250000000000000000]),
    UInt256(w: [0, 0, 0, 7812500000000000000]),
    UInt256(w: [0, 0, 0, 4882812500000000000]),
    UInt256(w: [0, 0, 0, 6103515625000000000]),
    UInt256(w: [0, 0, 0, 7629394531250000000]),
    UInt256(w: [0, 0, 0, 4768371582031250000]),
    UInt256(w: [0, 0, 0, 5960464477539062500]),
    UInt256(w: [0, 0, 0, 7450580596923828125]),
    UInt256(w: [0, 0, 2305843009213693952, 4656612873077392578]),
    UInt256(w: [0, 0, 12105675798371893248, 5820766091346740722]),
    UInt256(w: [0, 0, 5908722711110090752, 7275957614183425903]),
    UInt256(w: [0, 0, 2774217370460225536, 9094947017729282379]),
    UInt256(w: [0, 0, 17874786921033498624, 5684341886080801486]),
    UInt256(w: [0, 0, 13120111614437097472, 7105427357601001858]),
    UInt256(w: [0, 0, 7176767481191596032, 8881784197001252323]),
    UInt256(w: [0, 0, 2179636666531053568, 5551115123125782702]),
    UInt256(w: [0, 0, 11947917870018592768, 6938893903907228377]),
    UInt256(w: [0, 0, 1099839282241077248, 8673617379884035472]),
    UInt256(w: [0, 0, 687399551400673280, 5421010862427522170]),
    UInt256(w: [0, 0, 10082621476105617408, 6776263578034402712]),
    UInt256(w: [0, 0, 12603276845132021760, 8470329472543003390]),
    UInt256(w: [0, 0, 3265362009780125696, 5293955920339377119]),
    UInt256(w: [0, 0, 17916760567507320832, 6617444900424221398]),
    UInt256(w: [0, 0, 13172578672529375232, 8271806125530276748]),
    UInt256(w: [0, 0, 17456233707185635328, 5169878828456422967]),
    UInt256(w: [0, 0, 17208606115554656256, 6462348535570528709]),
    UInt256(w: [0, 0, 7675699589161156608, 8077935669463160887]),
    UInt256(w: [0, 0, 11714841270866804736, 5048709793414475554]),
    UInt256(w: [0, 0, 5420179551728730112, 6310887241768094443]),
    UInt256(w: [0, 0, 2163538421233524736, 7888609052210118054]),
    UInt256(w: [0, 0, 15187269568553116672, 4930380657631323783]),
    UInt256(w: [0, 0, 14372400942264007936, 6162975822039154729]),
    UInt256(w: [0, 0, 4130443122547846208, 7703719777548943412]),
    UInt256(w: [0, 0, 11804898988447179688, 4814824860968089632]),
    UInt256(w: [0, 0, 14756123735558974610, 6018531076210112040]),
    UInt256(w: [0, 9223372036854775808, 18445154669448718262, 7523163845262640050]),
    UInt256(w: [0, 1152921504606846976, 16139907686832836818, 4701977403289150031]),
    UInt256(w: [0, 10664523917613334528, 15563198590113658118, 5877471754111437539]),
    UInt256(w: [0, 4107282860161892352, 14842312219214684744, 7346839692639296924]),
    UInt256(w: [0, 5134103575202365440, 106146200308804314, 9183549615799121156]),
    UInt256(w: [0, 7820500752928866304, 9289713412047778504, 5739718509874450722]),
    UInt256(w: [0, 9775625941161082880, 2388769728204947322, 7174648137343063403]),
    UInt256(w: [0, 2996160389596577792, 16821020215538347865, 8968310171678829253]),
    UInt256(w: [0, 13401815289566330880, 12818980643925161367, 5605193857299268283]),
    UInt256(w: [0, 12140583093530525696, 11412039786479063805, 7006492321624085354]),
    UInt256(w: [0, 1340670811630993408, 5041677696244053949, 8758115402030106693]),
    UInt256(w: [0, 3143762266483064832, 5456891569366227670, 5473822126268816683]),
    UInt256(w: [0, 13153074869958606848, 2209428443280396683, 6842277657836020854]),
    UInt256(w: [0, 11829657569020870656, 11985157590955271662, 8552847072295026067]),
    UInt256(w: [0, 2781849962210656256, 5184880485133350837, 5345529420184391292]),
    UInt256(w: [0, 8088998471190708224, 6481100606416688546, 6681911775230489115]),
    UInt256(w: [0, 887876052133609472, 3489689739593472779, 8352389719038111394]),
    UInt256(w: [0, 16695823597079363584, 6792742105673308390, 5220243574398819621]),
    UInt256(w: [0, 11646407459494428672, 13102613650519023392, 6525304467998524526]),
    UInt256(w: [0, 14558009324368035840, 7154895026294003432, 8156630584998155658]),
    UInt256(w: [0, 9098755827730022400, 9083495409861140049, 5097894115623847286]),
    UInt256(w: [0, 15985130803089915904, 2130997225471649253, 6372367644529809108]),
    UInt256(w: [0, 6146355448580231168, 2663746531839561567, 7965459555662261385]),
    UInt256(w: [0, 10759001183003726336, 13194056628468195739, 4978412222288913365]),
    UInt256(w: [0, 8837065460327270016, 2657512730303080962, 6223015277861141707]),
    UInt256(w: [0, 1822959788554311712, 17156948968161014915, 7778769097326427133]),
    UInt256(w: [0, 17280250932342302484, 13028936114314328273, 4861730685829016958]),
    UInt256(w: [0, 7765255610145714393, 7062798106038134534, 6077163357286271198]),
    UInt256(w: [4611686018427387904, 483197475827367183, 18051869669402443976, 7596454196607838997]),
    UInt256(w: [9799832789158199296, 301998422392104489, 13588261552590221437, 4747783872879899373]),
    UInt256(w: [16861477004875137024, 4989184046417518515, 3150268885455613084, 5934729841099874217]),
    UInt256(w: [16465160237666533376, 6236480058021898144, 8549522125246904259, 7418412301374842771]),
    UInt256(w: [10290725148541583360, 1591957027049992388, 3037608319065621210, 4636507688359276732]),
    UInt256(w: [12863406435676979200, 11213318320667266293, 3797010398832026512, 5795634610449095915]),
    UInt256(w: [2244199989314060288, 14016647900834082867, 134576980112645236, 7244543263061369894]),
    UInt256(w: [16640308041924739072, 17520809876042603583, 9391593261995582353, 9055679078826712367]),
    UInt256(w: [17317721553844043776, 4032977144885545383, 12787274816388320827, 5659799424266695229]),
    UInt256(w: [17035465923877666816, 429535412679543825, 2149035465203237322, 7074749280333369037]),
    UInt256(w: [7459274349564919808, 9760291302704205590, 7297980349931434556, 8843436600416711296]),
    UInt256(w: [50360450050686976, 15323554101044904302, 4561237718707146597, 5527147875260444560]),
    UInt256(w: [9286322599418134528, 5319384571023966665, 5701547148383933247, 6908934844075555700]),
    UInt256(w: [16219589267700056064, 2037544695352570427, 7126933935479916559, 8636168555094444625]),
    UInt256(w: [7831400283098841088, 8190994462236438373, 15983548755743417609, 5397605346934027890]),
    UInt256(w: [14400936372300939264, 14850429096222935870, 10756063907824496203, 6747006683667534863]),
    UInt256(w: [8777798428521398272, 13951350351851281934, 8833393866353232350, 8433758354584418579]),
    UInt256(w: [874437999398486016, 4107907951479663305, 3215028157257076267, 5271098971615261612]),
    UInt256(w: [5704733517675495424, 523198920922191227, 4018785196571345334, 6588873714519077015]),
    UInt256(w: [2519230878666981376, 9877370688007514842, 411795477286793763, 8236092143148846269]),
    UInt256(w: [6186205317594251264, 3867513670791002824, 2563215182517940054, 5147557589468028918]),
    UInt256(w: [7732756646992814080, 14057764125343529338, 12427391015002200875, 6434446986835036147]),
    UInt256(w: [442573771886241792, 12960519138252023769, 10922552750325363190, 8043058733543795184]),
    UInt256(w: [11805823653497370880, 3488638442980126951, 6826595468953351994, 5026911708464871990]),
    UInt256(w: [10145593548444325696, 13584170090579934497, 17756616373046465800, 6283639635581089987]),
    UInt256(w: [17293677953982795024, 16980212613224918121, 17584084447880694346, 7854549544476362484]),
    UInt256(w: [3891019693598165034, 15224318901692961730, 1766680743070658158, 4909093465297726553]),
    UInt256(w: [14087146653852482101, 9807026590261426354, 6820036947265710602, 6136366831622158191]),
    UInt256(w: [8385561280460826818, 3035411200972007135, 3913360165654750349, 7670458539527697739]),
    UInt256(w: [12158504827929098618, 4202975009821198411, 140007094320525016, 4794036587204811087]),
    UInt256(w: [10586445016483985368, 5253718762276498014, 14010066923182819982, 5992545734006013858]),
    UInt256(w: [4009684233750205902, 15790520489700398326, 8289211617123749169, 7490682167507517323]),
    UInt256(w: [16341110701376042401, 2951546278421667097, 2874914251488649279, 4681676354692198327]),
    UInt256(w: [6591330321437889289, 17524490903309247584, 17428700869642975310, 5852095443365247908]),
    UInt256(w: [8239162901797361611, 12682241592281783672, 3339132013344167522, 7315119304206559886]),
    UInt256(w: [10298953627246702013, 6629429953497453782, 13397287053534985211, 9143899130258199857]),
    UInt256(w: [1825159998601800855, 1837550711722214662, 1455775380818283901, 5714936956411374911]),
    UInt256(w: [11504822035107026876, 6908624408080156231, 15654777281305018588, 7143671195514218638]),
    UInt256(w: [9769341525456395691, 8635780510100195289, 10345099564776497427, 8929588994392773298]),
    UInt256(w: [17635053499478717067, 3091519809598928103, 11077373246412698796, 5580993121495483311]),
    UInt256(w: [17432130855921008430, 3864399761998660129, 9235030539588485591, 6976241401869354139]),
    UInt256(w: [7955105514619096825, 218813684070937258, 6932102156058219085, 8720301752336692674]),
    UInt256(w: [9583626965064323420, 2442601561758029738, 8944249865963774832, 5450188595210432921]),
    UInt256(w: [2756161669475628467, 3053251952197537173, 15791998350882106444, 6812735744013041151]),
    UInt256(w: [8056888105271923487, 3816564940246921466, 15128311920175245151, 8515919680016301439]),
    UInt256(w: [9647241084222340084, 9302882115295407772, 16372723977750610075, 5322449800010188399]),
    UInt256(w: [12059051355277925104, 7016916625691871811, 15854218953760874690, 6653062250012735499]),
    UInt256(w: [10462128175670018476, 17994517818969615572, 15206087673773705458, 8316327812515919374]),
    UInt256(w: [15762202146648537356, 15858259655283397636, 4892118777681178007, 5197704882822449609]),
    UInt256(w: [1256008609601120079, 15211138550676859142, 10726834490528860413, 6497131103528062011]),
    UInt256(w: [10793382798856175906, 5178865133063910215, 8796857094733687613, 8121413879410077514]),
    UInt256(w: [13663393276926191798, 5542633717378637836, 10109721702635942662, 5075883674631298446]),
    UInt256(w: [17079241596157739747, 16151664183578073103, 3413780091440152519, 6344854593289123058]),
    UInt256(w: [16737365976769786780, 15577894211045203475, 13490597151154966457, 7931068241611403822]),
    UInt256(w: [8155010726267422785, 2818654854262170316, 3819937201044466132, 4956917651007127389]),
    UInt256(w: [10193763407834278482, 3523318567827712895, 9386607519732970569, 6196147063758909236]),
    UInt256(w: [8130518241365460198, 9015834228212029023, 11733259399666213211, 7745183829698636545]),
    UInt256(w: [11999102928494494480, 3329053383418824187, 415758097150301401, 4840739893561647841]),
    UInt256(w: [10387192642190730196, 8773002747700918138, 5131383639865264655, 6050924866952059801]),
    UInt256(w: [3760618765883636937, 6354567416198759769, 11025915568258968723, 7563656083690074751]),
    UInt256(w: [13879601774745742846, 1665761625910530903, 13808726257802937308, 4727285052306296719]),
    UInt256(w: [12737816200004790653, 2082202032388163629, 12649221803826283731, 5909106315382870899]),
    UInt256(w: [2087212194723824604, 16437810595767368249, 11199841236355466759, 7386382894228588624]),
    UInt256(w: [12833722667770860138, 17191160649995687011, 6999900772722166724, 4616489308892867890]),
    UInt256(w: [11430467316286187268, 3042206738785057148, 17973248002757484214, 5770611636116084862]),
    UInt256(w: [14288084145357734085, 13026130460336097243, 13243187966592079459, 7213264545145106078]),
    UInt256(w: [13248419163269779702, 11670977056992733650, 7330612921385323516, 9016580681431382598]),
    UInt256(w: [12891947995471000218, 16517732697475234339, 18416691131147990909, 5635362925894614123]),
    UInt256(w: [11503248975911362368, 6812107816561879212, 18409177895507600733, 7044203657368267654]),
    UInt256(w: [14379061219889202960, 13126820789129736919, 13788100332529725108, 8805254571710334568]),
    UInt256(w: [15904442290071833706, 17427635030060861382, 8617562707831078192, 5503284107318959105]),
    UInt256(w: [10657180825735016325, 3337799713866525112, 15383639403216235645, 6879105134148698881]),
    UInt256(w: [13321476032168770406, 8783935660760544294, 5394491198738130844, 8598881417685873602]),
    UInt256(w: [3714236501678093600, 14713331824830115992, 7983243017638719681, 5374300886053671001]),
    UInt256(w: [4642795627097617000, 4556606725755481278, 14590739790475787506, 6717876107567088751]),
    UInt256(w: [15026866570726797057, 14919130444049127405, 13626738719667346478, 8397345134458860939]),
    UInt256(w: [11697634615917942113, 4712770509103316724, 6210868690578397597, 5248340709036788087]),
    UInt256(w: [14622043269897427641, 10502649154806533809, 3151899844795609092, 6560425886295985109]),
    UInt256(w: [4442496032089620839, 13128311443508167262, 8551560824421899269, 8200532357869981386]),
    UInt256(w: [16611618075338176737, 10511037661406298490, 9956411533691074947, 5125332723668738366]),
    UInt256(w: [11541150557317945113, 8527111058330485209, 3222142380259067876, 6406665904585922958]),
    UInt256(w: [591380141365267679, 10658888822913106512, 13251050012178610653, 8008332380732403697]),
    UInt256(w: [369612588353292299, 8967648523534385522, 1364377229970549802, 5005207737957752311]),
    UInt256(w: [9685387772296391182, 1986188617563206094, 15540529592745350965, 6256509672447190388]),
    UInt256(w: [2883362678515713170, 7094421790381395522, 978917917222137090, 7820637090558987986]),
    UInt256(w: [6413787692499708635, 9045699637415760105, 5223509716691223585, 4887898181599367491]),
    UInt256(w: [12628920634052023698, 15918810565197088035, 1917701127436641577, 6109872726999209364]),
    UInt256(w: [11174464774137641718, 6063455151214196332, 2397126409295801972, 7637340908749011705]),
    UInt256(w: [16207412520690801882, 13013031506363648515, 13027419051878345992, 4773338067968132315]),
    UInt256(w: [15647579632436114448, 16266289382954560644, 11672587796420544586, 5966672584960165394]),
    UInt256(w: [1112730466835591444, 11109489691838424998, 5367362708670904925, 7458340731200206743]),
    UInt256(w: [14530514597054408365, 9249274066612709575, 10272130720560397434, 4661462957000129214]),
    UInt256(w: [13551457227890622552, 2338220546411111161, 3616791363845720985, 5826828696250161518]),
    UInt256(w: [3104263479581114478, 7534461701441276856, 13744361241661927039, 7283535870312701897]),
    UInt256(w: [3880329349476393097, 4806391108374208166, 3345393496795245087, 9104419837890877372]),
    UInt256(w: [16260263898704909398, 9921523470374961959, 11314242972351803987, 5690262398681798357]),
    UInt256(w: [15713643854953748843, 7790218319541314545, 307745660157591272, 7112827998352247947]),
    UInt256(w: [5806996763410022342, 9737772899426643182, 14219740130479152802, 8891034997940309933]),
    UInt256(w: [17464431032413427676, 10697794080569039892, 11193180590763164453, 5556896873712693708]),
    UInt256(w: [3383794716807232979, 17983928619138687770, 13991475738453955566, 6946121092140867135]),
    UInt256(w: [13453115432863817032, 13256538737068583904, 12877658654640056554, 8682651365176083919]),
    UInt256(w: [8408197145539885645, 12897022729095252844, 14966065686791117202, 5426657103235052449]),
    UInt256(w: [10510246431924857056, 6897906374514290247, 4872524053206732791, 6783321379043815562]),
    UInt256(w: [8526122021478683416, 4010696949715474905, 15314027103363191797, 8479151723804769452]),
    UInt256(w: [16858041309492646895, 4812528602785865767, 347894902747219065, 5299469827377980908]),
    UInt256(w: [16460865618438420715, 10627346771909720113, 434868628434023831, 6624337284222476135]),
    UInt256(w: [6741023967765862181, 8672497446459762238, 14378643840824693501, 8280421605278095168]),
    UInt256(w: [18048198035135827576, 7726153913251045350, 8986652400515433438, 5175263503298809480]),
    UInt256(w: [13336875507065008661, 434320354709030880, 11233315500644291798, 6469079379123511850]),
    UInt256(w: [16671094383831260826, 9766272480241064408, 4818272338950588939, 8086349223904389813]),
    UInt256(w: [10419433989894538017, 3798077290936971303, 5317263221057812039, 5053968264940243633]),
    UInt256(w: [8412606468940784617, 135910595243826225, 11258265044749652953, 6317460331175304541]),
    UInt256(w: [15127444104603368675, 4781574262482170685, 237773250654902479, 7896825413969130677]),
    UInt256(w: [11760495574590799374, 9906012941692438534, 2454451290873008001, 4935515883730706673]),
    UInt256(w: [5477247431383723409, 16994202195542936072, 7679750132018647905, 6169394854663383341]),
    UInt256(w: [6846559289229654262, 7407694689146506378, 14211373683450697786, 7711743568329229176]),
    UInt256(w: [8890785574195921818, 9241495199143954390, 8882108552156686116, 4819839730205768235]),
    UInt256(w: [1890109930890126464, 11551868998929942988, 6490949671768469741, 6024799662757210294]),
    UInt256(w: [2362637413612658080, 604778193380265023, 17337059126565362985, 7530999578446512867]),
    UInt256(w: [8394177411148993156, 11907201416931135399, 8529818944889657913, 4706874736529070542]),
    UInt256(w: [5881035745508853541, 1048943715881755537, 1438901644257296584, 5883593420661338178]),
    UInt256(w: [11962980700313454830, 1311179644852194421, 11021999092176396538, 7354491775826672722]),
    UInt256(w: [1118667820109654825, 10862346592920018835, 4554126828365719864, 9193114719783340903]),
    UInt256(w: [16840068452064391930, 6788966620575011771, 9763858295369656771, 5745696699864588064]),
    UInt256(w: [16438399546653102009, 3874522257291376810, 12204822869212070964, 7182120874830735080]),
    UInt256(w: [11324627396461601703, 4843152821614221013, 15256028586515088705, 8977651093538418850]),
    UInt256(w: [9383735132002195016, 14556185559577357893, 14146703884999318344, 5611031933461511781]),
    UInt256(w: [16341354933430131674, 18195231949471697366, 3848321800966984218, 7013789916826889727]),
    UInt256(w: [11203321629932888785, 13520667899984845900, 198716232781342369, 8767237396033612159]),
    UInt256(w: [16225448055562831299, 1532888409849446831, 7041726673129420837, 5479523372521007599]),
    UInt256(w: [15670124051026151219, 6527796530739196443, 4190472322984388142, 6849404215651259499]),
    UInt256(w: [14975969045355301120, 17383117700278771362, 626404385303097273, 8561755269564074374]),
    UInt256(w: [13971666671774451104, 3946919535033150245, 14226560796096599508, 5351097043477546483]),
    UInt256(w: [3629525284435900168, 4933649418791437807, 13171514976693361481, 6688871304346933104]),
    UInt256(w: [18371964660827038922, 10778747791916685162, 16464393720866701851, 8361089130433666380]),
    UInt256(w: [16094163931444287230, 4430874360734234274, 1066874038686912849, 5225680706521041488]),
    UInt256(w: [10894332877450583230, 10150278969345180747, 1333592548358641061, 6532100883151301860]),
    UInt256(w: [9006230078385841133, 17299534730108863838, 1666990685448301326, 8165126103939127325]),
    UInt256(w: [1017207780563762804, 6200523187890651995, 3347712187618882281, 5103203814961954578]),
    UInt256(w: [15106567780986867217, 12362340003290702897, 13408012271378378659, 6379004768702443222]),
    UInt256(w: [5048151670951420310, 10841238985685990718, 7536643302368197516, 7973755960878054028]),
    UInt256(w: [16990152849626801406, 15999146402908520006, 13933774100834899255, 4983597475548783767]),
    UInt256(w: [12014319025178725949, 15387246985208262104, 12805531607616236165, 6229496844435979709]),
    UInt256(w: [15017898781473407436, 5399000676228163918, 2171856454238131495, 7786871055544974637]),
    UInt256(w: [4774500719993491744, 10291904450283684305, 3663253293112526136, 4866794409715609148]),
    UInt256(w: [10579811918419252584, 12864880562854605381, 4579066616390657670, 6083493012144511435]),
    UInt256(w: [17836450916451453633, 6857728666713480918, 1112147252060934184, 7604366265180639294]),
    UInt256(w: [6536095804354770617, 4286080416695925574, 14530150087820247577, 4752728915737899558]),
    UInt256(w: [17393491792298239079, 9969286539297294871, 8939315572920533663, 5940911144672374448]),
    UInt256(w: [17130178721945410945, 7849922155694230685, 11174144466150667079, 7426138930840468060]),
    UInt256(w: [13012204710429575793, 11823730374949976034, 16207212328198942732, 4641336831775292537]),
    UInt256(w: [7041883851182193933, 14779662968687470043, 6423957354966514703, 5801671039719115672]),
    UInt256(w: [4190668795550354512, 13862892692431949650, 8029946693708143379, 7252088799648894590]),
    UInt256(w: [14461708031292718948, 12716929847112549158, 814061330280403416, 9065110999561118238]),
    UInt256(w: [4426881501130561438, 7948081154445343224, 14343846386707415847, 5665694374725698898]),
    UInt256(w: [5533601876413201798, 5323415424629291126, 8706435946529494001, 7082117968407123623]),
    UInt256(w: [16140374382371278055, 11265955299214001811, 6271358914734479597, 8852647460508904529]),
    UInt256(w: [7781890979768354833, 9347065071222445084, 15448814367777519508, 5532904662818065330]),
    UInt256(w: [9727363724710443541, 11683831339028056355, 10087645922867123577, 6916130828522581663]),
    UInt256(w: [7547518637460666522, 769731118502906732, 7997871385156516568, 8645163535653227079]),
    UInt256(w: [13940571185267692384, 481081949064316707, 11916198643363904711, 5403227209783266924]),
    UInt256(w: [12814027963157227576, 14436410491612559596, 14895248304204880888, 6754034012229083655]),
    UInt256(w: [16017534953946534470, 18045513114515699495, 14007374361828713206, 8442542515286354569]),
    UInt256(w: [16928488373857665900, 6666759678144924280, 1837079948501863898, 5276589072053971606]),
    UInt256(w: [2713866393612530759, 17556821634535931159, 11519721972482105680, 6595736340067464507]),
    UInt256(w: [17227391047297827161, 3499282969460362332, 9787966447175244197, 8244670425084330634]),
    UInt256(w: [1543747367706366168, 4492894865126420410, 10729165047911915527, 5152919015677706646]),
    UInt256(w: [11153056246487733517, 1004432562980637608, 4188084273035118601, 6441148769597133308]),
    UInt256(w: [13941320308109666897, 5867226722153184914, 5235105341293898251, 8051435961996416635]),
    UInt256(w: [13325011210995929715, 1361173692132046619, 966097829094992455, 5032147476247760397]),
    UInt256(w: [12044577995317524239, 15536525170447221986, 5819308304796128472, 6290184345309700496]),
    UInt256(w: [5832350457292129491, 973912389349475867, 7274135380995160591, 7862730431637125620]),
    UInt256(w: [1339376026593886980, 7526224270984504273, 13769706649976751177, 4914206519773203512]),
    UInt256(w: [6285906051669746629, 14019466357158018245, 17212133312470938971, 6142758149716504390]),
    UInt256(w: [12469068583014571190, 12912646928020134902, 12291794603733897906, 7678447687145630488]),
    UInt256(w: [3181481845956719090, 12682090348439972218, 7682371627333686191, 4799029804466019055]),
    UInt256(w: [13200224344300674670, 11240926917122577368, 4991278515739719835, 5998787255582523819]),
    UInt256(w: [16500280430375843337, 9439472627975833806, 1627412126247261890, 7498484069478154774]),
    UInt256(w: [5700989250557514182, 10511356410912284033, 14852190634186702393, 4686552543423846733]),
    UInt256(w: [11737922581624280632, 17750881532067742945, 4730180237451214279, 5858190679279808417]),
    UInt256(w: [837345171748187077, 17576915896657290778, 10524411315241405753, 7322738349099760521]),
    UInt256(w: [10270053501540009654, 8136086815539449760, 17767200162479145096, 9153422936374700651]),
    UInt256(w: [6418783438462506034, 5085054259712156100, 8798657092335771733, 5720889335234187907]),
    UInt256(w: [8023479298078132543, 10968003843067583029, 6386635346992326762, 7151111669042734884]),
    UInt256(w: [14641035141025053582, 4486632766979702978, 7983294183740408453, 8938889586303418605]),
    UInt256(w: [13762332981568046393, 5109988488576008313, 7295401874051449235, 5586805991439636628]),
    UInt256(w: [3367858171677894279, 1775799592292622488, 9119252342564311544, 6983507489299545785]),
    UInt256(w: [4209822714597367849, 2219749490365778110, 16010751446632777334, 8729384361624432231]),
    UInt256(w: [16466197251905518618, 15222401486760775030, 16924248681786567689, 5455865226015270144]),
    UInt256(w: [11359374528027122464, 5192943803168805076, 2708566778523657996, 6819831532519087681]),
    UInt256(w: [14199218160033903080, 6491179753961006345, 7997394491581960399, 8524789415648859601]),
    UInt256(w: [1956982322380107569, 10974516373866710822, 16527586603307195009, 5327993384780537250]),
    UInt256(w: [11669599939829910269, 18329831485760776431, 11436111217279217953, 6659991730975671563]),
    UInt256(w: [9975313906359999932, 9077231301918806827, 9683453003171634538, 8324989663719589454]),
    UInt256(w: [3928728182261306006, 10284955582126642171, 1440472108554883682, 5203118539824743409]),
    UInt256(w: [299224209399244603, 3632822440803526906, 6412276154120992507, 6503898174780929261]),
    UInt256(w: [9597402298603831562, 18376086106286572344, 12627031211078628537, 8129872718476161576]),
    UInt256(w: [5998376436627394726, 4567524788788025859, 7891894506924142836, 5081170449047600985]),
    UInt256(w: [2886284527356855504, 5709405985985032324, 14476554152082566449, 6351463061309501231]),
    UInt256(w: [3607855659196069380, 11748443500908678309, 13484006671675820157, 7939328826636876539]),
    UInt256(w: [4560752796211237315, 9648620197281617895, 6121661160583693646, 4962080516648047837]),
    UInt256(w: [1089254976836658739, 2837403209747246561, 12263762469157004962, 6202600645810059796]),
    UInt256(w: [5973254739473211328, 12770126049038834009, 15329703086446256202, 7753250807262574745]),
    UInt256(w: [15262499258239226840, 12593014799076659159, 2663535401387828270, 4845781754539109216]),
    UInt256(w: [14466438054371645646, 6517896461991048141, 3329419251734785338, 6057227193173886520]),
    UInt256(w: [4247989512682393345, 17370742614343585985, 4161774064668481672, 7571533991467358150]),
    UInt256(w: [14184208491494965601, 10856714133964741240, 16436166845699964757, 4732208744667098843]),
    UInt256(w: [17730260614368707001, 18182578685883314454, 15933522538697568042, 5915260930833873554]),
    UInt256(w: [12939453731106107943, 13504851320499367260, 10693531136517184245, 7394076163542341943]),
    UInt256(w: [17310530618796093273, 10746375084525798489, 13600985987964322009, 4621297602213963714]),
    UInt256(w: [7803105218212952879, 18044654874084636016, 7777860448100626703, 5776622002767454643]),
    UInt256(w: [9753881522766191098, 17944132574178407116, 5110639541698395475, 7220777503459318304]),
    UInt256(w: [12192351903457738873, 17818479699295620991, 6388299427122994344, 9025971879324147880]),
    UInt256(w: [14537748967302168652, 11136549812059763119, 3992687141951871465, 5641232424577592425]),
    UInt256(w: [13560500190700322911, 85629209792540187, 9602544945867227236, 7051540530721990531]),
    UInt256(w: [12338939219948015734, 107036512240675234, 7391495163906646141, 8814425663402488164]),
    UInt256(w: [12323523030894897738, 2372740829364115973, 13843056514296429646, 5509016039626555102]),
    UInt256(w: [1569345733336458460, 12189298073559920775, 8080448606015761249, 6886270049533193878]),
    UInt256(w: [15796740221952736787, 1401564536667737256, 877188720664925754, 8607837561916492348]),
    UInt256(w: [9872962638720460492, 5487663853844723689, 9771614987270354404, 5379898476197807717]),
    UInt256(w: [16952889316827963519, 6859579817305904611, 16826204752515330909, 6724873095247259646]),
    UInt256(w: [16579425627607566495, 13186160790059768668, 11809383903789387828, 8406091369059074558]),
    UInt256(w: [1138768980399953251, 17464722530642131226, 2769178921440979488, 5253807105661921599]),
    UInt256(w: [10646833262354717372, 3384159089593112416, 17296531707083388073, 6567258882077401998]),
    UInt256(w: [13308541577943396715, 8841884880418778424, 12397292596999459283, 8209073602596752498]),
    UInt256(w: [8317838486214622947, 3220335041048042563, 12359993891552049956, 5130671001622970311]),
    UInt256(w: [5785612089340890780, 4025418801310053204, 10838306346012674541, 6413338752028712889])
  ]
  
  static let bid_multipliers2_binary64: [UInt256] = [
    UInt256(w: [918777112239470733, 5138421092069233273,     15049182825743144437, 3877256883183270350]),
    UInt256(w: [12103450741218138969, 5517356191756964747, 4794053247662077369, 2423285551989543969]),
    UInt256(w: [10517627408095285807, 11508381258123593838, 10604252578004984615, 3029106939986929961]),
    UInt256(w: [3923662223264331450, 9773790554227104394, 17867001740933618673, 3786383674983662451]),
    UInt256(w: [7063974907967595060, 17637834142460410006, 8861033078869817718, 2366489796864789032]),
    UInt256(w: [18053340671814269633, 12823920641220736699, 11076291348587272148, 2958112246080986290]),
    UInt256(w: [17954989821340449138, 16029900801525920874, 4621992148879314377, 3697640307601232863]),
    UInt256(w: [15833554656765168615, 3101158973312618690, 9806274120690653342, 2311025192250770539]),
    UInt256(w: [10568571284101684961, 13099820753495549171, 7646156632435928773, 2888781490313463174]),
    UInt256(w: [8599028086699718297, 2539717886587272752, 334323753690135159, 3610976862891828968]),
    UInt256(w: [10748785108374647871, 17009705413516254652, 417904692112668948, 4513721078614786210]),
    UInt256(w: [15941362729588930728, 1407693846592883349, 4872876450997805997, 2821075674134241381]),
    UInt256(w: [6091645356703999697, 6371303326668492091, 10702781582174645400, 3526344592667801726]),
    UInt256(w: [3002870677452611718, 7964129158335615114, 4155104940863530942, 4407930740834752158]),
    UInt256(w: [6488480191835270228, 365894705532371542, 16431998643321870551, 2754956713021720098]),
    UInt256(w: [17333972276648863593, 14292426437197628139, 11316626267297562380, 3443695891277150123]),
    UInt256(w: [17055779327383691587, 17865533046497035174, 9534096815694565071, 4304619864096437654]),
    UInt256(w: [6048176061187419338, 18083487181701728840, 1347124491381715265, 2690387415060273534]),
    UInt256(w: [7560220076484274172, 8769300921844997338, 10907277651081919890, 3362984268825341917]),
    UInt256(w: [226903058750566907, 1738254115451470865, 18245783082279787767, 4203730336031677396]),
    UInt256(w: [11671029457787574077, 8003937849798251146, 2180242389570091546, 2627331460019798373]),
    UInt256(w: [5365414785379691788, 781550275393038125, 7336989005390002337, 3284164325024747966]),
    UInt256(w: [11318454500152002639, 5588623862668685560, 18394608293592278729, 4105205406280934957]),
    UInt256(w: [7074034062595001650, 15022104960236398235, 13802473192708868157, 2565753378925584348]),
    UInt256(w: [4230856559816364158, 4942573145013334082, 17253091490886085197, 3207191723656980435]),
    UInt256(w: [14511942736625231005, 10789902449694055506, 16954678345180218592, 4008989654571225544]),
    UInt256(w: [13681650228818157283, 6743689031058784691, 10596673965737636620, 2505618534107015965]),
    UInt256(w: [12490376767595308699, 8429611288823480864, 17857528475599433679, 3132023167633769956]),
    UInt256(w: [15612970959494135874, 5925328092601963176, 3875166520789740483, 3915028959542212446]),
    UInt256(w: [9758106849683834921, 1397487048662533033, 16257037130775751514, 2446893099713882778]),
    UInt256(w: [16809319580532181555, 10970230847682942099, 11097924376614913584, 3058616374642353473]),
    UInt256(w: [16399963457237839040, 13712788559603677624, 37347415486478268, 3823270468302941842]),
    UInt256(w: [10249977160773649400, 17793864886607074323, 4635028153106436821, 2389544042689338651]),
    UInt256(w: [8200785432539673846, 8407273052976679192, 1182099172955658123, 2986930053361673314]),
    UInt256(w: [10250981790674592308, 5897405297793461086, 10700996003049348462, 3733662566702091642]),
    UInt256(w: [1795177600744232288, 17520936366403076891, 11299808520333230692, 2333539104188807276]),
    UInt256(w: [16079030056212454072, 3454426384294294497, 14124760650416538366, 2916923880236009095]),
    UInt256(w: [6263729514983403878, 13541405017222643930, 13044264794593285053, 3646154850295011369]),
    UInt256(w: [17053033930584030656, 3091698216246141200, 2470272937959442605, 4557693562868764212]),
    UInt256(w: [10658146206615019160, 4238154394367532202, 10767292623079427436, 2848558476792977632]),
    UInt256(w: [4099310721413998142, 5297692992959415253, 13459115778849284295, 3560698095991222040]),
    UInt256(w: [9735824420194885581, 2010430222771881162, 16823894723561605369, 4450872619989027550]),
    UInt256(w: [10696576281049191393, 12785733935300895486, 5903248183798615451, 2781795387493142219]),
    UInt256(w: [4147348314456713433, 11370481400698731454, 2767374211320881410, 3477244234366427774]),
    UInt256(w: [14407557429925667599, 4989729714018638509, 12682589801005877571, 4346555292958034717]),
    UInt256(w: [11310566402917236201, 812738062047955116, 10232461634842367434, 2716597058098771698]),
    UInt256(w: [14138208003646545252, 10239294614414719703, 3567205006698183484, 3395746322623464623]),
    UInt256(w: [13061073986130793660, 12799118268018399629, 18294064313654893067, 4244682903279330778]),
    UInt256(w: [10469014250545439990, 5693605908297805816, 16045476214461696071, 2652926814549581736]),
    UInt256(w: [13086267813181799987, 2505321366944869366, 1610101194367568473, 3316158518186977171]),
    UInt256(w: [7134462729622474176, 7743337727108474612, 15847684548241624303, 4145198147733721463]),
    UInt256(w: [13682411242868822168, 11757115107083878488, 16822331870292097045, 2590748842333575914]),
    UInt256(w: [17103014053586027710, 861335828572684398, 11804542801010345499, 3238436052916969893]),
    UInt256(w: [12155395530127758829, 14911727840998019210, 920620445980768161, 4048045066146212367]),
    UInt256(w: [12208808224757237173, 2402300872982680150, 7492916806379061957, 2530028166341382729]),
    UInt256(w: [6037638244091770658, 7614562109655738092, 13977832026401215350, 3162535207926728411]),
    UInt256(w: [7547047805114713322, 294830600214896807, 12860604014574131284, 3953169009908410514]),
    UInt256(w: [11634433905837777682, 9407641161989086312, 12649563527536219956, 2470730631192756571]),
    UInt256(w: [14543042382297222103, 11759551452486357890, 11200268390992887041, 3088413288990945714]),
    UInt256(w: [8955430941016751820, 864381260325783651, 4776963451886332994, 3860516611238682143]),
    UInt256(w: [3291301328921775936, 5151924306131002686, 9903131185070039977, 2412822882024176339]),
    UInt256(w: [13337498698006995728, 11051591401091141261, 7767227962910162067, 3016028602530220424]),
    UInt256(w: [2836815317226580948, 9202803232936538673, 9709034953637702584, 3770035753162775530]),
    UInt256(w: [13302224619335082852, 5751752020585336670, 10679832864450952019, 2356272345726734706]),
    UInt256(w: [7404408737314077757, 2578004007304282934, 4126419043708914216, 2945340432158418383]),
    UInt256(w: [32138884787821389, 3222505009130353668, 546337786208754866, 3681675540198022979]),
    UInt256(w: [40173605984776736, 13251503298267717893, 14517980288043107294, 4602094425247528723]),
    UInt256(w: [2330951512954179412, 3670503542989935779, 6767894670813248107, 2876309015779705452]),
    UInt256(w: [16748747446474887977, 18423187484019583435, 8459868338516560133, 3595386269724631815]),
    UInt256(w: [16324248289666222067, 9193926299742315582, 5963149404718312263, 4494232837155789769]),
    UInt256(w: [5590969162614000888, 12663732964980029095, 15256183424017414924, 2808895523222368605]),
    UInt256(w: [2377025434840113206, 15829666206225036369, 5235171224739604943, 3511119404027960757]),
    UInt256(w: [7582967811977529411, 15175396739353907557, 11155650049351894083, 4388899255034950946]),
    UInt256(w: [7045197891699649834, 7178779952882498271, 11583967299272321706, 2743062034396844341]),
    UInt256(w: [4194811346197174389, 18196846977957898647, 644901068808238420, 3428827542996055427]),
    UInt256(w: [631828164319080082, 4299314648737821693, 14641184391292461738, 4286034428745069283]),
    UInt256(w: [2700735611913119003, 7298757673888526462, 6844897235344094634, 2678771517965668302]),
    UInt256(w: [12599291551746174562, 18346819129215433885, 17779493581034894100, 3348464397457085377]),
    UInt256(w: [1914056384400554490, 4486779837809740741, 8389308921011453914, 4185580496821356722]),
    UInt256(w: [3502128249464040509, 7415923417058475867, 9855004094059546600, 2615987810513347951]),
    UInt256(w: [18212718367112214348, 9269904271323094833, 7707069099147045346, 3269984763141684939]),
    UInt256(w: [8930839903608104222, 2364008302299092734, 5022150355506418779, 4087480953927106174]),
    UInt256(w: [970088921327677235, 17618406253432790623, 16973902027473675448, 2554675596204441358]),
    UInt256(w: [15047669206941760256, 3576263743081436662, 11994005497487318503, 3193344495255551698]),
    UInt256(w: [9586214471822424512, 18305387734133959540, 5769134835004372320, 3991680619069439623]),
    UInt256(w: [15214756081743791128, 11440867333833724712, 10523238299518814556, 2494800386918399764]),
    UInt256(w: [571701028470187294, 14301084167292155891, 13154047874398518195, 3118500483647999705]),
    UInt256(w: [14549684340869897829, 13264669190687806959, 2607501787715984032, 3898125604559999632]),
    UInt256(w: [16011081740684767999, 8290418244179879349, 1629688617322490020, 2436328502849999770]),
    UInt256(w: [6178794120573796287, 10363022805224849187, 11260482808507888333, 3045410628562499712]),
    UInt256(w: [3111806632289857455, 17565464524958449388, 14075603510634860416, 3806763285703124640]),
    UInt256(w: [11168251182035936718, 10978415328099030867, 8797252194146787760, 2379227053564452900]),
    UInt256(w: [9348627959117532993, 13723019160123788584, 10996565242683484700, 2974033816955566125]),
    UInt256(w: [11685784948896916241, 17153773950154735730, 18357392571781743779, 3717542271194457656]),
    UInt256(w: [11915301611487960555, 8415265709633015879, 11473370357363589862, 2323463919496536035]),
    UInt256(w: [10282440995932562789, 1295710100186494041, 9730026928277099424, 2904329899370670044]),
    UInt256(w: [17464737263343091390, 1619637625233117551, 12162533660346374280, 3630412374213337555]),
    UInt256(w: [17219235560751476334, 2024547031541396939, 10591481057005579946, 4538015467766671944]),
    UInt256(w: [8456179216255978757, 5877027913140760991, 6619675660628487466, 2836259667354169965]),
    UInt256(w: [5958538001892585542, 16569656928280727047, 12886280594212997236, 3545324584192712456]),
    UInt256(w: [2836486483938344023, 2265327086641357193, 16107850742766246546, 4431655730240890570]),
    UInt256(w: [13302019098529934775, 6027515447578236149, 14679092732656291995, 2769784831400556606]),
    UInt256(w: [2792465817880254756, 2922708291045407283, 9125493878965589186, 3462231039250695758]),
    UInt256(w: [17325640327632482157, 12876757400661534911, 2183495311852210674, 4327788799063369698]),
    UInt256(w: [17746054232411383205, 12659659393840847223, 5976370588335019575, 2704867999414606061]),
    UInt256(w: [17570881772086841102, 11212888223873671125, 12082149253846162373, 3381084999268257576]),
    UInt256(w: [8128544159826387665, 181052224559925195, 15102686567307702967, 4226356249085321970]),
    UInt256(w: [2774497090677798339, 7030686667991035103, 14050865122994702258, 2641472655678326231]),
    UInt256(w: [17303179418629411635, 18011730371843569686, 12951895385315989918, 3301840819597907789]),
    UInt256(w: [12405602236431988736, 13291290927949686300, 2354811176362823686, 4127301024497384737]),
    UInt256(w: [16976873434624768768, 3695370811541166033, 13000972031295234564, 2579563140310865460]),
    UInt256(w: [7386033737998797248, 4619213514426457542, 16251215039119043205, 3224453925388581825]),
    UInt256(w: [9170135643720752, 10385702911460459832, 6478960743616640294, 4030567406735727282]),
    UInt256(w: [5731334777325470, 1879378301235399491, 8661036483187788088, 2519104629209829551]),
    UInt256(w: [13842222223753820550, 2349222876544249363, 6214609585557347206, 3148880786512286939]),
    UInt256(w: [12691091761264887783, 12159900632535087512, 3156575963519296103, 3936100983140358674]),
    UInt256(w: [7931932350790554864, 14517466922975511551, 6584545995626947968, 2460063114462724171]),
    UInt256(w: [5303229420060805676, 18146833653719389439, 3618996476106297056, 3075078893078405214]),
    UInt256(w: [2017350756648619191, 4236797993439685183, 13747117631987647129, 3843848616348006517]),
    UInt256(w: [8178373250546468851, 14177213791968272999, 10897791529205973407, 2402405385217504073]),
    UInt256(w: [5611280544755698159, 13109831221532953345, 18233925429934854663, 3003006731521880091]),
    UInt256(w: [11625786699372010603, 11775603008488803777, 18180720768991180425, 3753758414402350114]),
    UInt256(w: [348587659466424771, 442222852664420505, 15974636499046875670, 2346099009001468821]),
    UInt256(w: [5047420592760418868, 9776150602685301439, 6133237568526430875, 2932623761251836027]),
    UInt256(w: [1697589722523135681, 7608502234929238895, 3054860942230650690, 3665779701564795034]),
    UInt256(w: [15957045208436083313, 287255756806772810, 13041948214643089171, 4582224626955993792]),
    UInt256(w: [14584839273699939975, 16320435912500090670, 8151217634151930731, 2863890391847496120]),
    UInt256(w: [9007677055270149160, 15788858872197725434, 10189022042689913414, 3579862989809370150]),
    UInt256(w: [2036224282232910642, 10512701553392380985, 3512905516507615960, 4474828737261712688]),
    UInt256(w: [12801855222464038911, 6570438470870238115, 2195565947817259975, 2796767960788570430]),
    UInt256(w: [11390633009652660735, 3601362070160409740, 11967829471626350777, 3495959950985713037]),
    UInt256(w: [14238291262065825919, 9113388606127900079, 1124728784250774759, 4369949938732141297]),
    UInt256(w: [15816461066432223055, 12613396906471019405, 12232170536225203984, 2731218711707588310]),
    UInt256(w: [5935518277758115107, 15766746133088774257, 6066841133426729172, 3414023389634485388]),
    UInt256(w: [12031083865625031788, 1261688592651416205, 7583551416783411466, 4267529237043106735]),
    UInt256(w: [9825270425229338820, 5400241388834523032, 11657248663130714022, 2667205773151941709]),
    UInt256(w: [12281588031536673524, 15973673772897929598, 736502773631228815, 3334007216439927137]),
    UInt256(w: [6128613002566066097, 15355406197695024094, 5532314485466423923, 4167509020549908921]),
    UInt256(w: [17665441181885955023, 7291285864345696106, 14986911599484984712, 2604693137843693075]),
    UInt256(w: [12858429440502667971, 9114107330432120133, 14121953480928842986, 3255866422304616344]),
    UInt256(w: [2237978745346171251, 2169262126185374359, 17652441851161053733, 4069833027880770430]),
    UInt256(w: [8316265743482438888, 3661631838079552926, 6421090138548270679, 2543645642425481519]),
    UInt256(w: [1171960142498272802, 18412097852881604870, 3414676654757950444, 3179557053031851899]),
    UInt256(w: [10688322214977616811, 4568378242392454471, 18103403873729601768, 3974446316289814873]),
    UInt256(w: [13597730412002092363, 2855236401495284044, 4397098393439919249, 2484028947681134296]),
    UInt256(w: [16997163015002615453, 8180731520296492959, 5496372991799899061, 3105036184601417870]),
    UInt256(w: [16634767750325881413, 14837600418798004103, 16093838276604649634, 3881295230751772337]),
    UInt256(w: [17314258871594757739, 13885186280176140468, 3141119895236824165, 2425809519219857711]),
    UInt256(w: [3196079515783895558, 3521424794938011874, 17761457924328193919, 3032261899024822138]),
    UInt256(w: [13218471431584645255, 18236839048954678554, 12978450368555466590, 3790327373781027673]),
    UInt256(w: [12873230663167791189, 6786338387169286192, 1194002452706084763, 2368954608613142296]),
    UInt256(w: [16091538328959738986, 3871236965534219836, 1492503065882605954, 2961193260766427870]),
    UInt256(w: [1667678837490122116, 14062418243772550604, 11089000869208033250, 3701491575958034837]),
    UInt256(w: [10265671310286102131, 13400697420785232031, 9236468552468714733, 2313432234973771773]),
    UInt256(w: [8220403119430239759, 2915813720699376327, 16157271709013281321, 2891790293717214716]),
    UInt256(w: [5663817880860411795, 8256453169301608313, 1749845562557050035, 3614737867146518396]),
    UInt256(w: [11691458369502902647, 5708880443199622487, 2187306953196312544, 4518422333933147995]),
    UInt256(w: [14224690508580396011, 3568050276999764054, 17507967910243553004, 2824013958708217496]),
    UInt256(w: [8557491098870719205, 4460062846249705068, 3438215814094889639, 3530017448385271871]),
    UInt256(w: [10696863873588399007, 963392539384743431, 18132827822900775761, 4412521810481589838]),
    UInt256(w: [13603068948633831235, 12131335383183934404, 6721331370885596946, 2757826131550993649]),
    UInt256(w: [17003836185792289044, 5940797192125142197, 13013350232034384087, 3447282664438742061]),
    UInt256(w: [7419737176958197593, 2814310471729039843, 2431629734760816397, 4309103330548427577]),
    UInt256(w: [2331492726385179544, 4064787054044343854, 13048983630293980008, 2693189581592767235]),
    UInt256(w: [12137737944836250237, 5080983817555429817, 11699543519440087106, 3366486976990959044]),
    UInt256(w: [1337114375763149085, 15574601808799063080, 14624429399300108882, 4208108721238698805]),
    UInt256(w: [835696484851968178, 14345812148926802329, 11446111383776262003, 2630067950774186753]),
    UInt256(w: [5656306624492348127, 13320579167731115007, 472581174438163792, 3287584938467733442]),
    UInt256(w: [2458697262188047254, 16650723959663893759, 9814098504902480548, 4109481173084666802]),
    UInt256(w: [8454214816508611390, 1183330437935157791, 10745497583991438247, 2568425733177916751]),
    UInt256(w: [5956082502208376333, 15314221102701110951, 8820185961561909904, 3210532166472395939]),
    UInt256(w: [2833417109333082513, 696032304666837073, 6413546433524999477, 4013165208090494924]),
    UInt256(w: [13300100739401646331, 2740863199630467122, 13231838557807900481, 2508228255056559327]),
    UInt256(w: [7401753887397282105, 8037765017965471807, 11928112178832487697, 3135285318820699159]),
    UInt256(w: [4640506340819214727, 14658892290884227663, 10298454205113221717, 3919106648525873949]),
    UInt256(w: [9817845490653091061, 11467650691016336241, 8742376887409457525, 2449441655328671218]),
    UInt256(w: [16883992881743751730, 499505308488256589, 1704599072407046099, 3061802069160839023]),
    UInt256(w: [7269933046897525950, 14459439690892484449, 15965806895790971335, 3827252586451048778]),
    UInt256(w: [16072923200379423479, 15954678834448884636, 14590315328296744988, 2392032866531905486]),
    UInt256(w: [1644409926764727733, 1496604469351554180, 9014522123516155428, 2990041083164881858]),
    UInt256(w: [2055512408455909666, 1870755586689442725, 2044780617540418477, 3737551353956102323]),
    UInt256(w: [3590538264498637493, 3475065250894595655, 17418888950458619212, 2335969596222563951]),
    UInt256(w: [18323230885905460578, 4343831563618244568, 17161925169645886111, 2919961995278204939]),
    UInt256(w: [4457294533672274107, 818103436095417807, 16840720443629969735, 3649952494097756174]),
    UInt256(w: [959932148662954729, 14857687350401435971, 11827528517682686360, 4562440617622195218]),
    UInt256(w: [16740858657410204370, 9286054594000897481, 12003891341979066879, 2851525386013872011]),
    UInt256(w: [7091015266480591750, 6995882224073733948, 10393178159046445695, 3564406732517340014]),
    UInt256(w: [8863769083100739688, 4133166761664779531, 3768100661953281311, 4455508415646675018]),
    UInt256(w: [3234012667724268353, 9500758253681569063, 6966748932148188723, 2784692759779171886]),
    UInt256(w: [17877573889937499153, 7264261798674573424, 17931808202040011712, 3480865949723964857]),
    UInt256(w: [3900223288712322326, 9080327248343216781, 8579702197267850928, 4351082437154956072]),
    UInt256(w: [4743482564658895406, 5675204530214510488, 5362313873292406830, 2719426523221847545]),
    UInt256(w: [5929353205823619257, 16317377699622913918, 11314578360042896441, 3399283154027309431]),
    UInt256(w: [16635063544134299879, 6561664069246478685, 9531536931626232648, 4249103942534136789]),
    UInt256(w: [12702757724297631377, 4101040043279049178, 8263053591480089357, 2655689964083835493]),
    UInt256(w: [6655075118517263413, 9737986072526199377, 14940503007777499600, 3319612455104794366]),
    UInt256(w: [12930529916573967170, 12172482590657749221, 9452256722867098692, 4149515568880992958]),
    UInt256(w: [10387424207072423433, 16831173656015869071, 1295974433364548778, 2593447230550620599]),
    UInt256(w: [8372594240413141387, 11815595033165060531, 15455026096987849685, 3241809038188275748]),
    UInt256(w: [5854056782089038830, 934435736174161952, 872038547525260491, 4052261297735344686]),
    UInt256(w: [3658785488805649269, 16724923399604708884, 14380082147485451518, 2532663311084590428]),
    UInt256(w: [4573481861007061586, 11682782212651110297, 17975102684356814398, 3165829138855738035]),
    UInt256(w: [10328538344686214887, 5380105728959112063, 17857192337018630094, 3957286423569672544]),
    UInt256(w: [13372865493069966160, 17197624135881608751, 11160745210636643808, 2473304014731045340]),
    UInt256(w: [12104395847910069796, 3050286096142459323, 13950931513295804761, 3091630018413806675]),
    UInt256(w: [10518808791460199341, 8424543638605462058, 12826978373192368047, 3864537523017258344]),
    UInt256(w: [11185941513090012492, 12182868801769495642, 8016861483245230029, 2415335951885786465]),
    UInt256(w: [4759054854507739807, 1393527946929705841, 14632762872483925441, 3019169939857233081]),
    UInt256(w: [10560504586562062663, 6353595952089520205, 4455895535322743089, 3773962424821541352]),
    UInt256(w: [8906158375814983117, 15500212516124419888, 2784934709576714430, 2358726515513463345]),
    UInt256(w: [11132697969768728896, 10151893608300749052, 8092854405398280942, 2948408144391829181]),
    UInt256(w: [13915872462210911119, 3466494973521160507, 14727754025175239082, 3685510180489786476]),
    UInt256(w: [12783154559336250995, 13556490753756226442, 18409692531469048852, 4606887725612233095]),
    UInt256(w: [12601157618012544776, 17696178757952417334, 18423586859809237388, 2879304828507645684]),
    UInt256(w: [6528074985660905162, 3673479373730970052, 4582739501051995120, 3599131035634557106]),
    UInt256(w: [8160093732076131452, 4591849217163712565, 14951796413169769708, 4498913794543196382]),
    UInt256(w: [7405901591761276110, 12093277797582096161, 4733186739803718163, 2811821121589497739]),
    UInt256(w: [13869063008128983041, 10504911228550232297, 1304797406327259800, 3514776401986872174]),
    UInt256(w: [3501270704879065090, 13131139035687790372, 10854368794763850558, 4393470502483590217]),
    UInt256(w: [11411666227404191489, 3595275878877481078, 18313195542795876359, 2745919064052243885]),
    UInt256(w: [5041210747400463553, 18329152903879015060, 9056436373212681736, 3432398830065304857]),
    UInt256(w: [6301513434250579442, 4464697056139217209, 15932231484943240075, 4290498537581631071]),
    UInt256(w: [15467660942475081911, 484592650873316803, 16875173705730606903, 2681561585988519419]),
    UInt256(w: [14722890159666464485, 14440798868873809716, 16482281113735870724, 3351951982485649274]),
    UInt256(w: [18403612699583080606, 18050998586092262145, 11379479355315062597, 4189939978107061593]),
    UInt256(w: [4584728909598343523, 13587717125521357793, 194645569430832267, 2618712486316913496]),
    UInt256(w: [10342597155425317307, 12372960388474309337, 243306961788540334, 3273390607896141870]),
    UInt256(w: [17539932462709034538, 6242828448738110863, 9527505739090451226, 4091738259870177337]),
    UInt256(w: [17879986816834228443, 8513453798888707193, 17483906133000001776, 2557336412418860835]),
    UInt256(w: [8514925465760621841, 10641817248610883992, 17243196647822614316, 3196670515523576044]),
    UInt256(w: [10643656832200777301, 13302271560763604990, 3107251736068716279, 3995838144404470056]),
    UInt256(w: [2040599501698097909, 15231448753118334975, 1942032335042947674, 2497398840252793785]),
    UInt256(w: [16385807432404786099, 9815938904543142910, 7039226437231072497, 3121748550315992231]),
    UInt256(w: [11258887253651206815, 16881609649106316542, 4187347028111452717, 3902185687894990289]),
    UInt256(w: [2425118515104616356, 12856849039905141791, 14146306938638127708, 2438866054934368930]),
    UInt256(w: [16866456199162934157, 16071061299881427238, 8459511636442883827, 3048582568667961163]),
    UInt256(w: [11859698212098891888, 15477140606424396144, 5962703527126216880, 3810728210834951454]),
    UInt256(w: [7412311382561807430, 9673212879015247590, 17561747759736049262, 2381705131771844658]),
    UInt256(w: [42017191347483479, 2868144061914283680, 12728812662815285770, 2977131414714805823]),
    UInt256(w: [52521489184354349, 12808552114247630408, 11299329810091719308, 3721414268393507279]),
    UInt256(w: [32825930740221468, 17228717108259544813, 13979610158948406423, 2325883917745942049]),
    UInt256(w: [4652718431852664739, 16924210366897043112, 3639454643403344317, 2907354897182427562]),
    UInt256(w: [5815898039815830924, 7320204903339140178, 13772690341108956205, 3634193621478034452]),
    UInt256(w: [16493244586624564463, 13761942147601313126, 17215862926386195256, 4542742026847543065]),
    UInt256(w: [5696591848212964885, 8601213842250820704, 3842385301350290179, 2839213766779714416]),
    UInt256(w: [7120739810266206107, 6139831284386137976, 4802981626687862724, 3549017208474643020]),
    UInt256(w: [8900924762832757633, 7674789105482672470, 6003727033359828405, 4436271510593303775]),
    UInt256(w: [951391958343085617, 7102586200140364246, 10669858423490974609, 2772669694120814859]),
    UInt256(w: [10412611984783632829, 13489918768602843211, 8725637010936330357, 3465837117651018574]),
    UInt256(w: [8404078962552153132, 3027340405471390302, 1683674226815637139, 4332296397063773218]),
    UInt256(w: [640863333167707804, 18032988817915476603, 5663982410187161115, 2707685248164858261]),
    UInt256(w: [14636137221741798467, 17929550003966957849, 11691664031161339298, 3384606560206072826]),
    UInt256(w: [4460113471895084371, 13188565468103921504, 5391208002096898315, 4230758200257591033]),
    UInt256(w: [2787570919934427732, 5937010408351256988, 14898720047379031207, 2644223875160994395]),
    UInt256(w: [3484463649918034665, 2809576992011683331, 14011714040796401105, 3305279843951242994]),
    UInt256(w: [18190637617679707043, 8123657258441992067, 8291270514140725573, 4131599804939053743]),
    UInt256(w: [9063305501836122950, 7383128795739938994, 12099573098979035339, 2582249878086908589]),
    UInt256(w: [2105759840440377880, 4617224976247535839, 1289408318441630462, 3227812347608635737]),
    UInt256(w: [16467257855832636061, 14994903257164195606, 6223446416479425981, 4034765434510794671]),
    UInt256(w: [5680350141468009635, 11677657544941316206, 10807183037940723094, 2521728396569246669]),
    UInt256(w: [16323809713689787851, 5373699894321869449, 18120664815853291772, 3152160495711558336]),
    UInt256(w: [6569704086830071102, 6717124867902336812, 4204086946107063099, 3940200619639447921]),
    UInt256(w: [13329437091123570247, 1892360033225266555, 14156769387385384197, 2462625387274654950]),
    UInt256(w: [12050110345477074904, 6977136059958971098, 8472589697376954438, 3078281734093318688]),
    UInt256(w: [5839265894991567822, 17944792111803489681, 10590737121721193047, 3847852167616648360]),
    UInt256(w: [15178756230438199649, 18133024097518262906, 6619210701075745654, 2404907604760405225]),
    UInt256(w: [9750073251192973753, 13442908085043052825, 12885699394772069972, 3006134505950506531]),
    UInt256(w: [16799277582418605095, 16803635106303816031, 11495438225037699561, 3757668132438133164]),
    UInt256(w: [17417077516652710041, 3584742913798803163, 16408020927503338034, 2348542582773833227]),
    UInt256(w: [17159660877388499647, 13704300679103279762, 15898340140951784638, 2935678228467291534]),
    UInt256(w: [12226204059880848751, 7907003812024323895, 10649553139334954990, 3669597785584114418]),
    UInt256(w: [10671069056423673034, 660382728175629061, 4088569387313917930, 4586997231980143023]),
    UInt256(w: [8975261169478489598, 5024425223537156067, 9472884894712280562, 2866873269987589389]),
    UInt256(w: [6607390443420724094, 15503903566276220892, 16452792136817738606, 3583591587484486736]),
    UInt256(w: [8259238054275905117, 10156507420990500307, 2119246097312621642, 4479489484355608421]),
    UInt256(w: [2856180774708746746, 10959503156546450596, 3630371820034082478, 2799680927722255263]),
    UInt256(w: [3570225968385933433, 4476006908828287437, 18373022830324766810, 3499601159652819078]),
    UInt256(w: [9074468478909804695, 14818380672890135104, 13742906501051182704, 4374501449566023848]),
    UInt256(w: [5671542799318627935, 9261487920556334440, 8589316563156989190, 2734063405978764905]),
    UInt256(w: [7089428499148284918, 2353487863840642242, 15348331722373624392, 3417579257473456131]),
    UInt256(w: [18085157660790131956, 2941859829800802802, 14573728634539642586, 4271974071841820164]),
    UInt256(w: [15914909556421220376, 6450348412052889655, 18331952433442052424, 2669983794901137602]),
    UInt256(w: [15281950927099137566, 8062935515066112069, 13691568504947789722, 3337479743626422003]),
    UInt256(w: [5267380603591758246, 855297356977864279, 12502774612757349249, 4171849679533027504]),
    UInt256(w: [10209641904885930760, 12063775894179634934, 7814234132973343280, 2607406049708142190]),
    UInt256(w: [3538680344252637642, 15079719867724543668, 544420629361903292, 3259257562135177738]),
    UInt256(w: [4423350430315797052, 402905760946127969, 9903897823557154924, 4074071952668972172]),
    UInt256(w: [14293809065015842918, 9475188137446105788, 15413308176577997635, 2546294970418107607]),
    UInt256(w: [17867261331269803647, 7232299153380244331, 14654949202295109140, 3182868713022634509]),
    UInt256(w: [17722390645659866655, 9040373941725305414, 4483628447586722713, 3978585891278293137]),
    UInt256(w: [6464808135110028755, 17179448759646785644, 14331482825810171455, 2486616182048933210]),
    UInt256(w: [8081010168887535944, 16862624931131094151, 8690981495407938511, 3108270227561166513]),
    UInt256(w: [5489576692682032026, 16466595145486479785, 15475412887687311043, 3885337784451458141]),
    UInt256(w: [14960200478994739776, 7985778956715355913, 11977976064018263354, 2428336115282161338]),
    UInt256(w: [4865192543461261008, 758851659039419084, 5749098043168053385, 3035420144102701673]),
    UInt256(w: [6081490679326576260, 5560250592226661759, 11798058572387454635, 3794275180128377091]),
    UInt256(w: [10718460702220192019, 1169313610927969647, 5067943598528465195, 2371421987580235682]),
    UInt256(w: [8786389859347852119, 15296700068942125771, 15558301535015357301, 2964277484475294602]),
    UInt256(w: [6371301305757427245, 5285817030895493502, 10224504881914420819, 3705346855594118253]),
    UInt256(w: [17817121371380555740, 997792635095989486, 8696158560410206964, 2315841784746323908]),
    UInt256(w: [13048029677370918867, 1247240793869986858, 10870198200512758705, 2894802230932904885]),
    UInt256(w: [7086665059858872776, 6170737010764871477, 18199433769068336285, 3618502788666131106]),
    UInt256(w: [13470017343250978874, 12325107281883477250, 13525920174480644548, 4523128485832663883]),
    UInt256(w: [13030446857959249701, 16926564088031949089, 6147857099836708890, 2826955303645414927]),
    UInt256(w: [2453000517166898414, 11934833073185160554, 3073135356368498209, 3533694129556768659]),
    UInt256(w: [12289622683313398825, 1083483286199286980, 17676477250742786474, 4417117661945960823]),
    UInt256(w: [16904386213925650074, 5288863072301942266, 17965327309355323402, 2760698538716225514]),
    UInt256(w: [11907110730552286784, 15834450877232203641, 13233287099839378444, 3450873173395281893]),
    UInt256(w: [1048830357908194768, 1346319522830702936, 2706550819517059344, 4313591466744102367]),
    UInt256(w: [655518973692621730, 841449701769189335, 8609123289839243946, 2695994666715063979]),
    UInt256(w: [14654456772397940874, 10275184164066262476, 6149718093871667028, 3369993333393829974]),
    UInt256(w: [18318070965497426093, 12843980205082828095, 16910519654194359593, 4212491666742287467]),
    UInt256(w: [18366323381076973164, 1109958600535685703, 8263231774657780794, 2632807291713929667]),
    UInt256(w: [18346218207918828551, 10610820287524382937, 5717353699894838088, 3291009114642412084]),
    UInt256(w: [9097714704616371977, 13263525359405478672, 7146692124868547610, 4113761393303015105]),
    UInt256(w: [5686071690385232486, 12901389368055812074, 15995897624111312016, 2571100870814384440]),
    UInt256(w: [16330961649836316415, 16126736710069765092, 1548127956429588404, 3213876088517980551]),
    UInt256(w: [1966957988585843903, 1711676813877654750, 15770218000819149218, 4017345110647475688]),
    UInt256(w: [15064406798148316151, 5681484027100922122, 9856386250511968261, 2510840694154672305]),
    UInt256(w: [9607136460830619381, 11713541052303540557, 16932168831567348230, 3138550867693340381]),
    UInt256(w: [16620606594465662130, 5418554278524649888, 7330152984177021576, 3923188584616675477]),
    UInt256(w: [10387879121541038832, 3386596424077906180, 6887188624324332437, 2451992865385422173]),
    UInt256(w: [12984848901926298539, 8844931548524770629, 13220671798832803450, 3064991081731777716]),
    UInt256(w: [2396003072125709462, 1832792398801187479, 16525839748541004313, 3831238852164722145]),
    UInt256(w: [8415030947719650270, 12674710295319211934, 3411120815197045839, 2394524282602951341]),
    UInt256(w: [1295416647794787029, 11231701850721627014, 8875587037423695203, 2993155353253689176]),
    UInt256(w: [10842642846598259595, 9427941294974645863, 11094483796779619004, 3741444191567111470]),
    UInt256(w: [13694180806764994103, 15115835346213929472, 2322366354559873973, 2338402619729444669]),
    UInt256(w: [17117726008456242628, 5059736127485248128, 7514643961627230371, 2923003274661805836]),
    UInt256(w: [2950413436860751669, 1712984140929172257, 9393304952034037964, 3653754093327257295]),
    UInt256(w: [8299702814503327491, 2141230176161465321, 7129945171615159551, 4567192616659071619]),
    UInt256(w: [16716529305133049442, 8255797887741997681, 2150372723045780767, 2854495385411919762]),
    UInt256(w: [7060603576134148090, 5708061341250109198, 11911337940662001767, 3568119231764899702]),
    UInt256(w: [18049126507022460921, 2523390658135248593, 5665800388972726401, 4460149039706124628]),
    UInt256(w: [4363175039247956220, 13106334207403000131, 12764497279962729808, 2787593149816327892]),
    UInt256(w: [842282780632557370, 16382917759253750164, 15955621599953412260, 3484491437270409865]),
    UInt256(w: [1052853475790696713, 2031903125357636089, 6109468944659601614, 4355614296588012332]),
    UInt256(w: [12187248468437655206, 15104997508630686267, 13041790127267026816, 2722258935367507707]),
    UInt256(w: [10622374567119681103, 434502812078806218, 11690551640656395617, 3402823669209384634]),
    UInt256(w: [4054596172044825571, 5154814533525895677, 5389817513965718713, 4253529586511730793]),
    UInt256(w: [4839965616741709934, 14750974129522154558, 14897850992297043955, 2658455991569831745]),
    UInt256(w: [15273329057781913225, 13827031643475305293, 4787255685089141232, 3323069989462289682]),
    UInt256(w: [5256603266945227819, 17283789554344131617, 15207441643216202348, 4153837486827862102]),
    UInt256(w: [14814592087909237147, 1578996434610306452, 4892965008582738564, 2596148429267413814]),
    UInt256(w: [71496036176994818, 1973745543262883066, 15339578297583199013, 3245185536584267267]),
    UInt256(w: [9312742082076019330, 7078867947505991736, 14562786853551610862, 4056481920730334084]),
    UInt256(w: [5820463801297512082, 18259350522473408547, 18325113820324532596, 2535301200456458802]),
    UInt256(w: [2663893733194502198, 4377444079382209068, 13683020238550889938, 3169126500570573503]),
    UInt256(w: [3329867166493127747, 14695177136082537143, 12492089279761224518, 3961408125713216879]),
    UInt256(w: [8998696006699286698, 4572799691624197810, 14725084827491847180, 2475880078570760549]),
    UInt256(w: [2024997971519332565, 5715999614530247263, 4571297979082645263, 3094850098213450687]),
    UInt256(w: [16366305519681329418, 2533313499735421174, 1102436455425918675, 3868562622766813359]),
    UInt256(w: [5617254931373442982, 17724222001830495898, 7606551812282281027, 2417851639229258349]),
    UInt256(w: [16244940701071579536, 17543591483860731968, 14119875783780239188, 3022314549036572936]),
    UInt256(w: [1859431802629922803, 3482745281116363345, 17649844729725298986, 3777893186295716170]),
    UInt256(w: [12691359922712171512, 6788401819125114994, 15642838974505699770, 2361183241434822606]),
    UInt256(w: [6640827866535438582, 17708874310761169551, 10330176681277348904, 2951479051793528258]),
    UInt256(w: [3689348814741910324, 3689348814741910323, 3689348814741910323, 3689348814741910323]),
    UInt256(w: [0, 0, 0, 4611686018427387904]),
    UInt256(w: [0, 0, 0, 2882303761517117440]),
    UInt256(w: [0, 0, 0, 3602879701896396800]),
    UInt256(w: [0, 0, 0, 4503599627370496000]),
    UInt256(w: [0, 0, 0, 2814749767106560000]),
    UInt256(w: [0, 0, 0, 3518437208883200000]),
    UInt256(w: [0, 0, 0, 4398046511104000000]),
    UInt256(w: [0, 0, 0, 2748779069440000000]),
    UInt256(w: [0, 0, 0, 3435973836800000000]),
    UInt256(w: [0, 0, 0, 4294967296000000000]),
    UInt256(w: [0, 0, 0, 2684354560000000000]),
    UInt256(w: [0, 0, 0, 3355443200000000000]),
    UInt256(w: [0, 0, 0, 4194304000000000000]),
    UInt256(w: [0, 0, 0, 2621440000000000000]),
    UInt256(w: [0, 0, 0, 3276800000000000000]),
    UInt256(w: [0, 0, 0, 4096000000000000000]),
    UInt256(w: [0, 0, 0, 2560000000000000000]),
    UInt256(w: [0, 0, 0, 3200000000000000000]),
    UInt256(w: [0, 0, 0, 4000000000000000000]),
    UInt256(w: [0, 0, 0, 2500000000000000000]),
    UInt256(w: [0, 0, 0, 3125000000000000000]),
    UInt256(w: [0, 0, 0, 3906250000000000000]),
    UInt256(w: [0, 0, 0, 2441406250000000000]),
    UInt256(w: [0, 0, 0, 3051757812500000000]),
    UInt256(w: [0, 0, 0, 3814697265625000000]),
    UInt256(w: [0, 0, 0, 2384185791015625000]),
    UInt256(w: [0, 0, 0, 2980232238769531250]),
    UInt256(w: [0, 0, 9223372036854775808, 3725290298461914062]),
    UInt256(w: [0, 0, 1152921504606846976, 2328306436538696289]),
    UInt256(w: [0, 0, 6052837899185946624, 2910383045673370361]),
    UInt256(w: [0, 0, 12177733392409821184, 3637978807091712951]),
    UInt256(w: [0, 0, 10610480722084888576, 4547473508864641189]),
    UInt256(w: [0, 0, 8937393460516749312, 2842170943040400743]),
    UInt256(w: [0, 0, 6560055807218548736, 3552713678800500929]),
    UInt256(w: [0, 0, 12811755777450573824, 4440892098500626161]),
    UInt256(w: [0, 0, 1089818333265526784, 2775557561562891351]),
    UInt256(w: [0, 0, 15197330971864072192, 3469446951953614188]),
    UInt256(w: [0, 0, 549919641120538624, 4336808689942017736]),
    UInt256(w: [0, 0, 343699775700336640, 2710505431213761085]),
    UInt256(w: [0, 0, 5041310738052808704, 3388131789017201356]),
    UInt256(w: [0, 0, 6301638422566010880, 4235164736271501695]),
    UInt256(w: [0, 0, 10856053041744838656, 2646977960169688559]),
    UInt256(w: [0, 0, 8958380283753660416, 3308722450212110699]),
    UInt256(w: [0, 0, 6586289336264687616, 4135903062765138374]),
    UInt256(w: [0, 0, 17951488890447593472, 2584939414228211483]),
    UInt256(w: [0, 0, 17827675094632103936, 3231174267785264354]),
    UInt256(w: [0, 0, 13061221831435354112, 4038967834731580443]),
    UInt256(w: [0, 0, 5857420635433402368, 2524354896707237777]),
    UInt256(w: [0, 0, 11933461812719140864, 3155443620884047221]),
    UInt256(w: [0, 0, 1081769210616762368, 3944304526105059027]),
    UInt256(w: [0, 0, 16817006821131334144, 2465190328815661891]),
    UInt256(w: [0, 0, 16409572507986779776, 3081487911019577364]),
    UInt256(w: [0, 0, 2065221561273923104, 3851859888774471706]),
    UInt256(w: [0, 0, 5902449494223589844, 2407412430484044816]),
    UInt256(w: [0, 0, 7378061867779487305, 3009265538105056020]),
    UInt256(w: [0, 4611686018427387904, 9222577334724359131, 3761581922631320025]),
    UInt256(w: [0, 576460752303423488, 17293325880271194217, 2350988701644575015]),
    UInt256(w: [0, 5332261958806667264, 17004971331911604867, 2938735877055718769]),
    UInt256(w: [0, 2053641430080946176, 7421156109607342372, 3673419846319648462]),
    UInt256(w: [0, 2567051787601182720, 53073100154402157, 4591774807899560578]),
    UInt256(w: [0, 3910250376464433152, 4644856706023889252, 2869859254937225361]),
    UInt256(w: [0, 4887812970580541440, 10417756900957249469, 3587324068671531701]),
    UInt256(w: [0, 10721452231653064704, 17633882144623949740, 4484155085839414626]),
    UInt256(w: [0, 15924279681637941248, 15632862358817356491, 2802596928649634141]),
    UInt256(w: [0, 15293663583620038656, 5706019893239531902, 3503246160812042677]),
    UInt256(w: [0, 9893707442670272512, 11744210884976802782, 4379057701015053346]),
    UInt256(w: [0, 1571881133241532416, 11951817821537889643, 2736911063134408341]),
    UInt256(w: [0, 15799909471834079232, 1104714221640198341, 3421138828918010427]),
    UInt256(w: [0, 5914828784510435328, 15215950832332411639, 4276423536147513033]),
    UInt256(w: [0, 10614297017960103936, 2592440242566675418, 2672764710092195646]),
    UInt256(w: [0, 4044499235595354112, 12463922340063120081, 3340955887615244557]),
    UInt256(w: [0, 9667310062921580544, 1744844869796736389, 4176194859519055697]),
    UInt256(w: [0, 8347911798539681792, 12619743089691430003, 2610121787199409810]),
    UInt256(w: [0, 5823203729747214336, 6551306825259511696, 3262652233999262263]),
    UInt256(w: [0, 7279004662184017920, 3577447513147001716, 4078315292499077829]),
    UInt256(w: [0, 13772749950719787008, 4541747704930570024, 2548947057811923643]),
    UInt256(w: [0, 17215937438399733760, 1065498612735824626, 3186183822264904554]),
    UInt256(w: [0, 12296549761144891392, 10555245302774556591, 3982729777831130692]),
    UInt256(w: [0, 14602872628356638976, 15820400351088873677, 2489206111144456682]),
    UInt256(w: [0, 4418532730163635008, 10552128402006316289, 3111507638930570853]),
    UInt256(w: [0, 10134851931131931664, 17801846520935283265, 3889384548663213566]),
    UInt256(w: [0, 17863497503025927050, 6514468057157164136, 2430865342914508479]),
    UInt256(w: [9223372036854775808, 3882627805072857196, 3531399053019067267, 3038581678643135599]),
    UInt256(w: [11529215046068469760, 241598737913683591, 18249306871555997796, 3798227098303919498]),
    UInt256(w: [14123288431433875456, 9374371248050828052, 16017502813149886526, 2373891936439949686]),
    UInt256(w: [17654110539292344320, 2494592023208759257, 10798506479582582350, 2967364920549937108]),
    UInt256(w: [8232580118833266688, 12341612065865724880, 13498133099478227937, 3709206150687421385]),
    UInt256(w: [5145362574270791680, 795978513524996194, 1518804159532810605, 2318253844179638366]),
    UInt256(w: [15655075254693265408, 5606659160333633146, 11121877236270789064, 2897817305224547957]),
    UInt256(w: [10345472031511805952, 7008323950417041433, 67288490056322618, 3622271631530684947]),
    UInt256(w: [17543526057817145344, 17983776974876077599, 13919168667852566984, 4527839539413356183]),
    UInt256(w: [17882232813776797696, 11239860609297548499, 15617009445048936221, 2829899712133347614]),
    UInt256(w: [17741104998793609216, 214767706339771912, 10297889769456394469, 3537374640166684518]),
    UInt256(w: [3729637174782459904, 4880145651352102795, 3648990174965717278, 4421718300208355648]),
    UInt256(w: [25180225025343488, 16885149087377227959, 2280618859353573298, 2763573937630222280]),
    UInt256(w: [13866533336563843072, 11883064322366759140, 2850773574191966623, 3454467422037777850]),
    UInt256(w: [17333166670704803840, 10242144384531061021, 12786839004594734087, 4318084277547222312]),
    UInt256(w: [13139072178404196352, 13318869267972994994, 7991774377871708804, 2698802673467013945]),
    UInt256(w: [7200468186150469632, 16648586584966243743, 14601403990767023909, 3373503341833767431]),
    UInt256(w: [4388899214260699136, 6975675175925640967, 13640068970031391983, 4216879177292209289]),
    UInt256(w: [9660591036554018816, 11277326012594607460, 1607514078628538133, 2635549485807630806]),
    UInt256(w: [12075738795692523520, 261599460461095613, 11232764635140448475, 3294436857259538507]),
    UInt256(w: [1259615439333490688, 14162057380858533229, 9429269775498172689, 4118046071574423134]),
    UInt256(w: [3093102658797125632, 1933756835395501412, 1281607591258970027, 2573778794734014459]),
    UInt256(w: [3866378323496407040, 16252254099526540477, 15437067544355876245, 3217223493417518073]),
    UInt256(w: [9444658922797896704, 6480259569126011884, 5461276375162681595, 4021529366771897592]),
    UInt256(w: [15126283863603461248, 1744319221490063475, 3413297734476675997, 2513455854232435995]),
    UInt256(w: [14296168811076938656, 6792085045289967248, 18101680223378008708, 3141819817790544993]),
    UInt256(w: [17870211013846173320, 8490106306612459060, 8792042223940347173, 3927274772238181242]),
    UInt256(w: [1945509846799082517, 7612159450846480865, 10106712408390104887, 2454546732648863276]),
    UInt256(w: [7043573326926241051, 4903513295130713177, 12633390510487631109, 3068183415811079095]),
    UInt256(w: [13416152677085189217, 10741077637340779375, 11180052119682150982, 3835229269763848869]),
    UInt256(w: [15302624450819325117, 2101487504910599205, 9293375584015038316, 2397018293602405543]),
    UInt256(w: [5293222508241992684, 2626859381138249007, 7005033461591409991, 2996272867003006929]),
    UInt256(w: [2004842116875102951, 17118632281704974971, 13367977845416650392, 3745341083753758661]),
    UInt256(w: [17393927387542797009, 10699145176065609356, 10660829162599100447, 2340838177346099163]),
    UInt256(w: [3295665160718944645, 8762245451654623792, 8714350434821487655, 2926047721682623954]),
    UInt256(w: [4119581450898680806, 6341120796140891836, 1669566006672083761, 3657559652103279943]),
    UInt256(w: [5149476813623351007, 12538087013603502699, 15922015563622268413, 4571949565129099928]),
    UInt256(w: [912579999300900428, 10142147392715883139, 9951259727263917758, 2857468478205687455]),
    UInt256(w: [14975783054408289246, 3454312204040078115, 7827388640652509294, 3571835597757109319]),
    UInt256(w: [14108042799582973654, 13541262291904873452, 5172549782388248713, 4464794497196386649]),
    UInt256(w: [18040898786594134342, 1545759904799464051, 14762058660061125206, 2790496560747741655]),
    UInt256(w: [17939437464815280023, 11155571917854105872, 13840887306649018603, 3488120700934677069]),
    UInt256(w: [3977552757309548413, 9332778878890244437, 3466051078029109542, 4360150876168346337]),
    UInt256(w: [4791813482532161710, 1221300780879014869, 13695496969836663224, 2725094297605216460]),
    UInt256(w: [10601452871592590042, 1526625976098768586, 17119371212295829030, 3406367872006520575]),
    UInt256(w: [4028444052635961744, 11131654506978236541, 16787527996942398383, 4257959840008150719]),
    UInt256(w: [4823620542111170042, 13874813094502479694, 17409734025730080845, 2661224900005094199]),
    UInt256(w: [15252897714493738360, 3508458312845935905, 17150481513735213153, 3326531125006367749]),
    UInt256(w: [5231064087835009238, 8997258909484807786, 7603043836886852729, 4158163906257959687]),
    UInt256(w: [7881101073324268678, 17152501864496474626, 11669431425695364811, 2598852441411224804]),
    UInt256(w: [628004304800560040, 16828941312193205379, 14586789282119206014, 3248565551764031005]),
    UInt256(w: [14620063436282863761, 11812804603386730915, 4398428547366843806, 4060706939705038757]),
    UInt256(w: [6831696638463095899, 2771316858689318918, 5054860851317971331, 2537941837315649223]),
    UInt256(w: [17762992834933645682, 17299204128643812359, 1706890045720076259, 3172427296644561529]),
    UInt256(w: [17592055025239669198, 17012319142377377545, 6745298575577483228, 3965534120805701911]),
    UInt256(w: [4077505363133711393, 1409327427131085158, 11133340637377008874, 2478458825503563694]),
    UInt256(w: [14320253740771915049, 10985031320768632255, 4693303759866485284, 3098073531879454618]),
    UInt256(w: [13288631157537505907, 13731289150960790319, 15090001736687882413, 3872591914849318272]),
    UInt256(w: [15222923501102023048, 10887898728564187901, 9431251085429926508, 2420369946780823920]),
    UInt256(w: [5193596321095365098, 13609873410705234877, 11789063856787408135, 3025462433476029900]),
    UInt256(w: [11103681419796594277, 12400655744954155692, 14736329820984260169, 3781828041845037375]),
    UInt256(w: [16163172924227647231, 832880812955265451, 16127735165756244462, 2363642526153148359]),
    UInt256(w: [15592280136857171135, 10264473053048857622, 15547982938767917673, 2954553157691435449]),
    UInt256(w: [10266978134216688110, 17442277334738459932, 5599920618177733379, 3693191447114294312]),
    UInt256(w: [15640233370740205877, 8595580324997843505, 3499950386361083362, 2308244654446433945]),
    UInt256(w: [5715233658143093634, 1521103369392528574, 8986624001378742107, 2885305818058042431]),
    UInt256(w: [16367414109533642851, 15736437267022824429, 6621593983296039729, 3606632272572553039]),
    UInt256(w: [6624209581634889851, 5835488528496366825, 3665306460692661758, 4508290340715691299]),
    UInt256(w: [15669346034590275917, 17482238385592392977, 18431717602428771262, 2817681462947307061]),
    UInt256(w: [5751624487955681184, 12629425945135715414, 9204588947753800366, 3522101828684133827]),
    UInt256(w: [16412902646799377288, 6563410394564868459, 6894050166264862554, 4402627285855167284]),
    UInt256(w: [7952221145035916853, 8713817515030430691, 13532153390770314904, 2751642053659479552]),
    UInt256(w: [5328590412867508163, 10892271893788038364, 16915191738462893630, 3439552567074349440]),
    UInt256(w: [6660738016084385203, 4391967830380272147, 2697245599369065422, 4299440708842936801]),
    UInt256(w: [1857118250839046800, 16580037949269833804, 13214993545674135648, 2687150443026835500]),
    UInt256(w: [2321397813548808500, 2278303362877740639, 16518741932092669561, 3358938053783544375]),
    UInt256(w: [16736805322218174337, 7459565222024563702, 16036741396688449047, 4198672567229430469]),
    UInt256(w: [5848817307958971057, 11579757291406434170, 12328806382143974606, 2624170354518394043]),
    UInt256(w: [16534393671803489629, 5251324577403266904, 10799321959252580354, 3280212943147992554]),
    UInt256(w: [2221248016044810420, 15787527758608859439, 4275780412210949634, 4100266178934990693]),
    UInt256(w: [8305809037669088369, 14478890867557925053, 4978205766845537473, 2562666361834369183]),
    UInt256(w: [14993947315513748365, 4263555529165242604, 1611071190129533938, 3203332952292961479]),
    UInt256(w: [295690070682633840, 14552816448311329064, 15848897042944081134, 4004166190366201848]),
    UInt256(w: [184806294176646150, 4483824261767192761, 9905560651840050709, 2502603868978876155]),
    UInt256(w: [4842693886148195591, 10216466345636378855, 7770264796372675482, 3128254836223595194]),
    UInt256(w: [1441681339257856585, 3547210895190697761, 489458958611068545, 3910318545279493993]),
    UInt256(w: [12430265883104630126, 13746221855562655860, 11835126895200387600, 2443949090799683745]),
    UInt256(w: [15537832353880787657, 17182777319453319825, 958850563718320788, 3054936363499604682]),
    UInt256(w: [5587232387068820859, 3031727575607098166, 10421935241502676794, 3818670454374505852]),
    UInt256(w: [17327078297200176749, 6506515753181824257, 15737081562793948804, 2386669033984066157]),
    UInt256(w: [7823789816218057224, 8133144691477280322, 5836293898210272293, 2983336292480082697]),
    UInt256(w: [556365233417795722, 14778116882773988307, 11907053391190228270, 3729170365600103371]),
    UInt256(w: [16488629335381979991, 4624637033306354787, 5136065360280198717, 2330731478500064607]),
    UInt256(w: [15999100650800087084, 10392482310060331388, 1808395681922860492, 2913414348125080759]),
    UInt256(w: [1552131739790557239, 12990602887575414236, 16095552657685739327, 3641767935156350948]),
    UInt256(w: [1940164674738196549, 11626567591041879891, 1672696748397622543, 4552209918945438686]),
    UInt256(w: [17353503986207230507, 14184133772042256787, 14880493523030677801, 2845131199340899178]),
    UInt256(w: [17080193964331650230, 3895109159770657272, 9377244866933571444, 3556413999176123973]),
    UInt256(w: [2903498381705011171, 4868886449713321591, 16333242102094352209, 4445517498970154966]),
    UInt256(w: [8732215516206713838, 14572269077139295754, 5596590295381582226, 2778448436856346854]),
    UInt256(w: [1691897358403616490, 8991964309569343885, 16219109906081753591, 3473060546070433567]),
    UInt256(w: [6726557716431908516, 6628269368534291952, 15662201364174804085, 4341325682588041959]),
    UInt256(w: [4204098572769942823, 6448511364547626422, 16706404880250334409, 2713328551617526224]),
    UInt256(w: [14478495252817204336, 12672325224111920931, 2436262026603366395, 3391660689521907781]),
    UInt256(w: [13486433047594117516, 11228720511712513260, 7657013551681595898, 4239575861902384726]),
    UInt256(w: [17652392691601099256, 11629636338247708691, 173947451373609532, 2649734913688990454]),
    UInt256(w: [17453804846073986166, 14537045422809635864, 9440806351071787723, 3312168642111238067]),
    UInt256(w: [3370511983882931091, 13559620760084656927, 7189321920412346750, 4140210802639047584]),
    UInt256(w: [9024099017567913788, 3863076956625522675, 4493326200257716719, 2587631751649404740]),
    UInt256(w: [6668437753532504331, 217160177354515440, 5616657750322145899, 3234539689561755925]),
    UInt256(w: [8335547191915630413, 14106508276975308012, 11632508206330070277, 4043174611952194906]),
    UInt256(w: [14433089031802044817, 11122410682323261459, 11882003647383681827, 2526984132470121816]),
    UInt256(w: [13429675271325168117, 9291327334476688920, 14852504559229602284, 3158730165587652270]),
    UInt256(w: [16787094089156460146, 11614159168095861150, 9342258662182227047, 3948412706984565338]),
    UInt256(w: [5880247787295399687, 14176378507700995075, 10450597682291279808, 2467757941865353336]),
    UInt256(w: [2738623715691861705, 17720473134626243844, 13063247102864099760, 3084697427331691670]),
    UInt256(w: [3423279644614827131, 3703847344573253189, 7105686841725348893, 3855871784164614588]),
    UInt256(w: [4445392787097960909, 4620747599571977195, 13664426312933118866, 2409919865102884117]),
    UInt256(w: [945054965445063232, 14999306536319747302, 3245474835884234870, 3012399831378605147]),
    UInt256(w: [10404690743661104848, 9525761133544908319, 17891901600137457300, 3765499789223256433]),
    UInt256(w: [13420460742429272386, 15176972745320343507, 4264909472444828956, 2353437368264535271]),
    UInt256(w: [12163889909609202579, 524471857940877768, 719450822128648292, 2941796710330669089]),
    UInt256(w: [15204862387011503223, 655589822426097210, 5510999546088198269, 3677245887913336361]),
    UInt256(w: [9782705946909603221, 5431173296460009417, 11500435451037635740, 4596557359891670451]),
    UInt256(w: [17643406262886971773, 12617855347142281693, 4881929147684828385, 2872848349932294032]),
    UInt256(w: [8219199773326551005, 1937261128645688405, 6102411434606035482, 3591060437415367540]),
    UInt256(w: [14885685735085576660, 11644948447661886314, 7628014293257544352, 4488825546769209425]),
    UInt256(w: [13915239602855873316, 7278092779788678946, 16296723979354434980, 2805515966730755890]),
    UInt256(w: [8170677466715065837, 9097615974735848683, 11147532937338267917, 3506894958413444863]),
    UInt256(w: [5601660814966444393, 15983705986847198758, 9322730153245446992, 4383618698016806079]),
    UInt256(w: [17336096064636191458, 9989816241779499223, 12744235373419486226, 2739761686260503799]),
    UInt256(w: [17058434062367851418, 3263898265369598221, 11318608198346969879, 3424702107825629749]),
    UInt256(w: [7487984522677650560, 17914930886994161489, 313202192651548636, 4280877634782037187]),
    UInt256(w: [16209205372742001360, 1973459767516575122, 16336652434903075562, 2675548521738773241]),
    UInt256(w: [11038134679072725892, 11690196746250494711, 6585757488346680740, 3344435652173466552]),
    UInt256(w: [9185982330413519461, 14612745932813118389, 8232196860433350925, 4180544565216833190]),
    UInt256(w: [8047081965722143615, 11438809217221892945, 533437019343456424, 2612840353260520744]),
    UInt256(w: [14670538475580067423, 14298511521527366181, 666796274179320530, 3266050441575650930]),
    UInt256(w: [4503115039192920567, 8649767365054431919, 10056867379578926471, 4082563051969563662]),
    UInt256(w: [9731975927136657210, 12323633630800101805, 1673856093809441140, 2551601907480977289]),
    UInt256(w: [16776655927348209417, 15404542038500127256, 6704006135689189329, 3189502384351221611]),
    UInt256(w: [2524075835475710155, 5420619492842995359, 3768321651184098758, 3986877980439027014]),
    UInt256(w: [8495076424813400703, 17222945238309035811, 16190259087272225435, 2491798737774391883]),
    UInt256(w: [6007159512589362975, 16916995529458906860, 15626137840662893890, 3114748422217989854]),
    UInt256(w: [7508949390736703718, 11922872374968857767, 10309300263973841555, 3893435527772487318]),
    UInt256(w: [11610622396851521680, 5145952225141842152, 1831626646556263068, 2433397204857804574]),
    UInt256(w: [14513277996064402100, 6432440281427302690, 11512905345050104643, 3041746506072255717]),
    UInt256(w: [8918225458225726817, 3428864333356740459, 556073626030467092, 3802183132590319647]),
    UInt256(w: [3268047902177385309, 11366412245202738595, 7265075043910123788, 2376364457868949779]),
    UInt256(w: [17920117933003895348, 14208015306503423243, 4469657786460266831, 2970455572336187224]),
    UInt256(w: [17788461397827481281, 13148333114701891150, 5587072233075333539, 3713069465420234030]),
    UInt256(w: [6506102355214787897, 5911865187474988017, 17326978200954247174, 2320668415887646268]),
    UInt256(w: [12744313962445872775, 16613203521198510829, 3211978677483257351, 2900835519859557836]),
    UInt256(w: [2095334397775177256, 16154818383070750633, 4014973346854071689, 3626044399824447295]),
    UInt256(w: [7230854015646359474, 6358464923556274579, 407030665140201708, 4532555499780559119]),
    UInt256(w: [2213440750565280719, 13197412614077447420, 7171923193353707923, 2832847187362849449]),
    UInt256(w: [2766800938206600899, 11885079749169421371, 13576590010119522808, 3541058984203561811]),
    UInt256(w: [17293559228040414836, 14856349686461776713, 12359051494222015606, 4426323730254452264]),
    UInt256(w: [3890945489884177417, 4673532535611222542, 7724407183888759754, 2766452331409032665]),
    UInt256(w: [14087053899209997579, 15065287706368803985, 14267194998288337596, 3458065414261290831]),
    UInt256(w: [3773759318730333261, 384865559251453366, 13222307729433034092, 4322581767826613539]),
    UInt256(w: [16193657629488622000, 9463913011386934161, 5958099321681952355, 2701613604891633462]),
    UInt256(w: [6407013981578613788, 7218205245806279798, 16670996188957216252, 3377017006114541827]),
    UInt256(w: [17232139513828043043, 9022756557257849747, 16227059217769132411, 4221271257643177284]),
    UInt256(w: [8464244186928832950, 3333379839072462140, 918539974250931949, 2638294536026985803]),
    UInt256(w: [10580305233661041188, 8778410817267965579, 14983233023095828648, 3297868170033732253]),
    UInt256(w: [8613695523648913581, 10973013521584956974, 4893983223587622098, 4122335212542165317]),
    UInt256(w: [771873683853183084, 11469819469417986013, 5364582523955957763, 2576459507838853323]),
    UInt256(w: [5576528123243866759, 9725588318345094612, 2094042136517559300, 3220574384798566654]),
    UInt256(w: [6970660154054833449, 12156985397931368265, 11840924707501724933, 4025717980998208317]),
    UInt256(w: [15885877642352740666, 9903958882920799117, 9706420951402272035, 2516073738123880198]),
    UInt256(w: [6022288997658762120, 7768262585223610993, 2909654152398064236, 3145092172654850248]),
    UInt256(w: [12139547265500840554, 9710328231529513741, 3637067690497580295, 3931365215818562810]),
    UInt256(w: [9893060050151719298, 12986484172347027944, 6884853324988375588, 2457103259886601756]),
    UInt256(w: [12366325062689649123, 16233105215433784930, 8606066656235469485, 3071379074858252195]),
    UInt256(w: [6234534291507285595, 6456323464010067451, 6145897301866948953, 3839223843572815244]),
    UInt256(w: [1590740922978359545, 15564417211074761917, 13064557850521618903, 2399514902233009527]),
    UInt256(w: [6600112172150337335, 14843835495416064492, 11719011294724635725, 2999393627791261909]),
    UInt256(w: [8250140215187921669, 4719736313987916903, 813706063123630945, 3749242034739077387]),
    UInt256(w: [12073866662133532899, 14479050242310917824, 16649467353948127004, 2343276271711923366]),
    UInt256(w: [15092333327666916124, 18098812802888647280, 11588462155580382947, 2929095339639904208]),
    UInt256(w: [418672585874093539, 18011829985183421197, 14485577694475478684, 3661369174549880260]),
    UInt256(w: [5135026750770004827, 4068043407769724880, 18106972118094348356, 4576711468187350325]),
    UInt256(w: [3209391719231253017, 11765899166710853858, 13622700583022661674, 2860444667617093953]),
    UInt256(w: [13235111685893842080, 5484001921533791514, 3193317673496163381, 3575555834521367442]),
    UInt256(w: [7320517570512526791, 11466688420344627297, 13215019128724980034, 4469444793151709302]),
    UInt256(w: [16104538527638799005, 11778366281142779964, 3647700937025724617, 2793402995719818314]),
    UInt256(w: [1683929085838947140, 887899796146311244, 13782998208136931580, 3491753744649772892]),
    UInt256(w: [2104911357298683925, 1109874745182889055, 17228747760171164475, 4364692180812216115]),
    UInt256(w: [8233098625952759309, 16834572780235163323, 8462124340893283844, 2727932613007635072]),
    UInt256(w: [5679687264013561232, 2596471901584402538, 10577655426116604806, 3409915766259543840]),
    UInt256(w: [16322981116871727348, 12468961913835278980, 13222069282645756007, 4262394707824429800]),
    UInt256(w: [978491161190053785, 14710630223788131219, 8263793301653597504, 2663996692390268625]),
    UInt256(w: [15058172006769730943, 18388287779735164023, 14941427645494384784, 3329995865487835781]),
    UInt256(w: [14211028990034775774, 4538615650959403413, 4841726501585817269, 4162494831859794727]),
    UInt256(w: [11187736127985428811, 5142477791063321085, 9943608091132217649, 2601559269912371704]),
    UInt256(w: [149612104699622302, 11039783257256539261, 12429510113915272061, 3251949087390464630]),
    UInt256(w: [4798701149301915781, 18411415089998061980, 6313515605539314268, 4064936359238080788]),
    UInt256(w: [12222560255168473171, 2283762394394012929, 13169319290316847226, 2540585224523800492]),
    UInt256(w: [1443142263678427752, 12078075029847291970, 16461649112896059032, 3175731530654750615]),
    UInt256(w: [11027299866452810498, 15097593787309114962, 15965375372692685886, 3969664413318438269]),
    UInt256(w: [11503748434960394466, 4824310098640808947, 12284202617146622631, 2481040258324023918]),
    UInt256(w: [9767999525273105178, 1418701604873623280, 6131881234578502481, 3101300322905029898]),
    UInt256(w: [12209999406591381472, 6385063024519417004, 16888223580077903909, 3876625403631287372]),
    UInt256(w: [16854621665974389228, 6296507399538329579, 1331767700693914135, 2422890877269554608]),
    UInt256(w: [16456591064040598631, 3258948230995524070, 1664709625867392669, 3028613596586943260]),
    UInt256(w: [11347366793195972481, 8685371307171792992, 2080887032334240836, 3785766995733679075]),
    UInt256(w: [7092104245747482801, 14651729103837146428, 17441455459704758186, 2366104372333549421]),
    UInt256(w: [8865130307184353501, 9091289342941657227, 7966761269348784021, 2957630465416936777]),
    UInt256(w: [6469726865553053972, 15975797697104459438, 14570137605113367930, 3697038081771170971]),
    UInt256(w: [17878637346252822445, 14596559579117675052, 6800492993982161004, 2310648801106981857]),
    UInt256(w: [3901552609106476440, 18245699473897093816, 13112302260905089159, 2888311001383727321]),
    UInt256(w: [4876940761383095549, 18195438323943979366, 2555319770849197737, 3610388751729659152]),
    UInt256(w: [15319547988583645245, 8909239849647810495, 3194149713561497172, 4512985939662073940]),
    UInt256(w: [16492246520505860134, 14791646942884657367, 11219715607830711540, 2820616212288796212]),
    UInt256(w: [16003622132204937264, 42814604896270093, 14024644509788389426, 3525770265360995265]),
    UInt256(w: [6169469609974007867, 9276890292975113425, 3695747581953323070, 4407212831701244082]),
    UInt256(w: [15385133552302224677, 1186370414682057986, 6921528257148214823, 2754508019813277551]),
    UInt256(w: [10008044903523005038, 15318021073634736195, 4040224303007880624, 3443135024766596939]),
    UInt256(w: [7898370110976368394, 700782268333868628, 438594360332462877, 4303918780958246174]),
    UInt256(w: [14159853356215006054, 2743831926922361844, 14109179530489953010, 2689949238098903858]),
    UInt256(w: [17699816695268757568, 12653161945507728113, 8413102376257665454, 3362436547623629823]),
    UInt256(w: [8289712813803783248, 6593080395029884334, 5904691951894693914, 4203045684529537279]),
    UInt256(w: [569384490199976626, 8732361265321065613, 10607961497575265552, 2626903552830960799]),
    UInt256(w: [5323416631177358686, 10915451581651332016, 8648265853541694036, 3283629441038700999]),
    UInt256(w: [6654270788971698358, 13644314477064165020, 6198646298499729641, 4104536801298376249]),
    UInt256(w: [13382291279962087282, 1610167520524021281, 15403368982630800786, 2565335500811485155]),
    UInt256(w: [2892806044670445390, 11236081437509802410, 14642525209861113078, 3206669376014356444])
  ]
  
  
  // **********************************************************************
  static let bid_breakpoints_binary64 : [UInt128] = [
    UInt128(w: [5261314576080512960, 21426681862861333]),
    UInt128(w: [4728754506986910400, 34282690980578133]),
    UInt128(w: [11161701235073348928, 27426152784462506]),
    UInt128(w: [5240012173316768832, 21940922227570005]),
    UInt128(w: [8384019477306830144, 35105475564112008]),
    UInt128(w: [14085913211329284736, 28084380451289606]),
    UInt128(w: [7579381754321517504, 22467504361031685]),
    UInt128(w: [12127010806914427968, 35948006977650696]),
    UInt128(w: [6012259830789632064, 28758405582120557]),
    UInt128(w: [15877854308857436608, 23006724465696445]),
    UInt128(w: [12702283447085949312, 18405379572557156]),
    UInt128(w: [12944955885853698240, 29448607316091450]),
    UInt128(w: [10355964708682958592, 23558885852873160]),
    UInt128(w: [8284771766946366848, 18847108682298528]),
    UInt128(w: [9566286012372276672, 30155373891677645]),
    UInt128(w: [7653028809897821312, 24124299113342116]),
    UInt128(w: [2433074233176346752, 19299439290673693]),
    UInt128(w: [203569958340244480, 30879102865077909]),
    UInt128(w: [3852204781414105920, 24703282292062327]),
    UInt128(w: [14149810269357015680, 19762625833649861]),
    UInt128(w: [15260998801487404480, 31620201333839778]),
    UInt128(w: [1140752596964192576, 25296161067071823]),
    UInt128(w: [8291299707055174720, 20236928853657458]),
    UInt128(w: [9576730716546369216, 32379086165851933]),
    UInt128(w: [15040082202720916032, 25903268932681546]),
    UInt128(w: [8342716947434822464, 20722615146145237]),
    UInt128(w: [17037695930637626304, 33156184233832379]),
    UInt128(w: [17319505559252011392, 26524947387065903]),
    UInt128(w: [2787558003175878144, 21219957909652723]),
    UInt128(w: [770743990339494720, 33951932655444357]),
    UInt128(w: [11684641636497326720, 27161546124355485]),
    UInt128(w: [9347713309197861376, 21729236899484388]),
    UInt128(w: [11266992479974667904, 34766779039175021]),
    UInt128(w: [5324245169237824000, 27813423231340017]),
    UInt128(w: [15327442579615990144, 22250738585072013]),
    UInt128(w: [2387815238934122304, 35601181736115222]),
    UInt128(w: [12978298635373028800, 28480945388892177]),
    UInt128(w: [3003941278814602368, 22784756311113742]),
    UInt128(w: [13471199467277412864, 18227805048890993]),
    UInt128(w: [17864570332901950336, 29164488078225589]),
    UInt128(w: [17981005081063470592, 23331590462580471]),
    UInt128(w: [10695455250108866112, 18665272370064377]),
    UInt128(w: [2355333141206544512, 29864435792103004]),
    UInt128(w: [5573615327707145920, 23891548633682403]),
    UInt128(w: [11837589891649537408, 19113238906945922]),
    UInt128(w: [4182748567671618560, 30581182251113476]),
    UInt128(w: [18103594113104936128, 24464945800890780]),
    UInt128(w: [14482875290483948864, 19571956640712624]),
    UInt128(w: [12104554020548587264, 31315130625140199]),
    UInt128(w: [13372992031180780160, 25052104500112159]),
    UInt128(w: [14387742439686534400, 20041683600089727]),
    UInt128(w: [8262992644530813824, 32066693760143564]),
    UInt128(w: [10299742930366561344, 25653355008114851]),
    UInt128(w: [4550445529551338752, 20522684006491881]),
    UInt128(w: [18348759291507873024, 32836294410387009]),
    UInt128(w: [18368356247948208704, 26269035528309607]),
    UInt128(w: [7315987368874746304, 21015228422647686]),
    UInt128(w: [4326882160715773504, 33624365476236298]),
    UInt128(w: [10840203358056439424, 26899492380989038]),
    UInt128(w: [16050860315928972160, 21519593904791230]),
    UInt128(w: [7234632431776803904, 34431350247665969]),
    UInt128(w: [9477054760163353472, 27545080198132775]),
    UInt128(w: [7581643808130682752, 22036064158506220]),
    UInt128(w: [12130630093009092416, 35257702653609952]),
    UInt128(w: [2325806444923453248, 28206162122887962]),
    UInt128(w: [12928691600164493568, 22564929698310369]),
    UInt128(w: [14032302094873505216, 18051943758648295]),
    UInt128(w: [4004939278088056704, 28883110013837273]),
    UInt128(w: [10582649051954265984, 23106488011069818]),
    UInt128(w: [15844816871047233408, 18485190408855854]),
    UInt128(w: [14283660549449842560, 29576304654169367]),
    UInt128(w: [4048230810076053376, 23661043723335494]),
    UInt128(w: [6927933462802753024, 18928834978668395]),
    UInt128(w: [11084693540484404864, 30286135965869432]),
    UInt128(w: [1489057202903703232, 24228908772695546]),
    UInt128(w: [15948641021290603904, 19383127018156436]),
    UInt128(w: [18139128004581145600, 31013003229050298]),
    UInt128(w: [3443255959439185472, 24810402583240239]),
    UInt128(w: [6443953582293258688, 19848322066592191]),
    UInt128(w: [2931628102185393280, 31757315306547506]),
    UInt128(w: [17102697740715955904, 25405852245238004]),
    UInt128(w: [17371507007314675072, 20324681796190403]),
    UInt128(w: [5658318323252018176, 32519490873904646]),
    UInt128(w: [837305843859704192, 26015592699123717]),
    UInt128(w: [11737891119313494336, 20812474159298973]),
    UInt128(w: [15091276976159680640, 33299958654878357]),
    UInt128(w: [4694323951443923840, 26639966923902686]),
    UInt128(w: [66110346413228736, 21311973539122149]),
    UInt128(w: [7484474183744986688, 34099157662595438]),
    UInt128(w: [13366276976479809984, 27279326130076350]),
    UInt128(w: [10693021581183848000, 21823460904061080]),
    UInt128(w: [17108834529894156800, 34917537446497728]),
    UInt128(w: [2619021179689594432, 27934029957198183]),
    UInt128(w: [9473914573235496192, 22347223965758546]),
    UInt128(w: [7779565687692973312, 35755558345213674]),
    UInt128(w: [9913001364896288960, 28604446676170939]),
    UInt128(w: [11619749906658941440, 22883557340936751]),
    UInt128(w: [5606451110585242816, 18306845872749401]),
    UInt128(w: [1591624147452567936, 29290953396399042]),
    UInt128(w: [12341345762187785280, 23432762717119233]),
    UInt128(w: [17251774239234048896, 18746210173695386]),
    UInt128(w: [1777397079581105984, 29993936277912619]),
    UInt128(w: [5111266478406795072, 23995149022330095]),
    UInt128(w: [4089013182725436096, 19196119217864076]),
    UInt128(w: [17610467536586428672, 30713790748582521]),
    UInt128(w: [10399025214527232640, 24571032598866017]),
    UInt128(w: [940522542137965440, 19656826079092814]),
    UInt128(w: [8883533696904565376, 31450921726548502]),
    UInt128(w: [18174873401749383296, 25160737381238801]),
    UInt128(w: [10850549906657596288, 20128589904991041]),
    UInt128(w: [9982182221168333440, 32205743847985666]),
    UInt128(w: [4296396962192756416, 25764595078388533]),
    UInt128(w: [10815815199238025792, 20611676062710826]),
    UInt128(w: [9926606689297020608, 32978681700337322]),
    UInt128(w: [562587721953795840, 26382945360269858]),
    UInt128(w: [7828767807046857280, 21106356288215886]),
    UInt128(w: [5147330861791151040, 33770170061145418]),
    UInt128(w: [11496562318916741504, 27016136048916334]),
    UInt128(w: [12886598669875303488, 21612908839133067]),
    UInt128(w: [5861162612832844352, 34580654142612908]),
    UInt128(w: [12067627719750096128, 27664523314090326]),
    UInt128(w: [5964753361058166592, 22131618651272261]),
    UInt128(w: [2164907748209245888, 35410589842035618]),
    UInt128(w: [9110623828051217344, 28328471873628494]),
    UInt128(w: [10977847877182884160, 22662777498902795]),
    UInt128(w: [8782278301746307328, 18130221999122236]),
    UInt128(w: [6672947653310271104, 29008355198595578]),
    UInt128(w: [12717055752132037568, 23206684158876462]),
    UInt128(w: [2794946972221809408, 18565347327101170]),
    UInt128(w: [4471915155554895040, 29704555723361872]),
    UInt128(w: [14645578568669646976, 23763644578689497]),
    UInt128(w: [4337765225451896960, 19010915662951598]),
    UInt128(w: [3251075545981124800, 30417465060722557]),
    UInt128(w: [13668906881010630784, 24333972048578045]),
    UInt128(w: [10935125504808504640, 19467177638862436]),
    UInt128(w: [10117503178209786752, 31147484222179898]),
    UInt128(w: [15472700172051650048, 24917987377743918]),
    UInt128(w: [1310113693415589056, 19934389902195135]),
    UInt128(w: [2096181909464942528, 31895023843512216]),
    UInt128(w: [16434340786539595328, 25516019074809772]),
    UInt128(w: [5768774999747855616, 20412815259847818]),
    UInt128(w: [5540691184854658688, 32660504415756509]),
    UInt128(w: [8121901762625637248, 26128403532605207]),
    UInt128(w: [17565567854326240768, 20902722826084165]),
    UInt128(w: [9658164493212433600, 33444356521734665]),
    UInt128(w: [7726531594569946880, 26755485217387732]),
    UInt128(w: [17249271719881688448, 21404388173910185]),
    UInt128(w: [9152090678101149952, 34247021078256297]),
    UInt128(w: [18389718986706650944, 27397616862605037]),
    UInt128(w: [7333077559881500096, 21918093490084030]),
    UInt128(w: [11732924095810400192, 35068949584134448]),
    UInt128(w: [16765036906132140800, 28055159667307558]),
    UInt128(w: [2343983080679981632, 22444127733846047]),
    UInt128(w: [7439721743829880960, 35910604374153675]),
    UInt128(w: [5951777395063904768, 28728483499322940]),
    UInt128(w: [4761421916051123840, 22982786799458352]),
    UInt128(w: [14877183977066630016, 18386229439566681]),
    UInt128(w: [16424796733822787392, 29417967103306690]),
    UInt128(w: [13139837387058229888, 23534373682645352]),
    UInt128(w: [3133172280162763264, 18827498946116282]),
    UInt128(w: [8702424463002331584, 30123998313786051]),
    UInt128(w: [3272590755659954944, 24099198651028841]),
    UInt128(w: [17375467863495605248, 19279358920823072]),
    UInt128(w: [13043353322625327104, 30846974273316916]),
    UInt128(w: [6745333843358351360, 24677579418653533]),
    UInt128(w: [12774964704170501696, 19742063534922826]),
    UInt128(w: [13061245897188982144, 31587301655876522]),
    UInt128(w: [3070299088267365056, 25269841324701218]),
    UInt128(w: [9834936900097712704, 20215873059760974]),
    UInt128(w: [4667852595930609344, 32345396895617559]),
    UInt128(w: [7423630891486397760, 25876317516494047]),
    UInt128(w: [17006951157414849216, 20701054013195237]),
    UInt128(w: [12453726592896117440, 33121686421112380]),
    UInt128(w: [9962981274316893952, 26497349136889904]),
    UInt128(w: [11659733834195425472, 21197879309511923]),
    UInt128(w: [14966225319970770432, 33916606895219077]),
    UInt128(w: [4594282626492795712, 27133285516175262]),
    UInt128(w: [14743472545419967552, 21706628412940209]),
    UInt128(w: [12521509628446217088, 34730605460704335]),
    UInt128(w: [10017207702756973632, 27784484368563468]),
    UInt128(w: [15392463791689399552, 22227587494850774]),
    UInt128(w: [13559895622477308352, 35564139991761239]),
    UInt128(w: [14537265312723756992, 28451311993408991]),
    UInt128(w: [7940463435437095296, 22761049594727193]),
    UInt128(w: [13731068377833496832, 18208839675781754]),
    UInt128(w: [10901662960307864000, 29134143481250807]),
    UInt128(w: [1342632738762470592, 23307314785000646]),
    UInt128(w: [15831501449977617728, 18645851828000516]),
    UInt128(w: [17951704690480367744, 29833362924800826]),
    UInt128(w: [10672014937642383872, 23866690339840661]),
    UInt128(w: [4848263135371996800, 19093352271872529]),
    UInt128(w: [15135918646079015488, 30549363634996046]),
    UInt128(w: [8419386102121302080, 24439490907996837]),
    UInt128(w: [17803555325922772608, 19551592726397469]),
    UInt128(w: [17417642077250705216, 31282548362235951]),
    UInt128(w: [10244764847058653888, 25026038689788761]),
    UInt128(w: [4506463062905012736, 20020830951831009]),
    UInt128(w: [14589038530131841088, 32033329522929614]),
    UInt128(w: [15360579638847383168, 25626663618343691]),
    UInt128(w: [8599114896335996224, 20501330894674953]),
    UInt128(w: [10069235019395683648, 32802129431479925]),
    UInt128(w: [8055388015516546880, 26241703545183940]),
    UInt128(w: [6444310412413237504, 20993362836147152]),
    UInt128(w: [14000245474603090368, 33589380537835443]),
    UInt128(w: [132149935456741312, 26871504430268355]),
    UInt128(w: [105719948365393024, 21497203544214684]),
    UInt128(w: [7547849546868449536, 34395525670743494]),
    UInt128(w: [9727628452236669952, 27516420536594795]),
    UInt128(w: [7782102761789335936, 22013136429275836]),
    UInt128(w: [5072666789379116928, 35221018286841338]),
    UInt128(w: [11436831060987114176, 28176814629473070]),
    UInt128(w: [9149464848789691328, 22541451703578456]),
    UInt128(w: [3630223064289842752, 18033161362862765]),
    UInt128(w: [5808356902863748416, 28853058180580424]),
    UInt128(w: [8336034337032909056, 23082446544464339]),
    UInt128(w: [10358176284368237568, 18465957235571471]),
    UInt128(w: [9194384425505359424, 29545531576914354]),
    UInt128(w: [11044856355146197888, 23636425261531483]),
    UInt128(w: [16214582713600778944, 18909140209225186]),
    UInt128(w: [117890638567874048, 30254624334760299]),
    UInt128(w: [3783661325596209536, 24203699467808239]),
    UInt128(w: [6716277875218877952, 19362959574246591]),
    UInt128(w: [3367346970866384128, 30980735318794546]),
    UInt128(w: [17451272835660748544, 24784588255035636]),
    UInt128(w: [10271669453786688512, 19827670604028509]),
    UInt128(w: [5366624681832970688, 31724272966445615]),
    UInt128(w: [4293299745466376576, 25379418373156492]),
    UInt128(w: [14502686240598832192, 20303534698525193]),
    UInt128(w: [1068205096506669632, 32485655517640310]),
    UInt128(w: [854564077205335680, 25988524414112248]),
    UInt128(w: [8062348891248089216, 20790819531289798]),
    UInt128(w: [9210409411255032384, 33265311250063677]),
    UInt128(w: [18436373973229756864, 26612249000050941]),
    UInt128(w: [11059750363841895168, 21289799200040753]),
    UInt128(w: [14006251767405121984, 34063678720065205]),
    UInt128(w: [11205001413924097600, 27250942976052164]),
    UInt128(w: [12653349945881188352, 21800754380841731]),
    UInt128(w: [12866662283926080768, 34881207009346770]),
    UInt128(w: [10293329827140864640, 27904965607477416]),
    UInt128(w: [4545315046970781376, 22323972485981933]),
    UInt128(w: [3583155260411339840, 35718355977571093]),
    UInt128(w: [10245221837812892544, 28574684782056874]),
    UInt128(w: [11885526284992224320, 22859747825645499]),
    UInt128(w: [13197769842735689792, 18287798260516399]),
    UInt128(w: [10048385304151372736, 29260477216826239]),
    UInt128(w: [11728057058063008512, 23408381773460991]),
    UInt128(w: [5693096831708496448, 18726705418768793]),
    UInt128(w: [5419606115991684032, 29962728670030069]),
    UInt128(w: [8025033707535257536, 23970182936024055]),
    UInt128(w: [6420026966028206016, 19176146348819244]),
    UInt128(w: [17650740775128950336, 30681834158110790]),
    UInt128(w: [14120592620103160256, 24545467326488632]),
    UInt128(w: [3917776466598707520, 19636373861190906]),
    UInt128(w: [17336488790783663040, 31418198177905449]),
    UInt128(w: [17558539847368840768, 25134558542324359]),
    UInt128(w: [17736180692636982912, 20107646833859487]),
    UInt128(w: [13620493849251531392, 32172234934175180]),
    UInt128(w: [10896395079401225152, 25737787947340144]),
    UInt128(w: [12406464878262890432, 20590230357872115]),
    UInt128(w: [1403599731511073088, 32944368572595385]),
    UInt128(w: [1122879785208858432, 26355494858076308]),
    UInt128(w: [8277001457650907392, 21084395886461046]),
    UInt128(w: [5864504702757631232, 33735033418337674]),
    UInt128(w: [8380952576948015296, 26988026734670139]),
    UInt128(w: [10394110876300322560, 21590421387736111]),
    UInt128(w: [9251879772596695424, 34544674220377778]),
    UInt128(w: [14780201447561177024, 27635739376302222]),
    UInt128(w: [4445463528565120960, 22108591501041778]),
    UInt128(w: [3423392830962283200, 35373746401666845]),
    UInt128(w: [2738714264769826560, 28298997121333476]),
    UInt128(w: [16948366670783502528, 22639197697066780]),
    UInt128(w: [13558693336626802048, 18111358157653424]),
    UInt128(w: [10625862894377152256, 28978173052245479]),
    UInt128(w: [12190039130243632128, 23182538441796383]),
    UInt128(w: [17130728933678726336, 18546030753437106]),
    UInt128(w: [1583724590692589952, 29673649205499371]),
    UInt128(w: [16024374931521713216, 23738919364399496]),
    UInt128(w: [9130151130475460224, 18991135491519597]),
    UInt128(w: [18297590623502646720, 30385816786431355]),
    UInt128(w: [14638072498802117376, 24308653429145084]),
    UInt128(w: [15399806813783604224, 19446922743316067]),
    UInt128(w: [9882295643086125504, 31115076389305708]),
    UInt128(w: [15284534143952721024, 24892061111444566]),
    UInt128(w: [8538278500420266496, 19913648889155653]),
    UInt128(w: [9971896785930516096, 31861838222649045]),
    UInt128(w: [7977517428744412864, 25489470578119236]),
    UInt128(w: [2692665128253619968, 20391576462495389]),
    UInt128(w: [11686961834689612608, 32626522339992622]),
    UInt128(w: [1970871838267869440, 26101217871994098]),
    UInt128(w: [8955395100098116160, 20880974297595278]),
    UInt128(w: [10639283345415075584, 33409558876152445]),
    UInt128(w: [8511426676332060480, 26727647100921956]),
    UInt128(w: [3119792526323738048, 21382117680737565]),
    UInt128(w: [4991668042117980864, 34211388289180104]),
    UInt128(w: [7682683248436295040, 27369110631344083]),
    UInt128(w: [13524844228232856640, 21895288505075266]),
    UInt128(w: [14261053135688750016, 35032461608120426]),
    UInt128(w: [7719493693809089664, 28025969286496341]),
    UInt128(w: [2486246140305361408, 22420775429197073]),
    UInt128(w: [288645009746667968, 35873240686715317]),
    UInt128(w: [11298962452023065344, 28698592549372253]),
    UInt128(w: [16417867591102272896, 22958874039497802]),
    UInt128(w: [5755596443397997696, 18367099231598242]),
    UInt128(w: [12898303124178706624, 29387358770557187]),
    UInt128(w: [2939944869859144640, 23509887016445750]),
    UInt128(w: [2351955895887315712, 18807909613156600]),
    UInt128(w: [3763129433419705152, 30092655381050560]),
    UInt128(w: [3010503546735764096, 24074124304840448]),
    UInt128(w: [9787100466872431936, 19259299443872358]),
    UInt128(w: [11970011932253980800, 30814879110195773]),
    UInt128(w: [16954707175287005248, 24651903288156618]),
    UInt128(w: [2495719296003873216, 19721522630525295]),
    UInt128(w: [3993150873606197184, 31554436208840472]),
    UInt128(w: [14262567143110688704, 25243548967072377]),
    UInt128(w: [4031356085004730304, 20194839173657902]),
    UInt128(w: [10139518550749478848, 32311742677852643]),
    UInt128(w: [15490312470083403712, 25849394142282114]),
    UInt128(w: [16081598790808633280, 20679515313825691]),
    UInt128(w: [18351860435809992640, 33087224502121106]),
    UInt128(w: [10992139533906083776, 26469779601696885]),
    UInt128(w: [8793711627124867008, 21175823681357508]),
    UInt128(w: [10380589788657876928, 33881317890172013]),
    UInt128(w: [15683169460410122176, 27105054312137610]),
    UInt128(w: [12546535568328097728, 21684043449710088]),
    UInt128(w: [16385108094583046080, 34694469519536141]),
    UInt128(w: [9418737660924526528, 27755575615628913]),
    UInt128(w: [14913687758223441856, 22204460492503130]),
    UInt128(w: [5415156339447955392, 35527136788005009]),
    UInt128(w: [8021473886300274624, 28421709430404007]),
    UInt128(w: [17485225553265950656, 22737367544323205]),
    UInt128(w: [13988180442612760512, 18189894035458564]),
    UInt128(w: [11313042263954685888, 29103830456733703]),
    UInt128(w: [16429131440647569344, 23283064365386962]),
    UInt128(w: [5764607523034234816, 18626451492309570]),
    UInt128(w: [9223372036854775744, 29802322387695312]),
    UInt128(w: [18446744073709551552, 23841857910156249]),
    UInt128(w: [18446744073709551552, 19073486328124999]),
    UInt128(w: [18446744073709551552, 30517578124999999]),
    UInt128(w: [18446744073709551552, 24414062499999999]),
    UInt128(w: [18446744073709551552, 19531249999999999]),
    UInt128(w: [18446744073709551552, 31249999999999999]),
    UInt128(w: [18446744073709551552, 24999999999999999]),
    UInt128(w: [18446744073709551552, 19999999999999999]),
    UInt128(w: [18446744073709551552, 31999999999999999]),
    UInt128(w: [18446744073709551552, 25599999999999999]),
    UInt128(w: [18446744073709551552, 20479999999999999]),
    UInt128(w: [18446744073709551552, 32767999999999999]),
    UInt128(w: [18446744073709551552, 26214399999999999]),
    UInt128(w: [18446744073709551552, 20971519999999999]),
    UInt128(w: [18446744073709551552, 33554431999999999]),
    UInt128(w: [18446744073709551552, 26843545599999999]),
    UInt128(w: [18446744073709551552, 21474836479999999]),
    UInt128(w: [18446744073709551552, 34359738367999999]),
    UInt128(w: [18446744073709551552, 27487790694399999]),
    UInt128(w: [18446744073709551552, 21990232555519999]),
    UInt128(w: [18446744073709551552, 35184372088831999]),
    UInt128(w: [18446744073709551552, 28147497671065599]),
    UInt128(w: [18446744073709551552, 22517998136852479]),
    UInt128(w: [18446744073709551552, 18014398509481983]),
    UInt128(w: [7378697629483820608, 28823037615171174]),
    UInt128(w: [9592306918328966784, 23058430092136939]),
    UInt128(w: [11363194349405083776, 18446744073709551]),
    UInt128(w: [10802413329564313408, 29514790517935282]),
    UInt128(w: [1263233034167630080, 23611832414348226]),
    UInt128(w: [15767981686301745344, 18889465931478580]),
    UInt128(w: [6782026624373240960, 30223145490365729]),
    UInt128(w: [9114970114240503040, 24178516392292583]),
    UInt128(w: [14670673720876223104, 19342813113834066]),
    UInt128(w: [16094380323918136320, 30948500982134506]),
    UInt128(w: [9186155444392598720, 24758800785707605]),
    UInt128(w: [7348924355514078976, 19807040628566084]),
    UInt128(w: [690232524596795392, 31691265005705735]),
    UInt128(w: [552186019677436352, 25353012004564588]),
    UInt128(w: [7820446445225769728, 20282409603651670]),
    UInt128(w: [12512714312361231552, 32451855365842672]),
    UInt128(w: [2631473820405164608, 25961484292674138]),
    UInt128(w: [9483876685807952320, 20769187434139310]),
    UInt128(w: [15174202697292723712, 33230699894622896]),
    UInt128(w: [8450013343092268608, 26584559915698317]),
    UInt128(w: [17828057118699545856, 21267647932558653]),
    UInt128(w: [6388798501467811456, 34028236692093846]),
    UInt128(w: [1421689986432338880, 27222589353675077]),
    UInt128(w: [12205398433371602048, 21778071482940061]),
    UInt128(w: [12149939863910742656, 34844914372704098]),
    UInt128(w: [17098649520612414784, 27875931498163278]),
    UInt128(w: [2610873172264200832, 22300745198530623]),
    UInt128(w: [488048260880811008, 35681192317648997]),
    UInt128(w: [11458485052930379776, 28544953854119197]),
    UInt128(w: [1788090412860483200, 22835963083295358]),
    UInt128(w: [8809169959772207168, 18268770466636286]),
    UInt128(w: [6715974306151710848, 29230032746618058]),
    UInt128(w: [12751477074405189312, 23384026197294446]),
    UInt128(w: [6511832844782241152, 18707220957835557]),
    UInt128(w: [14108281366393496128, 29931553532536891]),
    UInt128(w: [7597276278372886592, 23945242826029513]),
    UInt128(w: [13456518652182129920, 19156194260823610]),
    UInt128(w: [3083685769781856256, 30649910817317777]),
    UInt128(w: [13534995060051216000, 24519928653854221]),
    UInt128(w: [7138647233299062464, 19615942923083377]),
    UInt128(w: [15111184388020410240, 31385508676933403]),
    UInt128(w: [1020901066190597248, 25108406941546723]),
    UInt128(w: [8195418482436298432, 20086725553237378]),
    UInt128(w: [9423320757156167168, 32138760885179805]),
    UInt128(w: [7538656605724933760, 25711008708143844]),
    UInt128(w: [9720274099321857280, 20568806966515075]),
    UInt128(w: [15552438558914971712, 32910091146424120]),
    UInt128(w: [12441950847131977344, 26328072917139296]),
    UInt128(w: [6264211862963671552, 21062458333711437]),
    UInt128(w: [13712087795483784832, 33699933333938299]),
    UInt128(w: [14659019051128938176, 26959946667150639]),
    UInt128(w: [15416564055645060864, 21567957333720511]),
    UInt128(w: [17287804859548276736, 34508731733952818]),
    UInt128(w: [2762197443412890432, 27606985387162255]),
    UInt128(w: [2209757954730312320, 22085588309729804]),
    UInt128(w: [10914310357052320384, 35336941295567686]),
    UInt128(w: [5042099470899945984, 28269553036454149]),
    UInt128(w: [7723028391461867136, 22615642429163319]),
    UInt128(w: [9867771527911404032, 18092513943330655]),
    UInt128(w: [15788434444658246400, 28948022309329048]),
    UInt128(w: [1562701111500866176, 23158417847463239]),
    UInt128(w: [4939509703942603264, 18526734277970591]),
    UInt128(w: [524517896824344576, 29642774844752946]),
    UInt128(w: [15177009576427116928, 23714219875802356]),
    UInt128(w: [8452258846399783232, 18971375900641885]),
    UInt128(w: [13523614154239653184, 30354201441027016]),
    UInt128(w: [7129542508649812224, 24283361152821613]),
    UInt128(w: [13082331636403670400, 19426688922257290]),
    UInt128(w: [2484986544536321088, 31082702275611665]),
    UInt128(w: [1987989235629056832, 24866161820489332]),
    UInt128(w: [12658437832728976448, 19892929456391465]),
    UInt128(w: [1806756458656810688, 31828687130226345]),
    UInt128(w: [1445405166925448576, 25462949704181076]),
    UInt128(w: [15913719392508000128, 20370359763344860]),
    UInt128(w: [7015206954303248640, 32592575621351777]),
    UInt128(w: [16680212007668329856, 26074060497081421]),
    UInt128(w: [9654820791392753536, 20859248397665137]),
    UInt128(w: [690318007260764416, 33374797436264220]),
    UInt128(w: [552254405808611520, 26699837949011376]),
    UInt128(w: [15199198783614530496, 21359870359209100]),
    UInt128(w: [5871973980073697216, 34175792574734561]),
    UInt128(w: [1008230369317047424, 27340634059787649]),
    UInt128(w: [4495933110195548288, 21872507247830119]),
    UInt128(w: [14572190605796697920, 34996011596528190]),
    UInt128(w: [11657752484637358336, 27996809277222552]),
    UInt128(w: [1947504358226065984, 22397447421778042]),
    UInt128(w: [6805355787903615936, 35835915874844867]),
    UInt128(w: [16512331074548623680, 28668732699875893]),
    UInt128(w: [2141818415413168000, 22934986159900715]),
    UInt128(w: [1713454732330534400, 18347988927920572]),
    UInt128(w: [6430876386470765376, 29356782284672915]),
    UInt128(w: [5144701109176612288, 23485425827738332]),
    UInt128(w: [15183807331567020800, 18788340662190665]),
    UInt128(w: [5847347656797681664, 30061345059505065]),
    UInt128(w: [4677878125438145344, 24049076047604052]),
    UInt128(w: [14810348944576247232, 19239260838083241]),
    UInt128(w: [16317860681838174912, 30782817340933186]),
    UInt128(w: [9364939730728629632, 24626253872746549]),
    UInt128(w: [11181300599324814016, 19701003098197239]),
    UInt128(w: [6822034514693971456, 31521604957115583]),
    UInt128(w: [12836325241238997824, 25217283965692466]),
    UInt128(w: [6579711378249287936, 20173827172553973]),
    UInt128(w: [6838189390456950400, 32278123476086357]),
    UInt128(w: [16538597956591291264, 25822498780869085]),
    UInt128(w: [13230878365273033024, 20657999024695268]),
    UInt128(w: [17480056569694942528, 33052798439512429]),
    UInt128(w: [17673394070497864320, 26442238751609943]),
    UInt128(w: [3070668812172560448, 21153791001287955]),
    UInt128(w: [4913070099476096768, 33846065602060728]),
    UInt128(w: [11309153709064698048, 27076852481648582]),
    UInt128(w: [1668625337767937792, 21661481985318866]),
    UInt128(w: [13737846984654431488, 34658371176510185]),
    UInt128(w: [10990277587723545152, 27726696941208148]),
    UInt128(w: [16170919699662656768, 22181357552966518]),
    UInt128(w: [3737378631008788928, 35490172084746430]),
    UInt128(w: [2989902904807031104, 28392137667797144]),
    UInt128(w: [6081271138587535232, 22713710134237715]),
    UInt128(w: [4865016910870028160, 18170968107390172]),
    UInt128(w: [11473375872133955392, 29073548971824275]),
    UInt128(w: [9178700697707164352, 23258839177459420]),
    UInt128(w: [7342960558165731456, 18607071341967536]),
    UInt128(w: [4370039263581349696, 29771314147148058]),
    UInt128(w: [10874729040348900416, 23817051317718446]),
    UInt128(w: [5010434417537209984, 19053641054174757]),
    UInt128(w: [11706043882801446336, 30485825686679611]),
    UInt128(w: [5675486291499246720, 24388660549343689]),
    UInt128(w: [8229737847941307712, 19510928439474951]),
    UInt128(w: [5788882927222271680, 31217485503159922]),
    UInt128(w: [15699152786003548288, 24973988402527937]),
    UInt128(w: [5180624599319017984, 19979190722022350]),
    UInt128(w: [8288999358910428800, 31966705155235760]),
    UInt128(w: [6631199487128343040, 25573364124188608]),
    UInt128(w: [12683657219186495104, 20458691299350886]),
    UInt128(w: [12915153921214571520, 32733906078961418]),
    UInt128(w: [17710820766455477824, 26187124863169134]),
    UInt128(w: [17858005427906292608, 20949699890535307]),
    UInt128(w: [13815413425682426880, 33519519824856492]),
    UInt128(w: [3673633111062120832, 26815615859885194]),
    UInt128(w: [6628255303591606976, 21452492687908155]),
    UInt128(w: [10605208485746571200, 34323988300653048]),
    UInt128(w: [15862864418081077632, 27459190640522438]),
    UInt128(w: [1622245090239131136, 21967352512417951]),
    UInt128(w: [13663638588608340736, 35147764019868721]),
    UInt128(w: [7241562056144762304, 28118211215894977]),
    UInt128(w: [16861296089141540800, 22494568972715981]),
    UInt128(w: [1152632039433092992, 35991310356345571]),
    UInt128(w: [15679500890514115712, 28793048285076456]),
    UInt128(w: [8854251897669382208, 23034438628061165]),
    UInt128(w: [7083401518135505792, 18427550902448932]),
    UInt128(w: [15022791243758719616, 29484081443918291]),
    UInt128(w: [8328884180265065344, 23587265155134633]),
    UInt128(w: [14041804973695872896, 18869812124107706]),
    UInt128(w: [15088190328429576000, 30191699398572330]),
    UInt128(w: [12070552262743660800, 24153359518857864]),
    UInt128(w: [13345790624936838976, 19322687615086291]),
    UInt128(w: [13974567370415121728, 30916300184138066]),
    UInt128(w: [7490305081590187072, 24733040147310453]),
    UInt128(w: [13370941694755970304, 19786432117848362]),
    UInt128(w: [6636111452641911168, 31658291388557380]),
    UInt128(w: [5308889162113528960, 25326633110845904]),
    UInt128(w: [7936460144432733440, 20261306488676723]),
    UInt128(w: [9008987416350463232, 32418090381882757]),
    UInt128(w: [18275236377306101568, 25934472305506205]),
    UInt128(w: [14620189101844881216, 20747577844404964]),
    UInt128(w: [12324256118726079040, 33196124551047943]),
    UInt128(w: [17238102524464683840, 26556899640838354]),
    UInt128(w: [17479830834313657408, 21245519712670683]),
    UInt128(w: [5831636446450389952, 33992831540273094]),
    UInt128(w: [8354657971902222272, 27194265232218475]),
    UInt128(w: [6683726377521777792, 21755412185774780]),
    UInt128(w: [10693962204034844480, 34808659497239648]),
    UInt128(w: [15933867392711696256, 27846927597791718]),
    UInt128(w: [1679047469943626048, 22277542078233375]),
    UInt128(w: [2686475951909801664, 35644067325173400]),
    UInt128(w: [2149180761527841344, 28515253860138720]),
    UInt128(w: [1719344609222273024, 22812203088110976]),
    UInt128(w: [16132870946345459712, 18249762470488780]),
    UInt128(w: [7365849440443183936, 29199619952782049]),
    UInt128(w: [9582028367096457472, 23359695962225639]),
    UInt128(w: [11354971508419076288, 18687756769780511]),
    UInt128(w: [10789256783986701440, 29900410831648818]),
    UInt128(w: [16010103056673181824, 23920328665319054]),
    UInt128(w: [16497431260080455744, 19136262932255243]),
    UInt128(w: [4259797127677267328, 30618020691608390]),
    UInt128(w: [3407837702141813824, 24494416553286712]),
    UInt128(w: [13794316605939182016, 19595533242629369]),
    UInt128(w: [11002860125276960320, 31352853188206991]),
    UInt128(w: [5112939285479657920, 25082282550565593]),
    UInt128(w: [11469049057867546944, 20065826040452474]),
    UInt128(w: [7282432048362344192, 32105321664723959]),
    UInt128(w: [9515294453431785664, 25684257331779167]),
    UInt128(w: [233537933261607872, 20547405865423334]),
    UInt128(w: [7752358322702393280, 32875849384677334]),
    UInt128(w: [9891235472903824960, 26300679507741867]),
    UInt128(w: [534290748839239296, 21040543606193494]),
    UInt128(w: [8233562827626603520, 33664869769909590]),
    UInt128(w: [6586850262101282816, 26931895815927672]),
    UInt128(w: [16337526653906757248, 21545516652742137]),
    UInt128(w: [11382647387283170304, 34472826644387420]),
    UInt128(w: [9106117909826536256, 27578261315509936]),
    UInt128(w: [3595545513119318656, 22062609052407949]),
    UInt128(w: [13131570450474730496, 35300174483852718]),
    UInt128(w: [17883953989863605056, 28240139587082174]),
    UInt128(w: [17996512006632794368, 22592111669665739]),
    UInt128(w: [18086558420048145792, 18073689335732591]),
    UInt128(w: [3113051768883661056, 28917902937172147]),
    UInt128(w: [13558487859332659776, 23134322349737717]),
    UInt128(w: [3468092657982307200, 18507457879790174]),
    UInt128(w: [12927645882255512128, 29611932607664278]),
    UInt128(w: [17720814335288230336, 23689546086131422]),
    UInt128(w: [6797953838746763648, 18951636868905138]),
    UInt128(w: [7187377327252911552, 30322618990248221]),
    UInt128(w: [2060553047060418880, 24258095192198577]),
    UInt128(w: [12716488881874066048, 19406476153758861]),
    UInt128(w: [12967684581514685120, 31050361846014178]),
    UInt128(w: [17752845294695568704, 24840289476811342]),
    UInt128(w: [6823578606272634304, 19872231581449074]),
    UInt128(w: [18296423399520035584, 31795570530318518]),
    UInt128(w: [3569092275390297472, 25436456424254815]),
    UInt128(w: [2855273820312237952, 20349165139403852]),
    UInt128(w: [8257786927241491136, 32558664223046163]),
    UInt128(w: [13984927171277013504, 26046931378436930]),
    UInt128(w: [11187941737021610816, 20837545102749544]),
    UInt128(w: [6832660335008846336, 33340072164399271]),
    UInt128(w: [1776779453265166784, 26672057731519417]),
    UInt128(w: [12489470006837864384, 21337646185215533]),
    UInt128(w: [16293803196198672704, 34140233896344853]),
    UInt128(w: [1966996112733207168, 27312187117075883]),
    UInt128(w: [8952294519670386368, 21849749693660706]),
    UInt128(w: [6944973601988797568, 34959599509857130]),
    UInt128(w: [5555978881591038080, 27967679607885704]),
    UInt128(w: [8134131920014740736, 22374143686308563]),
    UInt128(w: [9325262257281674880, 35798629898093701]),
    UInt128(w: [3770860991083429568, 28638903918474961]),
    UInt128(w: [17774084051834384960, 22911123134779968]),
    UInt128(w: [3151220797241777024, 18328898507823975]),
    UInt128(w: [5041953275586843200, 29326237612518360]),
    UInt128(w: [4033562620469474560, 23460990090014688]),
    UInt128(w: [10605547725859400320, 18768792072011750]),
    UInt128(w: [16968876361375040512, 30030067315218800]),
    UInt128(w: [13575101089100032384, 24024053852175040]),
    UInt128(w: [10860080871280025920, 19219243081740032]),
    UInt128(w: [2618734135080400192, 30750788930784052]),
    UInt128(w: [13163033752290051072, 24600631144627241]),
    UInt128(w: [6841078187090130560, 19680504915701793]),
    UInt128(w: [7256376284602298560, 31488807865122869]),
    UInt128(w: [9494449842423749184, 25191046292098295]),
    UInt128(w: [7595559873938999360, 20152837033678636]),
    UInt128(w: [4774198168818578304, 32244539253885818]),
    UInt128(w: [11198056164538683264, 25795631403108654]),
    UInt128(w: [12647793746372856960, 20636505122486923]),
    UInt128(w: [16547121179454660800, 33018408195979077]),
    UInt128(w: [5858999314079907968, 26414726556783262]),
    UInt128(w: [15755245895489657344, 21131781245426609]),
    UInt128(w: [14140346988557720832, 33810849992682575]),
    UInt128(w: [11312277590846176640, 27048679994146060]),
    UInt128(w: [9049822072676941312, 21638943995316848]),
    UInt128(w: [10790366501541195776, 34622310392506957]),
    UInt128(w: [1253595571749136000, 27697848314005566]),
    UInt128(w: [15760271716366950080, 22158278651204452]),
    UInt128(w: [10459039487219478848, 35453245841927124]),
    UInt128(w: [12056580404517493376, 28362596673541699]),
    UInt128(w: [13334613138355905024, 22690077338833359]),
    UInt128(w: [14357039325426634368, 18152061871066687]),
    UInt128(w: [8213867661714973696, 29043298993706700]),
    UInt128(w: [6571094129371978944, 23234639194965360]),
    UInt128(w: [5256875303497583168, 18587711355972288]),
    UInt128(w: [4721651670854222720, 29740338169555661]),
    UInt128(w: [87972521941467840, 23792270535644529]),
    UInt128(w: [3759726832295084608, 19033816428515623]),
    UInt128(w: [2326214116930225024, 30454106285624997]),
    UInt128(w: [12929017737769910976, 24363285028499997]),
    UInt128(w: [2964516560732108160, 19490628022799998]),
    UInt128(w: [1053877682429462720, 31185004836479997]),
    UInt128(w: [11911148590169301120, 24948003869183997]),
    UInt128(w: [2150221242651620288, 19958403095347198]),
    UInt128(w: [18197749247210233728, 31933444952555516]),
    UInt128(w: [10868850583026276672, 25546755962044413]),
    UInt128(w: [16073778095904841984, 20437404769635530]),
    UInt128(w: [7271300879738195520, 32699847631416849]),
    UInt128(w: [9506389518532466752, 26159878105133479]),
    UInt128(w: [11294460429567883712, 20927902484106783]),
    UInt128(w: [14381787872566703680, 33484643974570853]),
    UInt128(w: [437383853827631936, 26787715179656683]),
    UInt128(w: [7728604712545926208, 21430172143725346]),
    UInt128(w: [4987069910589661312, 34288275429960554]),
    UInt128(w: [7679004743213639360, 27430620343968443]),
    UInt128(w: [13521901424054732096, 21944496275174754]),
    UInt128(w: [10566995834261840448, 35111194040279607]),
    UInt128(w: [1074899037925651712, 28088955232223686]),
    UInt128(w: [15617314489308162624, 22471164185778948]),
    UInt128(w: [2851610294441598336, 35953862697246318]),
    UInt128(w: [9659985865037099264, 28763090157797054]),
    UInt128(w: [11417337506771589760, 23010472126237643]),
    UInt128(w: [16512567634901092416, 18408377700990114]),
    UInt128(w: [15352061771616016960, 29453404321584183]),
    UInt128(w: [1213602973067082560, 23562723457267347]),
    UInt128(w: [12038928822679397056, 18850178765813877]),
    UInt128(w: [4504890857319393984, 30160286025302204]),
    UInt128(w: [7293261500597425472, 24128228820241763]),
    UInt128(w: [13213306829961761024, 19302583056193410]),
    UInt128(w: [2694546854229266048, 30884132889909457]),
    UInt128(w: [13223683927609143808, 24707306311927565]),
    UInt128(w: [10578947142087315072, 19765845049542052]),
    UInt128(w: [2168920168372062784, 31625352079267284]),
    UInt128(w: [5424484949439560576, 25300281663413827]),
    UInt128(w: [15407634403777379392, 20240225330731061]),
    UInt128(w: [17273517416559986432, 32384360529169698]),
    UInt128(w: [2750767489022258176, 25907488423335759])
  ]
  
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
  
  
  static let bid_power10_table_128: [UInt128] = [
    UInt128(w: [0x0000000000000001, 0x0000000000000000]),    // 10^0
    UInt128(w: [0x000000000000000a, 0x0000000000000000]),    // 10^1
    UInt128(w: [0x0000000000000064, 0x0000000000000000]),    // 10^2
    UInt128(w: [0x00000000000003e8, 0x0000000000000000]),    // 10^3
    UInt128(w: [0x0000000000002710, 0x0000000000000000]),    // 10^4
    UInt128(w: [0x00000000000186a0, 0x0000000000000000]),    // 10^5
    UInt128(w: [0x00000000000f4240, 0x0000000000000000]),    // 10^6
    UInt128(w: [0x0000000000989680, 0x0000000000000000]),    // 10^7
    UInt128(w: [0x0000000005f5e100, 0x0000000000000000]),    // 10^8
    UInt128(w: [0x000000003b9aca00, 0x0000000000000000]),    // 10^9
    UInt128(w: [0x00000002540be400, 0x0000000000000000]),    // 10^10
    UInt128(w: [0x000000174876e800, 0x0000000000000000]),    // 10^11
    UInt128(w: [0x000000e8d4a51000, 0x0000000000000000]),    // 10^12
    UInt128(w: [0x000009184e72a000, 0x0000000000000000]),    // 10^13
    UInt128(w: [0x00005af3107a4000, 0x0000000000000000]),    // 10^14
    UInt128(w: [0x00038d7ea4c68000, 0x0000000000000000]),    // 10^15
    UInt128(w: [0x002386f26fc10000, 0x0000000000000000]),    // 10^16
    UInt128(w: [0x016345785d8a0000, 0x0000000000000000]),    // 10^17
    UInt128(w: [0x0de0b6b3a7640000, 0x0000000000000000]),    // 10^18
    UInt128(w: [0x8ac7230489e80000, 0x0000000000000000]),    // 10^19
    UInt128(w: [0x6bc75e2d63100000, 0x0000000000000005]),    // 10^20
    UInt128(w: [0x35c9adc5dea00000, 0x0000000000000036]),    // 10^21
    UInt128(w: [0x19e0c9bab2400000, 0x000000000000021e]),    // 10^22
    UInt128(w: [0x02c7e14af6800000, 0x000000000000152d]),    // 10^23
    UInt128(w: [0x1bcecceda1000000, 0x000000000000d3c2]),    // 10^24
    UInt128(w: [0x161401484a000000, 0x0000000000084595]),    // 10^25
    UInt128(w: [0xdcc80cd2e4000000, 0x000000000052b7d2]),    // 10^26
    UInt128(w: [0x9fd0803ce8000000, 0x00000000033b2e3c]),    // 10^27
    UInt128(w: [0x3e25026110000000, 0x00000000204fce5e]),    // 10^28
    UInt128(w: [0x6d7217caa0000000, 0x00000001431e0fae]),    // 10^29
    UInt128(w: [0x4674edea40000000, 0x0000000c9f2c9cd0]),    // 10^30
    UInt128(w: [0xc0914b2680000000, 0x0000007e37be2022]),    // 10^31
    UInt128(w: [0x85acef8100000000, 0x000004ee2d6d415b]),    // 10^32
    UInt128(w: [0x38c15b0a00000000, 0x0000314dc6448d93]),    // 10^33
    UInt128(w: [0x378d8e6400000000, 0x0001ed09bead87c0]),    // 10^34
    UInt128(w: [0x2b878fe800000000, 0x0013426172c74d82]),    // 10^35
    UInt128(w: [0xb34b9f1000000000, 0x00c097ce7bc90715]),    // 10^36
    UInt128(w: [0x00f436a000000000, 0x0785ee10d5da46d9]),    // 10^37
    UInt128(w: [0x098a224000000000, 0x4b3b4ca85a86c47a]),    // 10^38
  ]
  
  // tables used in computation
  static let bid_estimate_decimal_digits: [Int8] = [
    1,    //2^0 =1     < 10^0
    1,    //2^1 =2     < 10^1
    1,    //2^2 =4     < 10^1
    1,    //2^3 =8     < 10^1
    2,    //2^4 =16    < 10^2
    2,    //2^5 =32    < 10^2
    2,    //2^6 =64    < 10^2
    3,    //2^7 =128   < 10^3
    3,    //2^8 =256   < 10^3
    3,    //2^9 =512   < 10^3
    4,    //2^10=1024  < 10^4
    4,    //2^11=2048  < 10^4
    4,    //2^12=4096  < 10^4
    4,    //2^13=8192  < 10^4
    5,    //2^14=16384 < 10^5
    5,    //2^15=32768 < 10^5
    
    5,    //2^16=65536     < 10^5
    6,    //2^17=131072    < 10^6
    6,    //2^18=262144    < 10^6
    6,    //2^19=524288    < 10^6
    7,    //2^20=1048576   < 10^7
    7,    //2^21=2097152   < 10^7
    7,    //2^22=4194304   < 10^7
    7,    //2^23=8388608   < 10^7
    8,    //2^24=16777216  < 10^8
    8,    //2^25=33554432  < 10^8
    8,    //2^26=67108864  < 10^8
    9,    //2^27=134217728 < 10^9
    9,    //2^28=268435456 < 10^9
    9,    //2^29=536870912 < 10^9
    10,    //2^30=1073741824< 10^10
    10,    //2^31=2147483648< 10^10
    
    10,    //2^32=4294967296     < 10^10
    10,    //2^33=8589934592     < 10^10
    11,    //2^34=17179869184    < 10^11
    11,    //2^35=34359738368    < 10^11
    11,    //2^36=68719476736    < 10^11
    12,    //2^37=137438953472   < 10^12
    12,    //2^38=274877906944   < 10^12
    12,    //2^39=549755813888   < 10^12
    13,    //2^40=1099511627776  < 10^13
    13,    //2^41=2199023255552  < 10^13
    13,    //2^42=4398046511104  < 10^13
    13,    //2^43=8796093022208  < 10^13
    14,    //2^44=17592186044416 < 10^14
    14,    //2^45=35184372088832 < 10^14
    14,    //2^46=70368744177664 < 10^14
    15,    //2^47=140737488355328< 10^15
    
    15,    //2^48=281474976710656    < 10^15
    15,    //2^49=562949953421312    < 10^15
    16,    //2^50=1125899906842624   < 10^16
    16,    //2^51=2251799813685248   < 10^16
    16,    //2^52=4503599627370496   < 10^16
    16,    //2^53=9007199254740992   < 10^16
    17,    //2^54=18014398509481984  < 10^17
    17,    //2^55=36028797018963968  < 10^17
    17,    //2^56=72057594037927936  < 10^17
    18,    //2^57=144115188075855872 < 10^18
    18,    //2^58=288230376151711744 < 10^18
    18,    //2^59=576460752303423488 < 10^18
    19,    //2^60=1152921504606846976< 10^19
    19,    //2^61=2305843009213693952< 10^19
    19,    //2^62=4611686018427387904< 10^19
    19,    //2^63=9223372036854775808< 10^19
    
    20,    //2^64=18446744073709551616
    20,    //2^65=36893488147419103232
    20,    //2^66=73786976294838206464
    21,    //2^67=147573952589676412928
    21,    //2^68=295147905179352825856
    21,    //2^69=590295810358705651712
    22,    //2^70=1180591620717411303424
    22,    //2^71=2361183241434822606848
    22,    //2^72=4722366482869645213696
    22,    //2^73=9444732965739290427392
    23,    //2^74=18889465931478580854784
    23,    //2^75=37778931862957161709568
    23,    //2^76=75557863725914323419136
    24,    //2^77=151115727451828646838272
    24,    //2^78=302231454903657293676544
    24,    //2^79=604462909807314587353088
    
    25,    //2^80=1208925819614629174706176
    25,    //2^81=2417851639229258349412352
    25,    //2^82=4835703278458516698824704
    25,    //2^83=9671406556917033397649408
    26,    //2^84=19342813113834066795298816
    26,    //2^85=38685626227668133590597632
    26,    //2^86=77371252455336267181195264
    27,    //2^87=154742504910672534362390528
    27,    //2^88=309485009821345068724781056
    27,    //2^89=618970019642690137449562112
    28,    //2^90=1237940039285380274899124224
    28,    //2^91=2475880078570760549798248448
    28,    //2^92=4951760157141521099596496896
    28,    //2^93=9903520314283042199192993792
    29,    //2^94=19807040628566084398385987584
    29,    //2^95=39614081257132168796771975168
    29,    //2^96=79228162514264337593543950336
    
    30,    //2^97=158456325028528675187087900672
    30,    //2^98=316912650057057350374175801344
    30,    //2^99=633825300114114700748351602688
    31,    //2^100=1267650600228229401496703205376
    31,    //2^101=2535301200456458802993406410752
    31,    //2^102=5070602400912917605986812821504
    32,    //2^103=10141204801825835211973625643008
    32,    //2^104=20282409603651670423947251286016
    32,    //2^105=40564819207303340847894502572032
    32,    //2^106=81129638414606681695789005144064
    33,    //2^107=162259276829213363391578010288128
    33,    // 2^108
    33,    // 2^109
    34,    // 2^110
    34,    // 2^111
    34,    // 2^112
    35,    // 2^113
    35,    // 2^114
    35,    // 2^115
    35,    // 2^116
    36,    // 2^117
    36,    // 2^118
    36,    // 2^119
    37,    // 2^120
    37,    // 2^121
    37,    // 2^122
    38,    // 2^123
    38,    // 2^124
    38,    // 2^125
    38,    // 2^126
    39,    // 2^127
    39    // 2^128
  ]
  
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
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 180, 181, 900, 901, 980, 981,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 190, 191, 910, 911, 990, 991,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 182, 183, 920, 921, 908, 909,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 192, 193, 930, 931, 918, 919,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 184, 185, 940, 941, 188, 189,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 194, 195, 950, 951, 198, 199,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 186, 187, 960, 961, 988, 989,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 196, 197, 970, 971, 998, 999,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 280, 281, 802, 803, 882, 883,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 290, 291, 812, 813, 892, 893,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 282, 283, 822, 823, 828, 829,
    230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 292, 293, 832, 833, 838, 839,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 284, 285, 842, 843, 288, 289,
    250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 294, 295, 852, 853, 298, 299,
    260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 286, 287, 862, 863, 888, 889,
    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 296, 297, 872, 873, 898, 899,
    300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 380, 381, 902, 903, 982, 983,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 390, 391, 912, 913, 992, 993,
    320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 382, 383, 922, 923, 928, 929,
    330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 392, 393, 932, 933, 938, 939,
    340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 384, 385, 942, 943, 388, 389,
    350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 394, 395, 952, 953, 398, 399,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 386, 387, 962, 963, 988, 989,
    370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 396, 397, 972, 973, 998, 999,
    400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 480, 481, 804, 805, 884, 885,
    410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 490, 491, 814, 815, 894, 895,
    420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 482, 483, 824, 825, 848, 849,
    430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 492, 493, 834, 835, 858, 859,
    440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 484, 485, 844, 845, 488, 489,
    450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 494, 495, 854, 855, 498, 499,
    460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 486, 487, 864, 865, 888, 889,
    470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 496, 497, 874, 875, 898, 899,
    500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 580, 581, 904, 905, 984, 985,
    510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 590, 591, 914, 915, 994, 995,
    520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 582, 583, 924, 925, 948, 949,
    530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 592, 593, 934, 935, 958, 959,
    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 584, 585, 944, 945, 588, 589,
    550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 594, 595, 954, 955, 598, 599,
    560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 586, 587, 964, 965, 988, 989,
    570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 596, 597, 974, 975, 998, 999,
    600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 680, 681, 806, 807, 886, 887,
    610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 690, 691, 816, 817, 896, 897,
    620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 682, 683, 826, 827, 868, 869,
    630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 692, 693, 836, 837, 878, 879,
    640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 684, 685, 846, 847, 688, 689,
    650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 694, 695, 856, 857, 698, 699,
    660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 686, 687, 866, 867, 888, 889,
    670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 696, 697, 876, 877, 898, 899,
    700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 780, 781, 906, 907, 986, 987,
    710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 790, 791, 916, 917, 996, 997,
    720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 782, 783, 926, 927, 968, 969,
    730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 792, 793, 936, 937, 978, 979,
    740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 784, 785, 946, 947, 788, 789,
    750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 794, 795, 956, 957, 798, 799,
    760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 786, 787, 966, 967, 988, 989,
    770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 796, 797, 976, 977, 998, 999
  ]
  
  static func bid_d2b2(_ i: Int) -> UInt64 { bid_d2b[i] * 1000 }
  
  // Table of powers of 5
  static let bid_power_five: [UInt128] = [
    UInt128(w: [1, 0]),
    UInt128(w: [5, 0]),
    UInt128(w: [25, 0]),
    UInt128(w: [125, 0]),
    UInt128(w: [625, 0]),
    UInt128(w: [3125, 0]),
    UInt128(w: [15625, 0]),
    UInt128(w: [78125, 0]),
    UInt128(w: [390625, 0]),
    UInt128(w: [1953125, 0]),
    UInt128(w: [9765625, 0]),
    UInt128(w: [48828125, 0]),
    UInt128(w: [244140625, 0]),
    UInt128(w: [1220703125, 0]),
    UInt128(w: [6103515625, 0]),
    UInt128(w: [30517578125, 0]),
    UInt128(w: [152587890625, 0]),
    UInt128(w: [762939453125, 0]),
    UInt128(w: [3814697265625, 0]),
    UInt128(w: [19073486328125, 0]),
    UInt128(w: [95367431640625, 0]),
    UInt128(w: [476837158203125, 0]),
    UInt128(w: [2384185791015625, 0]),
    UInt128(w: [11920928955078125, 0]),
    UInt128(w: [59604644775390625, 0]),
    UInt128(w: [298023223876953125, 0]),
    UInt128(w: [1490116119384765625, 0]),
    UInt128(w: [7450580596923828125, 0]),
    UInt128(w: [359414837200037393, 2]),
    UInt128(w: [1797074186000186965, 10]),
    UInt128(w: [8985370930000934825, 50]),
    UInt128(w: [8033366502585570893, 252]),
    UInt128(w: [3273344365508751233, 1262]),
    UInt128(w: [16366721827543756165, 6310]),
    UInt128(w: [8046632842880574361, 31554]),
    UInt128(w: [3339676066983768573, 157772]),
    UInt128(w: [16698380334918842865, 788860]),
    UInt128(w: [9704925379756007861, 3944304]),
    UInt128(w: [11631138751360936073, 19721522]),
    UInt128(w: [2815461535676025517, 98607613]),
    UInt128(w: [14077307678380127585, 493038065]),
    UInt128(w: [15046306170771983077, 2465190328]),
    UInt128(w: [1444554559021708921, 12325951644]),
    UInt128(w: [7222772795108544605, 61629758220]),
    UInt128(w: [17667119901833171409, 308148791101]),
    UInt128(w: [14548623214327650581, 1540743955509]),
    UInt128(w: [17402883850509598057, 7703719777548]),
    UInt128(w: [13227442957709783821, 38518598887744]),
    UInt128(w: [10796982567420264257, 192592994438723])
  ]
  
  static let coefflimits: [UInt64] = [
    10000000, 2000000, 400000, 80000, 16000, 3200, 640, 128, 25, 5, 1
  ]
  
  static func bid_coefflimits_bid32(_ i: Int) -> UInt128 {
    i > 10 ? 0 : UInt128(coefflimits[i])
  }
  
//  static let bid_coefflimits_bid32 : [UInt128] = [
//    UInt128(w: [10000000, 0]),
//    UInt128(w: [2000000, 0]),
//    UInt128(w: [400000, 0]),
//    UInt128(w: [80000, 0]),
//    UInt128(w: [16000, 0]),
//    UInt128(w: [3200, 0]),
//    UInt128(w: [640, 0]),
//    UInt128(w: [128, 0]),
//    UInt128(w: [25, 0]),
//    UInt128(w: [5, 0]),
//    UInt128(w: [1, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0]),
//    UInt128(w: [0, 0])
//  ]
  
  
  static let bid_multipliers1_bid32: [UInt256] = [
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [0, 0, 0, 0]),
    UInt256(w: [8022453891189237964, 4305922861044245892,  15091728617112590342, 392727477223]),
    UInt256(w: [16044907782378475927, 8611845722088491784,  11736713160515629068, 785454954447]),
    UInt256(w: [13643071491047400238, 17223691444176983569,  5026682247321706520, 1570909908895]),
    UInt256(w: [8839398908385248859, 16000638814644415523,  10053364494643413041, 3141819817790]),
    UInt256(w: [16525275040644691065, 6889476577670793427,  2010672898928682608, 628363963558]),
    UInt256(w: [14603806007579830513, 13778953155341586855,  4021345797857365216, 1256727927116]),
    UInt256(w: [10760867941450109410, 9111162236973622095,  8042691595714730433, 2513455854232]),
    UInt256(w: [2152173588290021882, 1822232447394724419,  8987235948626766733, 502691170846]),
    UInt256(w: [4304347176580043764, 3644464894789448838,  17974471897253533466, 1005382341692]),
    UInt256(w: [8608694353160087528, 7288929789578897676,  17502199720797515316, 2010764683385]),
    UInt256(w: [9100436500115838152, 5147134772657689858,  3500439944159503063, 402152936677]),
    UInt256(w: [18200873000231676304, 10294269545315379716,  7000879888319006126, 804305873354]),
    UInt256(w: [17955001926753800992, 2141795016921207817,  14001759776638012253, 1608611746708]),
    UInt256(w: [17463259779798050368, 4283590033842415635,  9556775479566472890, 3217223493417]),
    UInt256(w: [10871349585443430720, 8235415636252303773,  9290052725397115224, 643444698683]),
    UInt256(w: [3295955097177309824, 16470831272504607547,  133361377084678832, 1286889397367]),
    UInt256(w: [6591910194354619648, 14494918471299663478,  266722754169357665, 2573778794734]),
    UInt256(w: [8697079668354744576, 17656378953227573988,  14810739809801512825, 514755758946]),
    UInt256(w: [17394159336709489152, 16866013832745596360,  11174735545893474035, 1029511517893]),
    UInt256(w: [16341574599709426688, 15285283591781641105,  3902727018077396455, 2059023035787]),
    UInt256(w: [10647012549425705984, 10435754347840148867,  8159243033099299937, 411804607157]),
    UInt256(w: [2847281025141860352, 2424764621970746119,  16318486066198599875, 823609214314]),
    UInt256(w: [5694562050283720704, 4849529243941492238,  14190228058687648134, 1647218428629]),
    UInt256(w: [4828261224798654464, 12037952293014029417,  17595440870705170919, 329443685725]),
    UInt256(w: [9656522449597308928, 5629160512318507218,  16744137667700790223, 658887371451]),
    UInt256(w: [866300825485066240, 11258321024637014437,  15041531261692028830, 1317774742903]),
    UInt256(w: [1732601650970132480, 4069897975564477258,  11636318449674506045, 2635549485807]),
    UInt256(w: [346520330194026496, 8192677224596716098, 9705961319418721855,  527109897161]),
    UInt256(w: [693040660388052992, 16385354449193432196, 965178565127892094,  1054219794323]),
    UInt256(w: [1386081320776105984, 14323964824677312776,  1930357130255784189, 2108439588646]),
    UInt256(w: [3966565078897131520, 2864792964935462555,  4075420240793067161, 421687917729]),
    UInt256(w: [7933130157794263040, 5729585929870925110,  8150840481586134322, 843375835458]),
    UInt256(w: [15866260315588526080, 11459171859741850220,  16301680963172268644, 1686751670916]),
    UInt256(w: [3173252063117705216, 2291834371948370044,  6949685007376364052, 337350334183]),
    UInt256(w: [6346504126235410432, 4583668743896740088,  13899370014752728104, 674700668366]),
    UInt256(w: [12693008252470820864, 9167337487793480176,  9351995955795904592, 1349401336733]),
    UInt256(w: [6939272431232090112, 18334674975586960353,  257247837882257568, 2698802673467]),
    UInt256(w: [12455900930472148992, 3666934995117392070,  7430147197060272160, 539760534693]),
    UInt256(w: [6465057787234746368, 7333869990234784141,  14860294394120544320, 1079521069386]),
    UInt256(w: [12930115574469492736, 14667739980469568282,  11273844714531537024, 2159042138773]),
    UInt256(w: [17343418373861539840, 10312245625577734302,  13322815387132038374, 431808427754]),
    UInt256(w: [16240092674013528064, 2177747177445916989,  8198886700554525133, 863616855509]),
    UInt256(w: [14033441274317504512, 4355494354891833979,  16397773401109050266, 1727233711018]),
    UInt256(w: [13874734699089231872, 15628494129946008088,  14347601124447541022, 345446742203]),
    UInt256(w: [9302725324468912128, 12810244186182464561,  10248458175185530429, 690893484407]),
    UInt256(w: [158706575228272640, 7173744298655377507, 2050172276661509243,  1381786968815]),
    UInt256(w: [317413150456545280, 14347488597310755014,  4100344553323018486, 2763573937630]),
    UInt256(w: [63482630091309056, 6558846534204061326, 820068910664603697,  552714787526]),
    UInt256(w: [126965260182618112, 13117693068408122652,  1640137821329207394, 1105429575052]),
    UInt256(w: [253930520365236224, 7788642063106693688, 3280275642658414789,  2210859150104]),
    UInt256(w: [3740134918814957568, 12625774856847069707,  15413450387499324250, 442171830020]),
    UInt256(w: [7480269837629915136, 6804805639984587798,  12380156701289096885, 884343660041]),
    UInt256(w: [14960539675259830272, 13609611279969175596,  6313569328868642154, 1768687320083]),
    UInt256(w: [14060154379277697024, 10100619885477655765,  12330760309999459400, 353737464016]),
    UInt256(w: [9673564684845842432, 1754495697245759915,  6214776546289367185, 707474928033]),
    UInt256(w: [900385295982133248, 3508991394491519831,  12429553092578734370, 1414949856066]),
    UInt256(w: [1800770591964266496, 7017982788983039662,  6412362111447917124, 2829899712133]),
    UInt256(w: [15117549377360494592, 8782294187280428578,  12350518866515314394, 565979942426]),
    UInt256(w: [11788354681011437568, 17564588374560857157,  6254293659321077172, 1131959884853]),
    UInt256(w: [5129965288313323520, 16682432675412162699,  12508587318642154345, 2263919769706]),
    UInt256(w: [1025993057662664704, 7025835349824342863,  6191066278470341192, 452783953941]),
    UInt256(w: [2051986115325329408, 14051670699648685726,  12382132556940682384, 905567907882]),
    UInt256(w: [4103972230650658816, 9656597325587819836,  6317521040171813153, 1811135815765]),
    UInt256(w: [15578189705097773056, 12999365909343294936,  1263504208034362630, 362227163153]),
    UInt256(w: [12709635336485994496, 7551987744977038257,  2527008416068725261, 724454326306]),
    UInt256(w: [6972526599262437376, 15103975489954076515,  5054016832137450522, 1448908652612]),
    UInt256(w: [13945053198524874752, 11761206906198601414,  10108033664274901045, 2897817305224]),
    UInt256(w: [13857057083930705920, 17109636640207361575,  16779001991822621501, 579563461044]),
    UInt256(w: [9267370094151860224, 15772529206705171535,  15111259909935691387, 1159126922089]),
    UInt256(w: [87996114594168832, 13098314339700791455,  11775775746161831159, 2318253844179]),
    UInt256(w: [11085645667144564736, 13687709312165889260,  17112550408200007524, 463650768835]),
    UInt256(w: [3724547260579577856, 8928674550622226905,  15778356742690463433, 927301537671]),
    UInt256(w: [7449094521159155712, 17857349101244453810,  13109969411671375250, 1854603075343]),
    UInt256(w: [12557865348457562112, 14639516264474621731,  13690040326560006019, 370920615068]),
    UInt256(w: [6668986623205572608, 10832288455239691847,  8933336579410460423, 741841230137]),
    UInt256(w: [13337973246411145216, 3217832836769832078,  17866673158820920847, 1483682460274]),
    UInt256(w: [8229202419112738816, 6435665673539664157,  17286602243932290078, 2967364920549]),
    UInt256(w: [16403235742790189056, 8665830764191753477,  18214715707754099308, 593472984109]),
    UInt256(w: [14359727411870826496, 17331661528383506955,  17982687341798647000, 1186945968219]),
    UInt256(w: [10272710750032101376, 16216578983057462295,  17518630609887742385, 2373891936439]),
    UInt256(w: [16811937408974061568, 18000711055579133751,  18261121380945189769, 474778387287]),
    UInt256(w: [15177130744238571520, 17554678037448715887,  18075498688180827923, 949556774575]),
    UInt256(w: [11907517414767591424, 16662612001187880159,  17704253302652104231, 1899113549151]),
    UInt256(w: [6070852297695428608, 10711220029721396678,  7230199475272331169, 379822709830]),
    UInt256(w: [12141704595390857216, 2975695985733241740,  14460398950544662339, 759645419660]),
    UInt256(w: [5836665117072162816, 5951391971466483481,  10474053827379773062, 1519290839321]),
    UInt256(w: [11673330234144325632, 11902783942932966962,  2501363581049994508, 3038581678643]),
    UInt256(w: [13402712491054596096, 6069905603328503715,  11568319160435729871, 607716335728]),
    UInt256(w: [8358680908399640576, 12139811206657007431,  4689894247161908126, 1215432671457]),
    UInt256(w: [16717361816799281152, 5832878339604463246,  9379788494323816253, 2430865342914]),
    UInt256(w: [14411518807585587200, 8545273297404713295,  16633352957832404543, 486173068582]),
    UInt256(w: [10376293541461622784, 17090546594809426591,  14819961841955257470, 972346137165]),
    UInt256(w: [2305843009213693952, 15734349115909301567,  11193179610200963325, 1944692274331]),
    UInt256(w: [11529215046068469760, 6836218637923770636,  5927984736782102988, 388938454866]),
    UInt256(w: [4611686018427387904, 13672437275847541273,  11855969473564205976, 777876909732]),
    UInt256(w: [9223372036854775808, 8898130477985530930,  5265194873418860337, 1555753819465]),
    UInt256(w: [0, 17796260955971061861, 10530389746837720674,  3111507638930]),
    UInt256(w: [0, 18316647450161853665, 2106077949367544134,  622301527786]),
    UInt256(w: [0, 18186550826614155714, 4212155898735088269,  1244603055572]),
    UInt256(w: [0, 17926357579518759812, 8424311797470176539,  2489206111144]),
    UInt256(w: [0, 14653317960129482932, 16442257618461676600,  497841222228]),
    UInt256(w: [0, 10859891846549414248, 14437771163213801585,  995682444457]),
    UInt256(w: [0, 3273039619389276880, 10428798252718051555,  1991364888915]),
    UInt256(w: [0, 654607923877855376, 2085759650543610311,  398272977783]),
    UInt256(w: [0, 1309215847755710752, 4171519301087220622,  796545955566]),
    UInt256(w: [0, 2618431695511421504, 8343038602174441244,  1593091911132]),
    UInt256(w: [0, 5236863391022843008, 16686077204348882488,  3186183822264]),
    UInt256(w: [0, 8426070307688389248, 18094610699837417790,  637236764452]),
    UInt256(w: [0, 16852140615376778496, 17742477325965283964,  1274473528905]),
    UInt256(w: [0, 15257537157044005376, 17038210578221016313,  2548947057811]),
    UInt256(w: [0, 17808902690376442368, 7096990930386113585,  509789411562]),
    UInt256(w: [0, 17171061307043333120, 14193981860772227171,  1019578823124]),
    UInt256(w: [0, 15895378540377114624, 9941219647834902727,  2039157646249]),
    UInt256(w: [0, 6868424522817333248, 16745639188534621838,  407831529249]),
    UInt256(w: [0, 13736849045634666496, 15044534303359692060,  815663058499]),
    UInt256(w: [0, 9026954017559781376, 11642324533009832505,  1631326116999]),
    UInt256(w: [0, 18053908035119562752, 4837904992310113394,  3262652233999]),
    UInt256(w: [0, 14678828051249643520, 15724976257429663971,  652530446799]),
    UInt256(w: [0, 10910912028789735424, 13003208441149776327,  1305060893599]),
    UInt256(w: [0, 3375079983869919232, 7559672808590001039,  2610121787199]),
    UInt256(w: [0, 11743062440999714816, 16269329820685641500,  522024357439]),
    UInt256(w: [0, 5039380808289878016, 14091915567661731385,  1044048714879]),
    UInt256(w: [0, 10078761616579756032, 9737087061613911154,  2088097429759]),
    UInt256(w: [0, 13083798767541682176, 16704812671290423523,  417619485951]),
    UInt256(w: [0, 7720853461373812736, 14962881268871295431,  835238971903]),
    UInt256(w: [0, 15441706922747625472, 11479018464033039246,  1670477943807]),
    UInt256(w: [0, 14156387828775256064, 9674501322290428495,  334095588761]),
    UInt256(w: [0, 9866031583840960512, 902258570871305375,  668191177523]),
    UInt256(w: [0, 1285319093972369408, 1804517141742610751,  1336382355046]),
    UInt256(w: [0, 2570638187944738816, 3609034283485221502,  2672764710092]),
    UInt256(w: [0, 15271522896556589056, 8100504486180864946,  534552942018]),
    UInt256(w: [0, 12096301719403626496, 16201008972361729893,  1069105884036]),
    UInt256(w: [0, 5745859365097701376, 13955273871013908171,  2138211768073]),
    UInt256(w: [0, 15906567131987181568, 13859101218428512603,  427642353614]),
    UInt256(w: [0, 13366390190264811520, 9271458363147473591,  855284707229]),
    UInt256(w: [0, 8286036306820071424, 96172652585395567,  1710569414459]),
    UInt256(w: [0, 5346556076105924608, 14776629789484720406,  342113882891]),
    UInt256(w: [0, 10693112152211849216, 11106515505259889196,  684227765783]),
    UInt256(w: [0, 2939480230714146816, 3766286936810226777,  1368455531567]),
    UInt256(w: [0, 5878960461428293632, 7532573873620453554,  2736911063134]),
    UInt256(w: [0, 12243838536511389696, 16263910033691732003,  547382212626]),
    UInt256(w: [0, 6040932999313227776, 14081075993673912391,  1094764425253]),
    UInt256(w: [0, 12081865998626455552, 9715407913638273166,  2189528850507]),
    UInt256(w: [0, 13484419643951022080, 9321779212211475279,  437905770101]),
    UInt256(w: [0, 8522095214192492544, 196814350713398943,  875811540203]),
    UInt256(w: [0, 17044190428384985088, 393628701426797886,  1751623080406]),
    UInt256(w: [0, 10787535715160817664, 3768074555027269900,  350324616081]),
    UInt256(w: [0, 3128327356612083712, 7536149110054539801,  700649232162]),
    UInt256(w: [0, 6256654713224167424, 15072298220109079602,  1401298464324]),
    UInt256(w: [0, 12513309426448334848, 11697852366508607588,  2802596928649]),
    UInt256(w: [0, 9881359514773487616, 17096965732269362810,  560519385729]),
    UInt256(w: [0, 1315974955837423616, 15747187390829174005,  1121038771459]),
    UInt256(w: [0, 2631949911674847232, 13047630707948796394,  2242077542919]),
    UInt256(w: [0, 11594436426560700416, 17366921400557400571,  448415508583]),
    UInt256(w: [0, 4742128779411849216, 16287098727405249527,  896831017167]),
    UInt256(w: [0, 9484257558823698432, 14127453381100947438,  1793662034335]),
    UInt256(w: [0, 12964897955990470656, 2825490676220189487,  358732406867]),
    UInt256(w: [0, 7483051838271389696, 5650981352440378975,  717464813734]),
    UInt256(w: [0, 14966103676542779392, 11301962704880757950,  1434929627468]),
    UInt256(w: [0, 11485463279376007168, 4157181336051964285,  2869859254937]),
    UInt256(w: [0, 9675790285359022080, 8210133896694213503,  573971850987]),
    UInt256(w: [0, 904836497008492544, 16420267793388427007,  1147943701974]),
    UInt256(w: [0, 1809672994016985088, 14393791513067302398,  2295887403949]),
    UInt256(w: [0, 7740632228287217664, 17636153561581101772,  459177480789]),
    UInt256(w: [0, 15481264456574435328, 16825563049452651928,  918354961579]),
    UInt256(w: [0, 12515784839439319040, 15204382025195752241,  1836709923159]),
    UInt256(w: [0, 2503156967887863808, 17798271664006791741,  367341984631]),
    UInt256(w: [0, 5006313935775727616, 17149799254304031866,  734683969263]),
    UInt256(w: [0, 10012627871551455232, 15852854434898512116,  1469367938527]),
    UInt256(w: [0, 1578511669393358848, 13258964796087472617,  2938735877055]),
    UInt256(w: [0, 7694399963362492416, 2651792959217494523,  587747175411]),
    UInt256(w: [0, 15388799926724984832, 5303585918434989046,  1175494350822]),
    UInt256(w: [0, 12330855779740418048, 10607171836869978093,  2350988701644]),
    UInt256(w: [0, 9844868785431904256, 16878829626341636911,  470197740328]),
    UInt256(w: [0, 1242993497154256896, 15310915178973722207,  940395480657]),
    UInt256(w: [0, 2485986994308513792, 12175086284237892798,  1880790961315]),
    UInt256(w: [0, 11565243843087433728, 2435017256847578559,  376158192263]),
    UInt256(w: [0, 4683743612465315840, 4870034513695157119,  752316384526]),
    UInt256(w: [0, 9367487224930631680, 9740069027390314238,  1504632769052]),
    UInt256(w: [0, 288230376151711744, 1033393981071076861,  3009265538105]),
    UInt256(w: [0, 3746994889972252672, 206678796214215372,  601853107621]),
    UInt256(w: [0, 7493989779944505344, 413357592428430744,  1203706215242]),
    UInt256(w: [0, 14987979559889010688, 826715184856861488,  2407412430484]),
    UInt256(w: [0, 10376293541461622784, 14922738295939013590,  481482486096]),
    UInt256(w: [0, 2305843009213693952, 11398732518168475565,  962964972193]),
    UInt256(w: [0, 4611686018427387904, 4350720962627399514,  1925929944387]),
    UInt256(w: [0, 4611686018427387904, 8248841822009300549,  385185988877]),
    UInt256(w: [0, 9223372036854775808, 16497683644018601098,  770371977754]),
    UInt256(w: [0, 0, 14548623214327650581, 1540743955509]),
    UInt256(w: [0, 0, 10650502354945749546, 3081487911019]),
    UInt256(w: [0, 0, 16887495729956791202, 616297582203]),
    UInt256(w: [0, 0, 15328247386204030788, 1232595164407]),
    UInt256(w: [0, 0, 12209750698698509960, 2465190328815]),
    UInt256(w: [0, 0, 2441950139739701992, 493038065763]),
    UInt256(w: [0, 0, 4883900279479403984, 986076131526]),
    UInt256(w: [0, 0, 9767800558958807968, 1972152263052]),
    UInt256(w: [0, 0, 9332257741275582240, 394430452610]),
    UInt256(w: [0, 0, 217771408841612864, 788860905221]),
    UInt256(w: [0, 0, 435542817683225728, 1577721810442]),
    UInt256(w: [0, 0, 871085635366451456, 3155443620884]),
    UInt256(w: [0, 0, 14931612386040931584, 631088724176]),
    UInt256(w: [0, 0, 11416480698372311552, 1262177448353]),
    UInt256(w: [0, 0, 4386217323035071488, 2524354896707]),
    UInt256(w: [0, 0, 8255941094090834944, 504870979341]),
    UInt256(w: [0, 0, 16511882188181669888, 1009741958682]),
    UInt256(w: [0, 0, 14577020302653788160, 2019483917365]),
    UInt256(w: [0, 0, 2915404060530757632, 403896783473]),
    UInt256(w: [0, 0, 5830808121061515264, 807793566946]),
    UInt256(w: [0, 0, 11661616242123030528, 1615587133892]),
    UInt256(w: [0, 0, 4876488410536509440, 3231174267785]),
    UInt256(w: [0, 0, 975297682107301888, 646234853557]),
    UInt256(w: [0, 0, 1950595364214603776, 1292469707114]),
    UInt256(w: [0, 0, 3901190728429207552, 2584939414228]),
    UInt256(w: [0, 0, 11848284589911572480, 516987882845]),
    UInt256(w: [0, 0, 5249825106113593344, 1033975765691]),
    UInt256(w: [0, 0, 10499650212227186688, 2067951531382]),
    UInt256(w: [0, 0, 9478627671929257984, 413590306276]),
    UInt256(w: [0, 0, 510511270148964352, 827180612553]),
    UInt256(w: [0, 0, 1021022540297928704, 1654361225106]),
    UInt256(w: [0, 0, 3893553322801496064, 330872245021]),
    UInt256(w: [0, 0, 7787106645602992128, 661744490042]),
    UInt256(w: [0, 0, 15574213291205984256, 1323488980084]),
    UInt256(w: [0, 0, 12701682508702416896, 2646977960169]),
    UInt256(w: [0, 0, 17297731760708124672, 529395592033]),
    UInt256(w: [0, 0, 16148719447706697728, 1058791184067]),
    UInt256(w: [0, 0, 13850694821703843840, 2117582368135]),
    UInt256(w: [0, 0, 2770138964340768768, 423516473627]),
    UInt256(w: [0, 0, 5540277928681537536, 847032947254]),
    UInt256(w: [0, 0, 11080555857363075072, 1694065894508]),
    UInt256(w: [0, 0, 13284157615698345984, 338813178901]),
    UInt256(w: [0, 0, 8121571157687140352, 677626357803]),
    UInt256(w: [0, 0, 16243142315374280704, 1355252715606]),
    UInt256(w: [0, 0, 14039540557039009792, 2710505431213]),
    UInt256(w: [0, 0, 13875954555633532928, 542101086242]),
    UInt256(w: [0, 0, 9305165037557514240, 1084202172485]),
    UInt256(w: [0, 0, 163586001405476864, 2168404344971]),
    UInt256(w: [0, 0, 3722066015023005696, 433680868994]),
    UInt256(w: [0, 0, 7444132030046011392, 867361737988]),
    UInt256(w: [0, 0, 14888264060092022784, 1734723475976]),
    UInt256(w: [0, 0, 6667001626760314880, 346944695195]),
    UInt256(w: [0, 0, 13334003253520629760, 693889390390]),
    UInt256(w: [0, 0, 8221262433331707904, 1387778780781]),
    UInt256(w: [0, 0, 16442524866663415808, 2775557561562]),
    UInt256(w: [0, 0, 10667202602816503808, 555111512312]),
    UInt256(w: [0, 0, 2887661131923456000, 1110223024625]),
    UInt256(w: [0, 0, 5775322263846912000, 2220446049250]),
    UInt256(w: [0, 0, 1155064452769382400, 444089209850]),
    UInt256(w: [0, 0, 2310128905538764800, 888178419700]),
    UInt256(w: [0, 0, 4620257811077529600, 1776356839400]),
    UInt256(w: [0, 0, 924051562215505920, 355271367880]),
    UInt256(w: [0, 0, 1848103124431011840, 710542735760]),
    UInt256(w: [0, 0, 3696206248862023680, 1421085471520]),
    UInt256(w: [0, 0, 7392412497724047360, 2842170943040]),
    UInt256(w: [0, 0, 1478482499544809472, 568434188608]),
    UInt256(w: [0, 0, 2956964999089618944, 1136868377216]),
    UInt256(w: [0, 0, 5913929998179237888, 2273736754432]),
    UInt256(w: [0, 0, 8561483629119668224, 454747350886]),
    UInt256(w: [0, 0, 17122967258239336448, 909494701772]),
    UInt256(w: [0, 0, 15799190442769121280, 1818989403545]),
    UInt256(w: [0, 0, 3159838088553824256, 363797880709]),
    UInt256(w: [0, 0, 6319676177107648512, 727595761418]),
    UInt256(w: [0, 0, 12639352354215297024, 1455191522836]),
    UInt256(w: [0, 0, 6831960634721042432, 2910383045673]),
    UInt256(w: [0, 0, 12434438571169939456, 582076609134]),
    UInt256(w: [0, 0, 6422133068630327296, 1164153218269]),
    UInt256(w: [0, 0, 12844266137260654592, 2328306436538]),
    UInt256(w: [0, 0, 13636899671677861888, 465661287307]),
    UInt256(w: [0, 0, 8827055269646172160, 931322574615]),
    UInt256(w: [0, 0, 17654110539292344320, 1862645149230]),
    UInt256(w: [0, 0, 3530822107858468864, 372529029846]),
    UInt256(w: [0, 0, 7061644215716937728, 745058059692]),
    UInt256(w: [0, 0, 14123288431433875456, 1490116119384]),
    UInt256(w: [0, 0, 9799832789158199296, 2980232238769]),
    UInt256(w: [0, 0, 16717361816799281152, 596046447753]),
    UInt256(w: [0, 0, 14987979559889010688, 1192092895507]),
    UInt256(w: [0, 0, 11529215046068469760, 2384185791015]),
    UInt256(w: [0, 0, 2305843009213693952, 476837158203]),
    UInt256(w: [0, 0, 4611686018427387904, 953674316406]),
    UInt256(w: [0, 0, 9223372036854775808, 1907348632812]),
    UInt256(w: [0, 0, 9223372036854775808, 381469726562]),
    UInt256(w: [0, 0, 0, 762939453125]),
    UInt256(w: [0, 0, 0, 1525878906250]),
    UInt256(w: [0, 0, 0, 3051757812500]),
    UInt256(w: [0, 0, 0, 610351562500]),
    UInt256(w: [0, 0, 0, 1220703125000]),
    UInt256(w: [0, 0, 0, 2441406250000]),
    UInt256(w: [0, 0, 0, 488281250000]),
    UInt256(w: [0, 0, 0, 976562500000]),
    UInt256(w: [0, 0, 0, 1953125000000]),
    UInt256(w: [0, 0, 0, 390625000000]),
    UInt256(w: [0, 0, 0, 781250000000]),
    UInt256(w: [0, 0, 0, 1562500000000]),
    UInt256(w: [0, 0, 0, 3125000000000]),
    UInt256(w: [0, 0, 0, 625000000000]),
    UInt256(w: [0, 0, 0, 1250000000000]),
    UInt256(w: [0, 0, 0, 2500000000000]),
    UInt256(w: [0, 0, 0, 500000000000]),
    UInt256(w: [0, 0, 0, 1000000000000]),
    UInt256(w: [0, 0, 0, 2000000000000]),
    UInt256(w: [0, 0, 0, 400000000000]),
    UInt256(w: [0, 0, 0, 800000000000]),
    UInt256(w: [0, 0, 0, 1600000000000]),
    UInt256(w: [0, 0, 0, 3200000000000]),
    UInt256(w: [0, 0, 0, 640000000000]),
    UInt256(w: [0, 0, 0, 1280000000000]),
    UInt256(w: [0, 0, 0, 2560000000000]),
    UInt256(w: [0, 0, 0, 512000000000]),
    UInt256(w: [0, 0, 0, 1024000000000]),
    UInt256(w: [0, 0, 0, 2048000000000]),
    UInt256(w: [0, 0, 0, 409600000000]),
    UInt256(w: [0, 0, 0, 819200000000]),
    UInt256(w: [0, 0, 0, 1638400000000]),
    UInt256(w: [0, 0, 0, 3276800000000]),
    UInt256(w: [0, 0, 0, 655360000000]),
    UInt256(w: [0, 0, 0, 1310720000000]),
    UInt256(w: [0, 0, 0, 2621440000000]),
    UInt256(w: [0, 0, 0, 524288000000]),
    UInt256(w: [0, 0, 0, 1048576000000]),
    UInt256(w: [0, 0, 0, 2097152000000]),
    UInt256(w: [0, 0, 0, 419430400000]),
    UInt256(w: [0, 0, 0, 838860800000]),
    UInt256(w: [0, 0, 0, 1677721600000]),
    UInt256(w: [0, 0, 0, 335544320000]),
    UInt256(w: [0, 0, 0, 671088640000]),
    UInt256(w: [0, 0, 0, 1342177280000]),
    UInt256(w: [0, 0, 0, 2684354560000]),
    UInt256(w: [0, 0, 0, 536870912000]),
    UInt256(w: [0, 0, 0, 1073741824000]),
    UInt256(w: [0, 0, 0, 2147483648000]),
    UInt256(w: [0, 0, 0, 429496729600]),
    UInt256(w: [0, 0, 0, 858993459200]),
    UInt256(w: [0, 0, 0, 1717986918400]),
    UInt256(w: [0, 0, 0, 343597383680]),
    UInt256(w: [0, 0, 0, 687194767360]),
    UInt256(w: [0, 0, 0, 1374389534720]),
    UInt256(w: [0, 0, 0, 2748779069440]),
    UInt256(w: [0, 0, 0, 549755813888]),
    UInt256(w: [0, 0, 0, 1099511627776]),
    UInt256(w: [0, 0, 0, 2199023255552]),
    UInt256(w: [7378697629483820647, 7378697629483820646,  7378697629483820646, 439804651110]),
    UInt256(w: [14757395258967641293, 14757395258967641292,  14757395258967641292, 879609302220]),
    UInt256(w: [11068046444225730970, 11068046444225730969,  11068046444225730969, 1759218604441]),
    UInt256(w: [16971004547812787487, 2213609288845146193,  5902958103587056517, 351843720888]),
    UInt256(w: [15495265021916023358, 4427218577690292387,  11805916207174113034, 703687441776]),
    UInt256(w: [12543785970122495099, 8854437155380584775,  5165088340638674452, 1407374883553]),
    UInt256(w: [6640827866535438582, 17708874310761169551,  10330176681277348904, 2814749767106]),
    UInt256(w: [5017514388048998040, 3541774862152233910,  5755384150997380104, 562949953421]),
    UInt256(w: [10035028776097996080, 7083549724304467820,  11510768301994760208, 1125899906842]),
    UInt256(w: [1623313478486440543, 14167099448608935641,  4574792530279968800, 2251799813685]),
    UInt256(w: [4014011510439198432, 2833419889721787128, 914958506055993760,  450359962737]),
    UInt256(w: [8028023020878396864, 5666839779443574256,  1829917012111987520, 900719925474]),
    UInt256(w: [16056046041756793727, 11333679558887148512,  3659834024223975040, 1801439850948]),
    UInt256(w: [3211209208351358746, 13334782356003160672,  11800013249070525977, 360287970189]),
    UInt256(w: [6422418416702717491, 8222820638296769728,  5153282424431500339, 720575940379]),
    UInt256(w: [12844836833405434982, 16445641276593539456,  10306564848863000678, 1441151880758]),
    UInt256(w: [7242929593101318347, 14444538479477527297,  2166385624016449741, 2882303761517]),
    UInt256(w: [1448585918620263670, 13956954140121236429,  7811974754287110594, 576460752303]),
    UInt256(w: [2897171837240527339, 9467164206532921242,  15623949508574221189, 1152921504606]),
    UInt256(w: [5794343674481054678, 487584339356290868,  12801154943438890763, 2305843009213]),
    UInt256(w: [15916263993863852229, 3786865682613168496,  13628277432913509122, 461168601842]),
    UInt256(w: [13385783914018152841, 7573731365226336993,  8809810792117466628, 922337203685]),
    UInt256(w: [8324823754326754065, 15147462730452673987,  17619621584234933256, 1844674407370]),
    UInt256(w: [12733011195091081783, 6718841360832445120,  3523924316846986651, 368934881474]),
    UInt256(w: [7019278316472611950, 13437682721664890241,  7047848633693973302, 737869762948]),
    UInt256(w: [14038556632945223899, 8428621369620228866,  14095697267387946605, 1475739525896]),
    UInt256(w: [9630369192180896181, 16857242739240457733,  9744650461066341594, 2951479051793]),
    UInt256(w: [1926073838436179237, 10750146177331912193,  13016976536438999288, 590295810358]),
    UInt256(w: [3852147676872358473, 3053548280954272770,  7587208999168446961, 1180591620717]),
    UInt256(w: [7704295353744716945, 6107096561908545540,  15174417998336893922, 2361183241434]),
    UInt256(w: [5230207885490853713, 4910768127123619431,  17792278858635020077, 472236648286]),
    UInt256(w: [10460415770981707425, 9821536254247238862,  17137813643560488538, 944473296573]),
    UInt256(w: [2474087468253863233, 1196328434784926109,  15828883213411425461, 1888946593147]),
    UInt256(w: [7873515123134593293, 11307312131182716191,  10544474272166105738, 377789318629]),
    UInt256(w: [15747030246269186586, 4167880188655880766,  2642204470622659861, 755578637259]),
    UInt256(w: [13047316418828821556, 8335760377311761533,  5284408941245319722, 1511157274518]),
    UInt256(w: [7647888763948091496, 16671520754623523067,  10568817882490639444, 3022314549036]),
    UInt256(w: [8908275382273438946, 3334304150924704613,  5803112391240038212, 604462909807]),
    UInt256(w: [17816550764546877891, 6668608301849409226,  11606224782480076424, 1208925819614]),
    UInt256(w: [17186357455384204166, 13337216603698818453,  4765705491250601232, 2417851639229]),
    UInt256(w: [18194666750044482126, 6356792135481674013,  15710536357217761539, 483570327845]),
    UInt256(w: [17942589426379412636, 12713584270963348027,  12974328640725971462, 967140655691]),
    UInt256(w: [17438434779049273656, 6980424468217144439,  7501913207742391309, 1934281311383]),
    UInt256(w: [7177035770551765055, 8774782523127249534,  12568429085774209231, 386856262276]),
    UInt256(w: [14354071541103530109, 17549565046254499068,  6690114097838866846, 773712524553]),
    UInt256(w: [10261399008497508602, 16652386018799446521,  13380228195677733693, 1547425049106]),
    UInt256(w: [2076053943285465587, 14858027963889341427,  8313712317645915771, 3094850098213]),
    UInt256(w: [4104559603399003441, 17729000851745509578,  12730788907754914123, 618970019642]),
    UInt256(w: [8209119206798006882, 17011257629781467540,  7014833741800276631, 1237940039285]),
    UInt256(w: [16418238413596013763, 15575771185853383464,  14029667483600553263, 2475880078570]),
    UInt256(w: [10662345312203023399, 14183200681396407662,  2805933496720110652, 495176015714]),
    UInt256(w: [2877946550696495182, 9919657289083263709,  5611866993440221305, 990352031428]),
    UInt256(w: [5755893101392990364, 1392570504456975802,  11223733986880442611, 1980704062856]),
    UInt256(w: [15908573879246239366, 7657211730375215806,  5934095612117998845, 396140812571]),
    UInt256(w: [13370403684782927115, 15314423460750431613,  11868191224235997690, 792281625142]),
    UInt256(w: [8294063295856302614, 12182102847791311611,  5289638374762443765, 1584563250285]),
    UInt256(w: [16588126591712605228, 5917461621873071606,  10579276749524887531, 3169126500570]),
    UInt256(w: [10696322947826341692, 4872841139116524644,  2115855349904977506, 633825300114]),
    UInt256(w: [2945901821943131768, 9745682278233049289,  4231710699809955012, 1267650600228]),
    UInt256(w: [5891803643886263536, 1044620482756546962,  8463421399619910025, 2535301200456]),
    UInt256(w: [12246407173002983677, 3898272911293219715,  5382033094665892328, 507060240091]),
    UInt256(w: [6046070272296415738, 7796545822586439431,  10764066189331784656, 1014120480182]),
    UInt256(w: [12092140544592831476, 15593091645172878862,  3081388304954017696, 2028240960365]),
    UInt256(w: [13486474553144297265, 6807967143776486095,  616277660990803539, 405648192073]),
    UInt256(w: [8526205032579042914, 13615934287552972191,  1232555321981607078, 811296384146]),
    UInt256(w: [17052410065158085827, 8785124501396392766,  2465110643963214157, 1622592768292]),
    UInt256(w: [15658076056606620037, 17570249002792785533,  4930221287926428314, 3245185536584]),
    UInt256(w: [6820964026063234331, 14582096244784288076,  15743439516552926955, 649037107316]),
    UInt256(w: [13641928052126468662, 10717448415859024536,  13040134959396302295, 1298074214633]),
    UInt256(w: [8837112030543385707, 2988152758008497457,  7633525845083052975, 2596148429267]),
    UInt256(w: [16524817665076318435, 7976328181085520137,  8905402798500431241, 519229685853]),
    UInt256(w: [14602891256443085253, 15952656362171040275,  17810805597000862482, 1038459371706]),
    UInt256(w: [10759038439176618889, 13458568650632528935,  17174867120292173349, 2076918743413]),
    UInt256(w: [9530505317319144425, 10070411359610326433,  14503019868284165639, 415383748682]),
    UInt256(w: [614266560928737233, 1694078645511101251,  10559295662858779663, 830767497365]),
    UInt256(w: [1228533121857474465, 3388157291022202502,  2671847252008007710, 1661534994731]),
    UInt256(w: [11313753068597225863, 4366980272946350823,  4223718265143511865, 332306998946]),
    UInt256(w: [4180762063484900109, 8733960545892701647,  8447436530287023730, 664613997892]),
    UInt256(w: [8361524126969800218, 17467921091785403294,  16894873060574047460, 1329227995784]),
    UInt256(w: [16723048253939600436, 16489098109861254972,  15343002047438543305, 2658455991569]),
    UInt256(w: [7033958465529830411, 18055214880939892287,  17825995668455349953, 531691198313]),
    UInt256(w: [14067916931059660821, 17663685688170232958,  17205247263201148291, 1063382396627]),
    UInt256(w: [9689089788409770026, 16880627302630914301,  15963750452692744967, 2126764793255]),
    UInt256(w: [13005864401907684975, 10754823090010003506,  3192750090538548993, 425352958651]),
    UInt256(w: [7564984730105818334, 3062902106310455397,  6385500181077097987, 850705917302]),
    UInt256(w: [15129969460211636667, 6125804212620910794,  12771000362154195974, 1701411834604]),
    UInt256(w: [10404691521526147980, 12293207286749913128,  17311595331398480487, 340282366920]),
    UInt256(w: [2362638969342744344, 6139670499790274641,  16176446589087409359, 680564733841]),
    UInt256(w: [4725277938685488687, 12279340999580549282,  13906149104465267102, 1361129467683]),
    UInt256(w: [9450555877370977374, 6111937925451546948,  9365554135220982589, 2722258935367]),
    UInt256(w: [16647506434441836768, 4911736399832219712,  9251808456528017164, 544451787073]),
    UInt256(w: [14848268795174121920, 9823472799664439425, 56872839346482712,  1088903574147]),
    UInt256(w: [11249793516638692223, 1200201525619327235,  113745678692965425, 2177807148294]),
    UInt256(w: [17007353962295379738, 14997435564091506739,  14780144394706234377, 435561429658]),
    UInt256(w: [15567963850881207859, 11548127054473461863,  11113544715702917139, 871122859317]),
    UInt256(w: [12689183628052864101, 4649510035237372111,  3780345357696282663, 1742245718635]),
    UInt256(w: [17295231984578214113, 11997948451273205391,  756069071539256532, 348449143727]),
    UInt256(w: [16143719895446876610, 5549152828836859167,  1512138143078513065, 696898287454]),
    UInt256(w: [13840695717184201604, 11098305657673718335,  3024276286157026130, 1393796574908]),
    UInt256(w: [9234647360658851592, 3749867241637885055,  6048552572314052261, 2787593149816]),
    UInt256(w: [9225627101615590965, 8128671077811397657,  4899059329204720775, 557518629963]),
    UInt256(w: [4510129521630314, 16257342155622795315, 9798118658409441550,  1115037259926]),
    UInt256(w: [9020259043260628, 14067940237536039014, 1149493243109331485,  2230074519853]),
    UInt256(w: [7380501681292472772, 13881634491732938772,  11297945092847597266, 446014903970]),
    UInt256(w: [14761003362584945544, 9316524909756325928,  4149146111985642917, 892029807941]),
    UInt256(w: [11075262651460339472, 186305745803100241,  8298292223971285835, 1784059615882]),
    UInt256(w: [13283098974517798864, 7415958778644440694,  9038356074278077813, 356811923176]),
    UInt256(w: [8119453875326046112, 14831917557288881389,  18076712148556155626, 713623846352]),
    UInt256(w: [16238907750652092224, 11217091040868211162,  17706680223402759637, 1427247692705]),
    UInt256(w: [14031071427594632831, 3987438008026870709,  16966616373095967659, 2854495385411]),
    UInt256(w: [17563609544486567859, 797487601605374141,  7082672089361103855, 570899077082]),
    UInt256(w: [16680475015263584102, 1594975203210748283,  14165344178722207710, 1141798154164]),
    UInt256(w: [14914205956817616588, 3189950406421496567,  9883944283734863804, 2283596308329]),
    UInt256(w: [2982841191363523318, 11706036525510030283,  16734184115714614053, 456719261665]),
    UInt256(w: [5965682382727046636, 4965328977310508950,  15021624157719676491, 913438523331]),
    UInt256(w: [11931364765454093271, 9930657954621017900,  11596504241729801366, 1826877046663]),
    UInt256(w: [17143668212058459947, 16743526849891844872,  13387347292571691242, 365375409332]),
    UInt256(w: [15840592350407368278, 15040309626074138129,  8327950511433830869, 730750818665]),
    UInt256(w: [13234440627105184940, 11633875178438724643,  16655901022867661739, 1461501637330]),
    UInt256(w: [8022137180500818264, 4821006283167897671,  14865057972025771863, 2923003274661]),
    UInt256(w: [1604427436100163653, 15721596515601220827,  6662360409147064695, 584600654932]),
    UInt256(w: [3208854872200327306, 12996448957492890038,  13324720818294129391, 1169201309864]),
    UInt256(w: [6417709744400654611, 7546153841276228460,  8202697562878707167, 2338402619729]),
    UInt256(w: [4972890763622041246, 5198579582997156015,  16397934771543382726, 467680523945]),
    UInt256(w: [9945781527244082491, 10397159165994312030,  14349125469377213836, 935361047891]),
    UInt256(w: [1444818980778613366, 2347574258279072445,  10251506865044876057, 1870722095783]),
    UInt256(w: [288963796155722674, 469514851655814489, 13118347817234706181,  374144419156]),
    UInt256(w: [577927592311445347, 939029703311628978, 7789951560759860746,  748288838313]),
    UInt256(w: [1155855184622890693, 1878059406623257956,  15579903121519721492, 1496577676626]),
    UInt256(w: [2311710369245781385, 3756118813246515912,  12713062169329891368, 2993155353253]),
    UInt256(w: [11530388518074887247, 4440572577391213505,  13610658878091709243, 598631070650]),
    UInt256(w: [4614032962440222877, 8881145154782427011,  8774573682473866870, 1197262141301]),
    UInt256(w: [9228065924880445754, 17762290309564854022,  17549147364947733740, 2394524282602]),
    UInt256(w: [16603008443943730444, 10931155691396791450,  10888527102473367394, 478904856520]),
    UInt256(w: [14759272814177909272, 3415567309084031285,  3330310131237183173, 957809713041]),
    UInt256(w: [11071801554646266927, 6831134618168062571,  6660620262474366346, 1915619426082]),
    UInt256(w: [16971755569896894679, 12434273367859343483,  8710821681978693915, 383123885216]),
    UInt256(w: [15496767066084237741, 6421802662009135351,  17421643363957387831, 766247770432]),
    UInt256(w: [12546790058458923865, 12843605324018270703,  16396542654205224046, 1532495540865]),
    UInt256(w: [6646836043208296113, 7240466574326989791,  14346341234700896477, 3064991081731]),
    UInt256(w: [16086762467609300516, 12516139759091128927,  6558617061682089618, 612998216346]),
    UInt256(w: [13726780861509049415, 6585535444472706239,  13117234123364179237, 1225996432692]),
    UInt256(w: [9006817649308547214, 13171070888945412479,  7787724173018806858, 2451992865385]),
    UInt256(w: [9180061159345530090, 13702260622014813465,  1557544834603761371, 490398573077]),
    UInt256(w: [18360122318691060179, 8957777170320075314,  3115089669207522743, 980797146154]),
    UInt256(w: [18273500563672568741, 17915554340640150629,  6230179338415045486, 1961594292308]),
    UInt256(w: [14722746556960244718, 18340506127095671418,  12314082311908740066, 392318858461]),
    UInt256(w: [10998749040210937820, 18234268180481791221,  6181420550107928517, 784637716923]),
    UInt256(w: [3550754006712324023, 18021792287254030827,  12362841100215857035, 1569275433846]),
    UInt256(w: [7101508013424648045, 17596840500798510038,  6278938126722162455, 3138550867693]),
    UInt256(w: [5109650417426839933, 14587414544385432977,  12323834069570163460, 627710173538]),
    UInt256(w: [10219300834853679865, 10728085015061314338,  6200924065430775305, 1255420347077]),
    UInt256(w: [1991857595997808113, 3009425956413077061,  12401848130861550611, 2510840694154]),
    UInt256(w: [4087720333941471946, 601885191282615412,  17237764885139951415, 502168138830]),
    UInt256(w: [8175440667882943892, 1203770382565230824,  16028785696570351214, 1004336277661]),
    UInt256(w: [16350881335765887783, 2407540765130461648,  13610827319431150812, 2008672555323]),
    UInt256(w: [14338222711378908527, 481508153026092329,  13790211908111961132, 401734511064]),
    UInt256(w: [10229701349048265437, 963016306052184659,  9133679742514370648, 803469022129]),
    UInt256(w: [2012658624386979257, 1926032612104369319,  18267359485028741296, 1606938044258]),
    UInt256(w: [4025317248773958514, 3852065224208738638,  18087974896347930976, 3213876088517]),
    UInt256(w: [4494412264496702026, 11838459489067478697,  10996292608753406841, 642775217703]),
    UInt256(w: [8988824528993404052, 5230174904425405778,  3545841143797262067, 1285550435407]),
    UInt256(w: [17977649057986808104, 10460349808850811556,  7091682287594524134, 2571100870814]),
    UInt256(w: [18352925070565002914, 13160116405995893280,  16175731716486546119, 514220174162]),
    UInt256(w: [18259106067420454212, 7873488738282234945,  13904719359263540623, 1028440348325]),
    UInt256(w: [18071468061131356807, 15746977476564469891,  9362694644817529630, 2056880696651]),
    UInt256(w: [10992991241710092008, 6838744310054804301,  5561887743705416249, 411376139330]),
    UInt256(w: [3539238409710632400, 13677488620109608603,  11123775487410832498, 822752278660]),
    UInt256(w: [7078476819421264799, 8908233166509665590,  3800806901112113381, 1645504557321]),
    UInt256(w: [8794392993368073607, 9160344262785753764,  4449510194964332999, 329100911464]),
    UInt256(w: [17588785986736147213, 18320688525571507528,  8899020389928665998, 658201822928]),
    UInt256(w: [16730827899762742809, 18194632977433463441,  17798040779857331997, 1316403645856]),
    UInt256(w: [15014911725815934001, 17942521881157375267,  17149337486005112379, 2632807291713]),
    UInt256(w: [17760377604130828093, 10967202005715295699,  14497913941426753445, 526561458342]),
    UInt256(w: [17074011134552104570, 3487659937721039783,  10549083809143955275, 1053122916685]),
    UInt256(w: [15701278195394657524, 6975319875442079567,  2651423544578358934, 2106245833371]),
    UInt256(w: [10518953268562752152, 1395063975088415913,  4219633523657582110, 421249166674]),
    UInt256(w: [2591162463415952687, 2790127950176831827,  8439267047315164220, 842498333348]),
    UInt256(w: [5182324926831905373, 5580255900353663654,  16878534094630328440, 1684996666696]),
    UInt256(w: [1036464985366381075, 4805399994812643054,  7065055633667976011, 336999333339]),
    UInt256(w: [2072929970732762150, 9610799989625286108,  14130111267335952022, 673998666678]),
    UInt256(w: [4145859941465524299, 774855905541020600, 9813478460962352429,  1347997333357]),
    UInt256(w: [8291719882931048597, 1549711811082041200,  1180212848215153242, 2695994666715]),
    UInt256(w: [9037041606070030366, 7688639991700228886, 236042569643030648,  539198933343]),
    UInt256(w: [18074083212140060732, 15377279983400457772,  472085139286061296, 1078397866686]),
    UInt256(w: [17701422350570569847, 12307815893091363929,  944170278572122593, 2156795733372]),
    UInt256(w: [18297679729081755263, 2461563178618272785,  7567531685198245165, 431359146674]),
    UInt256(w: [18148615384453958909, 4923126357236545571,  15135063370396490330, 862718293348]),
    UInt256(w: [17850486695198366201, 9846252714473091143,  11823382667083429044, 1725436586697]),
    UInt256(w: [18327492598007314533, 5658599357636528551,  9743374162900506455, 345087317339]),
    UInt256(w: [18208241122305077450, 11317198715273057103,  1040004252091461294, 690174634679]),
    UInt256(w: [17969738170900603284, 4187653356836562591,  2080008504182922589, 1380349269358]),
    UInt256(w: [17492732268091654952, 8375306713673125183,  4160017008365845178, 2760698538716]),
    UInt256(w: [10877244083102151637, 16432456601702266329,  4521352216415079358, 552139707743]),
    UInt256(w: [3307744092494751658, 14418169129694981043,  9042704432830158717, 1104279415486]),
    UInt256(w: [6615488184989503315, 10389594185680410470,  18085408865660317435, 2208558830972]),
    UInt256(w: [8701795266481721310, 9456616466619902740,  10995779402615884133, 441711766194]),
    UInt256(w: [17403590532963442619, 466488859530253864,  3544814731522216651, 883423532389]),
    UInt256(w: [16360436992217333622, 932977719060507729,  7089629463044433302, 1766847064778]),
    UInt256(w: [18029482657411108018, 186595543812101545,  12485972336834617630, 353369412955]),
    UInt256(w: [17612221241112664419, 373191087624203091,  6525200599959683644, 706738825911]),
    UInt256(w: [16777698408515777221, 746382175248406183,  13050401199919367288, 1413477651822]),
    UInt256(w: [15108652743322002825, 1492764350496812367,  7654058326129182960, 2826955303645]),
    UInt256(w: [10400428178148221212, 298552870099362473,  1530811665225836592, 565391060729]),
    UInt256(w: [2354112282586890807, 597105740198724947, 3061623330451673184,  1130782121458]),
    UInt256(w: [4708224565173781614, 1194211480397449894,  6123246660903346368, 2261564242916]),
    UInt256(w: [12009691357260487293, 14996237555047131271,  4913998146922579596, 452312848583]),
    UInt256(w: [5572638640811422969, 11545731036384710927,  9827996293845159193, 904625697166]),
    UInt256(w: [11145277281622845937, 4644717999059870238,  1209248513980766771, 1809251394333]),
    UInt256(w: [9607753085808389834, 15686338858779615340,  11309896147021884323, 361850278866]),
    UInt256(w: [768762097907228052, 12925933643849679065,  4173048220334217031, 723700557733]),
    UInt256(w: [1537524195814456104, 7405123213989806514,  8346096440668434063, 1447401115466]),
    UInt256(w: [3075048391628912207, 14810246427979613028,  16692192881336868126, 2894802230932]),
    UInt256(w: [4304358493067692765, 14030095729821653575,  10717136205751194271, 578960446186]),
    UInt256(w: [8608716986135385529, 9613447385933755534,  2987528337792836927, 1157920892373]),
    UInt256(w: [17217433972270771058, 780150698157959452,  5975056675585673855, 2315841784746]),
    UInt256(w: [14511533238679885182, 3845378954373502213,  4884360149859045094, 463168356949]),
    UInt256(w: [10576322403650218747, 7690757908747004427,  9768720299718090188, 926336713898]),
    UInt256(w: [2705900733590885877, 15381515817494008855,  1090696525726628760, 1852673427797]),
    UInt256(w: [7919877776201997822, 10455000792982622417,  7596836934629146398, 370534685559]),
    UInt256(w: [15839755552403995644, 2463257512255693218,  15193673869258292797, 741069371118]),
    UInt256(w: [13232767031098439671, 4926515024511386437,  11940603664807033978, 1482138742237]),
    UInt256(w: [8018789988487327726, 9853030049022772875,  5434463255904516340, 2964277484475]),
    UInt256(w: [1603757997697465546, 1970606009804554575,  1086892651180903268, 592855496895]),
    UInt256(w: [3207515995394931091, 3941212019609109150,  2173785302361806536, 1185710993790]),
    UInt256(w: [6415031990789862181, 7882424039218218300,  4347570604723613072, 2371421987580]),
    UInt256(w: [8661704027641793083, 8955182437327464306, 869514120944722614,  474284397516]),
    UInt256(w: [17323408055283586166, 17910364874654928612,  1739028241889445228, 948568795032]),
    UInt256(w: [16200072036857620715, 17373985675600305609,  3478056483778890457, 1897137590064]),
    UInt256(w: [3240014407371524143, 7164145949861971445,  15453006555723419384, 379427518012]),
    UInt256(w: [6480028814743048286, 14328291899723942890,  12459269037737287152, 758855036025]),
    UInt256(w: [12960057629486096572, 10209839725738334164,  6471794001765022689, 1517710072051]),
    UInt256(w: [7473371185262641527, 1972935377767116713,  12943588003530045379, 3035420144102]),
    UInt256(w: [16252069496020169599, 4083935890295333665,  9967415230189829722, 607084028820]),
    UInt256(w: [14057394918330787581, 8167871780590667331,  1488086386670107828, 1214168057641]),
    UInt256(w: [9668045762952023545, 16335743561181334663,  2976172773340215656, 2428336115282]),
    UInt256(w: [5622957967332315033, 14335195156461997902,  7973932184151863777, 485667223056]),
    UInt256(w: [11245915934664630065, 10223646239214444188,  15947864368303727555, 971334446112]),
    UInt256(w: [4045087795619708513, 2000548404719336761,  13448984662897903495, 1942668892225]),
    UInt256(w: [4498366373865852026, 400109680943867352, 2689796932579580699,  388533778445]),
    UInt256(w: [8996732747731704052, 800219361887734704, 5379593865159161398,  777067556890]),
    UInt256(w: [17993465495463408103, 1600438723775469408,  10759187730318322796, 1554135113780]),
    UInt256(w: [17540186917217264590, 3200877447550938817,  3071631386927093976, 3108270227561]),
    UInt256(w: [18265432642411094211, 8018873118994008409,  4303675092127329118, 621654045512]),
    UInt256(w: [18084121211112636806, 16037746237988016819,  8607350184254658236, 1243308091024]),
    UInt256(w: [17721498348515721995, 13628748402266482023,  17214700368509316473, 2486616182048]),
    UInt256(w: [18301694928670785692, 6415098495195206727,  14510986517927594264, 497323236409]),
    UInt256(w: [18156645783632019768, 12830196990390413455,  10575228962145636912, 994646472819]),
    UInt256(w: [17866547493554487919, 7213649907071275295,  2703713850581722209, 1989292945639]),
    UInt256(w: [14641355942936628554, 12510776425639986028,  15298138029083985734, 397858589127]),
    UInt256(w: [10835967812163705491, 6574808777570420441,  12149531984458419853, 795717178255]),
    UInt256(w: [3225191550617859366, 13149617555140840883,  5852319895207288090, 1591434356511]),
    UInt256(w: [6450383101235718732, 7852491036572130150,  11704639790414576181, 3182868713022]),
    UInt256(w: [12358123064472874716, 12638544651540156999,  9719625587566735882, 636573742604]),
    UInt256(w: [6269502055236197816, 6830345229370762383, 992507101423920149,  1273147485209]),
    UInt256(w: [12539004110472395632, 13660690458741524766,  1985014202847840298, 2546294970418]),
    UInt256(w: [9886498451578299773, 6421486906490215276,  11465049284795299029, 509258994083]),
    UInt256(w: [1326252829447047930, 12842973812980430553,  4483354495881046442, 1018517988167]),
    UInt256(w: [2652505658894095859, 7239203552251309490,  8966708991762092885, 2037035976334]),
    UInt256(w: [15287896390746460465, 16205235969417903190,  16550737057320059869, 407407195266]),
    UInt256(w: [12129048707783369314, 13963727865126254765,  14654730040930568123, 814814390533]),
    UInt256(w: [5811353341857187011, 9480711656542957915,  10862716008151584631, 1629628781067]),
    UInt256(w: [11622706683714374021, 514679239376364214,  3278687942593617647, 3259257562135]),
    UInt256(w: [6013890151484785128, 7481633477359093489, 655737588518723529,  651851512427]),
    UInt256(w: [12027780302969570255, 14963266954718186978,  1311475177037447058, 1303703024854]),
    UInt256(w: [5608816532229588893, 11479789835726822341,  2622950354074894117, 2607406049708]),
    UInt256(w: [4811112121187828102, 2295957967145364468,  11592636515040709793, 521481209941]),
    UInt256(w: [9622224242375656204, 4591915934290728936,  4738528956371867970, 1042962419883]),
    UInt256(w: [797704411041760792, 9183831868581457873, 9477057912743735940,  2085924839766]),
    UInt256(w: [14916936141175993452, 5526115188458201897,  5584760397290657511, 417184967953]),
    UInt256(w: [11387128208642435287, 11052230376916403795,  11169520794581315022, 834369935906]),
    UInt256(w: [4327512343575318957, 3657716680123255975,  3892297515453078429, 1668739871813]),
    UInt256(w: [8244200098198884438, 8110240965508471841,  11846505947316346655, 333747974362]),
    UInt256(w: [16488400196397768876, 16220481931016943682,  5246267820923141694, 667495948725]),
    UInt256(w: [14530056319085986135, 13994219788324335749,  10492535641846283389, 1334991897450]),
    UInt256(w: [10613368564462420654, 9541695502939119883,  2538327209983015163, 2669983794901]),
    UInt256(w: [9501371342376304778, 16665734359555465269,  4197014256738513355, 533996758980]),
    UInt256(w: [555998611043057939, 14884724645401378923,  8394028513477026711, 1067993517960]),
    UInt256(w: [1111997222086115877, 11322705217093206230,  16788057026954053423, 2135987035920]),
    UInt256(w: [11290445888642954145, 13332587487644372215,  3357611405390810684, 427197407184]),
    UInt256(w: [4134147703576356674, 8218430901579192815,  6715222810781621369, 854394814368]),
    UInt256(w: [8268295407152713348, 16436861803158385630,  13430445621563242738, 1708789628736]),
    UInt256(w: [16411054340398183963, 18044767619599318418,  6375437939054558870, 341757925747]),
    UInt256(w: [14375364607086816309, 17642791165489085221,  12750875878109117741, 683515851494]),
    UInt256(w: [10303985140464081001, 16838838257268618827,  7055007682508683867, 1367031702989]),
    UInt256(w: [2161226207218610386, 15230932440827686039,  14110015365017367735, 2734063405978]),
    UInt256(w: [7810942870927542724, 14114232932391268177,  13890049517229204516, 546812681195]),
    UInt256(w: [15621885741855085448, 9781721791072984738,  9333354960748857417, 1093625362391]),
    UInt256(w: [12797027410000619279, 1116699508436417861,  219965847788163219, 2187250724783]),
    UInt256(w: [13627451926225854826, 7602037531171104218,  11112039613783363613, 437450144956]),
    UInt256(w: [8808159778742158035, 15204075062342208437,  3777335153857175610, 874900289913]),
    UInt256(w: [17616319557484316070, 11961406050974865258,  7554670307714351221, 1749800579826]),
    UInt256(w: [3523263911496863214, 9770978839678793698,  5200282876284780567, 349960115965]),
    UInt256(w: [7046527822993726428, 1095213605648035780,  10400565752569561135, 699920231930]),
    UInt256(w: [14093055645987452856, 2190427211296071560,  2354387431429570654, 1399840463861]),
    UInt256(w: [9739367218265354095, 4380854422592143121,  4708774862859141308, 2799680927722]),
    UInt256(w: [5637222258394981143, 876170884518428624, 8320452602055648908,  559936185544]),
    UInt256(w: [11274444516789962285, 1752341769036857248,  16640905204111297816, 1119872371088])
  ]
  
  static let bid_multipliers2_bid32: [UInt256] = [
    UInt256(w: [7156996302188685206, 14694123111064470433, 3521238664523520994, 11704]),
    UInt256(w: [14313992604377370412, 10941502148419389250, 7042477329047041989, 23408]),
    UInt256(w: [10181241135045189207, 3436260223129226885,  14084954658094083979, 46816]),
    UInt256(w: [1915738196380826798, 6872520446258453771,  9723165242478616342, 93633]),
    UInt256(w: [3831476392761653595, 13745040892516907542,  999586411247681068, 187267]),
    UInt256(w: [7662952785523307189, 9043337711324263468,  1999172822495362137, 374534]),
    UInt256(w: [15325905571046614378, 18086675422648526936,  3998345644990724274, 749068]),
    UInt256(w: [12205067068383677139, 17726606771587502257,  7996691289981448549, 1498136]),
    UInt256(w: [5963390063057802661, 17006469469465452899,  15993382579962897099, 2996272]),
    UInt256(w: [11926780126115605321, 15566194865221354182,  13540021086216242583, 5992545]),
    UInt256(w: [5406816178521659026, 12685645656733156749,  8633298098722933551, 11985091]),
    UInt256(w: [10813632357043318052, 6924547239756761882,  17266596197445867103, 23970182]),
    UInt256(w: [3180520640377084488, 13849094479513523765,  16086448321182182590, 47940365]),
    UInt256(w: [6361041280754168975, 9251444885317495914,  13726152568654813565, 95880731]),
    UInt256(w: [12722082561508337950, 56145696925440212, 9005561063600075515,  191761463]),
    UInt256(w: [6997421049307124283, 112291393850880425,  18011122127200151030, 383522926]),
    UInt256(w: [13994842098614248565, 224582787701760850,  17575500180690750444, 767045853]),
    UInt256(w: [9542940123518945513, 449165575403521701,  16704256287671949272, 1534091707]),
    UInt256(w: [639136173328339410, 898331150807043403, 14961768501634346928,  3068183415]),
    UInt256(w: [1278272346656678820, 1796662301614086806,  11476792929559142240, 6136366831]),
    UInt256(w: [2556544693313357639, 3593324603228173612,  4506841785408732864, 12272733663]),
    UInt256(w: [5113089386626715277, 7186649206456347224,  9013683570817465728, 24545467326]),
    UInt256(w: [10226178773253430554, 14373298412912694448,  18027367141634931456, 49090934652]),
    UInt256(w: [2005613472797309491, 10299852752115837281,  17607990209560311297, 98181869305]),
    UInt256(w: [4011226945594618982, 2152961430522122946,  16769236345411070979, 196363738611]),
    UInt256(w: [4491594203860834120, 430592286104424589, 7043196083824124519,  39272747722]),
    UInt256(w: [8983188407721668240, 861184572208849178,  14086392167648249038, 78545495444]),
    UInt256(w: [17966376815443336479, 1722369144417698356,  9726040261586946460, 157090990889]),
    UInt256(w: [17486009557177121341, 3444738288835396713,  1005336449464341304, 314181981779]),
    UInt256(w: [7186550726177334592, 11756994101992810312,  14958462548860509553, 62836396355]),
    UInt256(w: [14373101452354669183, 5067244130276069008,  11470181024011467491, 125672792711]),
    UInt256(w: [10299458830999786749, 10134488260552138017,  4493617974313383366, 251345585423]),
    UInt256(w: [5749240580941867673, 16784292911078068896,  11966770039088407642, 50269117084]),
    UInt256(w: [11498481161883735346, 15121841748446586176,  5486796004467263669, 100538234169]),
    UInt256(w: [4550218250057919076, 11796939423183620737,  10973592008934527339, 201076468338]),
    UInt256(w: [15667438908979225108, 9738085514120544793,  13262764846012636437, 40215293667]),
    UInt256(w: [12888133744248898600, 1029426954531537971,  8078785618315721259, 80430587335]),
    UInt256(w: [7329523414788245584, 2058853909063075943,  16157571236631442518, 160861174670]),
    UInt256(w: [14659046829576491168, 4117707818126151886,  13868398399553333420, 321722349341]),
    UInt256(w: [10310506995399118880, 4512890378367140700,  6463028494652577007, 64344469868]),
    UInt256(w: [2174269917088686144, 9025780756734281401,  12926056989305154014, 128688939736]),
    UInt256(w: [4348539834177372288, 18051561513468562802,  7405369904900756412, 257377879473]),
    UInt256(w: [8248405596319295104, 3610312302693712560,  12549120425205882252, 51475575894]),
    UInt256(w: [16496811192638590208, 7220624605387425120,  6651496776702212888, 102951151789]),
    UInt256(w: [14546878311567628800, 14441249210774850241,  13302993553404425776, 205902303578]),
    UInt256(w: [2909375662313525760, 17645645101122611341,  13728645154906616124, 41180460715]),
    UInt256(w: [5818751324627051520, 16844546128535671066,  9010546236103680633, 82360921431]),
    UInt256(w: [11637502649254103040, 15242348183361790516,  18021092472207361267, 164721842862]),
    UInt256(w: [2327500529850820608, 17805864895639999396,  10982916123925292899, 32944368572]),
    UInt256(w: [4655001059701641216, 17164985717570447176,  3519088174141034183, 65888737145]),
    UInt256(w: [9310002119403282432, 15883227361431342736,  7038176348282068367, 131777474290]),
    UInt256(w: [173260165097013248, 13319710649153133857,  14076352696564136735, 263554948580]),
    UInt256(w: [7413349662503223296, 2663942129830626771,  2815270539312827347, 52710989716]),
    UInt256(w: [14826699325006446592, 5327884259661253542,  5630541078625654694, 105421979432]),
    UInt256(w: [11206654576303341568, 10655768519322507085,  11261082157251309388, 210843958864]),
    UInt256(w: [9620028544744488960, 9509851333348322063,  17009611690417903170, 42168791772]),
    UInt256(w: [793313015779426304, 572958592987092511, 15572479307126254725,  84337583545]),
    UInt256(w: [1586626031558852608, 1145917185974185022,  12698214540542957834, 168675167091]),
    UInt256(w: [7696022835795591168, 229183437194837004, 6228991722850501890,  33735033418]),
    UInt256(w: [15392045671591182336, 458366874389674008,  12457983445701003780, 67470066836]),
    UInt256(w: [12337347269472813056, 916733748779348017,  6469222817692455944, 134940133673]),
    UInt256(w: [6227950465236074496, 1833467497558696035,  12938445635384911888, 269880267346]),
    UInt256(w: [16002985352014856192, 15124088758479380499,  6277037941818892700, 53976053469]),
    UInt256(w: [13559226630320160768, 11801433443249209383,  12554075883637785401, 107952106938]),
    UInt256(w: [8671709186930769920, 5156122812788867151,  6661407693566019187, 215904213877]),
    UInt256(w: [1734341837386153984, 15788619821525414723,  8710979168197024483, 43180842775]),
    UInt256(w: [3468683674772307968, 13130495569341277830,  17421958336394048967, 86361685550]),
    UInt256(w: [6937367349544615936, 7814247064973004044,  16397172599078546319, 172723371101]),
    UInt256(w: [16144868728876564480, 1562849412994600808,  6968783334557619587, 34544674220]),
    UInt256(w: [13842993384043577344, 3125698825989201617,  13937566669115239174, 69089348440]),
    UInt256(w: [9239242694377603072, 6251397651978403235,  9428389264520926732, 138178696881]),
    UInt256(w: [31741315045654528, 12502795303956806471, 410034455332301848,  276357393763]),
    UInt256(w: [7385045892492951552, 6189907875533271617,  11150053335292191339, 55271478752]),
    UInt256(w: [14770091784985903104, 12379815751066543234,  3853362596874831062, 110542957505]),
    UInt256(w: [11093439496262254592, 6312887428423534853,  7706725193749662125, 221085915010]),
    UInt256(w: [13286734343478181888, 1262577485684706970,  1541345038749932425, 44217183002]),
    UInt256(w: [8126724613246812160, 2525154971369413941,  3082690077499864850, 88434366004]),
    UInt256(w: [16253449226493624320, 5050309942738827882,  6165380154999729700, 176868732008]),
    UInt256(w: [3250689845298724864, 12078108432773496546,  12301122475225676909, 35373746401]),
    UInt256(w: [6501379690597449728, 5709472791837441476,  6155500876741802203, 70747492803]),
    UInt256(w: [13002759381194899456, 11418945583674882952,  12311001753483604406, 141494985606]),
    UInt256(w: [7558774688680247296, 4391147093640214289,  6175259433257657197, 282989971213]),
    UInt256(w: [16269150196703690752, 878229418728042857,  12303098330877262409, 56597994242]),
    UInt256(w: [14091556319697829888, 1756458837456085715,  6159452588044973202, 113195988485]),
    UInt256(w: [9736368565686108160, 3512917674912171431,  12318905176089946404, 226391976970]),
    UInt256(w: [1947273713137221632, 15459978793950075579,  2463781035217989280, 45278395394]),
    UInt256(w: [3894547426274443264, 12473213514190599542,  4927562070435978561, 90556790788]),
    UInt256(w: [7789094852548886528, 6499682954671647468,  9855124140871957123, 181113581576]),
    UInt256(w: [8936516599993597952, 16057331849901970786,  5660373642916301747, 36222716315]),
    UInt256(w: [17873033199987195904, 13667919626094389956,  11320747285832603495, 72445432630]),
    UInt256(w: [17299322326264840192, 8889095178479228297,  4194750497955655375, 144890865261]),
    UInt256(w: [16151900578820128768, 17778190356958456595,  8389500995911310750, 289781730522]),
    UInt256(w: [10609077745247846400, 10934335700875511965,  9056597828666082796, 57956346104]),
    UInt256(w: [2771411416786141184, 3421927328041472315,  18113195657332165593, 115912692208]),
    UInt256(w: [5542822833572282368, 6843854656082944630,  17779647240954779570, 231825384417]),
    UInt256(w: [8487262196198277120, 8747468560700409572,  10934627077674776560, 46365076883]),
    UInt256(w: [16974524392396554240, 17494937121400819144,  3422510081640001504, 92730153767]),
    UInt256(w: [15502304711083556864, 16543130169092086673,  6845020163280003009, 185460307534]),
    UInt256(w: [6789809756958621696, 14376672478044148304,  16126399291623641894, 37092061506]),
    UInt256(w: [13579619513917243392, 10306600882378744992,  13806054509537732173, 74184123013]),
    UInt256(w: [8712494954124935168, 2166457691047938369,  9165364945365912731, 148368246027]),
    UInt256(w: [17424989908249870336, 4332915382095876738,  18330729890731825462, 296736492054]),
    UInt256(w: [18242393240617615360, 4555931891161085670,  18423541237114006385, 59347298410]),
    UInt256(w: [18038042407525679104, 9111863782322171341,  18400338400518461154, 118694596821]),
    UInt256(w: [17629340741341806592, 18223727564644342683,  18353932727327370692, 237389193643]),
    UInt256(w: [14593914592494092288, 3644745512928868536,  14738832989691205108, 47477838728]),
    UInt256(w: [10741085111278632960, 7289491025857737073,  11030921905672858600, 94955677457]),
    UInt256(w: [3035426148847714304, 14578982051715474147,  3615099737636165584, 189911354915]),
    UInt256(w: [4296434044511453184, 17673191669310736122,  723019947527233116, 37982270983]),
    UInt256(w: [8592868089022906368, 16899639264911920628,  1446039895054466233, 75964541966]),
    UInt256(w: [17185736178045812736, 15352534456114289640,  2892079790108932467, 151929083932]),
    UInt256(w: [15924728282382073856, 12258324838519027665,  5784159580217864935, 303858167864]),
    UInt256(w: [17942340915444056064, 17209060226671446825,  15914227175011214279, 60771633572]),
    UInt256(w: [17437937757178560512, 15971376379633342035,  13381710276312876943, 121543267145]),
    UInt256(w: [16429131440647569408, 13496008685557132455,  8316676478916202271, 243086534291]),
    UInt256(w: [10664523917613334528, 10077899366595247137,  5352684110525150777, 48617306858]),
    UInt256(w: [2882303761517117440, 1709054659480942659,  10705368221050301555, 97234613716]),
    UInt256(w: [5764607523034234880, 3418109318961885318,  2963992368391051494, 194469227433]),
    UInt256(w: [1152921504606846976, 8062319493276197710,  11660844917903941268, 38893845486]),
    UInt256(w: [2305843009213693952, 16124638986552395420,  4874945762098330920, 77787690973]),
    UInt256(w: [4611686018427387904, 13802533899395239224,  9749891524196661841, 155575381946]),
    UInt256(w: [9223372036854775808, 9158323725080926832,  1053038974683772067, 311150763893]),
    UInt256(w: [9223372036854775808, 1831664745016185366,  11278654239162485383, 62230152778]),
    UInt256(w: [0, 3663329490032370733, 4110564404615419150,  124460305557]),
    UInt256(w: [0, 7326658980064741466, 8221128809230838300,  248920611114]),
    UInt256(w: [0, 16222727054980589586, 16401621020813808952,  49784122222]),
    UInt256(w: [0, 13998710036251627556, 14356497967918066289,  99568244445]),
    UInt256(w: [0, 9550675998793703496, 10266251862126580963,  199136488891]),
    UInt256(w: [0, 16667530458726381992, 5742599187167226515,  39827297778]),
    UInt256(w: [0, 14888316843743212368, 11485198374334453031,  79654595556]),
    UInt256(w: [0, 11329889613776873120, 4523652674959354447,  159309191113]),
    UInt256(w: [0, 4213035153844194624, 9047305349918708895,  318618382226]),
    UInt256(w: [0, 4531955845510749248, 5498809884725652102,  63723676445]),
    UInt256(w: [0, 9063911691021498496, 10997619769451304204,  127447352890]),
    UInt256(w: [0, 18127823382042996992, 3548495465193056792,  254894705781]),
    UInt256(w: [0, 14693611120634330368, 4399047907780521681,  50978941156]),
    UInt256(w: [0, 10940478167559109120, 8798095815561043363,  101957882312]),
    UInt256(w: [0, 3434212261408666624, 17596191631122086727,  203915764624]),
    UInt256(w: [0, 4376191267023643648, 18276633585192058638,  40783152924]),
    UInt256(w: [0, 8752382534047287296, 18106523096674565660,  81566305849]),
    UInt256(w: [0, 17504765068094574592, 17766302119639579704,  163132611699]),
    UInt256(w: [0, 16562786062479597568, 17085860165569607793,  326265223399]),
    UInt256(w: [0, 10691254841979740160, 18174567292081562851,  65253044679]),
    UInt256(w: [0, 2935765610249928704, 17902390510453574087,  130506089359]),
    UInt256(w: [0, 5871531220499857408, 17358036947197596558,  261012178719]),
    UInt256(w: [0, 8553003873583792128, 18229002648407160604,  52202435743]),
    UInt256(w: [0, 17106007747167584256, 18011261223104769592,  104404871487]),
    UInt256(w: [0, 15765271420625616896, 17575778372499987569,  208809742975]),
    UInt256(w: [0, 17910449543092764672, 3515155674499997513,  41761948595]),
    UInt256(w: [0, 17374155012475977728, 7030311348999995027,  83523897190]),
    UInt256(w: [0, 16301565951242403840, 14060622697999990055,  167047794380]),
    UInt256(w: [0, 3260313190248480768, 2812124539599998011,  33409558876]),
    UInt256(w: [0, 6520626380496961536, 5624249079199996022,  66819117752]),
    UInt256(w: [0, 13041252760993923072, 11248498158399992044,  133638235504]),
    UInt256(w: [0, 7635761448278294528, 4050252243090432473,  267276471009]),
    UInt256(w: [0, 8905849919139479552, 15567445707585727787,  53455294201]),
    UInt256(w: [0, 17811699838278959104, 12688147341461903958,  106910588403]),
    UInt256(w: [0, 17176655602848366592, 6929550609214256301,  213821176807]),
    UInt256(w: [0, 14503377564795404288, 8764607751326671906,  42764235361]),
    UInt256(w: [0, 10560011055881256960, 17529215502653343813,  85528470722]),
    UInt256(w: [0, 2673278038052962304, 16611686931597136011,  171056941445]),
    UInt256(w: [0, 4224004422352502784, 3322337386319427202,  34211388289]),
    UInt256(w: [0, 8448008844705005568, 6644674772638854404,  68422776578]),
    UInt256(w: [0, 16896017689410011136, 13289349545277708808,  136845553156]),
    UInt256(w: [0, 15345291305110470656, 8131955016845866001,  273691106313]),
    UInt256(w: [0, 17826453519989735424, 12694437447594904169,  54738221262]),
    UInt256(w: [0, 17206162966269919232, 6942130821480256723,  109476442525]),
    UInt256(w: [0, 15965581858830286848, 13884261642960513447,  218952885050]),
    UInt256(w: [0, 10571814001249878016, 2776852328592102689,  43790577010]),
    UInt256(w: [0, 2696883928790204416, 5553704657184205379,  87581154020]),
    UInt256(w: [0, 5393767857580408832, 11107409314368410758,  175162308040]),
    UInt256(w: [0, 12146800015741812736, 2221481862873682151,  35032461608]),
    UInt256(w: [0, 5846855957774073856, 4442963725747364303,  70064923216]),
    UInt256(w: [0, 11693711915548147712, 8885927451494728606,  140129846432]),
    UInt256(w: [0, 4940679757386743808, 17771854902989457213,  280259692864]),
    UInt256(w: [0, 8366833580961169408, 18311766239565532735,  56051938572]),
    UInt256(w: [0, 16733667161922338816, 18176788405421513854,  112103877145]),
    UInt256(w: [0, 15020590250135126016, 17906832737133476093,  224207754291]),
    UInt256(w: [0, 17761513308994666496, 7270715362168605541,  44841550858]),
    UInt256(w: [0, 17076282544279781376, 14541430724337211083,  89683101716]),
    UInt256(w: [0, 15705821014850011136, 10636117374964870551,  179366203433]),
    UInt256(w: [0, 17898559461937643520, 13195269919218705079,  35873240686]),
    UInt256(w: [0, 17350374850165735424, 7943795764727858543,  71746481373]),
    UInt256(w: [0, 16254005626621919232, 15887591529455717087,  143492962746]),
    UInt256(w: [0, 14061267179534286848, 13328438985201882559,  286985925493]),
    UInt256(w: [0, 10190951065390678016, 13733734241266107481,  57397185098]),
    UInt256(w: [0, 1935158057071804416, 9020724408822663347,  114794370197]),
    UInt256(w: [0, 3870316114143608832, 18041448817645326694,  229588740394]),
    UInt256(w: [0, 11842109667054452736, 18365685022496706631,  45917748078]),
    UInt256(w: [0, 5237475260399353856, 18284625971283861647,  91835496157]),
    UInt256(w: [0, 10474950520798707712, 18122507868858171678,  183670992315]),
    UInt256(w: [0, 13163036548385472512, 3624501573771634335,  36734198463]),
    UInt256(w: [0, 7879329023061393408, 7249003147543268671,  73468396926]),
    UInt256(w: [0, 15758658046122786816, 14498006295086537342,  146936793852]),
    UInt256(w: [0, 13070572018536022016, 10549268516463523069,  293873587705]),
    UInt256(w: [0, 17371509662674845696, 2109853703292704613,  58774717541]),
    UInt256(w: [0, 16296275251640139776, 4219707406585409227,  117549435082]),
    UInt256(w: [0, 14145806429570727936, 8439414813170818455,  235098870164]),
    UInt256(w: [0, 17586556544881786880, 16445278221601804983,  47019774032]),
    UInt256(w: [0, 16726369016054022144, 14443812369494058351,  94039548065]),
    UInt256(w: [0, 15005993958398492672, 10440880665278565087,  188079096131]),
    UInt256(w: [0, 14069245235905429504, 5777524947797623340,  37615819226]),
    UInt256(w: [0, 9691746398101307392, 11555049895595246681,  75231638452]),
    UInt256(w: [0, 936748722493063168, 4663355717480941747,  150463276905]),
    UInt256(w: [0, 1873497444986126336, 9326711434961883494,  300926553810]),
    UInt256(w: [0, 15132094747964866560, 1865342286992376698,  60185310762]),
    UInt256(w: [0, 11817445422220181504, 3730684573984753397,  120370621524]),
    UInt256(w: [0, 5188146770730811392, 7461369147969506795,  240741243048]),
    UInt256(w: [0, 12105675798371893248, 12560320273819632328,  48148248609]),
    UInt256(w: [0, 5764607523034234880, 6673896473929713041,  96296497219]),
    UInt256(w: [0, 11529215046068469760, 13347792947859426082,  192592994438]),
    UInt256(w: [0, 2305843009213693952, 13737605033797616186,  38518598887]),
    UInt256(w: [0, 4611686018427387904, 9028465993885680756,  77037197775]),
    UInt256(w: [0, 9223372036854775808, 18056931987771361512,  154074395550]),
    UInt256(w: [0, 0, 17667119901833171409, 308148791101]),
    UInt256(w: [0, 0, 7222772795108544605, 61629758220]),
    UInt256(w: [0, 0, 14445545590217089210, 123259516440]),
    UInt256(w: [0, 0, 10444347106724626804, 246519032881]),
    UInt256(w: [0, 0, 5778218236086835684, 49303806576]),
    UInt256(w: [0, 0, 11556436472173671368, 98607613152]),
    UInt256(w: [0, 0, 4666128870637791120, 197215226305]),
    UInt256(w: [0, 0, 933225774127558224, 39443045261]),
    UInt256(w: [0, 0, 1866451548255116448, 78886090522]),
    UInt256(w: [0, 0, 3732903096510232896, 157772181044]),
    UInt256(w: [0, 0, 7465806193020465792, 315544362088]),
    UInt256(w: [0, 0, 12561207682829824128, 63108872417]),
    UInt256(w: [0, 0, 6675671291950096640, 126217744835]),
    UInt256(w: [0, 0, 13351342583900193280, 252435489670]),
    UInt256(w: [0, 0, 2670268516780038656, 50487097934]),
    UInt256(w: [0, 0, 5340537033560077312, 100974195868]),
    UInt256(w: [0, 0, 10681074067120154624, 201948391736]),
    UInt256(w: [0, 0, 5825563628165941248, 40389678347]),
    UInt256(w: [0, 0, 11651127256331882496, 80779356694]),
    UInt256(w: [0, 0, 4855510438954213376, 161558713389]),
    UInt256(w: [0, 0, 9711020877908426752, 323117426778]),
    UInt256(w: [0, 0, 13010250619807416320, 64623485355]),
    UInt256(w: [0, 0, 7573757165905281024, 129246970711]),
    UInt256(w: [0, 0, 15147514331810562048, 258493941422]),
    UInt256(w: [0, 0, 10408200495845933056, 51698788284]),
    UInt256(w: [0, 0, 2369656917982314496, 103397576569]),
    UInt256(w: [0, 0, 4739313835964628992, 206795153138]),
    UInt256(w: [0, 0, 12015909211418656768, 41359030627]),
    UInt256(w: [0, 0, 5585074349127761920, 82718061255]),
    UInt256(w: [0, 0, 11170148698255523840, 165436122510]),
    UInt256(w: [0, 0, 2234029739651104768, 33087224502]),
    UInt256(w: [0, 0, 4468059479302209536, 66174449004]),
    UInt256(w: [0, 0, 8936118958604419072, 132348898008]),
    UInt256(w: [0, 0, 17872237917208838144, 264697796016]),
    UInt256(w: [0, 0, 7263796398183677952, 52939559203]),
    UInt256(w: [0, 0, 14527592796367355904, 105879118406]),
    UInt256(w: [0, 0, 10608441519025160192, 211758236813]),
    UInt256(w: [0, 0, 13189734748030763008, 42351647362]),
    UInt256(w: [0, 0, 7932725422351974400, 84703294725]),
    UInt256(w: [0, 0, 15865450844703948800, 169406589450]),
    UInt256(w: [0, 0, 3173090168940789760, 33881317890]),
    UInt256(w: [0, 0, 6346180337881579520, 67762635780]),
    UInt256(w: [0, 0, 12692360675763159040, 135525271560]),
    UInt256(w: [0, 0, 6937977277816766464, 271050543121]),
    UInt256(w: [0, 0, 5076944270305263616, 54210108624]),
    UInt256(w: [0, 0, 10153888540610527232, 108420217248]),
    UInt256(w: [0, 0, 1861033007511502848, 216840434497]),
    UInt256(w: [0, 0, 7750904230986121216, 43368086899]),
    UInt256(w: [0, 0, 15501808461972242432, 86736173798]),
    UInt256(w: [0, 0, 12556872850234933248, 173472347597]),
    UInt256(w: [0, 0, 9890072199530807296, 34694469519]),
    UInt256(w: [0, 0, 1333400325352062976, 69388939039]),
    UInt256(w: [0, 0, 2666800650704125952, 138777878078]),
    UInt256(w: [0, 0, 5333601301408251904, 277555756156]),
    UInt256(w: [0, 0, 4756069075023560704, 55511151231]),
    UInt256(w: [0, 0, 9512138150047121408, 111022302462]),
    UInt256(w: [0, 0, 577532226384691200, 222044604925]),
    UInt256(w: [0, 0, 115506445276938240, 44408920985]),
    UInt256(w: [0, 0, 231012890553876480, 88817841970]),
    UInt256(w: [0, 0, 462025781107752960, 177635683940]),
    UInt256(w: [0, 0, 92405156221550592, 35527136788]),
    UInt256(w: [0, 0, 184810312443101184, 71054273576]),
    UInt256(w: [0, 0, 369620624886202368, 142108547152]),
    UInt256(w: [0, 0, 739241249772404736, 284217094304]),
    UInt256(w: [0, 0, 14905243508922122240, 56843418860]),
    UInt256(w: [0, 0, 11363742944134692864, 113686837721]),
    UInt256(w: [0, 0, 4280741814559834112, 227373675443]),
    UInt256(w: [0, 0, 11924194807137697792, 45474735088]),
    UInt256(w: [0, 0, 5401645540565843968, 90949470177]),
    UInt256(w: [0, 0, 10803291081131687936, 181898940354]),
    UInt256(w: [0, 0, 16918053475193978880, 36379788070]),
    UInt256(w: [0, 0, 15389362876678406144, 72759576141]),
    UInt256(w: [0, 0, 12331981679647260672, 145519152283]),
    UInt256(w: [0, 0, 6217219285584969728, 291038304567]),
    UInt256(w: [0, 0, 8622141486600814592, 58207660913]),
    UInt256(w: [0, 0, 17244282973201629184, 116415321826]),
    UInt256(w: [0, 0, 16041821872693706752, 232830643653]),
    UInt256(w: [0, 0, 14276410818764472320, 46566128730]),
    UInt256(w: [0, 0, 10106077563819393024, 93132257461]),
    UInt256(w: [0, 0, 1765411053929234432, 186264514923]),
    UInt256(w: [0, 0, 11421128655011577856, 37252902984]),
    UInt256(w: [0, 0, 4395513236313604096, 74505805969]),
    UInt256(w: [0, 0, 8791026472627208192, 149011611938]),
    UInt256(w: [0, 0, 17582052945254416384, 298023223876]),
    UInt256(w: [0, 0, 7205759403792793600, 59604644775]),
    UInt256(w: [0, 0, 14411518807585587200, 119209289550]),
    UInt256(w: [0, 0, 10376293541461622784, 238418579101]),
    UInt256(w: [0, 0, 5764607523034234880, 47683715820]),
    UInt256(w: [0, 0, 11529215046068469760, 95367431640]),
    UInt256(w: [0, 0, 4611686018427387904, 190734863281]),
    UInt256(w: [0, 0, 4611686018427387904, 38146972656]),
    UInt256(w: [0, 0, 9223372036854775808, 76293945312]),
    UInt256(w: [0, 0, 0, 152587890625]),
    UInt256(w: [0, 0, 0, 305175781250]),
    UInt256(w: [0, 0, 0, 61035156250]),
    UInt256(w: [0, 0, 0, 122070312500]),
    UInt256(w: [0, 0, 0, 244140625000]),
    UInt256(w: [0, 0, 0, 48828125000]),
    UInt256(w: [0, 0, 0, 97656250000]),
    UInt256(w: [0, 0, 0, 195312500000]),
    UInt256(w: [0, 0, 0, 39062500000]),
    UInt256(w: [0, 0, 0, 78125000000]),
    UInt256(w: [0, 0, 0, 156250000000]),
    UInt256(w: [0, 0, 0, 312500000000]),
    UInt256(w: [0, 0, 0, 62500000000]),
    UInt256(w: [0, 0, 0, 125000000000]),
    UInt256(w: [0, 0, 0, 250000000000]),
    UInt256(w: [0, 0, 0, 50000000000]),
    UInt256(w: [0, 0, 0, 100000000000]),
    UInt256(w: [0, 0, 0, 200000000000]),
    UInt256(w: [0, 0, 0, 40000000000]),
    UInt256(w: [0, 0, 0, 80000000000]),
    UInt256(w: [0, 0, 0, 160000000000]),
    UInt256(w: [0, 0, 0, 320000000000]),
    UInt256(w: [0, 0, 0, 64000000000]),
    UInt256(w: [0, 0, 0, 128000000000]),
    UInt256(w: [0, 0, 0, 256000000000]),
    UInt256(w: [0, 0, 0, 51200000000]),
    UInt256(w: [0, 0, 0, 102400000000]),
    UInt256(w: [0, 0, 0, 204800000000]),
    UInt256(w: [0, 0, 0, 40960000000]),
    UInt256(w: [0, 0, 0, 81920000000]),
    UInt256(w: [0, 0, 0, 163840000000]),
    UInt256(w: [0, 0, 0, 327680000000]),
    UInt256(w: [0, 0, 0, 65536000000]),
    UInt256(w: [0, 0, 0, 131072000000]),
    UInt256(w: [0, 0, 0, 262144000000]),
    UInt256(w: [0, 0, 0, 52428800000]),
    UInt256(w: [0, 0, 0, 104857600000]),
    UInt256(w: [0, 0, 0, 209715200000]),
    UInt256(w: [0, 0, 0, 41943040000]),
    UInt256(w: [0, 0, 0, 83886080000]),
    UInt256(w: [0, 0, 0, 167772160000]),
    UInt256(w: [0, 0, 0, 33554432000]),
    UInt256(w: [0, 0, 0, 67108864000]),
    UInt256(w: [0, 0, 0, 134217728000]),
    UInt256(w: [0, 0, 0, 268435456000]),
    UInt256(w: [0, 0, 0, 53687091200]),
    UInt256(w: [0, 0, 0, 107374182400]),
    UInt256(w: [0, 0, 0, 214748364800]),
    UInt256(w: [0, 0, 0, 42949672960]),
    UInt256(w: [0, 0, 0, 85899345920]),
    UInt256(w: [0, 0, 0, 171798691840]),
    UInt256(w: [0, 0, 0, 34359738368]),
    UInt256(w: [0, 0, 0, 68719476736]),
    UInt256(w: [0, 0, 0, 137438953472]),
    UInt256(w: [0, 0, 0, 274877906944]),
    UInt256(w: [14757395258967641293, 14757395258967641292,  14757395258967641292, 54975581388]),
    UInt256(w: [11068046444225730970, 11068046444225730969,  11068046444225730969, 109951162777]),
    UInt256(w: [3689348814741910324, 3689348814741910323,  3689348814741910323, 219902325555]),
    UInt256(w: [4427218577690292388, 11805916207174113034,  737869762948382064, 43980465111]),
    UInt256(w: [8854437155380584776, 5165088340638674452,  1475739525896764129, 87960930222]),
    UInt256(w: [17708874310761169552, 10330176681277348904,  2951479051793528258, 175921860444]),
    UInt256(w: [7231123676894144234, 9444732965739290427,  15347691069326346944, 35184372088]),
    UInt256(w: [14462247353788288467, 442721857769029238,  12248638064943142273, 70368744177]),
    UInt256(w: [10477750633867025318, 885443715538058477,  6050532056176732930, 140737488355]),
    UInt256(w: [2508757194024499020, 1770887431076116955,  12101064112353465860, 281474976710]),
    UInt256(w: [501751438804899804, 354177486215223391, 2420212822470693172,  56294995342]),
    UInt256(w: [1003502877609799608, 708354972430446782, 4840425644941386344,  112589990684]),
    UInt256(w: [2007005755219599216, 1416709944860893564,  9680851289882772688, 225179981368]),
    UInt256(w: [401401151043919844, 3972690803714089036,  13004216702202285507, 45035996273]),
    UInt256(w: [802802302087839687, 7945381607428178072, 7561689330695019398,  90071992547]),
    UInt256(w: [1605604604175679373, 15890763214856356144,  15123378661390038796, 180143985094]),
    UInt256(w: [15078516179802777168, 3178152642971271228,  17782070991245649052, 36028797018]),
    UInt256(w: [11710288285896002719, 6356305285942542457,  17117397908781746488, 72057594037]),
    UInt256(w: [4973832498082453822, 12712610571885084915,  15788051743853941360, 144115188075]),
    UInt256(w: [9947664996164907643, 6978477070060618214,  13129359413998331105, 288230376151]),
    UInt256(w: [1989532999232981529, 5085044228754033966,  6315220697541576544, 57646075230]),
    UInt256(w: [3979065998465963058, 10170088457508067932,  12630441395083153088, 115292150460]),
    UInt256(w: [7958131996931926115, 1893432841306584248,  6814138716456754561, 230584300921]),
    UInt256(w: [1591626399386385223, 7757384197745137496,  5052176558033261235, 46116860184]),
    UInt256(w: [3183252798772770446, 15514768395490274992,  10104353116066522470, 92233720368]),
    UInt256(w: [6366505597545540892, 12582792717270998368,  1761962158423493325, 184467440737]),
    UInt256(w: [1273301119509108179, 9895256172938020320,  7731090061168519311, 36893488147]),
    UInt256(w: [2546602239018216357, 1343768272166489024,  15462180122337038623, 73786976294]),
    UInt256(w: [5093204478036432714, 2687536544332978048,  12477616170964525630, 147573952589]),
    UInt256(w: [10186408956072865427, 5375073088665956096,  6508488268219499644, 295147905179]),
    UInt256(w: [16794677050182214379, 12143061061958922188,  16059092912611541221, 59029581035]),
    UInt256(w: [15142610026654877141, 5839378050208292761,  13671441751513530827, 118059162071]),
    UInt256(w: [11838475979600202665, 11678756100416585523,  8896139429317510038, 236118324143]),
    UInt256(w: [17125090454887681826, 6025100034825227427,  12847274330089232977, 47223664828]),
    UInt256(w: [15803436836065812036, 12050200069650454855,  7247804586468914338, 94447329657]),
    UInt256(w: [13160129598422072455, 5653656065591358095,  14495609172937828677, 188894659314]),
    UInt256(w: [6321374734426324815, 4820080027860181942,  17656517093555207028, 37778931862]),
    UInt256(w: [12642749468852649629, 9640160055720363884,  16866290113400862440, 75557863725]),
    UInt256(w: [6838754863995747641, 833576037731176153,  15285836153092173265, 151115727451]),
    UInt256(w: [13677509727991495281, 1667152075462352306,  12124928232474794914, 302231454903]),
    UInt256(w: [13803548389824030026, 7712128044576291107,  13493032090720689952, 60446290980]),
    UInt256(w: [9160352705938508436, 15424256089152582215,  8539320107731828288, 120892581961]),
    UInt256(w: [18320705411877016871, 12401768104595612814,  17078640215463656577, 241785163922]),
    UInt256(w: [14732187526601134344, 17237748879886763855,  10794425672576551961, 48357032784]),
    UInt256(w: [11017630979492717072, 16028753686063976095,  3142107271443552307, 96714065569]),
    UInt256(w: [3588517885275882528, 13610763298418400575,  6284214542887104615, 193428131138]),
    UInt256(w: [11785750021280907476, 13790199103909411084,  12324889352803151892, 38685626227]),
    UInt256(w: [5124755968852263335, 9133654134109270553,  6203034631896752169, 77371252455]),
    UInt256(w: [10249511937704526669, 18267308268218541106,  12406069263793504338, 154742504910]),
    UInt256(w: [2052279801699501721, 18087872462727530597,  6365394453877457061, 309485009821]),
    UInt256(w: [15167851219307541637, 10996272122029326765,  4962427705517401735, 61897001964]),
    UInt256(w: [11888958364905531658, 3545800170349101915,  9924855411034803471, 123794003928]),
    UInt256(w: [5331172656101511700, 7091600340698203831,  1402966748360055326, 247588007857]),
    UInt256(w: [15823629790187943633, 12486366512365371735,  7659290979155831711, 49517601571]),
    UInt256(w: [13200515506666335650, 6525988951021191855,  15318581958311663423, 99035203142]),
    UInt256(w: [7954286939623119683, 13051977902042383711,  12190419842913775230, 198070406285]),
    UInt256(w: [5280206202666534260, 2610395580408476742,  2438083968582755046, 39614081257]),
    UInt256(w: [10560412405333068520, 5220791160816953484,  4876167937165510092, 79228162514]),
    UInt256(w: [2674080736956585423, 10441582321633906969,  9752335874331020184, 158456325028]),
    UInt256(w: [5348161473913170846, 2436420569558262322,  1057927674952488753, 316912650057]),
    UInt256(w: [8448329924266454816, 487284113911652464, 7590283164474318397,  63382530011]),
    UInt256(w: [16896659848532909632, 974568227823304928,  15180566328948636794, 126765060022]),
    UInt256(w: [15346575623356267647, 1949136455646609857,  11914388584187721972, 253530120045]),
    UInt256(w: [17826710383638894823, 7768524920613142617,  2382877716837544394, 50706024009]),
    UInt256(w: [17206676693568238029, 15537049841226285235,  4765755433675088788, 101412048018]),
    UInt256(w: [15966609313426924441, 12627355608743018855,  9531510867350177577, 202824096036]),
    UInt256(w: [14261368306911115858, 13593517565974334740,  5595650988211945838, 40564819207]),
    UInt256(w: [10075992540112680100, 8740291058239117865,  11191301976423891677, 81129638414]),
    UInt256(w: [1705241006515808583, 17480582116478235731,  3935859879138231738, 162259276829]),
    UInt256(w: [3410482013031617166, 16514420159246919846,  7871719758276463477, 324518553658]),
    UInt256(w: [4371445217348233757, 3302884031849383969,  12642390395881023665, 64903710731]),
    UInt256(w: [8742890434696467513, 6605768063698767938,  6838036718052495714, 129807421463]),
    UInt256(w: [17485780869392935026, 13211536127397535876,  13676073436104991428, 259614842926]),
    UInt256(w: [3497156173878587006, 17399702484447148468,  6424563501962908608, 51922968585]),
    UInt256(w: [6994312347757174011, 16352660895184745320,  12849127003925817217, 103845937170]),
    UInt256(w: [13988624695514348021, 14258577716659939024,  7251509934142082819, 207691874341]),
    UInt256(w: [17555120198070510897, 2851715543331987804,  5139650801570326887, 41538374868]),
    UInt256(w: [16663496322431470178, 5703431086663975609,  10279301603140653774, 83076749736]),
    UInt256(w: [14880248571153388740, 11406862173327951219,  2111859132571755932, 166153499473]),
    UInt256(w: [17733444973198319041, 2281372434665590243,  11490418270740082156, 33230699894]),
    UInt256(w: [17020145872687086466, 4562744869331180487,  4534092467770612696, 66461399789]),
    UInt256(w: [15593547671664621315, 9125489738662360975,  9068184935541225392, 132922799578]),
    UInt256(w: [12740351269619691014, 18250979477324721951,  18136369871082450784, 265845599156]),
    UInt256(w: [6237419068665848526, 3650195895464944390,  7316622788958400480, 53169119831]),
    UInt256(w: [12474838137331697052, 7300391790929888780,  14633245577916800960, 106338239662]),
    UInt256(w: [6502932200953842488, 14600783581859777561,  10819747082124050304, 212676479325]),
    UInt256(w: [1300586440190768498, 17677551975339596805,  2163949416424810060, 42535295865]),
    UInt256(w: [2601172880381536995, 16908359876969641994,  4327898832849620121, 85070591730]),
    UInt256(w: [5202345760763073990, 15369975680229732372,  8655797665699240243, 170141183460]),
    UInt256(w: [1040469152152614798, 14142041580271677444,  1731159533139848048, 34028236692]),
    UInt256(w: [2080938304305229596, 9837339086833803272,  3462319066279696097, 68056473384]),
    UInt256(w: [4161876608610459192, 1227934099958054928,  6924638132559392195, 136112946768]),
    UInt256(w: [8323753217220918384, 2455868199916109856,  13849276265118784390, 272225893536]),
    UInt256(w: [9043448272928004324, 4180522454725132294,  6459204067765667201, 54445178707]),
    UInt256(w: [18086896545856008647, 8361044909450264588,  12918408135531334402, 108890357414]),
    UInt256(w: [17727049018002465677, 16722089818900529177,  7390072197353117188, 217780714829]),
    UInt256(w: [18302805062568134429, 10723115593263926481,  16235409698438264730, 43556142965]),
    UInt256(w: [18158866051426717241, 2999487112818301347,  14024075323166977845, 87112285931]),
    UInt256(w: [17870988029143882865, 5998974225636602695,  9601406572624404074, 174224571863]),
    UInt256(w: [10952895235312597220, 8578492474611141185,  12988327758750611784, 34844914372]),
    UInt256(w: [3459046396915642823, 17156984949222282371,  7529911443791671952, 69689828745]),
    UInt256(w: [6918092793831285646, 15867225824735013126,  15059822887583343905, 139379657490]),
    UInt256(w: [13836185587662571291, 13287707575760474636,  11672901701457136195, 278759314981]),
    UInt256(w: [10145934747016334905, 6346890329894005250,  6023929155033337562, 55751862996]),
    UInt256(w: [1845125420323118193, 12693780659788010501,  12047858310066675124, 111503725992]),
    UInt256(w: [3690250840646236386, 6940817245866469386,  5648972546423798633, 223007451985]),
    UInt256(w: [15495445427096888570, 12456209893399024846,  1129794509284759726, 44601490397]),
    UInt256(w: [12544146780484225524, 6465675713088498077,  2259589018569519453, 89202980794]),
    UInt256(w: [6641549487258899432, 12931351426176996155,  4519178037139038906, 178405961588]),
    UInt256(w: [16085705156419421180, 17343665544203040523,  11971882051653538750, 35681192317]),
    UInt256(w: [13724666239129290743, 16240587014696529431,  5497020029597525885, 71362384635]),
    UInt256(w: [9002588404549029869, 14034429955683507247,  10994040059195051771, 142724769270]),
    UInt256(w: [18005176809098059738, 9622115837657462878,  3541336044680551927, 285449538541]),
    UInt256(w: [7290384176561522271, 12992469611757223545,  4397616023678020708, 57089907708]),
    UInt256(w: [14580768353123044542, 7538195149804895474,  8795232047356041417, 114179815416]),
    UInt256(w: [10714792632536537467, 15076390299609790949,  17590464094712082834, 228359630832]),
    UInt256(w: [2142958526507307494, 6704626874663868513,  10896790448426237213, 45671926166]),
    UInt256(w: [4285917053014614987, 13409253749327737026,  3346836823142922810, 91343852333]),
    UInt256(w: [8571834106029229974, 8371763424945922436,  6693673646285845621, 182687704666]),
    UInt256(w: [12782413265431576965, 9053050314473005133,  5028083543999079447, 36537540933]),
    UInt256(w: [7118082457153602313, 18106100628946010267,  10056167087998158894, 73075081866]),
    UInt256(w: [14236164914307204626, 17765457184182468918,  1665590102286766173, 146150163733]),
    UInt256(w: [10025585754904857635, 17084170294655386221,  3331180204573532347, 292300327466]),
    UInt256(w: [16762512409948612820, 14484880503156808213,  4355584855656616792, 58460065493]),
    UInt256(w: [15078280746187674024, 10523016932604064811,  8711169711313233585, 116920130986]),
    UInt256(w: [11709817418665796431, 2599289791498578007,  17422339422626467171, 233840261972]),
    UInt256(w: [2341963483733159287, 11587904402525446571,  10863165514009114080, 46768052394]),
    UInt256(w: [4683926967466318573, 4729064731341341526,  3279586954308676545, 93536104789]),
    UInt256(w: [9367853934932637145, 9458129462682683052,  6559173908617353090, 187072209578]),
    UInt256(w: [1873570786986527429, 12959672336762267580,  12379881225949201587, 37414441915]),
    UInt256(w: [3747141573973054858, 7472600599814983544,  6313018378188851559, 74828883831]),
    UInt256(w: [7494283147946109716, 14945201199629967088,  12626036756377703118, 149657767662]),
    UInt256(w: [14988566295892219432, 11443658325550382560,  6805329439045854621, 299315535325]),
    UInt256(w: [6687062073920354210, 5978080479851986835,  1361065887809170924, 59863107065]),
    UInt256(w: [13374124147840708419, 11956160959703973670,  2722131775618341848, 119726214130]),
    UInt256(w: [8301504221971865222, 5465577845698395725,  5444263551236683697, 239452428260]),
    UInt256(w: [9038998473878193691, 8471813198623499791,  1088852710247336739, 47890485652]),
    UInt256(w: [18077996947756387382, 16943626397246999582,  2177705420494673478, 95780971304]),
    UInt256(w: [17709249821803223148, 15440508720784447549,  4355410840989346957, 191561942608]),
    UInt256(w: [18299245223328285923, 3088101744156889509,  11939128612423600361, 38312388521]),
    UInt256(w: [18151746372947020229, 6176203488313779019,  5431513151137649106, 76624777043]),
    UInt256(w: [17856748672184488841, 12352406976627558039,  10863026302275298212, 153249554086]),
    UInt256(w: [17266753270659426066, 6258069879545564463,  3279308530841044809, 306499108173]),
    UInt256(w: [3453350654131885214, 8630311605392933539,  11723908150393939931, 61299821634]),
    UInt256(w: [6906701308263770427, 17260623210785867078,  5001072227078328246, 122599643269]),
    UInt256(w: [13813402616527540853, 16074502347862182540,  10002144454156656493, 245199286538]),
    UInt256(w: [6452029338047418494, 6904249284314346831,  13068475335057062268, 49039857307]),
    UInt256(w: [12904058676094836988, 13808498568628693662,  7690206596404572920, 98079714615]),
    UInt256(w: [7361373278480122359, 9170253063547835709,  15380413192809145841, 196159429230]),
    UInt256(w: [1472274655696024472, 5523399427451477465,  3076082638561829168, 39231885846]),
    UInt256(w: [2944549311392048944, 11046798854902954930,  6152165277123658336, 78463771692]),
    UInt256(w: [5889098622784097888, 3646853636096358244,  12304330554247316673, 156927543384]),
    UInt256(w: [11778197245568195775, 7293707272192716488,  6161917034785081730, 313855086769]),
    UInt256(w: [9734337078597459802, 16216136713406184590,  15989778665924657638, 62771017353]),
    UInt256(w: [1021930083485367987, 13985529353102817565,  13532813258139763661, 125542034707]),
    UInt256(w: [2043860166970735973, 9524314632496083514,  8618882442569975707, 251084069415]),
    UInt256(w: [4098120848136057518, 9283560555983037349,  1723776488513995141, 50216813883]),
    UInt256(w: [8196241696272115036, 120377038256523082, 3447552977027990283,  100433627766]),
    UInt256(w: [16392483392544230072, 240754076513046164,  6895105954055980566, 200867255532]),
    UInt256(w: [10657194307992666661, 11116197259528340202,  8757718820295016759, 40173451106]),
    UInt256(w: [2867644542275781706, 3785650445347128789,  17515437640590033519, 80346902212]),
    UInt256(w: [5735289084551563411, 7571300890694257578,  16584131207470515422, 160693804425]),
    UInt256(w: [11470578169103126821, 15142601781388515156,  14721518341231479228, 321387608851]),
    UInt256(w: [2294115633820625365, 17785915615245344324,  6633652482988206168, 64277521770]),
    UInt256(w: [4588231267641250729, 17125087156781137032,  13267304965976412337, 128555043540]),
    UInt256(w: [9176462535282501457, 15803430239852722448,  8087865858243273059, 257110087081]),
    UInt256(w: [12903338951282231261, 3160686047970544489,  5306921986390564935, 51422017416]),
    UInt256(w: [7359933828854910906, 6321372095941088979,  10613843972781129870, 102844034832]),
    UInt256(w: [14719867657709821812, 12642744191882177958,  2780943871852708124, 205688069665]),
    UInt256(w: [10322671161025785009, 17285944097344076884,  556188774370541624, 41137613933]),
    UInt256(w: [2198598248342018402, 16125144120978602153,  1112377548741083249, 82275227866]),
    UInt256(w: [4397196496684036804, 13803544168247652690,  2224755097482166499, 164550455732]),
    UInt256(w: [4568788114078717684, 6450057648391440861,  7823648648980253946, 32910091146]),
    UInt256(w: [9137576228157435368, 12900115296782881722,  15647297297960507892, 65820182292]),
    UInt256(w: [18275152456314870736, 7353486519856211828,  12847850522211464169, 131640364585]),
    UInt256(w: [18103560838920189855, 14706973039712423657,  7248956970713376722, 263280729171]),
    UInt256(w: [3620712167784037971, 14009441052168215701,  5139140208884585667, 52656145834]),
    UInt256(w: [7241424335568075942, 9572138030626879786,  10278280417769171335, 105312291668]),
    UInt256(w: [14482848671136151884, 697531987544207956,  2109816761828791055, 210624583337]),
    UInt256(w: [13964616178452961347, 7518204026992662237,  7800660981849578857, 42124916667]),
    UInt256(w: [9482488283196371077, 15036408053985324475,  15601321963699157714, 84249833334]),
    UInt256(w: [518232492683190538, 11626072034261097335,  12755899853688763813, 168499666669]),
    UInt256(w: [7482344128020458754, 9703912036336040113,  17308575229705394055, 33699933333]),
    UInt256(w: [14964688256040917508, 961079998962528610,  16170406385701236495, 67399866667]),
    UInt256(w: [11482632438372283400, 1922159997925057221,  13894068697692921374, 134799733335]),
    UInt256(w: [4518520803035015183, 3844319995850114443,  9341393321676291132, 269599466671]),
    UInt256(w: [4593052975348913360, 11836910443395753858,  5557627479077168549, 53919893334]),
    UInt256(w: [9186105950697826720, 5227076813081956100,  11115254958154337099, 107839786668]),
    UInt256(w: [18372211901395653440, 10454153626163912200,  3783765842599122582, 215679573337]),
    UInt256(w: [18431837639246771981, 16848225984200423732,  8135450798003645162, 43135914667]),
    UInt256(w: [18416931204783992346, 15249707894691295849,  16270901596007290325, 86271829334]),
    UInt256(w: [18387118335858433075, 12052671715673040083,  14095059118305029035, 172543658669]),
    UInt256(w: [11056121296655507262, 17167929602102249309,  17576407082628647099, 34508731733]),
    UInt256(w: [3665498519601462907, 15889115130494947003,  16706070091547742583, 69017463467]),
    UInt256(w: [7330997039202925814, 13331486187280342390,  14965396109385933551, 138034926935]),
    UInt256(w: [14661994078405851627, 8216228300851133164,  11484048145062315487, 276069853871]),
    UInt256(w: [10311096445164990972, 12711292104395957602,  5986158443754373420, 55213970774]),
    UInt256(w: [2175448816620430328, 6975840135082363589,  11972316887508746841, 110427941548]),
    UInt256(w: [4350897633240860655, 13951680270164727178,  5497889701307942066, 220855883097]),
    UInt256(w: [4559528341390082455, 13858382498258676405,  8478275569745409059, 44171176619]),
    UInt256(w: [9119056682780164909, 9270020922807801194,  16956551139490818119, 88342353238]),
    UInt256(w: [18238113365560329817, 93297771906050772,  15466358205272084623, 176684706477]),
    UInt256(w: [11026320302595886610, 18659554381210154,  10471969270538237571, 35336941295]),
    UInt256(w: [3605896531482221604, 37319108762420309, 2497194467366923526,  70673882591]),
    UInt256(w: [7211793062964443207, 74638217524840618, 4994388934733847052,  141347765182]),
    UInt256(w: [14423586125928886414, 149276435049681236,  9988777869467694104, 282695530364]),
    UInt256(w: [17642112484153418576, 11097901731235667216,  16755150832861180113, 56539106072]),
    UInt256(w: [16837480894597285536, 3749059388761782817,  15063557592012808611, 113078212145]),
    UInt256(w: [15228217715485019455, 7498118777523565635,  11680371110316065606, 226156424291]),
    UInt256(w: [10424341172580824538, 8878321384988533773,  6025423036805123444, 45231284858]),
    UInt256(w: [2401938271452097459, 17756642769977067547,  12050846073610246888, 90462569716]),
    UInt256(w: [4803876542904194917, 17066541466244583478,  5654948073510942161, 180925139433]),
    UInt256(w: [8339472938064659630, 18170703552216557988,  12199036058927919401, 36185027886]),
    UInt256(w: [16678945876129319260, 17894663030723564360,  5951328044146287187, 72370055773]),
    UInt256(w: [14911147678549086904, 17342581987737577105,  11902656088292574375, 144740111546]),
    UInt256(w: [11375551283388622191, 16238419901765602595,  5358568102875597135, 289480223093]),
    UInt256(w: [13343156700903455408, 14315730424578851488,  12139760064800850396, 57896044618]),
    UInt256(w: [8239569328097359200, 10184716775448151361,  5832776055892149177, 115792089237]),
    UInt256(w: [16479138656194718399, 1922689477186751106,  11665552111784298355, 231584178474]),
    UInt256(w: [3295827731238943680, 15141933154404991514,  17090505681324500963, 46316835694]),
    UInt256(w: [6591655462477887360, 11837122235100431412,  15734267288939450311, 92633671389]),
    UInt256(w: [13183310924955774719, 5227500396491311208,  13021790504169349007, 185267342779]),
    UInt256(w: [17394057443958796237, 4734848894040172564,  17361753359801511094, 37053468555]),
    UInt256(w: [16341370814208040858, 9469697788080345129,  16276762645893470572, 74106937111]),
    UInt256(w: [14235997554706530099, 492651502451138643,  14106781218077389529, 148213874223]),
    UInt256(w: [10025251035703508581, 985303004902277287,  9766818362445227442, 296427748447]),
    UInt256(w: [5694399021882612040, 14954455859948096750,  9332061301972866134, 59285549689]),
    UInt256(w: [11388798043765224079, 11462167646186641884,  217378530236180653, 118571099379]),
    UInt256(w: [4330852013820896542, 4477591218663732153, 434757060472361307,  237142198758]),
    UInt256(w: [11934216846989910278, 895518243732746430,  11154997856320203231, 47428439751]),
    UInt256(w: [5421689620270268940, 1791036487465492861,  3863251638930854846, 94856879503]),
    UInt256(w: [10843379240540537880, 3582072974930985722,  7726503277861709692, 189713759006]),
    UInt256(w: [2168675848108107576, 11784461039211928114,  5234649470314252261, 37942751801]),
    UInt256(w: [4337351696216215152, 5122178004714304612,  10469298940628504523, 75885503602]),
    UInt256(w: [8674703392432430304, 10244356009428609224,  2491853807547457430, 151771007205]),
    UInt256(w: [17349406784864860608, 2041967945147666832,  4983707615094914861, 303542014410]),
    UInt256(w: [14537927801198703092, 4097742403771443689,  996741523018982972, 60708402882]),
    UInt256(w: [10629111528687854567, 8195484807542887379,  1993483046037965944, 121416805764]),
    UInt256(w: [2811478983666157517, 16390969615085774759,  3986966092075931888, 242833611528]),
    UInt256(w: [562295796733231504, 6967542737759065275,  11865439662640917347, 48566722305]),
    UInt256(w: [1124591593466463007, 13935085475518130550,  5284135251572283078, 97133444611]),
    UInt256(w: [2249183186932926013, 9423426877326709484,  10568270503144566157, 194266889222]),
    UInt256(w: [11517883081612316173, 16642080634432983189,  9492351730112733877, 38853377844]),
    UInt256(w: [4589022089515080729, 14837417195156414763,  537959386515916139, 77706755689]),
    UInt256(w: [9178044179030161457, 11228090316603277910,  1075918773031832279, 155413511378]),
    UInt256(w: [18356088358060322914, 4009436559497004204,  2151837546063664559, 310827022756]),
    UInt256(w: [18428612930579705876, 801887311899400840,  4119716323954643235, 62165404551]),
    UInt256(w: [18410481787449860135, 1603774623798801681,  8239432647909286470, 124330809102]),
    UInt256(w: [18374219501190168654, 3207549247597603363,  16478865295818572940, 248661618204]),
    UInt256(w: [11053541529721854378, 15398905108487161965,  18053168318131355880, 49732323640]),
    UInt256(w: [3660338985734157139, 12351066143264772315,  17659592562553160145, 99464647281]),
    UInt256(w: [7320677971468314277, 6255388212819993014,  16872441051396768675, 198929294563]),
    UInt256(w: [8842833223777483502, 12319124086789729572,  14442534654505084704, 39785858912]),
    UInt256(w: [17685666447554967004, 6191504099869907528,  10438325235300617793, 79571717825]),
    UInt256(w: [16924588821400382391, 12383008199739815057,  2429906396891683970, 159143435651]),
    UInt256(w: [15402433569091213166, 6319272325770078499,  4859812793783367941, 318286871302]),
    UInt256(w: [10459184343302063280, 12331900909379746669,  8350660188240494234, 63657374260]),
    UInt256(w: [2471624612894574944, 6217057745049941723,  16701320376480988469, 127314748520]),
    UInt256(w: [4943249225789149887, 12434115490099883446,  14955896679252425322, 254629497041]),
    UInt256(w: [15746045104125471271, 13554869542245707658,  6680528150592395387, 50925899408]),
    UInt256(w: [13045346134541390925, 8662995010781863701,  13361056301184790775, 101851798816]),
    UInt256(w: [7643948195373230233, 17325990021563727403,  8275368528660029934, 203703597633]),
    UInt256(w: [1528789639074646047, 10843895633796566127,  12723120149957736956, 40740719526]),
    UInt256(w: [3057579278149292093, 3241047193883580638,  6999496226205922297, 81481439053]),
    UInt256(w: [6115158556298584186, 6482094387767161276,  13998992452411844594, 162962878106]),
    UInt256(w: [12230317112597168372, 12964188775534322552,  9551240831114137572, 325925756213]),
    UInt256(w: [9824761052003254321, 2592837755106864510,  12978294610448558484, 65185151242]),
    UInt256(w: [1202778030296957026, 5185675510213729021,  7509845147187565352, 130370302485]),
    UInt256(w: [2405556060593914051, 10371351020427458042,  15019690294375130704, 260740604970]),
    UInt256(w: [4170460026860693134, 16831665463053132901,  3003938058875026140, 52148120994]),
    UInt256(w: [8340920053721386267, 15216586852396714186,  6007876117750052281, 104296241988]),
    UInt256(w: [16681840107442772534, 11986429631083876756,  12015752235500104563, 208592483976]),
    UInt256(w: [3336368021488554507, 17154681185184416644,  6092499261841931235, 41718496795]),
    UInt256(w: [6672736042977109014, 15862618296659281672,  12184998523683862471, 83436993590]),
    UInt256(w: [13345472085954218027, 13278492519609011728,  5923252973658173327, 166873987181]),
    UInt256(w: [6358443231932753929, 13723744948147533315,  4873999409473544988, 33374797436]),
    UInt256(w: [12716886463865507858, 9000745822585515014,  9747998818947089977, 66749594872]),
    UInt256(w: [6987028854021464099, 18001491645171030029,  1049253564184628338, 133499189745]),
    UInt256(w: [13974057708042928197, 17556239216632508442,  2098507128369256677, 266998379490]),
    UInt256(w: [17552206800576226933, 10889945472810322334,  419701425673851335, 53399675898]),
    UInt256(w: [16657669527442902249, 3333146871911093053,  839402851347702671, 106799351796]),
    UInt256(w: [14868594981176252881, 6666293743822186107,  1678805702695405342, 213598703592]),
    UInt256(w: [6663067810977160900, 16090654007732078514,  7714458770022901714, 42719740718]),
    UInt256(w: [13326135621954321799, 13734563941754605412,  15428917540045803429, 85439481436]),
    UInt256(w: [8205527170199091982, 9022383809799659209,  12411091006382055243, 170878962873]),
    UInt256(w: [1641105434039818397, 5493825576701842165,  13550264645502142018, 34175792574]),
    UInt256(w: [3282210868079636793, 10987651153403684330,  8653785217294732420, 68351585149]),
    UInt256(w: [6564421736159273585, 3528558233097817044,  17307570434589464841, 136703170298]),
    UInt256(w: [13128843472318547170, 7057116466195634088,  16168396795469378066, 273406340597]),
    UInt256(w: [6315117509205619758, 12479469737464857787,  10612376988577696259, 54681268119]),
    UInt256(w: [12630235018411239515, 6512195401220163958,  2778009903445840903, 109362536239]),
    UInt256(w: [6813725963112927413, 13024390802440327917,  5556019806891681806, 218725072478]),
    UInt256(w: [5052094007364495806, 17362273419455706876,  12179250405604067330, 43745014495]),
    UInt256(w: [10104188014728991612, 16277802765201862136,  5911756737498583045, 87490028991]),
    UInt256(w: [1761631955748431607, 14108861456694172657,  11823513474997166091, 174980057982]),
    UInt256(w: [352326391149686322, 13889818735564565501,  9743400324483253864, 34996011596]),
    UInt256(w: [704652782299372643, 9332893397419579386, 1040056575256956113,  69992023193]),
    UInt256(w: [1409305564598745286, 219042721129607156, 2080113150513912227,  139984046386]),
    UInt256(w: [2818611129197490572, 438085442259214312, 4160226301027824454,  279968092772]),
    UInt256(w: [11631768670065229084, 3776965903193753185,  8210742889689385537, 55993618554]),
    UInt256(w: [4816793266420906552, 7553931806387506371,  16421485779378771074, 111987237108])
  ]
  
  static let bid_exponents_bid32: [Int16] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4,
     4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11,
     11, 11, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
     16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22,
     22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27,
     27, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32,
     33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38,
     38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 43,
     44, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 49,
     49, 49, 50, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 53, 54, 54,
     54, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 59,
     60, 60, 60, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65,
     65, 66, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 69, 70, 70, 70,
     71, 71, 71, 72, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 75, 76,
     76, 76, 77, 77, 77, 78, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81,
     81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87,
     87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 90, 91, 91, 91, 92, 92,
     92, 93, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 97,
     98, 98, 98, 99, 99, 99, 100, 100, 100, 100, 101, 101, 101, 102, 102, 102,
     103, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 106, 107,
     107, 107, 108, 108, 108, 109, 109, 109, 109, 110, 110, 110, 111, 111, 111,
     112, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 115, 116,
     116, 116, 117, 117, 117, 118, 118, 118, 118, 119, 119, 119, 120, 120, 120,
     121, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125,
     125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 128, 129, 129, 129,
     130, 130, 130, 131, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134,
     134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 137, 138, 138, 138,
     139, 139, 139, 140, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143,
     143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 146, 147, 147, 147,
     148, 148, 148, 149, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152,
     152, 153, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 156,
     157, 157, 157, 158, 158, 158, 159, 159, 159, 159, 160, 160, 160, 161, 161,
     161, 162, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 165,
     166, 166, 166, 167, 167, 167, 168, 168, 168, 168, 169, 169, 169, 170, 170,
     170, 171, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 174,
     175, 175, 175, 176, 176, 176, 177, 177, 177, 177, 178, 178, 178, 179, 179,
     179, 180, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184,
     184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 187, 188, 188,
     188, 189, 189, 189, 190, 190, 190, 190, 191, 191
  ]
  
  static let bid_breakpoints_bid32 : [UInt128] = [
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [0, 0]),
    UInt128(w: [11908810229357645280, 469708516554766]),
    UInt128(w: [5954405114678822640, 234854258277383]),
    UInt128(w: [12200574594194187128, 117427129138691]),
    UInt128(w: [15323659333951869372, 58713564569345]),
    UInt128(w: [2831320374921140396, 293567822846729]),
    UInt128(w: [10639032224315346006, 146783911423364]),
    UInt128(w: [5319516112157673003, 73391955711682]),
    UInt128(w: [8150836487078813399, 366959778558411]),
    UInt128(w: [13298790280394182507, 183479889279205]),
    UInt128(w: [15872767177051867061, 91739944639602]),
    UInt128(w: [5576859590421128845, 458699723198014]),
    UInt128(w: [2788429795210564422, 229349861599007]),
    UInt128(w: [10617586934460058019, 114674930799503]),
    UInt128(w: [14532165504084804817, 57337465399751]),
    UInt128(w: [17320595299295369240, 286687326998758]),
    UInt128(w: [8660297649647684620, 143343663499379]),
    UInt128(w: [13553520861678618118, 71671831749689]),
    UInt128(w: [12427372087264435742, 358359158748448]),
    UInt128(w: [6213686043632217871, 179179579374224]),
    UInt128(w: [3106843021816108935, 89589789687112]),
    UInt128(w: [15534215109080544677, 447948948435560]),
    UInt128(w: [7767107554540272338, 223974474217780]),
    UInt128(w: [3883553777270136169, 111987237108890]),
    UInt128(w: [971024812641129231, 559936185544451]),
    UInt128(w: [9708884443175340423, 279968092772225]),
    UInt128(w: [14077814258442446019, 139984046386112]),
    UInt128(w: [7038907129221223009, 69992023193056]),
    UInt128(w: [16747791572396563433, 349960115965281]),
    UInt128(w: [17597267823053057524, 174980057982640]),
    UInt128(w: [8798633911526528762, 87490028991320]),
    UInt128(w: [7099681410213540580, 437450144956602]),
    UInt128(w: [3549840705106770290, 218725072478301]),
    UInt128(w: [10998292389408160953, 109362536239150]),
    UInt128(w: [18097973799621701533, 546812681195752]),
    UInt128(w: [9048986899810850766, 273406340597876]),
    UInt128(w: [4524493449905425383, 136703170298938]),
    UInt128(w: [2262246724952712691, 68351585149469]),
    UInt128(w: [11311233624763563458, 341757925747345]),
    UInt128(w: [14878988849236557537, 170878962873672]),
    UInt128(w: [7439494424618278768, 85439481436836]),
    UInt128(w: [303983975672290610, 427197407184182]),
    UInt128(w: [151991987836145305, 213598703592091]),
    UInt128(w: [9299368030772848460, 106799351796045]),
    UInt128(w: [9603352006445139071, 533996758980227]),
    UInt128(w: [14025048040077345343, 266998379490113]),
    UInt128(w: [16235896056893448479, 133499189745056]),
    UInt128(w: [8117948028446724239, 66749594872528]),
    UInt128(w: [3696251994814517967, 333747974362642]),
    UInt128(w: [1848125997407258983, 166873987181321]),
    UInt128(w: [10147435035558405299, 83436993590660]),
    UInt128(w: [13843687030372923267, 417184967953302]),
    UInt128(w: [6921843515186461633, 208592483976651]),
    UInt128(w: [12684293794448006624, 104296241988325]),
    UInt128(w: [8081236751111378276, 521481209941628]),
    UInt128(w: [4040618375555689138, 260740604970814]),
    UInt128(w: [2020309187777844569, 130370302485407]),
    UInt128(w: [10233526630743698092, 65185151242703]),
    UInt128(w: [14274145006299387230, 325925756213517]),
    UInt128(w: [16360444540004469423, 162962878106758]),
    UInt128(w: [8180222270002234711, 81481439053379]),
    UInt128(w: [4007623202592070326, 407407195266897]),
    UInt128(w: [11227183638150810971, 203703597633448]),
    UInt128(w: [5613591819075405485, 101851798816724]),
    UInt128(w: [9621215021667475812, 509258994083621]),
    UInt128(w: [14033979547688513714, 254629497041810]),
    UInt128(w: [7016989773844256857, 127314748520905]),
    UInt128(w: [12731866923776904236, 63657374260452]),
    UInt128(w: [8319102397755866334, 318286871302263]),
    UInt128(w: [13382923235732708975, 159143435651131]),
    UInt128(w: [15914833654721130295, 79571717825565]),
    UInt128(w: [5787191978767445014, 397858589127829]),
    UInt128(w: [12116968026238498315, 198929294563914]),
    UInt128(w: [6058484013119249157, 99464647281957]),
    UInt128(w: [11845675991886694171, 497323236409786]),
    UInt128(w: [5922837995943347085, 248661618204893]),
    UInt128(w: [12184791034826449350, 124330809102446]),
    UInt128(w: [6092395517413224675, 62165404551223]),
    UInt128(w: [12015233513356571761, 310827022756116]),
    UInt128(w: [6007616756678285880, 155413511378058]),
    UInt128(w: [3003808378339142940, 77706755689029]),
    UInt128(w: [15019041891695714701, 388533778445145]),
    UInt128(w: [16732892982702633158, 194266889222572]),
    UInt128(w: [8366446491351316579, 97133444611286]),
    UInt128(w: [4938744309337479665, 485667223056432]),
    UInt128(w: [2469372154668739832, 242833611528216]),
    UInt128(w: [1234686077334369916, 121416805764108]),
    UInt128(w: [617343038667184958, 60708402882054]),
    UInt128(w: [3086715193335924790, 303542014410270]),
    UInt128(w: [1543357596667962395, 151771007205135]),
    UInt128(w: [9995050835188757005, 75885503602567]),
    UInt128(w: [13081766028524681796, 379427518012837]),
    UInt128(w: [15764255051117116706, 189713759006418]),
    UInt128(w: [7882127525558558353, 94856879503209]),
    UInt128(w: [2517149480373688533, 474284397516047]),
    UInt128(w: [10481946777041620074, 237142198758023]),
    UInt128(w: [14464345425375585845, 118571099379011]),
    UInt128(w: [16455544749542568730, 59285549689505]),
    UInt128(w: [8490747452874637189, 296427748447529]),
    UInt128(w: [13468745763292094402, 148213874223764]),
    UInt128(w: [6734372881646047201, 74106937111882]),
    UInt128(w: [15225120334520684390, 370534685559411]),
    UInt128(w: [16835932204115118003, 185267342779705]),
    UInt128(w: [17641338138912334809, 92633671389852]),
    UInt128(w: [14419714399723467584, 463168356949264]),
    UInt128(w: [7209857199861733792, 231584178474632]),
    UInt128(w: [3604928599930866896, 115792089237316]),
    UInt128(w: [1802464299965433448, 57896044618658]),
    UInt128(w: [9012321499827167240, 289480223093290]),
    UInt128(w: [4506160749913583620, 144740111546645]),
    UInt128(w: [11476452411811567618, 72370055773322]),
    UInt128(w: [2042029837929183242, 361850278866613]),
    UInt128(w: [10244386955819367429, 180925139433306]),
    UInt128(w: [5122193477909683714, 90462569716653]),
    UInt128(w: [7164223315838866956, 452312848583266]),
    UInt128(w: [3582111657919433478, 226156424291633]),
    UInt128(w: [11014427865814492547, 113078212145816]),
    UInt128(w: [5507213932907246273, 56539106072908]),
    UInt128(w: [9089325590826679752, 282695530364541]),
    UInt128(w: [13768034832268115684, 141347765182270]),
    UInt128(w: [6884017416134057842, 70673882591135]),
    UInt128(w: [15973343006960737594, 353369412955676]),
    UInt128(w: [7986671503480368797, 176684706477838]),
    UInt128(w: [3993335751740184398, 88342353238919]),
    UInt128(w: [1519934684991370376, 441711766194596]),
    UInt128(w: [759967342495685188, 220855883097298]),
    UInt128(w: [379983671247842594, 110427941548649]),
    UInt128(w: [1899918356239212970, 552139707743245]),
    UInt128(w: [10173331214974382293, 276069853871622]),
    UInt128(w: [5086665607487191146, 138034926935811]),
    UInt128(w: [11766704840598371381, 69017463467905]),
    UInt128(w: [3493291981863202058, 345087317339528]),
    UInt128(w: [1746645990931601029, 172543658669764]),
    UInt128(w: [873322995465800514, 86271829334882]),
    UInt128(w: [4366614977329002573, 431359146674410]),
    UInt128(w: [2183307488664501286, 215679573337205]),
    UInt128(w: [10315025781187026451, 107839786668602]),
    UInt128(w: [14681640758516029024, 539198933343012]),
    UInt128(w: [7340820379258014512, 269599466671506]),
    UInt128(w: [3670410189629007256, 134799733335753]),
    UInt128(w: [11058577131669279436, 67399866667876]),
    UInt128(w: [18399397510927293948, 336999333339382]),
    UInt128(w: [9199698755463646974, 168499666669691]),
    UInt128(w: [13823221414586599295, 84249833334845]),
    UInt128(w: [13775874851804341627, 421249166674228]),
    UInt128(w: [6887937425902170813, 210624583337114]),
    UInt128(w: [3443968712951085406, 105312291668557]),
    UInt128(w: [17219843564755427034, 526561458342785]),
    UInt128(w: [17833293819232489325, 263280729171392]),
    UInt128(w: [8916646909616244662, 131640364585696]),
    UInt128(w: [4458323454808122331, 65820182292848]),
    UInt128(w: [3844873200331060040, 329100911464241]),
    UInt128(w: [11145808637020305828, 164550455732120]),
    UInt128(w: [5572904318510152914, 82275227866060]),
    UInt128(w: [9417777518841212954, 411376139330301]),
    UInt128(w: [13932260796275382285, 205688069665150]),
    UInt128(w: [6966130398137691142, 102844034832575]),
    UInt128(w: [16383907916978904097, 514220174162876]),
    UInt128(w: [8191953958489452048, 257110087081438]),
    UInt128(w: [4095976979244726024, 128555043540719]),
    UInt128(w: [11271360526477138820, 64277521770359]),
    UInt128(w: [1016570411257039252, 321387608851798]),
    UInt128(w: [508285205628519626, 160693804425899]),
    UInt128(w: [9477514639669035621, 80346902212949]),
    UInt128(w: [10494085050926074874, 401734511064747]),
    UInt128(w: [14470414562317813245, 200867255532373]),
    UInt128(w: [16458579318013682430, 100433627766186]),
    UInt128(w: [8505920295230205688, 502168138830934]),
    UInt128(w: [4252960147615102844, 251084069415467]),
    UInt128(w: [11349852110662327230, 125542034707733]),
    UInt128(w: [14898298092185939423, 62771017353866]),
    UInt128(w: [704514166091490651, 313855086769334]),
    UInt128(w: [352257083045745325, 156927543384667]),
    UInt128(w: [9399500578377648470, 78463771692333]),
    UInt128(w: [10104014744469139122, 392318858461667]),
    UInt128(w: [14275379409089345369, 196159429230833]),
    UInt128(w: [16361061741399448492, 98079714615416]),
    UInt128(w: [8018332412159035998, 490398573077084]),
    UInt128(w: [4009166206079517999, 245199286538542]),
    UInt128(w: [2004583103039758999, 122599643269271]),
    UInt128(w: [10225663588374655307, 61299821634635]),
    UInt128(w: [14234829794454173307, 306499108173177]),
    UInt128(w: [16340786934081862461, 153249554086588]),
    UInt128(w: [8170393467040931230, 76624777043294]),
    UInt128(w: [3958479187785552922, 383123885216472]),
    UInt128(w: [1979239593892776461, 191561942608236]),
    UInt128(w: [989619796946388230, 95780971304118]),
    UInt128(w: [4948098984731941152, 478904856520590]),
    UInt128(w: [2474049492365970576, 239452428260295]),
    UInt128(w: [10460396783037761096, 119726214130147]),
    UInt128(w: [14453570428373656356, 59863107065073]),
    UInt128(w: [16927619920739626932, 299315535325368]),
    UInt128(w: [8463809960369813466, 149657767662684]),
    UInt128(w: [4231904980184906733, 74828883831342]),
    UInt128(w: [2712780827214982049, 374144419156711]),
    UInt128(w: [10579762450462266832, 187072209578355]),
    UInt128(w: [14513253262085909224, 93536104789177]),
    UInt128(w: [17226034089300891273, 467680523945888]),
    UInt128(w: [8613017044650445636, 233840261972944]),
    UInt128(w: [4306508522325222818, 116920130986472]),
    UInt128(w: [2153254261162611409, 58460065493236]),
    UInt128(w: [10766271305813057046, 292300327466180]),
    UInt128(w: [5383135652906528523, 146150163733090]),
    UInt128(w: [2691567826453264261, 73075081866545]),
    UInt128(w: [13457839132266321307, 365375409332725]),
    UInt128(w: [15952291602987936461, 182687704666362]),
    UInt128(w: [7976145801493968230, 91343852333181]),
    UInt128(w: [2987240860050737922, 456719261665907]),
    UInt128(w: [10716992466880144769, 228359630832953]),
    UInt128(w: [14581868270294848192, 114179815416476]),
    UInt128(w: [7290934135147424096, 57089907708238]),
    UInt128(w: [18007926602027568865, 285449538541191]),
    UInt128(w: [18227335337868560240, 142724769270595]),
    UInt128(w: [18337039705789055928, 71362384635297]),
    UInt128(w: [17898222234107073178, 356811923176489]),
    UInt128(w: [18172483153908312397, 178405961588244]),
    UInt128(w: [9086241576954156198, 89202980794122]),
    UInt128(w: [8537719737351677760, 446014903970612]),
    UInt128(w: [4268859868675838880, 223007451985306]),
    UInt128(w: [2134429934337919440, 111503725992653]),
    UInt128(w: [10672149671689597200, 557518629963265]),
    UInt128(w: [14559446872699574408, 278759314981632]),
    UInt128(w: [7279723436349787204, 139379657490816]),
    UInt128(w: [3639861718174893602, 69689828745408]),
    UInt128(w: [18199308590874468010, 348449143727040]),
    UInt128(w: [9099654295437234005, 174224571863520]),
    UInt128(w: [4549827147718617002, 87112285931760]),
    UInt128(w: [4302391664883533397, 435561429658801]),
    UInt128(w: [11374567869296542506, 217780714829400]),
    UInt128(w: [5687283934648271253, 108890357414700]),
    UInt128(w: [9989675599531804650, 544451787073501]),
    UInt128(w: [14218209836620678133, 272225893536750]),
    UInt128(w: [7109104918310339066, 136112946768375]),
    UInt128(w: [12777924496009945341, 68056473384187]),
    UInt128(w: [8549390258921071858, 340282366920938]),
    UInt128(w: [4274695129460535929, 170141183460469]),
    UInt128(w: [11360719601585043772, 85070591730234]),
    UInt128(w: [1463365786796564015, 425352958651173]),
    UInt128(w: [9955054930253057815, 212676479325586]),
    UInt128(w: [4977527465126528907, 106338239662793]),
    UInt128(w: [6440893251923092922, 531691198313966]),
    UInt128(w: [3220446625961546461, 265845599156983]),
    UInt128(w: [10833595349835549038, 132922799578491]),
    UInt128(w: [14640169711772550327, 66461399789245]),
    UInt128(w: [17860616337734096788, 332306998946228]),
    UInt128(w: [8930308168867048394, 166153499473114]),
    UInt128(w: [4465154084433524197, 83076749736557]),
    UInt128(w: [3879026348458069369, 415383748682786]),
    UInt128(w: [1939513174229034684, 207691874341393]),
    UInt128(w: [10193128623969293150, 103845937170696]),
    UInt128(w: [14072154972427362520, 519229685853482]),
    UInt128(w: [7036077486213681260, 259614842926741]),
    UInt128(w: [12741410779961616438, 129807421463370]),
    UInt128(w: [6370705389980808219, 64903710731685]),
    UInt128(w: [13406782876194489479, 324518553658426]),
    UInt128(w: [6703391438097244739, 162259276829213]),
    UInt128(w: [12575067755903398177, 81129638414606]),
    UInt128(w: [7535106558388336041, 405648192073033]),
    UInt128(w: [12990925316048943828, 202824096036516]),
    UInt128(w: [6495462658024471914, 101412048018258]),
    UInt128(w: [14030569216412807955, 507060240091291]),
    UInt128(w: [16238656645061179785, 253530120045645]),
    UInt128(w: [17342700359385365700, 126765060022822]),
    UInt128(w: [8671350179692682850, 63382530011411]),
    UInt128(w: [6463262751044311020, 316912650057057]),
    UInt128(w: [12455003412376931318, 158456325028528]),
    UInt128(w: [6227501706188465659, 79228162514264]),
    UInt128(w: [12690764457232776679, 396140812571321]),
    UInt128(w: [15568754265471164147, 198070406285660]),
    UInt128(w: [7784377132735582073, 99035203142830]),
    UInt128(w: [2028397516258807136, 495176015714152]),
    UInt128(w: [1014198758129403568, 247588007857076]),
    UInt128(w: [507099379064701784, 123794003928538]),
    UInt128(w: [253549689532350892, 61897001964269]),
    UInt128(w: [1267748447661754460, 309485009821345]),
    UInt128(w: [9857246260685653038, 154742504910672]),
    UInt128(w: [4928623130342826519, 77371252455336]),
    UInt128(w: [6196371578004580979, 386856262276681]),
    UInt128(w: [12321557825857066297, 193428131138340]),
    UInt128(w: [6160778912928533148, 96714065569170]),
    UInt128(w: [12357150490933114128, 483570327845851]),
    UInt128(w: [15401947282321332872, 241785163922925]),
    UInt128(w: [16924345678015442244, 120892581961462]),
    UInt128(w: [8462172839007721122, 60446290980731]),
    UInt128(w: [5417376047619502378, 302231454903657]),
    UInt128(w: [11932060060664526997, 151115727451828]),
    UInt128(w: [5966030030332263498, 75557863725914]),
    UInt128(w: [11383406077951765876, 377789318629571]),
    UInt128(w: [14915075075830658746, 188894659314785]),
    UInt128(w: [16680909574770105181, 94447329657392]),
    UInt128(w: [9617571579012319442, 472236648286964]),
    UInt128(w: [4808785789506159721, 236118324143482]),
    UInt128(w: [2404392894753079860, 118059162071741]),
    UInt128(w: [10425568484231315738, 59029581035870]),
    UInt128(w: [15234354273737475459, 295147905179352]),
    UInt128(w: [7617177136868737729, 147573952589676]),
    UInt128(w: [3808588568434368864, 73786976294838]),
    UInt128(w: [596198768462292708, 368934881474191]),
    UInt128(w: [9521471421085922162, 184467440737095]),
    UInt128(w: [13984107747397736889, 92233720368547]),
    UInt128(w: [14580306515860029597, 461168601842738]),
    UInt128(w: [7290153257930014798, 230584300921369]),
    UInt128(w: [12868448665819783207, 115292150460684]),
    UInt128(w: [6434224332909891603, 57646075230342]),
    UInt128(w: [13724377590839906402, 288230376151711]),
    UInt128(w: [16085560832274729009, 144115188075855]),
    UInt128(w: [17266152452992140312, 72057594037927]),
    UInt128(w: [12543785970122495098, 360287970189639]),
    UInt128(w: [15495265021916023357, 180143985094819]),
    UInt128(w: [16971004547812787486, 90071992547409]),
    UInt128(w: [11068046444225730969, 450359962737049]),
    UInt128(w: [14757395258967641292, 225179981368524]),
    UInt128(w: [7378697629483820646, 112589990684262]),
    UInt128(w: [3689348814741910323, 56294995342131]),
    UInt128(w: [18446744073709551615, 281474976710655]),
    UInt128(w: [18446744073709551615, 140737488355327]),
    UInt128(w: [18446744073709551615, 70368744177663]),
    UInt128(w: [18446744073709551615, 351843720888319]),
    UInt128(w: [18446744073709551615, 175921860444159]),
    UInt128(w: [18446744073709551615, 87960930222079]),
    UInt128(w: [18446744073709551615, 439804651110399]),
    UInt128(w: [18446744073709551615, 219902325555199]),
    UInt128(w: [18446744073709551615, 109951162777599]),
    UInt128(w: [18446744073709551615, 549755813887999]),
    UInt128(w: [18446744073709551615, 274877906943999]),
    UInt128(w: [18446744073709551615, 137438953471999]),
    UInt128(w: [18446744073709551615, 68719476735999]),
    UInt128(w: [18446744073709551615, 343597383679999]),
    UInt128(w: [18446744073709551615, 171798691839999]),
    UInt128(w: [18446744073709551615, 85899345919999]),
    UInt128(w: [18446744073709551615, 429496729599999]),
    UInt128(w: [18446744073709551615, 214748364799999]),
    UInt128(w: [18446744073709551615, 107374182399999]),
    UInt128(w: [18446744073709551615, 536870911999999]),
    UInt128(w: [18446744073709551615, 268435455999999]),
    UInt128(w: [18446744073709551615, 134217727999999]),
    UInt128(w: [18446744073709551615, 67108863999999]),
    UInt128(w: [18446744073709551615, 335544319999999]),
    UInt128(w: [18446744073709551615, 167772159999999]),
    UInt128(w: [18446744073709551615, 83886079999999]),
    UInt128(w: [18446744073709551615, 419430399999999]),
    UInt128(w: [18446744073709551615, 209715199999999]),
    UInt128(w: [18446744073709551615, 104857599999999]),
    UInt128(w: [18446744073709551615, 524287999999999]),
    UInt128(w: [18446744073709551615, 262143999999999]),
    UInt128(w: [18446744073709551615, 131071999999999]),
    UInt128(w: [18446744073709551615, 65535999999999]),
    UInt128(w: [18446744073709551615, 327679999999999]),
    UInt128(w: [18446744073709551615, 163839999999999]),
    UInt128(w: [18446744073709551615, 81919999999999]),
    UInt128(w: [18446744073709551615, 409599999999999]),
    UInt128(w: [18446744073709551615, 204799999999999]),
    UInt128(w: [18446744073709551615, 102399999999999]),
    UInt128(w: [18446744073709551615, 511999999999999]),
    UInt128(w: [18446744073709551615, 255999999999999]),
    UInt128(w: [18446744073709551615, 127999999999999]),
    UInt128(w: [18446744073709551615, 63999999999999]),
    UInt128(w: [18446744073709551615, 319999999999999]),
    UInt128(w: [18446744073709551615, 159999999999999]),
    UInt128(w: [18446744073709551615, 79999999999999]),
    UInt128(w: [18446744073709551615, 399999999999999]),
    UInt128(w: [18446744073709551615, 199999999999999]),
    UInt128(w: [18446744073709551615, 99999999999999]),
    UInt128(w: [18446744073709551615, 499999999999999]),
    UInt128(w: [18446744073709551615, 249999999999999]),
    UInt128(w: [18446744073709551615, 124999999999999]),
    UInt128(w: [18446744073709551615, 62499999999999]),
    UInt128(w: [18446744073709551615, 312499999999999]),
    UInt128(w: [18446744073709551615, 156249999999999]),
    UInt128(w: [18446744073709551615, 78124999999999]),
    UInt128(w: [18446744073709551615, 390624999999999]),
    UInt128(w: [18446744073709551615, 195312499999999]),
    UInt128(w: [18446744073709551615, 97656249999999]),
    UInt128(w: [18446744073709551615, 488281249999999]),
    UInt128(w: [18446744073709551615, 244140624999999]),
    UInt128(w: [18446744073709551615, 122070312499999]),
    UInt128(w: [18446744073709551615, 61035156249999]),
    UInt128(w: [18446744073709551615, 305175781249999]),
    UInt128(w: [18446744073709551615, 152587890624999]),
    UInt128(w: [18446744073709551615, 76293945312499]),
    UInt128(w: [18446744073709551615, 381469726562499]),
    UInt128(w: [18446744073709551615, 190734863281249]),
    UInt128(w: [18446744073709551615, 95367431640624]),
    UInt128(w: [18446744073709551615, 476837158203124]),
    UInt128(w: [9223372036854775807, 238418579101562]),
    UInt128(w: [4611686018427387903, 119209289550781]),
    UInt128(w: [11529215046068469759, 59604644775390]),
    UInt128(w: [2305843009213693951, 298023223876953]),
    UInt128(w: [10376293541461622783, 149011611938476]),
    UInt128(w: [5188146770730811391, 74505805969238]),
    UInt128(w: [7493989779944505343, 372529029846191]),
    UInt128(w: [12970366926827028479, 186264514923095]),
    UInt128(w: [15708555500268290047, 93132257461547]),
    UInt128(w: [4755801206503243775, 465661287307739]),
    UInt128(w: [11601272640106397695, 232830643653869]),
    UInt128(w: [15024008356907974655, 116415321826934]),
    UInt128(w: [7512004178453987327, 58207660913467]),
    UInt128(w: [666532744850833407, 291038304567337]),
    UInt128(w: [9556638409280192511, 145519152283668]),
    UInt128(w: [4778319204640096255, 72759576141834]),
    UInt128(w: [5444851949490929663, 363797880709171]),
    UInt128(w: [11945798011600240639, 181898940354585]),
    UInt128(w: [15196271042654896127, 90949470177292]),
    UInt128(w: [2194378918436274175, 454747350886464]),
    UInt128(w: [1097189459218137087, 227373675443232]),
    UInt128(w: [548594729609068543, 113686837721616]),
    UInt128(w: [274297364804534271, 56843418860808]),
    UInt128(w: [1371486824022671359, 284217094304040]),
    UInt128(w: [685743412011335679, 142108547152020]),
    UInt128(w: [342871706005667839, 71054273576010]),
    UInt128(w: [1714358530028339199, 355271367880050]),
    UInt128(w: [857179265014169599, 177635683940025]),
    UInt128(w: [9651961669361860607, 88817841970012]),
    UInt128(w: [11366320199390199807, 444089209850062]),
    UInt128(w: [5683160099695099903, 222044604925031]),
    UInt128(w: [12064952086702325759, 111022302462515]),
    UInt128(w: [4984528212382973951, 555111512312578]),
    UInt128(w: [2492264106191486975, 277555756156289]),
    UInt128(w: [10469504089950519295, 138777878078144]),
    UInt128(w: [5234752044975259647, 69388939039072]),
    UInt128(w: [7727016151166746623, 346944695195361]),
    UInt128(w: [13086880112438149119, 173472347597680]),
    UInt128(w: [6543440056219074559, 86736173798840]),
    UInt128(w: [14270456207385821183, 433680868994201]),
    UInt128(w: [16358600140547686399, 216840434497100]),
    UInt128(w: [8179300070273843199, 108420217248550]),
    UInt128(w: [4003012203950112767, 542101086242752]),
    UInt128(w: [2001506101975056383, 271050543121376]),
    UInt128(w: [1000753050987528191, 135525271560688]),
    UInt128(w: [500376525493764095, 67762635780344]),
    UInt128(w: [2501882627468820479, 338813178901720]),
    UInt128(w: [1250941313734410239, 169406589450860]),
    UInt128(w: [625470656867205119, 84703294725430]),
    UInt128(w: [3127353284336025599, 423516473627150]),
    UInt128(w: [1563676642168012799, 211758236813575]),
    UInt128(w: [10005210357938782207, 105879118406787]),
    UInt128(w: [13132563642274807807, 529395592033937]),
    UInt128(w: [15789653857992179711, 264697796016968]),
    UInt128(w: [7894826928996089855, 132348898008484]),
    UInt128(w: [3947413464498044927, 66174449004242]),
    UInt128(w: [1290323248780673023, 330872245021211]),
    UInt128(w: [9868533661245112319, 165436122510605]),
    UInt128(w: [14157638867477331967, 82718061255302]),
    UInt128(w: [15447962116258004991, 413590306276513]),
    UInt128(w: [16947353094983778303, 206795153138256]),
    UInt128(w: [8473676547491889151, 103397576569128]),
    UInt128(w: [5474894590040342527, 516987882845642]),
    UInt128(w: [2737447295020171263, 258493941422821]),
    UInt128(w: [10592095684364861439, 129246970711410]),
    UInt128(w: [5296047842182430719, 64623485355705]),
    UInt128(w: [8033495137202601983, 323117426778526]),
    UInt128(w: [4016747568601300991, 161558713389263]),
    UInt128(w: [11231745821155426303, 80779356694631]),
    UInt128(w: [818496884648476671, 403896783473158]),
    UInt128(w: [409248442324238335, 201948391736579]),
    UInt128(w: [9427996258016894975, 100974195868289]),
    UInt128(w: [10246493142665371647, 504870979341447]),
    UInt128(w: [14346618608187461631, 252435489670723]),
    UInt128(w: [16396681340948506623, 126217744835361]),
    UInt128(w: [17421712707329029119, 63108872417680]),
    UInt128(w: [13321587241806939135, 315544362088404]),
    UInt128(w: [6660793620903469567, 157772181044202]),
    UInt128(w: [3330396810451734783, 78886090522101]),
    UInt128(w: [16651984052258673919, 394430452610505]),
    UInt128(w: [17549364062984112767, 197215226305252]),
    UInt128(w: [8774682031492056383, 98607613152626]),
    UInt128(w: [6979922010041178687, 493038065763132]),
    UInt128(w: [3489961005020589343, 246519032881566]),
    UInt128(w: [1744980502510294671, 123259516440783]),
    UInt128(w: [10095862288109923143, 61629758220391]),
    UInt128(w: [13585823293130512487, 308148791101957]),
    UInt128(w: [16016283683420032051, 154074395550978]),
    UInt128(w: [8008141841710016025, 77037197775489]),
    UInt128(w: [3147221061130976897, 385185988877447]),
    UInt128(w: [10796982567420264256, 192592994438723]),
    UInt128(w: [14621863320564907936, 96296497219361]),
    UInt128(w: [17769084381695884834, 481482486096808]),
    UInt128(w: [8884542190847942417, 240741243048404]),
    UInt128(w: [4442271095423971208, 120370621524202]),
    UInt128(w: [2221135547711985604, 60185310762101]),
    UInt128(w: [11105677738559928021, 300926553810505]),
    UInt128(w: [14776210906134739818, 150463276905252]),
    UInt128(w: [7388105453067369909, 75231638452626]),
    UInt128(w: [47039117917746314, 376158192263132]),
    UInt128(w: [23519558958873157, 188079096131566]),
    UInt128(w: [11759779479436578, 94039548065783]),
    UInt128(w: [58798897397182893, 470197740328915]),
    UInt128(w: [9252771485553367254, 235098870164457]),
    UInt128(w: [13849757779631459435, 117549435082228]),
    UInt128(w: [6924878889815729717, 58774717541114]),
    UInt128(w: [16177650375369096972, 293873587705571]),
    UInt128(w: [17312197224539324294, 146936793852785]),
    UInt128(w: [17879470649124437955, 73468396926392]),
    UInt128(w: [15610376950783983311, 367341984631964]),
    UInt128(w: [7805188475391991655, 183670992315982]),
    UInt128(w: [3902594237695995827, 91835496157991]),
    UInt128(w: [1066227114770427523, 459177480789956]),
    UInt128(w: [533113557385213761, 229588740394978]),
    UInt128(w: [266556778692606880, 114794370197489]),
    UInt128(w: [9356650426201079248, 57397185098744]),
    UInt128(w: [9889763983586293010, 286985925493722]),
    UInt128(w: [4944881991793146505, 143492962746861]),
    UInt128(w: [11695813032751349060, 71746481373430]),
    UInt128(w: [3138832942628090454, 358732406867153]),
    UInt128(w: [10792788508168821035, 179366203433576]),
    UInt128(w: [5396394254084410517, 89683101716788]),
    UInt128(w: [8535227196712500972, 448415508583941]),
    UInt128(w: [13490985635211026294, 224207754291970]),
    UInt128(w: [6745492817605513147, 112103877145985]),
    UInt128(w: [15280720014318014119, 560519385729926]),
    UInt128(w: [7640360007159007059, 280259692864963]),
    UInt128(w: [13043552040434279337, 140129846432481]),
    UInt128(w: [15745148057071915476, 70064923216240]),
    UInt128(w: [4938763990521370920, 350324616081204]),
    UInt128(w: [2469381995260685460, 175162308040602]),
    UInt128(w: [1234690997630342730, 87581154020301]),
    UInt128(w: [6173454988151713650, 437905770101505]),
    UInt128(w: [12310099530930632633, 218952885050752]),
    UInt128(w: [6155049765465316316, 109476442525376]),
    UInt128(w: [12328504753617029967, 547382212626881]),
    UInt128(w: [15387624413663290791, 273691106313440]),
    UInt128(w: [7693812206831645395, 136845553156720]),
    UInt128(w: [3846906103415822697, 68422776578360]),
    UInt128(w: [787786443369561873, 342113882891801]),
    UInt128(w: [9617265258539556744, 171056941445900]),
    UInt128(w: [4808632629269778372, 85528470722950]),
    UInt128(w: [5596419072639340246, 427642353614751]),
    UInt128(w: [12021581573174445931, 213821176807375]),
    UInt128(w: [15234162823441998773, 106910588403687]),
    UInt128(w: [2383837822371787403, 534552942018439]),
    UInt128(w: [10415290948040669509, 267276471009219]),
    UInt128(w: [14431017510875110562, 133638235504609]),
    UInt128(w: [16438880792292331089, 66819117752304]),
    UInt128(w: [8407427666623448983, 334095588761524]),
    UInt128(w: [4203713833311724491, 167047794380762]),
    UInt128(w: [2101856916655862245, 83523897190381]),
    UInt128(w: [10509284583279311229, 417619485951905]),
    UInt128(w: [14478014328494431422, 208809742975952]),
    UInt128(w: [7239007164247215711, 104404871487976]),
    UInt128(w: [17748291747526526940, 522024357439881]),
    UInt128(w: [18097517910618039278, 261012178719940]),
    UInt128(w: [9048758955309019639, 130506089359970]),
    UInt128(w: [4524379477654509819, 65253044679985]),
    UInt128(w: [4175153314562997481, 326265223399926]),
    UInt128(w: [2087576657281498740, 163132611699963]),
    UInt128(w: [10267160365495525178, 81566305849981]),
    UInt128(w: [14442313680058522660, 407831529249907]),
    UInt128(w: [16444528876884037138, 203915764624953]),
    UInt128(w: [17445636475296794377, 101957882312476]),
    UInt128(w: [13441206081645765421, 509789411562384]),
    UInt128(w: [6720603040822882710, 254894705781192]),
    UInt128(w: [3360301520411441355, 127447352890596]),
    UInt128(w: [1680150760205720677, 63723676445298]),
    UInt128(w: [8400753801028603388, 318618382226490]),
    UInt128(w: [4200376900514301694, 159309191113245]),
    UInt128(w: [11323560487111926655, 79654595556622]),
    UInt128(w: [1277570214430978427, 398272977783113]),
    UInt128(w: [9862157144070265021, 199136488891556]),
    UInt128(w: [4931078572035132510, 99568244445778]),
    UInt128(w: [6208648786466110938, 497841222228891]),
    UInt128(w: [12327696430087831277, 248920611114445]),
    UInt128(w: [15387220251898691446, 124460305557222]),
    UInt128(w: [7693610125949345723, 62230152778611]),
    UInt128(w: [1574562482327625384, 311150763893057]),
    UInt128(w: [10010653278018588500, 155575381946528]),
    UInt128(w: [5005326639009294250, 77787690973264]),
    UInt128(w: [6579889121336919634, 388938454866321]),
    UInt128(w: [12513316597523235625, 194469227433160]),
    UInt128(w: [6256658298761617812, 97234613716580]),
    UInt128(w: [12836547420098537447, 486173068582901]),
    UInt128(w: [15641645746904044531, 243086534291450]),
    UInt128(w: [7820822873452022265, 121543267145725]),
    UInt128(w: [13133783473580786940, 60771633572862]),
    UInt128(w: [10328685146775279856, 303858167864313]),
    UInt128(w: [14387714610242415736, 151929083932156]),
    UInt128(w: [7193857305121207868, 75964541966078]),
    UInt128(w: [17522542451896487724, 379822709830391]),
    UInt128(w: [17984643262803019670, 189911354915195]),
    UInt128(w: [18215693668256285643, 94955677457597]),
    UInt128(w: [17291492046443221751, 474778387287989]),
    UInt128(w: [17869118060076386683, 237389193643994]),
    UInt128(w: [8934559030038193341, 118694596821997]),
    UInt128(w: [13690651551873872478, 59347298410998]),
    UInt128(w: [13113025538240707546, 296736492054993]),
    UInt128(w: [15779884805975129581, 148368246027496]),
    UInt128(w: [7889942402987564790, 74184123013748]),
    UInt128(w: [2556223867518720721, 370920615068742]),
    UInt128(w: [1278111933759360360, 185460307534371]),
    UInt128(w: [9862428003734455988, 92730153767185]),
    UInt128(w: [12418651871253176710, 463650768835927]),
    UInt128(w: [15432697972481364163, 231825384417963]),
    UInt128(w: [16939721023095457889, 115912692208981]),
    UInt128(w: [17693232548402504752, 57956346104490]),
    UInt128(w: [14679186447174317299, 289781730522454]),
    UInt128(w: [7339593223587158649, 144890865261227]),
    UInt128(w: [12893168648648355132, 72445432630613]),
    UInt128(w: [9125611022113120816, 362227163153068]),
    UInt128(w: [4562805511056560408, 181113581576534]),
    UInt128(w: [2281402755528280204, 90556790788267]),
    UInt128(w: [11407013777641401020, 452783953941335]),
    UInt128(w: [14926878925675476318, 226391976970667]),
    UInt128(w: [16686811499692513967, 113195988485333]),
    UInt128(w: [17566777786701032791, 56597994242666]),
    UInt128(w: [14046912638666957494, 282989971213334]),
    UInt128(w: [7023456319333478747, 141494985606667]),
    UInt128(w: [12735100196521515181, 70747492803333]),
    UInt128(w: [8335268761478921059, 353737464016668]),
    UInt128(w: [4167634380739460529, 176868732008334]),
    UInt128(w: [2083817190369730264, 88434366004167]),
    UInt128(w: [10419085951848651324, 442171830020835]),
    UInt128(w: [14432915012779101470, 221085915010417]),
    UInt128(w: [16439829543244326543, 110542957505208]),
    UInt128(w: [8412171421383426251, 552714787526044]),
    UInt128(w: [4206085710691713125, 276357393763022]),
    UInt128(w: [2103042855345856562, 138178696881511]),
    UInt128(w: [10274893464527704089, 69089348440755]),
    UInt128(w: [14480979175219417215, 345446742203777]),
    UInt128(w: [16463861624464484415, 172723371101888]),
    UInt128(w: [8231930812232242207, 86361685550944]),
    UInt128(w: [4266165913742107807, 431808427754722]),
    UInt128(w: [2133082956871053903, 215904213877361]),
    UInt128(w: [10289913515290302759, 107952106938680]),
    UInt128(w: [14556079429032410566, 539760534693402]),
    UInt128(w: [7278039714516205283, 269880267346701]),
    UInt128(w: [12862391894112878449, 134940133673350]),
    UInt128(w: [6431195947056439224, 67470066836675]),
    UInt128(w: [13709235661572644508, 337350334183376]),
    UInt128(w: [6854617830786322254, 168675167091688]),
    UInt128(w: [3427308915393161127, 84337583545844]),
    UInt128(w: [17136544576965805635, 421687917729220]),
    UInt128(w: [8568272288482902817, 210843958864610]),
    UInt128(w: [4284136144241451408, 105421979432305]),
    UInt128(w: [2973936647497705428, 527109897161526]),
    UInt128(w: [1486968323748852714, 263554948580763]),
    UInt128(w: [9966856198729202165, 131777474290381]),
    UInt128(w: [14206800136219376890, 65888737145190]),
    UInt128(w: [15693768459968229604, 329443685725953]),
    UInt128(w: [17070256266838890610, 164721842862976])
  ]
  
  static let bid_roundbound_128: [UInt128] = [
    // BID_ROUNDING_TO_NEAREST
    UInt128(w: [0, (1 << 63)]),      // positive|even
    UInt128(w: [~0, (1 << 63) - 1]), // positive|odd
    UInt128(w: [0, (1 << 63)]),      // negative|even
    UInt128(w: [~0, (1 << 63) - 1]), // negative|odd

    // BID_ROUNDING_DOWN
    UInt128(w: [~0, ~0]),            // positive|even
    UInt128(w: [~0, ~0]),            // positive|odd
    UInt128(w: [0, 0]),              // negative|even
    UInt128(w: [0, 0]),              // negative|odd

    // BID_ROUNDING_UP
    UInt128(w: [0, 0]),              // positive|even
    UInt128(w: [0, 0]),              // positive|odd
    UInt128(w: [~0, ~0]),            // negative|even
    UInt128(w: [~0, ~0]),            // negative|odd

    // BID_ROUNDING_TO_ZERO
    UInt128(w: [~0, ~0]),            // positive|even
    UInt128(w: [~0, ~0]),            // positive|odd
    UInt128(w: [~0, ~0]),            // negative|even
    UInt128(w: [~0, ~0]),            // negative|odd

    // BID_ROUNDING_TIES_AWAY
    UInt128(w: [~0, (1 << 63) - 1]), // positive|even
    UInt128(w: [~0, (1 << 63) - 1]), // positive|odd
    UInt128(w: [~0, (1 << 63) - 1]), // negative|even
    UInt128(w: [~0, (1 << 63) - 1])  // negative|odd
  ]
  
  static let bid_short_recip_scale: [Int8] = [
    1, 1, 5, 7, 11, 12, 17, 21, 24, 27, 31, 34, 37, 41, 44, 47, 51, 54
  ]
  
  // bid_ten2k64[i] = 10^i, 0 <= i <= 19
  static func bid_ten2k64(_ i:Int) -> UInt64 {
    UInt64(bid_power10_table_128[i])
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
     UInt64((UInt128(1) << bid_powers[i]) / bid_power10_table_128[i+1])+1
  }
  
  // Values of 10^(-x) trancated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static func bid_ten2mxtrunc64(_ i:Int) -> UInt64 {
    UInt64((UInt128(1) << (64+bid_Ex64m64(i))) / bid_power10_table_128[i+1])
  }

  // Kx from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static func bid_Kx64(_ i:Int) -> UInt64 {
    bid_ten2mxtrunc64(i)+1
  }
   
  static func bid_reciprocals10_64(_ i: Int) -> UInt64 {
    if i == 0 { return 1 }
    let twoPower = bid_short_recip_scale[i]+64
    return UInt64(UInt128(1) << twoPower / bid_power10_table_128[i]) + 1
  }

  
  // bid_shiftright128[] contains the right shift count to obtain C2* from
  // the top 128 bits of the 128x128-bit product C2 * Kx
  static let bid_shiftright128: [UInt8] = [
    0, 0, 0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56,
    59, 63, 66, 69, 73, 76, 79, 83, 86, 89, 92, 96, 99, 102
  ]
  
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

//  static let bid_nr_digits : [DEC_DIGITS] = [
//    // only the first entry is used if it is not 0
//    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   1-bit n < 10^1
//    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   2-bit n < 10^1
//    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   3-bit n < 10^1
//    DEC_DIGITS(0, 0x0000000000000000, 0x000000000000000a, 1),    //   4-bit n ? 10^1
//    DEC_DIGITS(2, 0x0000000000000000, 0x0000000000000064, 2),    //   5-bit n < 10^2
//    DEC_DIGITS(2, 0x0000000000000000, 0x0000000000000064, 2),    //   6-bit n < 10^2
//    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000000064, 2),    //   7-bit n ? 10^2
//    DEC_DIGITS(3, 0x0000000000000000, 0x00000000000003e8, 3),    //   8-bit n < 10^3
//    DEC_DIGITS(3, 0x0000000000000000, 0x00000000000003e8, 3),    //   9-bit n < 10^3
//    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000003e8, 3),    //  10-bit n ? 10^3
//    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  11-bit n < 10^4
//    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  12-bit n < 10^4
//    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  13-bit n < 10^4
//    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000002710, 4),    //  14-bit n ? 10^4
//    DEC_DIGITS(5, 0x0000000000000000, 0x00000000000186a0, 5),    //  15-bit n < 10^5
//    DEC_DIGITS(5, 0x0000000000000000, 0x00000000000186a0, 5),    //  16-bit n < 10^5
//    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000186a0, 5),    //  17-bit n ? 10^5
//    DEC_DIGITS(6, 0x0000000000000000, 0x00000000000f4240, 6),    //  18-bit n < 10^6
//    DEC_DIGITS(6, 0x0000000000000000, 0x00000000000f4240, 6),    //  19-bit n < 10^6
//    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000f4240, 6),    //  20-bit n ? 10^6
//    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  21-bit n < 10^7
//    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  22-bit n < 10^7
//    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  23-bit n < 10^7
//    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000989680, 7),    //  24-bit n ? 10^7
//    DEC_DIGITS(8, 0x0000000000000000, 0x0000000005f5e100, 8),    //  25-bit n < 10^8
//    DEC_DIGITS(8, 0x0000000000000000, 0x0000000005f5e100, 8),    //  26-bit n < 10^8
//    DEC_DIGITS(0, 0x0000000000000000, 0x0000000005f5e100, 8),    //  27-bit n ? 10^8
//    DEC_DIGITS(9, 0x0000000000000000, 0x000000003b9aca00, 9),    //  28-bit n < 10^9
//    DEC_DIGITS(9, 0x0000000000000000, 0x000000003b9aca00, 9),    //  29-bit n < 10^9
//    DEC_DIGITS(0, 0x0000000000000000, 0x000000003b9aca00, 9),    //  30-bit n ? 10^9
//    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  31-bit n < 10^10
//    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  32-bit n < 10^10
//    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  33-bit n < 10^10
//    DEC_DIGITS(0, 0x0000000000000000, 0x00000002540be400, 10),    //  34-bit n ? 10^10
//    DEC_DIGITS(11, 0x0000000000000000, 0x000000174876e800, 11),    //  35-bit n < 10^11
//    DEC_DIGITS(11, 0x0000000000000000, 0x000000174876e800, 11),    //  36-bit n < 10^11
//    DEC_DIGITS(0, 0x0000000000000000, 0x000000174876e800, 11),    //  37-bit n ? 10^11
//    DEC_DIGITS(12, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  38-bit n < 10^12
//    DEC_DIGITS(12, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  39-bit n < 10^12
//    DEC_DIGITS(0, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  40-bit n ? 10^12
//    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  41-bit n < 10^13
//    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  42-bit n < 10^13
//    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  43-bit n < 10^13
//    DEC_DIGITS(0, 0x0000000000000000, 0x000009184e72a000, 13),    //  44-bit n ? 10^13
//    DEC_DIGITS(14, 0x0000000000000000, 0x00005af3107a4000, 14),    //  45-bit n < 10^14
//    DEC_DIGITS(14, 0x0000000000000000, 0x00005af3107a4000, 14),    //  46-bit n < 10^14
//    DEC_DIGITS(0, 0x0000000000000000, 0x00005af3107a4000, 14),    //  47-bit n ? 10^14
//    DEC_DIGITS(15, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  48-bit n < 10^15
//    DEC_DIGITS(15, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  49-bit n < 10^15
//    DEC_DIGITS(0, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  50-bit n ? 10^15
//    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  51-bit n < 10^16
//    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  52-bit n < 10^16
//    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  53-bit n < 10^16
//    DEC_DIGITS(0, 0x0000000000000000, 0x002386f26fc10000, 16),    //  54-bit n ? 10^16
//    DEC_DIGITS(17, 0x0000000000000000, 0x016345785d8a0000, 17),    //  55-bit n < 10^17
//    DEC_DIGITS(17, 0x0000000000000000, 0x016345785d8a0000, 17),    //  56-bit n < 10^17
//    DEC_DIGITS(0, 0x0000000000000000, 0x016345785d8a0000, 17),    //  57-bit n ? 10^17
//    DEC_DIGITS(18, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  58-bit n < 10^18
//    DEC_DIGITS(18, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  59-bit n < 10^18
//    DEC_DIGITS(0, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  60-bit n ? 10^18
//    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  61-bit n < 10^19
//    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  62-bit n < 10^19
//    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  63-bit n < 10^19
//    DEC_DIGITS(0, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  64-bit n ? 10^19
//    DEC_DIGITS(20, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  65-bit n < 10^20
//    DEC_DIGITS(20, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  66-bit n < 10^20
//    DEC_DIGITS(0, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  67-bit n ? 10^20
//    DEC_DIGITS(21, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  68-bit n < 10^21
//    DEC_DIGITS(21, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  69-bit n < 10^21
//    DEC_DIGITS(0, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  70-bit n ? 10^21
//    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  71-bit n < 10^22
//    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  72-bit n < 10^22
//    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  73-bit n < 10^22
//    DEC_DIGITS(0, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  74-bit n ? 10^22
//    DEC_DIGITS(23, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  75-bit n < 10^23
//    DEC_DIGITS(23, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  76-bit n < 10^23
//    DEC_DIGITS(0, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  77-bit n ? 10^23
//    DEC_DIGITS(24, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  78-bit n < 10^24
//    DEC_DIGITS(24, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  79-bit n < 10^24
//    DEC_DIGITS(0, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  80-bit n ? 10^24
//    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  81-bit n < 10^25
//    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  82-bit n < 10^25
//    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  83-bit n < 10^25
//    DEC_DIGITS(0, 0x0000000000084595, 0x161401484a000000, 25),    //  84-bit n ? 10^25
//    DEC_DIGITS(26, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  85-bit n < 10^26
//    DEC_DIGITS(26, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  86-bit n < 10^26
//    DEC_DIGITS(0, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  87-bit n ? 10^26
//    DEC_DIGITS(27, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  88-bit n < 10^27
//    DEC_DIGITS(27, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  89-bit n < 10^27
//    DEC_DIGITS(0, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  90-bit n ? 10^27
//    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  91-bit n < 10^28
//    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  92-bit n < 10^28
//    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  93-bit n < 10^28
//    DEC_DIGITS(0, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  94-bit n ? 10^28
//    DEC_DIGITS(29, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  95-bit n < 10^29
//    DEC_DIGITS(29, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  96-bit n < 10^29
//    DEC_DIGITS(0, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  97-bit n ? 10^29
//    DEC_DIGITS(30, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    //  98-bit n < 10^30
//    DEC_DIGITS(30, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    //  99-bit n < 10^30
//    DEC_DIGITS(0, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    // 100-bit n ? 10^30
//    DEC_DIGITS(31, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 101-bit n < 10^31
//    DEC_DIGITS(31, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 102-bit n < 10^31
//    DEC_DIGITS(0, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 103-bit n ? 10^31
//    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 104-bit n < 10^32
//    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 105-bit n < 10^32
//    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 106-bit n < 10^32
//    DEC_DIGITS(0, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 107-bit n ? 10^32
//    DEC_DIGITS(33, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 108-bit n < 10^33
//    DEC_DIGITS(33, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 109-bit n < 10^33
//    DEC_DIGITS(0, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 100-bit n ? 10^33
//    DEC_DIGITS(34, 0x0001ed09bead87c0, 0x378d8e6400000000, 34),    // 111-bit n < 10^34
//    DEC_DIGITS(34, 0x0001ed09bead87c0, 0x378d8e6400000000, 34),    // 112-bit n < 10^34
//    DEC_DIGITS(0, 0x0001ed09bead87c0, 0x378d8e6400000000, 34)    // 113-bit n ? 10^34
//    //{ 35, 0x0013426172c74d82, 0x2b878fe800000000, 35 }  // 114-bit n < 10^35
//  ]
  
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
  
  static func bid32_to_string (_ x: UInt32, _ showPlus: Bool = false) ->
      String {
    // unpack arguments, check for NaN or Infinity
    let plus = showPlus ? "+" : ""
    
    let (negative_x, exponent_x, coefficient_x, valid_x) = unpack(bid32: x)
    if valid_x {
      // x is not special
      let ps = String(coefficient_x)
      let exponent_x = exponent_x - EXPONENT_BIAS + (ps.count - 1)
      return (negative_x ? "-" : plus) +
              addDecimalPointAndExponent(ps, exponent_x, MAX_DIGITS)
    } else {
      // x is Inf. or NaN or 0
      var ps = negative_x ? "-" : plus
      if isNaN(x) {
        if isSNaN(x) { ps.append("S") }
        ps.append("NaN")
        return ps
      }
      if isInfinite(x) {
        ps.append("Inf")
        return ps
      }
      ps.append("0")
      return ps
    }
  }
  
  static func addDecimalPointAndExponent(_ ps:String, _ exponent:Int,
                                         _ maxDigits:Int) -> String {
    var digits = ps.count
    var ps = ps
    var exponent_x = exponent
    if exponent_x == 0 {
      ps.insert(".", at: ps.index(ps.startIndex, offsetBy: exponent_x+1))
    } else if abs(exponent_x) > maxDigits {
      ps.insert(".", at: ps.index(after: ps.startIndex))
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
  
  static func bid32_from_string (_ ps: String, _ rnd_mode: Rounding,
                                 _ pfpsf: inout Status) -> UInt32 {
    // eliminate leading whitespace
    var ps = ps.trimmingCharacters(in: .whitespaces).lowercased()
    var res: UInt32
    
    // get first non-whitespace character
    var c = ps.isEmpty ? "\0" : ps.removeFirst()
    
    // detect special cases (INF or NaN)
    if c == "\0" || (c != "." && c != "-" && c != "+" && (c < "0" || c > "9")){
      // Infinity?
      if c == "i" && (ps.hasPrefix("nfinity") || ps.hasPrefix("nf")) {
        return INFINITY_MASK
      }
      // return sNaN
      if c == "s" && ps.hasPrefix("nan") {
        // case insensitive check for snan
        return SNAN_MASK
      } else {
        // return qNaN
        return NAN_MASK
      }
    }
    
    // detect +INF or -INF
    if ps.hasPrefix("infinity") || ps.hasPrefix("inf") {
      if c == "+" {
        res = INFINITY_MASK
      } else if c == "-" {
        res = SINFINITY_MASK
      } else {
        res = NAN_MASK
      }
      return res
    }
    
    // if +sNaN, +SNaN, -sNaN, or -SNaN
    if ps.hasPrefix("snan") {
      if c == "-" {
        res = SSNAN_MASK
      } else {
        res = SNAN_MASK
      }
      return res
    }
    
    // determine sign
    var sign_x = UInt32(0)
    if c == "-" {
      sign_x = SIGN_MASK
    }
    
    // get next character if leading +/- sign
    if c == "-" || c == "+" {
      c = ps.isEmpty ? "\0" : ps.removeFirst()
    }
    
    // if c isn"t a decimal point or a decimal digit, return NaN
    if c != "." && (c < "0" || c > "9") {
      // return NaN
      return NAN_MASK | sign_x
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
            // if this is the first radix point, and the next character is NULL,
            // we have a zero
            if ps.isEmpty {
              right_radix_leading_zeros = EXPONENT_BIAS -
                                          right_radix_leading_zeros
              if right_radix_leading_zeros < 0 {
                right_radix_leading_zeros = 0
              }
              return (UInt32(right_radix_leading_zeros) << 23) | sign_x
            }
            c = ps.isEmpty ? "\0" : ps.removeFirst()
          } else {
            // if 2 radix points, return NaN
            return NAN_MASK | sign_x
          }
        } else if ps.isEmpty {
          right_radix_leading_zeros = EXPONENT_BIAS - right_radix_leading_zeros
          if right_radix_leading_zeros < 0 {
            right_radix_leading_zeros = 0
          }
          return (UInt32(right_radix_leading_zeros) << 23) | sign_x
        }
      }
    }
    
    var ndigits = 0
    var dec_expon_scale = 0
    var midpoint = 0
    var rounded_up = 0
    var add_expon = 0
    var rounded = 0
    while (c >= "0" && c <= "9") || c == "." {
      if c == "." {
        if rdx_pt_enc {
          // return NaN
          return NAN_MASK | sign_x
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
        switch rnd_mode {
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
            if sign_x != 0 { coefficient_x+=1; rounded_up=1 }
          case BID_ROUNDING_UP:
            if sign_x == 0 { coefficient_x+=1; rounded_up=1 }
          case BID_ROUNDING_TIES_AWAY:
            if c >= "5" { coefficient_x+=1; rounded_up=1 }
          default: break
        }
        if coefficient_x == 10000000 {
          coefficient_x = 1000000
          add_expon = 1;
        }
        if c > "0" {
          rounded = 1;
        }
        add_expon += 1;
      } else { // ndigits > 8
        add_expon+=1
        if midpoint != 0 && c > "0" {
          coefficient_x+=1
          midpoint = 0;
          rounded_up = 1;
        }
        if c > "0" {
          rounded = 1;
        }
      }
      c = ps.isEmpty ? "\0" : ps.removeFirst()
    }
    
    add_expon -= dec_expon_scale + Int(right_radix_leading_zeros)
    
    if c == "\0" {
      if rounded != 0 {
        pfpsf.insert(.inexact)
      }
      return bid32(sign_x, add_expon+EXPONENT_BIAS, Word(coefficient_x),
                   .toNearestOrEven, &pfpsf)
    }
    
    if c != "e" {
      // return NaN
      return NAN_MASK | sign_x
    }
    c = ps.isEmpty ? "\0" : ps.removeFirst()
    let sgn_expon = (c == "-") ? 1 : 0
    var expon_x = 0
    if c == "-" || c == "+" {
      c = ps.isEmpty ? "\0" : ps.removeFirst()
    }
    if c == "\0" || c < "0" || c > "9" {
      // return NaN
      return NAN_MASK | sign_x
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
      return NAN_MASK | sign_x
    }
    
    if rounded != 0 {
      pfpsf.insert(.inexact)
    }
    
    if sgn_expon != 0 {
      expon_x = -expon_x
    }
    
    expon_x += add_expon + EXPONENT_BIAS
    
    if expon_x < 0 {
      if rounded_up != 0 {
        coefficient_x-=1
      }
      return get_BID32_UF(sign_x, expon_x, UInt64(coefficient_x), rounded,
        .toNearestOrEven, &pfpsf)
    }
    return bid32(sign_x, expon_x, Word(coefficient_x), rnd_mode, &pfpsf)
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
  static func roundboundIndex(_ round:Rounding, _ negative:Bool=false,
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
}

extension UInt128 {
  
  public var high: UInt64 { UInt64(self.words[1]) }
  public var low:  UInt64 { UInt64(self.words[0]) }
  
  init(w: [UInt64]) { self.init(high: w[1], low: w[0]) }
  
}
