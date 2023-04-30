//
//  Decimal32.swift
//  
//
//  Created by Mike Griebling on 2022-03-07.
//

import UInt128


/// Implementation of the 32-bit Decimal32 floating-point operations from IEEE STD
/// 754-2000 for Floating-Point Arithmetic.
///
/// The IEEE Standard 754-2008 for Floating-Point Arithmetic supports two
/// encoding formats: the decimal encoding format, and the binary encoding format.
/// The Intel(R) Decimal Floating-Point Math Library supports primarily the binary
/// encoding format for decimal floating-point values, but the decimal encoding
/// format is supported too in the library, by means of conversion functions
/// between the two encoding formats.

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
  // MARK: - 32-bit Binary Integer Decimal (BID32) definitions
  
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
  // MARK: - Class State variables
  static public var state = Status.clearFlags
  static public var rounding = FloatingPointRoundingRule.toNearestOrEven
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Class State constants
  public static let zero = Decimal32(raw: bid32(0,EXPONENT_BIAS,0) )
  public static let radix = 10
  public static let pi = Decimal32(floatLiteral: Double.pi)
  public static let nan = Decimal32(raw: bid32(0, 0x1F<<3, 0) )
  public static let quietNaN = Decimal32(raw: bid32(0, 0x1F<<3, 0) )
  public static let signalingNaN = Decimal32(raw: SNAN_MASK)
  public static let infinity = Decimal32(raw: bid32(0,0xF<<4,0) )
  
  public static var greatestFiniteMagnitude: Decimal32 {
    Decimal32(raw: bid32(0, MAX_EXPON, MAX_NUMBER))
  }
  public static var leastNormalMagnitude: Decimal32 {
    Decimal32(raw: bid32(0, 0, 1_000_000))
  }
  public static var leastNonzeroMagnitude: Decimal32 {
    Decimal32(raw: bid32(0, 0, 1))
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers
  init(raw: UInt32) { x = raw } // only for internal use
  
  private func showState() {
    if Decimal32.enableStateOutput && !Decimal32.state.isEmpty {
      print("Warning: \(Decimal32.state)")
    }
    // Decimal32.state = .clearFlags
  }
  
  /// Binary Integer Decimal (BID) encoded 32-bit number
  public init(bid32: Word) { x = bid32 }
  
  /// Densely Packed Decimal encoded 32-bit number
  public init(dpd32: Word) { self.init() /* x = Decimal32.dpd_to_bid32(dpd32); showState() */ }
  
  public init(integerLiteral value: Int) {
    self = Decimal32.int64_to_BID32(Int64(value), Decimal32.rounding, &Decimal32.state)
  }
  
  //    public init(_ value: Decimal64) {
  //        x = Decimal64.bid64_to_bid32(value.x, Decimal32.rounding, &Decimal32.state)
  //        showState()
  //    }
  //
  //    public init(_ value: Decimal128) {
  //        x = Decimal128.bid128_to_bid32(value.x, Decimal32.rounding, &Decimal32.state)
  //        showState()
  //    }
  
  public init(_ value: Int = 0) { self.init(integerLiteral: value) }
  public init<Source>(_ value: Source) where Source : BinaryInteger { self.init(Int(value)) }
  
  public init?<T>(exactly source: T) where T : BinaryInteger {
    self.init(Int(source))  // FIX ME
    //        showState()
  }
  
  public init(floatLiteral value: Double) {
    x = Decimal32.double_to_bid32(value, Decimal32.rounding, &Decimal32.state)
  }
  
  public init(stringLiteral value: String) {
    if value.hasPrefix("0x") {
      var s = value; s.removeFirst(2)
      let n = Word(s, radix: 16) ?? 0
      x = n
    } else {
      x = Self.bid32_from_string(value, Decimal32.rounding, &Decimal32.state)
    }
  }
  
  public init(sign: FloatingPointSign, exponentBitPattern: UInt32, significandDigits: [UInt8]) {
    let mantissa = significandDigits.reduce(into: 0) { $0 = $0 * 10 + Int($1) }
    self.init(sign: sign, exponent: Int(exponentBitPattern), significand: Decimal32(mantissa))
  }
  
  public init(sign: FloatingPointSign, exponent: Int, significand: Decimal32) {
    let sgn = sign == .minus ? Decimal32.SIGN_MASK : 0
    self.init()
    if let s = Decimal32.unpack(bid32: significand.x) {
      x = Decimal32.get_BID32_UF(sgn, exponent, UInt64(s.coeff), 0, Self.rounding, &Self.state)
    }
  }
  
  public init(signOf: Decimal32, magnitudeOf: Decimal32) {
    let sign = signOf.isSignMinus
    self = sign ? -magnitudeOf.magnitude : magnitudeOf.magnitude
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Custom String Convertible compliance
  public var description: String { Decimal32.bid32_to_string(x) }
  
}

extension Decimal32 : AdditiveArithmetic, Comparable, SignedNumeric,
                      Strideable, FloatingPoint {
  
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    //        let dec64 = Decimal64.bid32_to_bid64(x, &Decimal32.state)
    //        let res = Decimal64.bid64_round_integral_exact(dec64, rule, &Decimal32.state)
    //        x = Decimal64.bid64_to_bid32(res, rule, &Decimal32.state)
  }
  
  public mutating func formRemainder(dividingBy other: Decimal32) {
    //       x = Decimal32.bid32_rem(self.x, other.x, &Decimal32.state)
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Decimal32) {
    let q = (self/other).rounded(.towardZero)
    self -= q * other
  }
  
  public mutating func formSquareRoot() {
    /* x = Decimal32.sqrt(x, Decimal32.rounding, &Decimal32.state) */
    
  }
  
  public mutating func addProduct(_ lhs: Decimal32, _ rhs: Decimal32) {
    // x = Decimal32.bid32_fma(lhs.x, rhs.x, self.x, Decimal32.rounding, &Decimal32.state)
  }
  
  public func distance(to other: Decimal32) -> Decimal32 { other - self }
  public func advanced(by n: Decimal32) -> Decimal32 { self + n }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Basic arithmetic operations
  
  public func isEqual(to other: Decimal32) -> Bool { self == other }
  public func isLess(than other: Decimal32) -> Bool { self < other }
  public func isLessThanOrEqualTo(_ other: Decimal32) -> Bool {
    self < other || self == other
  }
  
  public static func == (lhs: Self, rhs: Self) -> Bool {
    Self.equal(lhs, rhs, &Self.state)
  }
  
  public static func < (lhs: Self, rhs: Self) -> Bool {
    Self.lessThan(lhs, rhs, &Self.state)
  }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    lhs // Decimal32(raw: Decimal32.add(lhs.x, rhs.x, Decimal32.rounding, &Decimal32.state))
  }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    lhs // Decimal32(raw: Decimal32.div(lhs.x, rhs.x, Decimal32.rounding, &Decimal32.state))
  }
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    lhs // Decimal32(raw: Decimal32.mul(lhs.x, rhs.x, Decimal32.rounding, &Decimal32.state))
  }
  
  public static func /= (lhs: inout Decimal32, rhs: Decimal32) { lhs = lhs / rhs }
  public static func *= (lhs: inout Decimal32, rhs: Decimal32) { lhs = lhs * rhs }
  public static func - (lhs: Decimal32, rhs: Decimal32) -> Decimal32 { lhs + (-rhs) }
  
}

public extension Decimal32 {
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Numeric State variables
  var sign: FloatingPointSign { x & Decimal32.SIGN_MASK != 0 ? .minus : .plus }
  var magnitude: Decimal32    { Decimal32(raw: x & ~Decimal32.SIGN_MASK) }
  //    var decimal64: Decimal64    { Decimal64(raw: Decimal64.bid32_to_bid64(x, &Decimal32.state)) }
  //    var decimal128: Decimal128  { Decimal128(raw: Decimal128.bid32_to_bid128(x, &Decimal32.state)) }
  var dpd32: Word             { 0 /* Decimal32.bid_to_dpd32(x) */ }
  var int: Int                { Decimal32.bid32_to_int(x, Decimal32.rounding, &Decimal32.state) }
  var uint: UInt              { 0 /* Decimal32.bid32_to_uint(x, Decimal32.rounding, &Decimal32.state) */ }
  var double: Double          { 0 /* Decimal32.bid32_to_double(x, Decimal32.rounding, &Decimal32.state) */ }
  var isZero: Bool            { _isZero }
  var isSignMinus: Bool       { sign == .minus }
  var isInfinite: Bool        { ((x & Decimal32.INFINITY_MASK) == Decimal32.INFINITY_MASK) && !isNaN }
  var isNaN: Bool             { (x & Decimal32.NAN_MASK) == Decimal32.NAN_MASK }
  var isSignalingNaN: Bool    { (x & Decimal32.SNAN_MASK) == Decimal32.SNAN_MASK }
  var isFinite: Bool          { (x & Decimal32.INFINITY_MASK) != Decimal32.INFINITY_MASK }
  var isNormal: Bool          { _isNormal }
  var isSubnormal: Bool       { _isSubnormal }
  var isCanonical: Bool       { _isCanonical }
  var isBIDFormat: Bool       { true }
  var ulp: Decimal32          { self } // nextUp - self }
  var nextUp: Decimal32       { self } // Decimal32(raw: Decimal32.bid32_nextup(x, &Decimal32.state)) }
  
  mutating func negate()      { self.x = x ^ Decimal32.SIGN_MASK }
  
  private func unpack() -> (negative: Bool, exp: Int, coeff: UInt32)? {
    guard let s = Decimal32.unpack(bid32: x) else { return nil }
    return s
  }
  
  var significand: Decimal32 {
    let /* exp = 0, */ m = Word()
    //        Decimal32.frexp(x, &m, &exp)
    return Decimal32(raw: m)
  }
  
  //    var decimal: Decimal {
  //        // Not optimized but should be ok since this is rarely used -- feel free to fix me
  //        Decimal(string: self.description) ?? Decimal()
  //    }
  
  var exponent: Int {
    let exp = 0 //, m = Word()
    //        Decimal32.frexp(x, &m, &exp)
    return exp
  }
  
  private var _isZero: Bool {
    if (x & Decimal32.INFINITY_MASK) == Decimal32.INFINITY_MASK { return false }
    if (Decimal32.STEERING_BITS_MASK & x) == Decimal32.STEERING_BITS_MASK {
      return ((x & Decimal32.SMALL_COEFF_MASK) | Decimal32.LARGE_COEFF_HIGH_BIT) > Decimal32.MAX_NUMBER
    } else {
      return (x & Decimal32.LARGE_COEFF_MASK) == 0
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
    } else if (x & Decimal32.INFINITY_MASK) == Decimal32.INFINITY_MASK {
      return (x & 0x03ffffff) == 0
    } else if (x & Decimal32.STEERING_BITS_MASK) == Decimal32.STEERING_BITS_MASK{ // 24-bit
      return ((x & Decimal32.SMALL_COEFF_MASK) | Decimal32.LARGE_COEFF_HIGH_BIT) <= Decimal32.MAX_NUMBER
    } else { // 23-bit coeff.
      return true
    }
  }
  
  static private func validDecode(_ x: Word) -> (exp:Int, sig:Word)? {
    let exp_x:Int
    let sig_x:Word
    if (x & INFINITY_MASK) == INFINITY_MASK { return nil }
    if (x & STEERING_BITS_MASK) == STEERING_BITS_MASK {
      sig_x = (x & SMALL_COEFF_MASK) | LARGE_COEFF_HIGH_BIT
      // check for zero or non-canonical
      if sig_x > Decimal32.MAX_NUMBER || sig_x == 0 { return nil } // zero or non-canonical
      exp_x = Int((x & MASK_BINARY_EXPONENT2) >> 21)
    } else {
      sig_x = (x & LARGE_COEFF_MASK)
      if sig_x == 0 { return nil } // zero
      exp_x = Int((x & MASK_BINARY_EXPONENT1) >> 23)
    }
    return (exp_x, sig_x)
  }
  
  private var _isNormal: Bool {
    guard let result = Decimal32.validDecode(x) else { return false }
    
    // if exponent is less than -95, the number may be subnormal
    // if (exp_x - 101 = -95) the number may be subnormal
    if result.exp < 6 {
        let sig_x_prime = UInt64(result.sig) * UInt64(Decimal32.bid_mult_factor[result.exp])
        return sig_x_prime >= 1000000 // subnormal test
    } else {
        return true // normal
    }
  }
  
  private var _isSubnormal:Bool {
    guard let result = Decimal32.validDecode(x) else { return false }
    
    // if exponent is less than -95, the number may be subnormal
    // if (exp_x - 101 = -95) the number may be subnormal
    if result.exp < 6 {
        let sig_x_prime = UInt64(result.sig) * UInt64(Decimal32.bid_mult_factor[result.exp])
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
    return Decimal32.digitsIn(x.coeff)
  }
  
  public var exponentBitPattern: Word { Word(unpack()?.exp ?? 0) }
  
  public var significandDigits: [UInt8] {
    guard let x = unpack() else { return [] }
    return Array(String(x.coeff)).map { UInt8($0.wholeNumberValue!) }
  }
  
  public var decade: Decimal32 {
    let res = Word() // , exp = 0
    return Decimal32(raw: res)
    //        Decimal32.frexp(x, &res, &exp)
    //        return Decimal32(raw: return_bid32(0, exp+Decimal32.exponentBias, 1))
  }
  
}

extension Decimal32 {
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - BID32-based masking bits
  
  static let SIGN_MASK             = Word(1) << signBit   //0x8000_0000)
  static let SNAN_MASK             = Word(0x3f) << nanSignalBit //Word(0x7e00_0000)
  static let SSNAN_MASK            = SNAN_MASK | SIGN_MASK
  static let NAN_MASK              = Word(0x1f) << nanBit //Word(0x7c00_0000)
  static let INFINITY_MASK         = Word(0x0f) << infinityLowBit //Word(0x7800_0000)
  static let SINFINITY_MASK        = SIGN_MASK | INFINITY_MASK
  static let STEERING_BITS_MASK    = Word(3) << steeringLowBit //  Word(0x6000_0000)
  static let LARGE_COEFF_MASK      = LARGE_COEFF_HIGH_BIT - 1 // Word(0x007f_ffff)
  static let SMALL_COEFF_MASK      = (Word(1) << mantissaHighBit2) - 1 // Word(0x001f_ffff)
  static let LARGE_COEFF_HIGH_BIT  = Word(1) << mantissaHighBit1 // 0x0080_0000)
  static let MASK_BINARY_EXPONENT1 = Word.max ^ LARGE_COEFF_MASK // Word(0x7f80_0000)
  static let MASK_BINARY_EXPONENT2 = Word.max ^ SMALL_COEFF_MASK // Word(0x1fe0_0000)
  static let LARGEST_BID           = Word(0x77f8_967f)
  static let EXPONENT_MASK         = Word(0xff)
  
  /// Creates a 32-bit Binary Integer Decimal from `s`, the negative
  /// sign bit, `e`, the biased exponent, and `c`, the mantissa bits.
  /// There are two possible BID variants: one with a 23-bit mantissa and
  /// a second using 21 mantissa bits.
  static private func bid32(_ s:Int, _ e:Int, _ c:Int, _ R:Int, _ r:Rounding, _ fpsc: inout Status) -> Word {
    let sign = Word(s) << signBit
    var exp = Word(e)
    var man = Word(c), manHigh = Word(1) << mantissaHighBit1
    let steering = Word(3) << steeringLowBit
    
    // check for limits
    if man > MAX_NUMBER {
      exp += 1; man = 1_000_000
    }
    
    // check for overflow/underflow
    if exp > MAX_EXPON {
      return handleRounding(UInt32(s), Int(exp), Int(man), R, r, &fpsc)
    }
    
    if man < manHigh {
      return sign | (exp<<mantissaHighBit1) | man
    } else {
      return sign | steering | (exp<<mantissaHighBit2) | man
    }
  }
  
  static private func bid32(_ s:Int, _ e:Int, _ c:Int) -> Word {
    var status = Status.clearFlags
    return bid32(s, e, c, 0, .toNearestOrAwayFromZero, &status)
  }
  
  static private func unpack(bid32 x: Word) -> (negative: Bool, exp: Int, coeff: UInt32)? {
    let negative = (x & SIGN_MASK) == SIGN_MASK
    var coeff: UInt32
    var exp: Int
    if (x & STEERING_BITS_MASK) == STEERING_BITS_MASK {
      // special encodings
      if (x & INFINITY_MASK) == INFINITY_MASK {
        coeff = x & 0xfe0f_ffff
        if (x & 0x000f_ffff) >= 1_000_000 {
          coeff = x & SSNAN_MASK
        }
        if (x & NAN_MASK) == INFINITY_MASK {
          coeff = x & SINFINITY_MASK
        }
        exp = 0
        return nil    // NaN or Infinity
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
      return coeff != 0 ? (negative, exp, coeff) : nil
    }
    
    // exponent
    let tmp = x >> 23
    exp = Int(tmp & EXPONENT_MASK)
    
    // coefficient
    coeff = (x & LARGE_COEFF_MASK)
    return coeff != 0 ? (negative, exp, coeff) : nil
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
    if x == y {
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
    
    // if steering bits are 11 (condition will be 0), then exponent is G[0:w+1] =>
    //var exp_x, sig_x: UInt32; var non_canon_x: Bool
    var (exp_x, sig_x, non_canon_x) = extractExpSig(x.x)
    
    // if steering bits are 11 (condition will be 0), then exponent is G[0:w+1] =>
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
    if x == y {
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
    // if steering bits are 11 (condition will be 0), then exponent is G[0:w+1] =>
    let (exp_x, sig_x, non_canon_x) = extractExpSig(x.x)

    // if steering bits are 11 (condition will be 0), then exponent is G[0:w+1] =>
    let (exp_y, sig_y, non_canon_y) = extractExpSig(y.x)

    // ZERO (CASE4)
    // some properties:
    // (+ZERO==-ZERO) => therefore ignore the sign, and neither number is greater
    // (ZERO x 10^A == ZERO x 10^B) for any valid A, B =>
    //  therefore ignore the exponent field
    //    (Any non-canonical # is considered 0)
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
      __mul_64x64_to_128(&sig_n_prime, UInt64(sig_x), bid_mult_factor[exp_x - exp_y]);
      // return 0 if values are equal
      if (sig_n_prime.high == 0 && (sig_n_prime.low == sig_y)) {
        return false
      }
      // if postitive, return whichever significand abs is smaller
      // (converse if negative)
      return (((sig_n_prime.high == 0) && sig_n_prime.low < sig_y) != x.isSignMinus)
    }
    // adjust the y significand upwards
    __mul_64x64_to_128(&sig_n_prime, UInt64(sig_y), bid_mult_factor[exp_y - exp_x]);
    // return 0 if values are equal
    if (sig_n_prime.high == 0 && (sig_n_prime.low == sig_x)) {
      return false
    }
    // if positive, return whichever significand abs is smaller
    // (converse if negative)
    return (((sig_n_prime.high > 0) || (sig_x < sig_n_prime.low)) != x.isSignMinus)
  }
  
  fileprivate static func extractExpSig(_ x: UInt32) -> (exp: Int, sig: UInt32, non_canon: Bool) {
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
  
  fileprivate static func extractExpSig(_ x: UInt32) -> (exp: Int, sig: UInt32) {
    let (exp, sig, nonCanon) = extractExpSig(x)
    if nonCanon { return (0,0) }
    return (exp, sig)
  }
  
  //////////////////////////////////////////////////////////////////////////
  // MARK: - Conversions
  
  static func int64_to_BID32 (_ value:Int64, _ rnd_mode:Rounding, _ state: inout Status) -> Decimal32 {
    // Dealing with 64-bit integer
    let x_sign32 = (value < 0 ? 1 : 0)
    let C = UInt64(value.magnitude) // if the integer is negative, use the absolute value
    
    var res: UInt32
    if C <= UInt64(MAX_NUMBER) { // |C| <= 10^7-1 and the result is exact
      res = bid32(x_sign32, EXPONENT_BIAS, Int(C))
    } else { // |C| >= 10^7 and the result may be inexact
      // the smallest |C| is 10^7 which has 8 decimal digits
      // the largest |C| is SIGN_MASK64 = 9223372036854775808 w/ 19 digits
      var q, ind : Int
      switch C {
        case 0..<100_000_000:               q =  8; ind = 1  // number of digits to remove for q = 8
        case  ..<1_000_000_000:             q =  9; ind = 2  // number of digits to remove for q = 9
        case  ..<10_000_000_000:            q = 10; ind = 3  // number of digits to remove for q = 10
        case  ..<100_000_000_000:           q = 11; ind = 4  // number of digits to remove for q = 11
        case  ..<1_000_000_000_000:         q = 12; ind = 5  // number of digits to remove for q = 12
        case  ..<10_000_000_000_000:        q = 13; ind = 6  // number of digits to remove for q = 13
        case  ..<100_000_000_000_000:       q = 14; ind = 7  // number of digits to remove for q = 14
        case  ..<1_000_000_000_000_000:     q = 15; ind = 8  // number of digits to remove for q = 11
        case  ..<10_000_000_000_000_000:    q = 16; ind = 9  // number of digits to remove for q = 12
        case  ..<100_000_000_000_000_000:   q = 17; ind = 10 // number of digits to remove for q = 13
        case  ..<1_000_000_000_000_000_000: q = 18; ind = 11 // number of digits to remove for q = 14
        default:                            q = 19; ind = 12 // number of digits to remove for q = 19
      }
      
      // overflow and underflow are not possible
      // Note: performance can be improved by inlining this call
      var is_midpoint_lt_even = false, is_midpoint_gt_even = false, is_inexact_lt_midpoint = false
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
      if is_inexact_lt_midpoint || is_inexact_gt_midpoint || is_midpoint_lt_even || is_midpoint_gt_even {
        state.insert(.inexact)
      }
      // general correction from RN to RA, RM, RP, RZ; result uses ind for exp
      if rnd_mode != .toNearestOrAwayFromZero {
        let x_sign = value < 0
        if ((!x_sign && ((rnd_mode == .up && is_inexact_lt_midpoint) ||
                         ((rnd_mode == .toNearestOrEven || rnd_mode == .up) && is_midpoint_gt_even))) ||
            (x_sign && ((rnd_mode == .down && is_inexact_lt_midpoint) ||
                        ((rnd_mode == .toNearestOrEven || rnd_mode == .down) && is_midpoint_gt_even)))) {
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
    return Decimal32(raw: res)
  }
  
  static func double_to_bid32 (_ x:Double, _ rnd_mode:Rounding, _ state: inout Status) -> UInt32 {
      // Unpack the input
      var s = 0, e = 0, t = 0
      var low = UInt64(), high = UInt64()
      if let res = unpack_binary64 (x, &s, &e, &low, &t, &state) { return UInt32(res) }
      
      // Now -1126<=e<=971 (971 for max normal, -1074 for min normal, -1126 for min denormal)
      
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
      // This could be intercepted later, but it's convenient to keep tables smaller
      if e >= 211 {
          state.formUnion([.overflow, .inexact])
          return bid32_ovf(s)
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
          var cint:UInt128
          let a = -(e + t)
          cint = c
          if a <= 0 {
              cint = srl128(cint.high, cint.low, -e)
              if ((cint.high == 0) && (cint.low < MAX_NUMBERP1)) {
                  return bid32(s, EXPONENT_BIAS, Int(cint.low))
              }
          } else if a <= 48 {
              var pow5 = bid_coefflimits_bid32[a]
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
      var z:UInt384=UInt384()
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
      if lt128(bid_roundbound_128[ind].high, bid_roundbound_128[ind].low, z.w[4], z.w[3]) {
          c_prov += 1
          if c_prov == MAX_NUMBERP1 {
              c_prov = 1_000_000
              e_out += 1
          } else if c_prov == 1_000_000 && e_out == 0 {
              let ind = roundboundIndex(rnd_mode, false, 0) >> 2
              if ((((ind & 3) == 0) && (z.w[4] <= 17524406870024074035)) ||
                  ((ind + (s & 1) == 2) && (z.w[4] <= 16602069666338596454))) {
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
      return bid32(s, e_out, Int(c_prov))
  }
  
  // 128x256->384 bit multiplication (missing from existing macros)
  // I derived this by propagating (A).w[2] = 0 in __mul_192x256_to_448
  static func __mul_128x256_to_384(_  P: inout UInt384, _ A:UInt128, _ B:UInt256) {
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
  
  static func __mul_128x128_low(_ Ql: inout UInt128, _ A:UInt128, _ B:UInt128) {
    var ALBL = UInt128()
    __mul_64x64_to_128(&ALBL, A.low, B.low)
    let QM64 = B.low*A.high + A.low*B.high
    Ql = UInt128(high: QM64 + ALBL.high, low: ALBL.low)
  }
  
  @inlinable static func __add_carry_in_out(_ S: inout UInt64,
      _ CY: inout UInt64, _ X:UInt64, _ Y:UInt64, _ CI: UInt64) {
      let X1 = X + CI
      S = X1 &+ Y
      CY = ((S<X1) || (X1<CI)) ? 1 : 0
  }
  
  static func srl128(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
      if c == 0 { return UInt128(w: [lo, hi]) }
      if c >= 64 { return UInt128(w: [hi >> (c - 64), 0]) }
      else { return srl128_short(hi, lo, c) }
  }
  
  // Shift 2-part 2^64 * hi + lo right by "c" bits
  // The "short" form requires a shift 0 < c < 64 and will be faster
  // Note that shifts of 64 can't be relied on as ANSI
  static func srl128_short(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
      UInt128(w: [(hi << (64 - c)) + (lo >> c), hi >> c])
  }
  
  // Shift 2-part 2^64 * hi + lo left by "c" bits
  // The "short" form requires a shift 0 < c < 64 and will be faster
  // Note that shifts of 64 can't be relied on as ANSI

  static func sll128_short(_ hi:UInt64, _ lo:UInt64, _ c:Int) -> UInt128 {
      UInt128(w: [lo << c, (hi << c) + (lo>>(64-c))])
  }
  
  // Compare "<" two 2-part unsigned integers
  @inlinable static func lt128(_ x_hi:UInt64, _ x_lo:UInt64, _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (((x_hi) < (y_hi)) || (((x_hi) == (y_hi)) && ((x_lo) < (y_lo))))
  }

  // Likewise "<="
  @inlinable static func le128(_ x_hi:UInt64, _ x_lo:UInt64, _ y_hi:UInt64, _ y_lo:UInt64) -> Bool {
    (((x_hi) < (y_hi)) || (((x_hi) == (y_hi)) && ((x_lo) <= (y_lo))))
  }
  
  static func bid32_ovf(_ s:Int) -> UInt32 {
      let rnd_mode = Decimal32.rounding
      if ((rnd_mode == BID_ROUNDING_TO_ZERO) || (rnd_mode==(s != 0 ? BID_ROUNDING_UP : BID_ROUNDING_DOWN))) {
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
  
  static func unpack_binary64(_ x:Double, _ s: inout Int, _ e: inout Int, _ c: inout UInt64, _ t: inout Int, _ status: inout Status) -> UInt32? {
    let expMask = 1<<11 - 1
    e = Int(x.bitPattern >> 52) & expMask
    c = x.significandBitPattern
    s = x.sign == .minus ? 1 : 0
    if e == 0 {
      if c == 0 { return bid32_zero(s) } // number = 0
      
      // denormalized number
      let l = clz64(c) - (64 - 53)
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
  @inlinable static func clz64(_ n:UInt64) -> Int { n.leadingZeroBitCount }
  
  /*****************************************************************************
   *  BID32_to_int64_int
   ****************************************************************************/
  static func bid32_to_int (_ x: Word, _ rmode:Rounding, _ pfpsc: inout Status) -> Int {
    var res: Int = 0
    
    // check for NaN or Infinity and unpack `x`
    guard let (x_negative, x_exp, C1) = unpack(bid32: x)
    else { pfpsc.insert(.invalidOperation); return Int.min }
    
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
        
        __mul_64x64_to_128(&C, UInt64(C1), bid_ten2k64[20 - q]);
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
        __mul_64x64_to_128(&C, UInt64(C1), bid_ten2k64[20 - q])
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
        // kx = 10^(-x) = bid_ten2mk64[ind - 1]
        // C* = C1 * 10^(-x)
        // the approximation of 10^(-x) was rounded up to 54 bits
        var P128 = UInt128()
        __mul_64x64_to_128(&P128, UInt64(C1), bid_ten2mk64[ind - 1])
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
          res = -Int(UInt64(C1) * bid_ten2k64[exp])
        } else {
          res = Int(UInt64(C1) * bid_ten2k64[exp])
        }
      }
    }
    return res
  }
  
  // 64x64-bit product
  static func __mul_64x64_to_128(_ P128: inout UInt128, _ CX:UInt64, _ CY:UInt64)  {
    let r = CX.multipliedFullWidth(by: CY)
    P128 = UInt128(high: r.high, low: r.low)
  }
  
  static func digitsIn(_ sig_x: UInt32) -> Int {
    let tmp = Float(sig_x) // exact conversion
    let x_nr_bits = 1 + Int(((UInt(tmp.bitPattern >> 23)) & 0xff) - 0x7f)
    var q = Int(bid_nr_digits[x_nr_bits - 1].digits)
    if q == 0 {
      q = Int(bid_nr_digits[x_nr_bits - 1].digits1)
      if UInt64(sig_x) >= bid_nr_digits[x_nr_bits - 1].threshold_lo {
        q+=1
      }
    }
    return q
  }
  
  // the first entry of bid_nr_digits[i - 1] (where 1 <= i <= 113), indicates
  // the number of decimal digits needed to represent a binary number with i bits;
  // however, if a binary number of i bits may require either k or k + 1 decimal
  // digits, then the first entry of bid_nr_digits[i - 1] is 0; in this case if the
  // number is less than the value represented by the second and third entries
  // concatenated, then the number of decimal digits k is the fourth entry, else
  // the number of decimal digits is the fourth entry plus 1
  struct DEC_DIGITS {
    let digits: UInt
    let threshold_hi:UInt64
    let threshold_lo:UInt64
    let digits1: UInt
    
    init(_ d: UInt, _ hi: UInt64, _ lo: UInt64, _ d1: UInt) {
      digits = d; threshold_hi = hi; threshold_lo = lo; digits1 = d1
    }
  }
  
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
  
  static let bid_coefflimits_bid32 : [UInt128] = [
      UInt128(w: [10000000, 0]),
      UInt128(w: [2000000, 0]),
      UInt128(w: [400000, 0]),
      UInt128(w: [80000, 0]),
      UInt128(w: [16000, 0]),
      UInt128(w: [3200, 0]),
      UInt128(w: [640, 0]),
      UInt128(w: [128, 0]),
      UInt128(w: [25, 0]),
      UInt128(w: [5, 0]),
      UInt128(w: [1, 0]),
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
      UInt128(w: [0, 0])
  ]

  
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

  
  static let bid_exponents_bid32 = [
      -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       -1,
       0,
       0,
       0,
       0,
       1,
       1,
       1,
       2,
       2,
       2,
       3,
       3,
       3,
       3,
       4,
       4,
       4,
       5,
       5,
       5,
       6,
       6,
       6,
       7,
       7,
       7,
       7,
       8,
       8,
       8,
       9,
       9,
       9,
       10,
       10,
       10,
       10,
       11,
       11,
       11,
       12,
       12,
       12,
       13,
       13,
       13,
       13,
       14,
       14,
       14,
       15,
       15,
       15,
       16,
       16,
       16,
       16,
       17,
       17,
       17,
       18,
       18,
       18,
       19,
       19,
       19,
       19,
       20,
       20,
       20,
       21,
       21,
       21,
       22,
       22,
       22,
       22,
       23,
       23,
       23,
       24,
       24,
       24,
       25,
       25,
       25,
       25,
       26,
       26,
       26,
       27,
       27,
       27,
       28,
       28,
       28,
       28,
       29,
       29,
       29,
       30,
       30,
       30,
       31,
       31,
       31,
       31,
       32,
       32,
       32,
       33,
       33,
       33,
       34,
       34,
       34,
       34,
       35,
       35,
       35,
       36,
       36,
       36,
       37,
       37,
       37,
       38,
       38,
       38,
       38,
       39,
       39,
       39,
       40,
       40,
       40,
       41,
       41,
       41,
       41,
       42,
       42,
       42,
       43,
       43,
       43,
       44,
       44,
       44,
       44,
       45,
       45,
       45,
       46,
       46,
       46,
       47,
       47,
       47,
       47,
       48,
       48,
       48,
       49,
       49,
       49,
       50,
       50,
       50,
       50,
       51,
       51,
       51,
       52,
       52,
       52,
       53,
       53,
       53,
       53,
       54,
       54,
       54,
       55,
       55,
       55,
       56,
       56,
       56,
       56,
       57,
       57,
       57,
       58,
       58,
       58,
       59,
       59,
       59,
       59,
       60,
       60,
       60,
       61,
       61,
       61,
       62,
       62,
       62,
       62,
       63,
       63,
       63,
       64,
       64,
       64,
       65,
       65,
       65,
       66,
       66,
       66,
       66,
       67,
       67,
       67,
       68,
       68,
       68,
       69,
       69,
       69,
       69,
       70,
       70,
       70,
       71,
       71,
       71,
       72,
       72,
       72,
       72,
       73,
       73,
       73,
       74,
       74,
       74,
       75,
       75,
       75,
       75,
       76,
       76,
       76,
       77,
       77,
       77,
       78,
       78,
       78,
       78,
       79,
       79,
       79,
       80,
       80,
       80,
       81,
       81,
       81,
       81,
       82,
       82,
       82,
       83,
       83,
       83,
       84,
       84,
       84,
       84,
       85,
       85,
       85,
       86,
       86,
       86,
       87,
       87,
       87,
       87,
       88,
       88,
       88,
       89,
       89,
       89,
       90,
       90,
       90,
       90,
       91,
       91,
       91,
       92,
       92,
       92,
       93,
       93,
       93,
       93,
       94,
       94,
       94,
       95,
       95,
       95,
       96,
       96,
       96,
       97,
       97,
       97,
       97,
       98,
       98,
       98,
       99,
       99,
       99,
       100,
       100,
       100,
       100,
       101,
       101,
       101,
       102,
       102,
       102,
       103,
       103,
       103,
       103,
       104,
       104,
       104,
       105,
       105,
       105,
       106,
       106,
       106,
       106,
       107,
       107,
       107,
       108,
       108,
       108,
       109,
       109,
       109,
       109,
       110,
       110,
       110,
       111,
       111,
       111,
       112,
       112,
       112,
       112,
       113,
       113,
       113,
       114,
       114,
       114,
       115,
       115,
       115,
       115,
       116,
       116,
       116,
       117,
       117,
       117,
       118,
       118,
       118,
       118,
       119,
       119,
       119,
       120,
       120,
       120,
       121,
       121,
       121,
       121,
       122,
       122,
       122,
       123,
       123,
       123,
       124,
       124,
       124,
       125,
       125,
       125,
       125,
       126,
       126,
       126,
       127,
       127,
       127,
       128,
       128,
       128,
       128,
       129,
       129,
       129,
       130,
       130,
       130,
       131,
       131,
       131,
       131,
       132,
       132,
       132,
       133,
       133,
       133,
       134,
       134,
       134,
       134,
       135,
       135,
       135,
       136,
       136,
       136,
       137,
       137,
       137,
       137,
       138,
       138,
       138,
       139,
       139,
       139,
       140,
       140,
       140,
       140,
       141,
       141,
       141,
       142,
       142,
       142,
       143,
       143,
       143,
       143,
       144,
       144,
       144,
       145,
       145,
       145,
       146,
       146,
       146,
       146,
       147,
       147,
       147,
       148,
       148,
       148,
       149,
       149,
       149,
       149,
       150,
       150,
       150,
       151,
       151,
       151,
       152,
       152,
       152,
       153,
       153,
       153,
       153,
       154,
       154,
       154,
       155,
       155,
       155,
       156,
       156,
       156,
       156,
       157,
       157,
       157,
       158,
       158,
       158,
       159,
       159,
       159,
       159,
       160,
       160,
       160,
       161,
       161,
       161,
       162,
       162,
       162,
       162,
       163,
       163,
       163,
       164,
       164,
       164,
       165,
       165,
       165,
       165,
       166,
       166,
       166,
       167,
       167,
       167,
       168,
       168,
       168,
       168,
       169,
       169,
       169,
       170,
       170,
       170,
       171,
       171,
       171,
       171,
       172,
       172,
       172,
       173,
       173,
       173,
       174,
       174,
       174,
       174,
       175,
       175,
       175,
       176,
       176,
       176,
       177,
       177,
       177,
       177,
       178,
       178,
       178,
       179,
       179,
       179,
       180,
       180,
       180,
       180,
       181,
       181,
       181,
       182,
       182,
       182,
       183,
       183,
       183,
       184,
       184,
       184,
       184,
       185,
       185,
       185,
       186,
       186,
       186,
       187,
       187,
       187,
       187,
       188,
       188,
       188,
       189,
       189,
       189,
       190,
       190,
       190,
       190,
       191,
       191,
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
      UInt128(w: [0, (1 << 63)]),         // BID_ROUNDING_TO_NEAREST | positive | even
      UInt128(w: [~0, (1 << 63) - 1]),    // BID_ROUNDING_TO_NEAREST | positive | odd
      UInt128(w: [0, (1 << 63)]),         // BID_ROUNDING_TO_NEAREST | negative | even
      UInt128(w: [~0, (1 << 63) - 1]),    // BID_ROUNDING_TO_NEAREST | negative | odd
      
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_DOWN       | positive | even
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_DOWN       | positive | odd
      UInt128(w: [0, 0]),                 // BID_ROUNDING_DOWN       | negative | even
      UInt128(w: [0, 0]),                 // BID_ROUNDING_DOWN       | negative | odd
      
      UInt128(w: [0, 0]),                 // BID_ROUNDING_UP         | positive | even
      UInt128(w: [0, 0]),                 // BID_ROUNDING_UP         | positive | odd
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_UP         | negative | even
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_UP         | negative | odd
      
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_TO_ZERO    | positive | even
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_TO_ZERO    | positive | odd
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_TO_ZERO    | negative | even
      UInt128(w: [~0, ~0]),               // BID_ROUNDING_TO_ZERO    | negative | odd
      
      UInt128(w: [~0, (1 << 63) - 1]),    // BID_ROUNDING_TIES_AWAY  | positive | even
      UInt128(w: [~0, (1 << 63) - 1]),    // BID_ROUNDING_TIES_AWAY  | positive | odd
      UInt128(w: [~0, (1 << 63) - 1]),    // BID_ROUNDING_TIES_AWAY  | negative | even
      UInt128(w: [~0, (1 << 63) - 1])     // BID_ROUNDING_TIES_AWAY  | negative | odd
  ]
  
  static let bid_mult_factor: [UInt64] = [
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000,
    100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000
  ]
  
  // Values of 10^(-x) trancated to Ex bits beyond the binary point, and
  // in the right position to be compared with the fraction from C * kx,
  // 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the low 64 bits of the fraction)
  static let bid_ten2mxtrunc64 : [UInt64] = [
    0xcccccccccccccccc,    // (ten2mx >> 64) = cccccccccccccccc
    0xa3d70a3d70a3d70a,    // (ten2mx >> 64) = a3d70a3d70a3d70a
    0x83126e978d4fdf3b,    // (ten2mx >> 64) = 83126e978d4fdf3b
    0xd1b71758e219652b,    // (ten2mx >> 64) = d1b71758e219652b
    0xa7c5ac471b478423,    // (ten2mx >> 64) = a7c5ac471b478423
    0x8637bd05af6c69b5,    // (ten2mx >> 64) = 8637bd05af6c69b5
    0xd6bf94d5e57a42bc,    // (ten2mx >> 64) = d6bf94d5e57a42bc
    0xabcc77118461cefc,    // (ten2mx >> 64) = abcc77118461cefc
    0x89705f4136b4a597,    // (ten2mx >> 64) = 89705f4136b4a597
    0xdbe6fecebdedd5be,    // (ten2mx >> 64) = dbe6fecebdedd5be
    0xafebff0bcb24aafe,    // (ten2mx >> 64) = afebff0bcb24aafe
    0x8cbccc096f5088cb,    // (ten2mx >> 64) = 8cbccc096f5088cb
    0xe12e13424bb40e13,    // (ten2mx >> 64) = e12e13424bb40e13
    0xb424dc35095cd80f,    // (ten2mx >> 64) = b424dc35095cd80f
    0x901d7cf73ab0acd9,    // (ten2mx >> 64) = 901d7cf73ab0acd9
    0xe69594bec44de15b,    // (ten2mx >> 64) = e69594bec44de15b
    0xb877aa3236a4b449    // (ten2mx >> 64) = b877aa3236a4b449
  ]
  
  // Values of 1/2 in the right position to be compared with the fraction from
  // C * kx, 1 <= x <= 17; the fraction consists of the low Ex bits in C * kx
  // (these values are aligned with the high 64 bits of the fraction)
  static let bid_half64: [UInt64] = [
    0x0000000000000004,    // half / 2^64 = 4
    0x0000000000000020,    // half / 2^64 = 20
    0x0000000000000100,    // half / 2^64 = 100
    0x0000000000001000,    // half / 2^64 = 1000
    0x0000000000008000,    // half / 2^64 = 8000
    0x0000000000040000,    // half / 2^64 = 40000
    0x0000000000400000,    // half / 2^64 = 400000
    0x0000000002000000,    // half / 2^64 = 2000000
    0x0000000010000000,    // half / 2^64 = 10000000
    0x0000000100000000,    // half / 2^64 = 100000000
    0x0000000800000000,    // half / 2^64 = 800000000
    0x0000004000000000,    // half / 2^64 = 4000000000
    0x0000040000000000,    // half / 2^64 = 40000000000
    0x0000200000000000,    // half / 2^64 = 200000000000
    0x0001000000000000,    // half / 2^64 = 1000000000000
    0x0010000000000000,    // half / 2^64 = 10000000000000
    0x0080000000000000    // half / 2^64 = 80000000000000
  ]
  
  // Values of mask in the right position to obtain the high Ex - 64 bits
  // of the fraction from C * kx, 1 <= x <= 17; the fraction consists of
  // the low Ex bits in C * kx
  static let bid_mask64: [UInt64] = [
    0x0000000000000007,    // mask / 2^64
    0x000000000000003f,    // mask / 2^64
    0x00000000000001ff,    // mask / 2^64
    0x0000000000001fff,    // mask / 2^64
    0x000000000000ffff,    // mask / 2^64
    0x000000000007ffff,    // mask / 2^64
    0x00000000007fffff,    // mask / 2^64
    0x0000000003ffffff,    // mask / 2^64
    0x000000001fffffff,    // mask / 2^64
    0x00000001ffffffff,    // mask / 2^64
    0x0000000fffffffff,    // mask / 2^64
    0x0000007fffffffff,    // mask / 2^64
    0x000007ffffffffff,    // mask / 2^64
    0x00003fffffffffff,    // mask / 2^64
    0x0001ffffffffffff,    // mask / 2^64
    0x001fffffffffffff,    // mask / 2^64
    0x00ffffffffffffff    // mask / 2^64
  ]
  
  // Ex-64 from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static let bid_Ex64m64 : [UInt8] = [
    3,    // 67 - 64, Ex = 67
    6,    // 70 - 64, Ex = 70
    9,    // 73 - 64, Ex = 73
    13,    // 77 - 64, Ex = 77
    16,    // 80 - 64, Ex = 80
    19,    // 83 - 64, Ex = 83
    23,    // 87 - 64, Ex = 87
    26,    // 90 - 64, Ex = 90
    29,    // 93 - 64, Ex = 93
    33,    // 97 - 64, Ex = 97
    36,    // 100 - 64, Ex = 100
    39,    // 103 - 64, Ex = 103
    43,    // 107 - 64, Ex = 107
    46,    // 110 - 64, Ex = 110
    49,    // 113 - 64, Ex = 113
    53,    // 117 - 64, Ex = 117
    56    // 120 - 64, Ex = 120
  ]
  
  // bid_midpoint64[i - 1] = 1/2 * 10^i = 5 * 10^(i-1), 1 <= i <= 19
  static let bid_midpoint64 : [UInt64] = [
    0x0000000000000005,    // 1/2 * 10^1 = 5 * 10^0
    0x0000000000000032,    // 1/2 * 10^2 = 5 * 10^1
    0x00000000000001f4,    // 1/2 * 10^3 = 5 * 10^2
    0x0000000000001388,    // 1/2 * 10^4 = 5 * 10^3
    0x000000000000c350,    // 1/2 * 10^5 = 5 * 10^4
    0x000000000007a120,    // 1/2 * 10^6 = 5 * 10^5
    0x00000000004c4b40,    // 1/2 * 10^7 = 5 * 10^6
    0x0000000002faf080,    // 1/2 * 10^8 = 5 * 10^7
    0x000000001dcd6500,    // 1/2 * 10^9 = 5 * 10^8
    0x000000012a05f200,    // 1/2 * 10^10 = 5 * 10^9
    0x0000000ba43b7400,    // 1/2 * 10^11 = 5 * 10^10
    0x000000746a528800,    // 1/2 * 10^12 = 5 * 10^11
    0x0000048c27395000,    // 1/2 * 10^13 = 5 * 10^12
    0x00002d79883d2000,    // 1/2 * 10^14 = 5 * 10^13
    0x0001c6bf52634000,    // 1/2 * 10^15 = 5 * 10^14
    0x0011c37937e08000,    // 1/2 * 10^16 = 5 * 10^15
    0x00b1a2bc2ec50000,    // 1/2 * 10^17 = 5 * 10^16
    0x06f05b59d3b20000,    // 1/2 * 10^18 = 5 * 10^17
    0x4563918244f40000     // 1/2 * 10^19 = 5 * 10^18
  ]
  
  // Kx from 10^(-x) ~= Kx * 2^(-Ex); Kx rounded up to 64 bits, 1 <= x <= 17
  static let bid_Kx64: [UInt64] = [
    0xcccccccccccccccd,    // 10^-1 ~= cccccccccccccccd * 2^-67
    0xa3d70a3d70a3d70b,    // 10^-2 ~= a3d70a3d70a3d70b * 2^-70
    0x83126e978d4fdf3c,    // 10^-3 ~= 83126e978d4fdf3c * 2^-73
    0xd1b71758e219652c,    // 10^-4 ~= d1b71758e219652c * 2^-77
    0xa7c5ac471b478424,    // 10^-5 ~= a7c5ac471b478424 * 2^-80
    0x8637bd05af6c69b6,    // 10^-6 ~= 8637bd05af6c69b6 * 2^-83
    0xd6bf94d5e57a42bd,    // 10^-7 ~= d6bf94d5e57a42bd * 2^-87
    0xabcc77118461cefd,    // 10^-8 ~= abcc77118461cefd * 2^-90
    0x89705f4136b4a598,    // 10^-9 ~= 89705f4136b4a598 * 2^-93
    0xdbe6fecebdedd5bf,    // 10^-10 ~= dbe6fecebdedd5bf * 2^-97
    0xafebff0bcb24aaff,    // 10^-11 ~= afebff0bcb24aaff * 2^-100
    0x8cbccc096f5088cc,    // 10^-12 ~= 8cbccc096f5088cc * 2^-103
    0xe12e13424bb40e14,    // 10^-13 ~= e12e13424bb40e14 * 2^-107
    0xb424dc35095cd810,    // 10^-14 ~= b424dc35095cd810 * 2^-110
    0x901d7cf73ab0acda,    // 10^-15 ~= 901d7cf73ab0acda * 2^-113
    0xe69594bec44de15c,    // 10^-16 ~= e69594bec44de15c * 2^-117
    0xb877aa3236a4b44a     // 10^-17 ~= b877aa3236a4b44a * 2^-120
  ]
  
  
  static let bid_nr_digits : [DEC_DIGITS] = [   // only the first entry is used if it is not 0
    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   1-bit n < 10^1
    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   2-bit n < 10^1
    DEC_DIGITS(1, 0x0000000000000000, 0x000000000000000a, 1),    //   3-bit n < 10^1
    DEC_DIGITS(0, 0x0000000000000000, 0x000000000000000a, 1),    //   4-bit n ? 10^1
    DEC_DIGITS(2, 0x0000000000000000, 0x0000000000000064, 2),    //   5-bit n < 10^2
    DEC_DIGITS(2, 0x0000000000000000, 0x0000000000000064, 2),    //   6-bit n < 10^2
    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000000064, 2),    //   7-bit n ? 10^2
    DEC_DIGITS(3, 0x0000000000000000, 0x00000000000003e8, 3),    //   8-bit n < 10^3
    DEC_DIGITS(3, 0x0000000000000000, 0x00000000000003e8, 3),    //   9-bit n < 10^3
    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000003e8, 3),    //  10-bit n ? 10^3
    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  11-bit n < 10^4
    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  12-bit n < 10^4
    DEC_DIGITS(4, 0x0000000000000000, 0x0000000000002710, 4),    //  13-bit n < 10^4
    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000002710, 4),    //  14-bit n ? 10^4
    DEC_DIGITS(5, 0x0000000000000000, 0x00000000000186a0, 5),    //  15-bit n < 10^5
    DEC_DIGITS(5, 0x0000000000000000, 0x00000000000186a0, 5),    //  16-bit n < 10^5
    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000186a0, 5),    //  17-bit n ? 10^5
    DEC_DIGITS(6, 0x0000000000000000, 0x00000000000f4240, 6),    //  18-bit n < 10^6
    DEC_DIGITS(6, 0x0000000000000000, 0x00000000000f4240, 6),    //  19-bit n < 10^6
    DEC_DIGITS(0, 0x0000000000000000, 0x00000000000f4240, 6),    //  20-bit n ? 10^6
    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  21-bit n < 10^7
    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  22-bit n < 10^7
    DEC_DIGITS(7, 0x0000000000000000, 0x0000000000989680, 7),    //  23-bit n < 10^7
    DEC_DIGITS(0, 0x0000000000000000, 0x0000000000989680, 7),    //  24-bit n ? 10^7
    DEC_DIGITS(8, 0x0000000000000000, 0x0000000005f5e100, 8),    //  25-bit n < 10^8
    DEC_DIGITS(8, 0x0000000000000000, 0x0000000005f5e100, 8),    //  26-bit n < 10^8
    DEC_DIGITS(0, 0x0000000000000000, 0x0000000005f5e100, 8),    //  27-bit n ? 10^8
    DEC_DIGITS(9, 0x0000000000000000, 0x000000003b9aca00, 9),    //  28-bit n < 10^9
    DEC_DIGITS(9, 0x0000000000000000, 0x000000003b9aca00, 9),    //  29-bit n < 10^9
    DEC_DIGITS(0, 0x0000000000000000, 0x000000003b9aca00, 9),    //  30-bit n ? 10^9
    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  31-bit n < 10^10
    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  32-bit n < 10^10
    DEC_DIGITS(10, 0x0000000000000000, 0x00000002540be400, 10),    //  33-bit n < 10^10
    DEC_DIGITS(0, 0x0000000000000000, 0x00000002540be400, 10),    //  34-bit n ? 10^10
    DEC_DIGITS(11, 0x0000000000000000, 0x000000174876e800, 11),    //  35-bit n < 10^11
    DEC_DIGITS(11, 0x0000000000000000, 0x000000174876e800, 11),    //  36-bit n < 10^11
    DEC_DIGITS(0, 0x0000000000000000, 0x000000174876e800, 11),    //  37-bit n ? 10^11
    DEC_DIGITS(12, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  38-bit n < 10^12
    DEC_DIGITS(12, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  39-bit n < 10^12
    DEC_DIGITS(0, 0x0000000000000000, 0x000000e8d4a51000, 12),    //  40-bit n ? 10^12
    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  41-bit n < 10^13
    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  42-bit n < 10^13
    DEC_DIGITS(13, 0x0000000000000000, 0x000009184e72a000, 13),    //  43-bit n < 10^13
    DEC_DIGITS(0, 0x0000000000000000, 0x000009184e72a000, 13),    //  44-bit n ? 10^13
    DEC_DIGITS(14, 0x0000000000000000, 0x00005af3107a4000, 14),    //  45-bit n < 10^14
    DEC_DIGITS(14, 0x0000000000000000, 0x00005af3107a4000, 14),    //  46-bit n < 10^14
    DEC_DIGITS(0, 0x0000000000000000, 0x00005af3107a4000, 14),    //  47-bit n ? 10^14
    DEC_DIGITS(15, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  48-bit n < 10^15
    DEC_DIGITS(15, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  49-bit n < 10^15
    DEC_DIGITS(0, 0x0000000000000000, 0x00038d7ea4c68000, 15),    //  50-bit n ? 10^15
    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  51-bit n < 10^16
    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  52-bit n < 10^16
    DEC_DIGITS(16, 0x0000000000000000, 0x002386f26fc10000, 16),    //  53-bit n < 10^16
    DEC_DIGITS(0, 0x0000000000000000, 0x002386f26fc10000, 16),    //  54-bit n ? 10^16
    DEC_DIGITS(17, 0x0000000000000000, 0x016345785d8a0000, 17),    //  55-bit n < 10^17
    DEC_DIGITS(17, 0x0000000000000000, 0x016345785d8a0000, 17),    //  56-bit n < 10^17
    DEC_DIGITS(0, 0x0000000000000000, 0x016345785d8a0000, 17),    //  57-bit n ? 10^17
    DEC_DIGITS(18, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  58-bit n < 10^18
    DEC_DIGITS(18, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  59-bit n < 10^18
    DEC_DIGITS(0, 0x0000000000000000, 0x0de0b6b3a7640000, 18),    //  60-bit n ? 10^18
    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  61-bit n < 10^19
    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  62-bit n < 10^19
    DEC_DIGITS(19, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  63-bit n < 10^19
    DEC_DIGITS(0, 0x0000000000000000, 0x8ac7230489e80000, 19),    //  64-bit n ? 10^19
    DEC_DIGITS(20, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  65-bit n < 10^20
    DEC_DIGITS(20, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  66-bit n < 10^20
    DEC_DIGITS(0, 0x0000000000000005, 0x6bc75e2d63100000, 20),    //  67-bit n ? 10^20
    DEC_DIGITS(21, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  68-bit n < 10^21
    DEC_DIGITS(21, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  69-bit n < 10^21
    DEC_DIGITS(0, 0x0000000000000036, 0x35c9adc5dea00000, 21),    //  70-bit n ? 10^21
    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  71-bit n < 10^22
    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  72-bit n < 10^22
    DEC_DIGITS(22, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  73-bit n < 10^22
    DEC_DIGITS(0, 0x000000000000021e, 0x19e0c9bab2400000, 22),    //  74-bit n ? 10^22
    DEC_DIGITS(23, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  75-bit n < 10^23
    DEC_DIGITS(23, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  76-bit n < 10^23
    DEC_DIGITS(0, 0x000000000000152d, 0x02c7e14af6800000, 23),    //  77-bit n ? 10^23
    DEC_DIGITS(24, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  78-bit n < 10^24
    DEC_DIGITS(24, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  79-bit n < 10^24
    DEC_DIGITS(0, 0x000000000000d3c2, 0x1bcecceda1000000, 24),    //  80-bit n ? 10^24
    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  81-bit n < 10^25
    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  82-bit n < 10^25
    DEC_DIGITS(25, 0x0000000000084595, 0x161401484a000000, 25),    //  83-bit n < 10^25
    DEC_DIGITS(0, 0x0000000000084595, 0x161401484a000000, 25),    //  84-bit n ? 10^25
    DEC_DIGITS(26, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  85-bit n < 10^26
    DEC_DIGITS(26, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  86-bit n < 10^26
    DEC_DIGITS(0, 0x000000000052b7d2, 0xdcc80cd2e4000000, 26),    //  87-bit n ? 10^26
    DEC_DIGITS(27, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  88-bit n < 10^27
    DEC_DIGITS(27, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  89-bit n < 10^27
    DEC_DIGITS(0, 0x00000000033b2e3c, 0x9fd0803ce8000000, 27),    //  90-bit n ? 10^27
    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  91-bit n < 10^28
    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  92-bit n < 10^28
    DEC_DIGITS(28, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  93-bit n < 10^28
    DEC_DIGITS(0, 0x00000000204fce5e, 0x3e25026110000000, 28),    //  94-bit n ? 10^28
    DEC_DIGITS(29, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  95-bit n < 10^29
    DEC_DIGITS(29, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  96-bit n < 10^29
    DEC_DIGITS(0, 0x00000001431e0fae, 0x6d7217caa0000000, 29),    //  97-bit n ? 10^29
    DEC_DIGITS(30, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    //  98-bit n < 10^30
    DEC_DIGITS(30, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    //  99-bit n < 10^30
    DEC_DIGITS(0, 0x0000000c9f2c9cd0, 0x4674edea40000000, 30),    // 100-bit n ? 10^30
    DEC_DIGITS(31, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 101-bit n < 10^31
    DEC_DIGITS(31, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 102-bit n < 10^31
    DEC_DIGITS(0, 0x0000007e37be2022, 0xc0914b2680000000, 31),    // 103-bit n ? 10^31
    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 104-bit n < 10^32
    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 105-bit n < 10^32
    DEC_DIGITS(32, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 106-bit n < 10^32
    DEC_DIGITS(0, 0x000004ee2d6d415b, 0x85acef8100000000, 32),    // 107-bit n ? 10^32
    DEC_DIGITS(33, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 108-bit n < 10^33
    DEC_DIGITS(33, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 109-bit n < 10^33
    DEC_DIGITS(0, 0x0000314dc6448d93, 0x38c15b0a00000000, 33),    // 100-bit n ? 10^33
    DEC_DIGITS(34, 0x0001ed09bead87c0, 0x378d8e6400000000, 34),    // 111-bit n < 10^34
    DEC_DIGITS(34, 0x0001ed09bead87c0, 0x378d8e6400000000, 34),    // 112-bit n < 10^34
    DEC_DIGITS(0, 0x0001ed09bead87c0, 0x378d8e6400000000, 34)    // 113-bit n ? 10^34
    //{ 35, 0x0013426172c74d82, 0x2b878fe800000000, 35 }  // 114-bit n < 10^35
  ]
  
  // bid_ten2k64[i] = 10^i, 0 <= i <= 19
  static let bid_ten2k64 : [UInt64] = [
    0x0000000000000001,    // 10^0
    0x000000000000000a,    // 10^1
    0x0000000000000064,    // 10^2
    0x00000000000003e8,    // 10^3
    0x0000000000002710,    // 10^4
    0x00000000000186a0,    // 10^5
    0x00000000000f4240,    // 10^6
    0x0000000000989680,    // 10^7
    0x0000000005f5e100,    // 10^8
    0x000000003b9aca00,    // 10^9
    0x00000002540be400,    // 10^10
    0x000000174876e800,    // 10^11
    0x000000e8d4a51000,    // 10^12
    0x000009184e72a000,    // 10^13
    0x00005af3107a4000,    // 10^14
    0x00038d7ea4c68000,    // 10^15
    0x002386f26fc10000,    // 10^16
    0x016345785d8a0000,    // 10^17
    0x0de0b6b3a7640000,    // 10^18
    0x8ac7230489e80000    // 10^19 (20 digits)
  ]
  
  static let bid_ten2mk64 : [UInt64] = [
    0x199999999999999a,    //  10^(-1) * 2^ 64
    0x028f5c28f5c28f5d,    //  10^(-2) * 2^ 64
    0x004189374bc6a7f0,    //  10^(-3) * 2^ 64
    0x00346dc5d638865a,    //  10^(-4) * 2^ 67
    0x0029f16b11c6d1e2,    //  10^(-5) * 2^ 70
    0x00218def416bdb1b,    //  10^(-6) * 2^ 73
    0x0035afe535795e91,    //  10^(-7) * 2^ 77
    0x002af31dc4611874,    //  10^(-8) * 2^ 80
    0x00225c17d04dad2a,    //  10^(-9) * 2^ 83
    0x0036f9bfb3af7b76,    // 10^(-10) * 2^ 87
    0x002bfaffc2f2c92b,    // 10^(-11) * 2^ 90
    0x00232f33025bd423,    // 10^(-12) * 2^ 93
    0x00384b84d092ed04,    // 10^(-13) * 2^ 97
    0x002d09370d425737,    // 10^(-14) * 2^100
    0x0024075f3dceac2c,    // 10^(-15) * 2^103
    0x0039a5652fb11379,    // 10^(-16) * 2^107
  ]
  
  // bid_shiftright128[] contains the right shift count to obtain C2* from the top
  // 128 bits of the 128x128-bit product C2 * Kx
  static let bid_shiftright128: [Int] = [
    0,    // 128 - 128
    0,    // 128 - 128
    0,    // 128 - 128
    
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
    102    // 230 - 128
  ]
  
  static let bid_reciprocals10_64: [UInt64] = [
    1,    // dummy value for 0 extra digits
    0x3333333333333334,    // 1 extra digit
    0x51eb851eb851eb86,
    0x20c49ba5e353f7cf,
    0x346dc5d63886594b,
    0x29f16b11c6d1e109,
    0x218def416bdb1a6e,
    0x35afe535795e90b0,
    0x2af31dc4611873c0,
    0x225c17d04dad2966,
    0x36f9bfb3af7b7570,
    0x2bfaffc2f2c92ac0,
    0x232f33025bd42233,
    0x384b84d092ed0385,
    0x2d09370d42573604,
    0x24075f3dceac2b37,
    0x39a5652fb1137857,
    0x2e1dea8c8da92d13
  ]
  
  static let bid_short_recip_scale: [Int8] = [
    1,
    65 - 64,
    69 - 64,
    71 - 64,
    75 - 64,
    78 - 64,
    81 - 64,
    85 - 64,
    88 - 64,
    91 - 64,
    95 - 64,
    98 - 64,
    101 - 64,
    105 - 64,
    108 - 64,
    111 - 64,
    115 - 64,    //114 - 64
    118 - 64
  ]
  
  static let bid_round_const_table : [[UInt64]] = [
    [    // RN
      0,    // 0 extra digits
      5,    // 1 extra digits
      50,    // 2 extra digits
      500,    // 3 extra digits
      5000,    // 4 extra digits
      50000,    // 5 extra digits
      500000,    // 6 extra digits
      5000000,    // 7 extra digits
      50000000,    // 8 extra digits
      500000000,    // 9 extra digits
      5000000000,    // 10 extra digits
      50000000000,    // 11 extra digits
      500000000000,    // 12 extra digits
      5000000000000,    // 13 extra digits
      50000000000000,    // 14 extra digits
      500000000000000,    // 15 extra digits
      5000000000000000,    // 16 extra digits
      50000000000000000,    // 17 extra digits
      500000000000000000    // 18 extra digits
    ],
    [    // RD
      0,    // 0 extra digits
      0,    // 1 extra digits
      0,    // 2 extra digits
      00,    // 3 extra digits
      000,    // 4 extra digits
      0000,    // 5 extra digits
      00000,    // 6 extra digits
      000000,    // 7 extra digits
      0000000,    // 8 extra digits
      00000000,    // 9 extra digits
      000000000,    // 10 extra digits
      0000000000,    // 11 extra digits
      00000000000,    // 12 extra digits
      000000000000,    // 13 extra digits
      0000000000000,    // 14 extra digits
      00000000000000,    // 15 extra digits
      000000000000000,    // 16 extra digits
      0000000000000000,    // 17 extra digits
      00000000000000000    // 18 extra digits
    ],
    [    // round to Inf
      0,    // 0 extra digits
      9,    // 1 extra digits
      99,    // 2 extra digits
      999,    // 3 extra digits
      9999,    // 4 extra digits
      99999,    // 5 extra digits
      999999,    // 6 extra digits
      9999999,    // 7 extra digits
      99999999,    // 8 extra digits
      999999999,    // 9 extra digits
      9999999999,    // 10 extra digits
      99999999999,    // 11 extra digits
      999999999999,    // 12 extra digits
      9999999999999,    // 13 extra digits
      99999999999999,    // 14 extra digits
      999999999999999,    // 15 extra digits
      9999999999999999,    // 16 extra digits
      99999999999999999,    // 17 extra digits
      999999999999999999    // 18 extra digits
    ],
    [    // RZ
      0,    // 0 extra digits
      0,    // 1 extra digits
      0,    // 2 extra digits
      00,    // 3 extra digits
      000,    // 4 extra digits
      0000,    // 5 extra digits
      00000,    // 6 extra digits
      000000,    // 7 extra digits
      0000000,    // 8 extra digits
      00000000,    // 9 extra digits
      000000000,    // 10 extra digits
      0000000000,    // 11 extra digits
      00000000000,    // 12 extra digits
      000000000000,    // 13 extra digits
      0000000000000,    // 14 extra digits
      00000000000000,    // 15 extra digits
      000000000000000,    // 16 extra digits
      0000000000000000,    // 17 extra digits
      00000000000000000    // 18 extra digits
    ],
    [    // round ties away from 0
      0,    // 0 extra digits
      5,    // 1 extra digits
      50,    // 2 extra digits
      500,    // 3 extra digits
      5000,    // 4 extra digits
      50000,    // 5 extra digits
      500000,    // 6 extra digits
      5000000,    // 7 extra digits
      50000000,    // 8 extra digits
      500000000,    // 9 extra digits
      5000000000,    // 10 extra digits
      50000000000,    // 11 extra digits
      500000000000,    // 12 extra digits
      5000000000000,    // 13 extra digits
      50000000000000,    // 14 extra digits
      500000000000000,    // 15 extra digits
      5000000000000000,    // 16 extra digits
      50000000000000000,    // 17 extra digits
      500000000000000000    // 18 extra digits
    ]
  ]
  
  
  static func bid_round64_2_18 (_ q: Int, _ x:Int, _ C: UInt64, _ ptr_Cstar: inout UInt64, _ incr_exp: inout Int,
                                _ ptr_is_midpoint_lt_even: inout Bool, _ ptr_is_midpoint_gt_even: inout Bool,
                                _ ptr_is_inexact_lt_midpoint: inout Bool, _ ptr_is_inexact_gt_midpoint: inout Bool) {
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
    let C = C + bid_midpoint64[ind];
    // kx ~= 10^(-x), kx = bid_Kx64[ind] * 2^(-Ex), 0 <= ind <= 16
    // P128 = (C + 1/2 * 10^x) * kx * 2^Ex = (C + 1/2 * 10^x) * Kx
    // the approximation kx of 10^(-x) was rounded up to 64 bits
    var P128 = UInt128()
    __mul_64x64_to_128(&P128, C, bid_Kx64[ind]);
    // calculate C* = floor (P128) and f*
    // Cstar = P128 >> Ex
    // fstar = low Ex bits of P128
    let shift = bid_Ex64m64[ind]    // in [3, 56]
    var Cstar = P128.high >> shift
    let fstar = UInt128(high: P128.high & bid_mask64[ind], low: P128.low)
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
    if (fstar.high > bid_half64[ind] || (fstar.high == bid_half64[ind] && fstar.low != 0)) {
      // f* > 1/2 and the result may be exact
      // Calculate f* - 1/2
      let tmp64 = fstar.high - bid_half64[ind];
      if (tmp64 != 0 || fstar.low > bid_ten2mxtrunc64[ind]) {    // f* - 1/2 > 10^(-x)
        ptr_is_inexact_lt_midpoint = true
      }    // else the result is exact
    } else {    // the result is inexact; f2* <= 1/2
      ptr_is_inexact_gt_midpoint = true
    }
    // check for midpoints (could do this before determining inexactness)
    if (fstar.high == 0 && fstar.low <= bid_ten2mxtrunc64[ind]) {
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
    if (Cstar == bid_ten2k64[ind]) {    // if  Cstar = 10^(q-x)
      Cstar = bid_ten2k64[ind - 1];    // Cstar = 10^(q-x-1)
      incr_exp = 1;
    } else {    // 10^33 <= Cstar <= 10^34 - 1
      incr_exp = 0;
    }
    ptr_Cstar = Cstar;
  }
  
  static func bid32_to_string (_ x: UInt32, _ showPlus: Bool = false) -> String {
    
    func stripZeros(_ d: UInt64, _ addDecimal: Bool = false) -> String {
      var digs = getDigits(d)
      if digs.first! == "0" { digs.removeFirst() }
      if digs.first! == "0" && digs.count == 2 { digs.removeFirst() }
      return digs
    }
    
    func getDigits(_ n:UInt64) -> String { String(n) }
    
    // unpack arguments, check for NaN or Infinity
    let addDecimal = true
    let plus = showPlus ? "+" : ""
    
    if let (negative_x, exponent_x, coefficient_x) = unpack(bid32: x) {
      // x is not special
      var exponent_x = exponent_x
      var coefficient_x = coefficient_x
      var ps = ""
      if coefficient_x >= 1_000_000 {
        var CT = UInt64(coefficient_x) * 0x431B_DE83
        CT >>= 32
        var d = CT >> (50-32)
        
        // upper digit
        ps.append(String(d))
        coefficient_x -= UInt32(d * 1_000_000)
        
        // get lower 6 digits
        CT = UInt64(coefficient_x) * 0x20C4_9BA6
        CT >>= 32
        d = CT >> (39-32)
        ps += getDigits(d)
        d = UInt64(coefficient_x) - d * 1000
        ps += getDigits(d)
      } else if coefficient_x >= 1000 {
        // get 4 to 6 digits
        var CT = UInt64(coefficient_x) * 0x20C4_9BA6
        CT >>= 32
        var d = CT >> (39-32)
        ps += stripZeros(d, addDecimal)
        d = UInt64(coefficient_x) - d*1000
        ps += getDigits(d)
      } else {
        // get 1 to 3 digits
        ps += stripZeros(UInt64(coefficient_x), addDecimal)
      }
      
      exponent_x -= EXPONENT_BIAS - (ps.count - 1)
      return (negative_x ? "-" : plus) + addDecimalPointAndExponent(ps, exponent_x, MAX_DIGITS)
    } else {
      // x is Inf. or NaN or 0
      var ps = (x&SIGN_MASK) != 0 ? "-" : plus
      if (x&NAN_MASK) == NAN_MASK {
        if (x & SNAN_MASK) == SNAN_MASK { ps.append("S") }
        ps.append("NaN")
        return ps
      }
      if (x&INFINITY_MASK) == INFINITY_MASK {
        ps.append("Inf")
        return ps
      }
      ps.append("0")
      return ps
    }
  }
  
  static func addDecimalPointAndExponent(_ ps:String, _ exponent:Int, _ maxDigits:Int) -> String {
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
  
  static func bid32_from_string (_ ps: String, _ rnd_mode: Rounding, _ pfpsf: inout Status) -> UInt32 {
    // eliminate leading whitespace
    var ps = ps.trimmingCharacters(in: .whitespaces).lowercased()
    var res: UInt32
    
    // get first non-whitespace character
    var c = ps.isEmpty ? "\0" : ps.removeFirst()
    
    // detect special cases (INF or NaN)
    if c == "\0" || (c != "." && c != "-" && c != "+" && (c < "0" || c > "9")) {
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
      // if all numbers are zeros (with possibly 1 radix point, the number is zero
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
              right_radix_leading_zeros = EXPONENT_BIAS - right_radix_leading_zeros
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
          case .toNearestOrAwayFromZero:
            midpoint = (c == "5" && (coefficient_x & 1 == 0)) ? 1 : 0;
            // if coefficient is even and c is 5, prepare to round up if
            // subsequent digit is nonzero
            // if str[MAXDIG+1] > 5, we MUST round up
            // if str[MAXDIG+1] == 5 and coefficient is ODD, ROUND UP!
            if c > "5" || (c == "5" && (coefficient_x & 1) != 0) {
              coefficient_x+=1
              rounded_up = 1
            }
          case .down:
            if sign_x != 0 { coefficient_x+=1; rounded_up=1 }
          case .up:
            if sign_x == 0 { coefficient_x+=1; rounded_up=1 }
          case .awayFromZero:
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
      return bid32(Int(sign_x), add_expon+EXPONENT_BIAS, Int(coefficient_x), 0, .toNearestOrEven, &pfpsf)
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
      return bid32(Int(sign_x), expon_x, coefficient_x, rounded, .toNearestOrEven, &pfpsf)
    }
    return bid32(Int(sign_x), expon_x, coefficient_x, 0, rnd_mode, &pfpsf)
  }
  
  ///   General pack macro for BID32 with underflow resolution
  static func get_BID32_UF (_ sgn: UInt32, _ expon: Int, _ coeff: UInt64, _ R: Int, _ rmode:Rounding, _ fpsc: inout Status) -> UInt32 {
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
  
  
  static private func handleRounding(_ s:Word, _ exp:Int, _ c:Int, _ R: Int = 0, _ r:Rounding, _ fpsc: inout Status) -> Word {
    let sexp = exp-exponentBias
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
      let extra_digits = -sexp
      c += Int(bid_round_const_table[roundIndex][extra_digits])
      
      // get coeff*(2^M[extra_digits])/10^extra_digits
      var Q = UInt128()
      __mul_64x64_to_128(&Q, UInt64(c), bid_reciprocals10_64[extra_digits]);
      
      // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-128
      let amount = bid_short_recip_scale[extra_digits]
      
      var _C64 = Q.high >> amount
      var remainder_h = UInt64(0)
      
      if r == .toNearestOrAwayFromZero {
        if (_C64 & 1 != 0) {
          // check whether fractional part of initial_P/10^extra_digits is exactly .5
          
          // get remainder
          let amount2 = 64 - amount
          remainder_h = 0
          remainder_h &-= 1
          remainder_h >>= amount2
          remainder_h = remainder_h & Q.high
          
          if remainder_h == 0 && Q.low < bid_reciprocals10_64[extra_digits] {
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
            if (remainder_h == (UInt64(SIGN_MASK) << 32) && (Q.low < bid_reciprocals10_64[extra_digits])) {
              status = Status.clearFlags // BID_EXACT_STATUS;
            }
          case .down, .towardZero:
            if remainder_h == 0 && Q.low < bid_reciprocals10_64[extra_digits] {
              status = Status.clearFlags // BID_EXACT_STATUS;
            }
          default:
            // round up
            var Stemp = UInt64(0), carry = UInt64(0)
            __add_carry_out(&Stemp, &carry, Q.low, bid_reciprocals10_64[extra_digits]);
            if (remainder_h >> (64 - amount)) + carry >= UInt64(1) << amount {
              status = Status.clearFlags // BID_EXACT_STATUS;
            }
        }
        
        if !status.isEmpty {
          status.insert(.underflow)
          fpsc.formUnion(status)
        }
        
      }
      
      return (Word(s) << signBit) | UInt32(_C64)
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
  
  static func __add_carry_out(_ S: inout UInt64, _ CY: inout UInt64, _ X:UInt64, _ Y:UInt64) {
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
  static func roundboundIndex(_ round:Rounding, _ negative:Bool=false, _ lsb:Int=0) -> Int {
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
