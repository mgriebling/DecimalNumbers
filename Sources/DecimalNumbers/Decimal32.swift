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

/// Definition of the data storage for the Decimal32 floating-point data type.
/// the `IntegerDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal32 sign, exponent, and
/// mantissa fields.  By specifying some key bit positions, it is possible
/// to completely define many of the Decimal32 operations.  The `data` word
/// holds all 32 bits of the Decimal32 data type.
public struct IntegerDecimal32 : IntegerDecimal {
  public typealias RawDataFields = UInt32
  public typealias Mantissa = UInt
  
  public var data: RawDataFields = 0
  
  public init(_ word: RawDataFields) {
    self.data = word
  }
  
  public init(sign:FloatingPointSign = .plus, exponent:Int = 0,
              mantissa:Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  101 }
  public static var maximumExponent: Int {   90 } // unbiased
  public static var minimumExponent: Int { -101 } // unbiased
  public static var numberOfDigits:  Int {    7 }
  public static var exponentBits:    Int {    8 }
  
  public static var largestNumber: Mantissa { 9_999_999 }
  
  // Two mantissa sizes must be supported
  public static var exponentLMBits:    ClosedRange<Int> { 23...30 }
  public static var largeMantissaBits: ClosedRange<Int> {  0...22 }
  
  public static var exponentSMBits:    ClosedRange<Int> { 21...28 }
  public static var smallMantissaBits: ClosedRange<Int> {  0...20 }
}

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
  
  public typealias ID = IntegerDecimal32
  var bid: ID
  
  public init(bid: ID) {
    self.bid = bid
  }
  
  public init?(_ s: String) {
    if let n:ID = numberFromString(s, round: .toNearestOrEven) {
      bid = n
    }
    return nil
  }
  
  func add(_ y: Self, rounding: FloatingPointRoundingRule) -> Self {
    let xb = self.bid, yb = y.bid
    let (signX, exponentX, mantissaX, validX) = xb.unpack()
    let (signY, exponentY, mantissaY, validY) = yb.unpack()
    
    // Deal with illegal numbers
    if !validX {
      if xb.isNaN {
        if xb.isSNaN || yb.isSNaN { /* invalid Op */ }
        return Decimal32(bid: ID(mantissa:xb.nanQuiet()))
      }
      if xb.isInfinite {
        if yb.isNaNInf {
          if signX == signY {
            return Decimal32(bid: ID(mantissa: mantissaX)) }
          else {
            return y // invalid Op
          }
        }
        if yb.isNaN {
          if yb.isSNaN { /* invalid Op */ }
          return Decimal32(bid: ID(mantissa:yb.nanQuiet()))
        } else {
          // +/- infinity
          return self
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
        return Decimal32(bid: ID(mantissa:yb.nanQuiet()))
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
        return Decimal32(bid: ID(sign: sign, exponent: exp, mantissa: 0))
      } else if exponentY >= exponentX {
        return self
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
    if exponentDiff > ID.numberOfDigits {
      let binExpon = Double(mantissaA).exponent
      let scaleCA = ID.bid_estimate_decimal_digits(binExpon)
      let d2 = 16 - scaleCA
      if exponentDiff > d2 {
        exponentDiff = d2
        exponentB = exponentA - exponentDiff
      }
    }
    
    let signAB = signA != signB ? FloatingPointSign.minus : .plus
    let addIn = signAB == .minus ? Int64(1) : 0
    let CB = UInt64(bitPattern: (Int64(mantissaB) + addIn) ^ addIn)
    
    let SU = UInt64(mantissaA) *
              ID.bid_power10_table_128(exponentDiff).components.low
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
      n_digits = Int(ID.bid_estimate_decimal_digits(bin_expon))
      if P >= ID.bid_power10_table_128(n_digits).components.low {
        n_digits+=1
      }
    }
    
    if n_digits <= ID.numberOfDigits {
      return Decimal32(bid: ID(sign: signA, exponent: exponentB,
                               mantissa: ID.Mantissa(P)))
    }
    
    let extra_digits = n_digits - ID.numberOfDigits
    
    var irmode = roundboundIndex(rounding) >> 2
    if signA == .minus && (UInt(irmode) &- 1) < 2 {
      irmode = 3 - irmode
    }
    
    // add a constant to P, depending on rounding mode
    // 0.5*10^(digits_p - 16) for round-to-nearest
    P += ID.bid_round_const_table(irmode, extra_digits)
    //var Tmp = UInt128()
    let Tmp = P.multipliedFullWidth(by: ID.bid_reciprocals10_64(extra_digits))
    // __mul_64x64_to_128(&Tmp, P, bid_reciprocals10_64(extra_digits))
    
    // now get P/10^extra_digits: shift Q_high right by M[extra_digits]-64
    let amount = bid_short_recip_scale[extra_digits]
    var Q = Tmp.high >> amount
    
    // remainder
    let R = P - Q * ID.bid_power10_table_128(extra_digits).components.low
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
    return Decimal32(bid: ID(sign: signA, exponent: exponentB+extra_digits,
                             mantissa: ID.Mantissa(Q)))
  }
}

extension Decimal32 : AdditiveArithmetic {
  public static func - (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    var addIn = rhs
    addIn.negate()
    return lhs + addIn
  }
  
  public mutating func negate() {
    bid.sign = bid.sign == .minus ? FloatingPointSign.plus : .minus
  }
  
  public static func + (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    lhs.add(rhs, rounding: .toNearestOrEven)
  }
  
  public static var zero: Decimal32 { Self(bid: ID.zero) }
}

extension Decimal32 : Equatable {
  public static func == (lhs: Decimal32, rhs: Decimal32) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    
    // all data bits equsl case
    if lhs.bid.data == rhs.bid.data { return true }
    
    // infinity cases
    if lhs.isInfinite && rhs.isInfinite { return lhs.sign == rhs.sign }
    if lhs.isInfinite || rhs.isInfinite { return false }
    
    // zero cases
    let xisZero = lhs.isZero, yisZero = rhs.isZero
    if xisZero && yisZero { return true }
    if (xisZero && !yisZero) || (!xisZero && yisZero) { return false }
    
    // normal numbers
    var (xsign, xexp, xman, _) = lhs.bid.unpack()
    var (ysign, yexp, yman, _) = rhs.bid.unpack()
    
    // opposite signs
    if xsign != ysign { return false }
    
    // redundant representations
    if xexp > yexp {
      swap(&xexp, &yexp)
      swap(&xman, &yman)
    }
    if yexp - xexp > ID.numberOfDigits-1 { return false }
    for _ in 0..<(yexp - xexp) {
      // recalculate y's significand upwards
      yman *= 10
      if yman > ID.largestNumber { return false }
    }
    return xman == yman
  }
}

extension Decimal32 : Comparable {
  public static func < (lhs: Decimal32, rhs: Decimal32) -> Bool {
    guard !lhs.isNaN && !rhs.isNaN else { return false }
    
    // all data bits equsl case
    if lhs.bid.data == rhs.bid.data { return false }
    
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
    let (xsign, xexp, xman, _) = lhs.bid.unpack()
    let (ysign, yexp, yman, _) = rhs.bid.unpack()
    
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
    if xexp - yexp > ID.numberOfDigits-1 { return xsign == .plus }
    
    // need to compensate the mantissa
    var manPrime: ID.Mantissa
    if xexp > yexp {
      manPrime = xman * power(ID.Mantissa(10), to: xexp - yexp) // bid_mult_factor(xexp - yexp)
      if manPrime == yman { return false }
      return (manPrime < yman) != (xsign == .minus)
    }
    
    // adjust y mantissa upwards
    manPrime = yman * power(ID.Mantissa(10), to: yexp - xexp)
    if manPrime == xman { return false }
    
    // if positive, return whichever abs number is smaller
    return (xman < manPrime) != (xsign == .minus)
  }
}

extension Decimal32 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}

extension Decimal32 : ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    // FIXME: - floatLiteral
    self.init(bid: ID(0))
  }
}

extension Decimal32 : ExpressibleByIntegerLiteral {
  public init(integerLiteral value: IntegerLiteralType) {
    self.init(bid: ID(mantissa: ID.Mantissa(value))) // FIXME: - integer literal
  }
}

extension Decimal32 : ExpressibleByStringLiteral {
  public init(stringLiteral value: StringLiteralType) {
    bid = numberFromString(value, round: .toNearestOrEven) ?? ID.zero
  }
}

extension Decimal32 : Strideable {
  public func distance(to other: Decimal32) -> Decimal32 { other - self }
  public func advanced(by n: Decimal32) -> Decimal32 { self + n }
}

extension Decimal32 : FloatingPoint {
  
  public init<T:BinaryInteger>(_ value: T) {
    // FIXME: - Fix this init()
    self.init(integerLiteral: Int(value))
  }
  
  public init(_ value: Int) { self.init(integerLiteral: value) }
  
  public init?<T:BinaryInteger>(exactly source: T) {
    self.init(source)
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for FloatingPoint
  
  public init(sign: FloatingPointSign, exponent: Int, significand: Decimal32) {
    let x = significand.bid.unpack()
    bid = ID(sign: sign, exponent: exponent, mantissa: x.mantissa)
  }
    
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    // FIXME: - Round
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var exponentBitCount: Int         { ID.exponentBits }
  public static var exponentBias: Int             { ID.exponentBias }
  public static var significandDigitCount: Int    { ID.numberOfDigits }
  
  public static var rounding = FloatingPointRoundingRule.toNearestOrEven
  
  @inlinable public static var nan: Self          { Self(bid:ID.nan(.plus,0)) }
  @inlinable public static var signalingNaN: Self { Self(bid:ID.snan) }
  @inlinable public static var infinity: Self     { Self(bid:ID.infinite()) }
  
  @inlinable
  public static var greatestFiniteMagnitude: Self {
    Self(bid: ID(exponent: ID.maximumExponent, mantissa: ID.largestNumber))
  }
  
  @inlinable
  public static var leastNormalMagnitude: Self {
    Self(bid: ID(exponent: ID.minimumExponent, mantissa: ID.largestNumber))
  }
  
  @inlinable
  public static var leastNonzeroMagnitude: Self {
    Self(bid: ID(exponent: ID.minimumExponent, mantissa: 1))
  }
  
  @inlinable
  public static var pi: Self { Self(bid: ID(exponent: -6, mantissa: 3141593)) }
    
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var ulp: Self               { nextUp - self }
  public var sign: FloatingPointSign { bid.sign }
  public var exponent: Int           { bid.exponent + ID.numberOfDigits - 1 }
  public var isNormal: Bool          { bid.isNormal }
  public var isSubnormal: Bool       { bid.isSubnormal }
  public var isFinite: Bool          { !bid.isInfinite }
  public var isZero: Bool            { bid.isZero }
  public var isInfinite: Bool        { bid.isInfinite }
  public var isNaN: Bool             { bid.isNaN }
  public var isSignalingNaN: Bool    { bid.isSNaN }
  public var isCanonical: Bool       { bid.isCanonical }
  
  public var significand: Self {
    let (_, _, man, valid) = bid.unpack()
    if !valid { return self }
    return Self(bid: ID(mantissa: man))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Floating-point basic operations
  
  public static func * (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    lhs // FIXME: -
  }
  
  public static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    lhs // FIXME: -
  }
  
  public static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }
  
  public mutating func formRemainder(dividingBy other: Decimal32) {
    // FIXME: -
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Decimal32) {
    // FIXME: -
  }
  
  public mutating func formSquareRoot() {
    // FIXME: -
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    self += lhs * rhs // FIXME: -
  }
  
  public var nextUp: Self {
    Decimal32(0) // FIXME: -
  }
  
  public func isEqual(to other: Decimal32) -> Bool  { self == other }
  public func isLess(than other: Decimal32) -> Bool { self < other }
  
  public func isLessThanOrEqualTo(_ other: Decimal32) -> Bool {
    isEqual(to: other) || isLess(than: other)
  }
  
  public var magnitude: Decimal32 {
    let (_, exp, mag, _) = bid.unpack()
    return Self(bid: ID(exponent:exp, mantissa: mag))
  }
}

extension Decimal32 : DecimalFloatingPoint {

  public typealias RawExponent = UInt
  public typealias RawSignificand = UInt32

  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for DecimalFloatingPoint
  public init(bitPattern bits: RawSignificand, bidEncoding: Bool) {
    if bidEncoding {
      bid = ID(ID.RawDataFields(bits))
    } else {
      // convert from dpd to bid
      bid = ID(dpd: ID.RawDataFields(bits))
    }
  }
  
  public init(sign: FloatingPointSign, exponentBitPattern: RawExponent,
              significandBitPattern significantBitPattern: RawSignificand) {
    bid = ID(sign: sign, exponent: Int(exponentBitPattern) - ID.exponentBias,
             mantissa: ID.Mantissa(significantBitPattern))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var significandBitPattern: UInt32 { bid.data }
  public var exponentBitPattern: UInt   { UInt(bid.exponent+ID.exponentBias) }
  public var dpd: UInt32                   { bid.dpd }
  
  public var significandDigitCount: Int {
    guard bid.isValid else { return 0 }
    return ID.digitsIn(bid.mantissa)
  }
  
  public var significandDigits: [UInt8] {
    guard bid.isValid else { return [] }
    return Array(String(bid.mantissa)).map { UInt8($0.wholeNumberValue!) }
  }
  
  public var decade: Self {
    let (_, exp, _, valid) = bid.unpack()
    if !valid { return self } // For infinity, Nan, sNaN
    return Self(bid: ID(exponent: exp, mantissa: 1))
  }
  
  public func unpack() -> (sign: FloatingPointSign, exp: Int, coeff: UInt,
                           valid: Bool) {
    let x = bid.unpack()
    return (x.sign, x.exponent, x.mantissa, x.valid)
  }
  
}

