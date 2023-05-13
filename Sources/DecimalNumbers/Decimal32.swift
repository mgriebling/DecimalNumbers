//
//  Decimal.swift
//  
//
//  Created by Mike Griebling on 12.05.2023.
//

import UInt128

/// Define the data fields for the Decimal32 storage type
struct IntegerDecimal32 : IntegerDecimal {
  typealias RawDataFields = UInt32
  typealias Mantissa = UInt
  
  var data: RawDataFields = 0
  
  init(_ word: RawDataFields) {
    self.data = word
  }
  
  init(sign:FloatingPointSign = .plus, exponent:Int = 0, mantissa:Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  static var exponentBias:    Int { 101 }
  static var maximumExponent: Int {  96 } // unbiased & normal
  static var minimumExponent: Int { -95 } // unbiased & normal
  static var numberOfDigits:  Int {   7 }
  
  static var largestNumber: Mantissa { 9_999_999 }
  
  // Two mantissa sizes must be supported
  static var exponentLMBits:    ClosedRange<Int> { 23...30 }
  static var largeMantissaBits: ClosedRange<Int> { 0...22 }
  
  static var exponentSMBits:    ClosedRange<Int> { 21...28 }
  static var smallMantissaBits: ClosedRange<Int> { 0...20 }
}

/// Implementation of the Decimal32 data type
public struct Decimal32 : Codable, Hashable {
  
  typealias ID = IntegerDecimal32
  var bid: ID
  
  init(bid: ID) {
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
    lhs
  }
  
  public static func + (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    lhs // FIXME: - +
  }
  
  public static var zero: Decimal32 {
    Decimal32(bid: ID(mantissa: 0))
  }
  
  public static func == (lhs: Decimal32, rhs: Decimal32) -> Bool {
    false // FIXME: - ==
  }
}

//extension Decimal32 : Strideable {
//  public func distance(to other: Self) -> Self { other - self }
//  public func advanced(by n: Self) -> Self { self + n }
//}

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
  
  // MARK: - Initializers for FloatingPoint
  public init(sign: FloatingPointSign, exponent: Int, significand: Decimal32) {
    let x = significand.bid.unpack()
    bid = ID(sign: sign, exponent: exponent, mantissa: x.mantissa)
  }
  
  public init(signOf: Decimal32, magnitudeOf: Decimal32) {
    self.init(bid: ID(0))
    let (_, exponent, mantissa, _) = significand.bid.unpack()
    bid = ID(sign: signOf.sign, exponent: exponent, mantissa: mantissa)
  }
  
  
  public init<Source:BinaryInteger>(_ value: Source) {
    self.init(bid: ID(0)) // FIXME: - Binary Integer
  }
  
  public init?<T:BinaryInteger>(exactly source: T) {
    self.init(bid: ID(0)) // FIXME: - Binary Integer
  }
  
  public var exponent: Int { bid.exponent }
  
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    // FIXME: - Round
  }

  
  // MARK: - FloatingPont class properties and attributes
  
  public static var radix: Int { 10 }
  
  public static var nan: Decimal32 { Decimal32(0) } // FIXME: -
  
  public static var signalingNaN: Decimal32 { Decimal32(0) } // FIXME: -
  
  public static var infinity: Decimal32 { Decimal32(0) } // FIXME: -
  
  public static var greatestFiniteMagnitude: Decimal32 { Decimal32(0) } // FIXME: -
  public static var leastNormalMagnitude: Decimal32 {
    Decimal32(0) // FIXME: -
  }
  
  public static var leastNonzeroMagnitude: Decimal32 {
    Decimal32(0) // FIXME: -
  }
  
  public static var pi: Decimal32 { Decimal32(0) } // FIXME: -
    
  
  // MARK: - Instance properties and attributes
  
  public var ulp: Decimal32 {
    Decimal32(0) // FIXME: -
  }
    
  public var sign: FloatingPointSign { bid.sign }
  
  public var significand: Decimal32 {
    Decimal32(0) // FIXME: -
  }
  
  public static func * (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    lhs
  }
  
  public static func *= (lhs: inout Decimal32, rhs: Decimal32) { lhs = lhs * rhs }
  
  public static func / (lhs: Decimal32, rhs: Decimal32) -> Decimal32 {
    lhs
  }
  
  public static func /= (lhs: inout Decimal32, rhs: Decimal32) {
    lhs = lhs / rhs
  }
  
  public mutating func formRemainder(dividingBy other: Decimal32) {
    // FIXME: -
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Decimal32) {
    // FIXME: -
  }
  
  public mutating func formSquareRoot() {
    // FIXME: -
  }
  
  public mutating func addProduct(_ lhs: Decimal32, _ rhs: Decimal32) {
    self += lhs * rhs // FIXME: -
  }
  
  public var nextUp: Decimal32 {
    Decimal32(0) // FIXME: -
  }
  
  public func isEqual(to other: Decimal32) -> Bool {
    false // FIXME: -
  }
  
  public func isLess(than other: Decimal32) -> Bool {
    false // FIXME: -
  }
  
  public func isLessThanOrEqualTo(_ other: Decimal32) -> Bool {
    false // FIXME: -
  }
  
  public func isTotallyOrdered(belowOrEqualTo other: Decimal32) -> Bool {
    false // FIXME: -
  }
  
  public var isNormal: Bool {
    false // FIXME: -
  }
  
  public var isFinite: Bool {
    false // FIXME: -
  }
  
  public var isZero: Bool { bid.isZero }
  
  public var isSubnormal: Bool {
    false // FIXME: -
  }
  
  public var isInfinite: Bool { bid.isInfinite }
  public var isNaN: Bool { bid.isNaN }
  public var isSignalingNaN: Bool { bid.isSNaN }
  
  public var isCanonical: Bool {
    false // FIXME: -
  }
  
  public var magnitude: Decimal32 {
    let (_, exp, mag, _) = bid.unpack()
    return Self(bid: ID(exponent:exp, mantissa: mag))
  }
}

extension Decimal32 : DecimalFloatingPoint {

  // MARK: - Initializers for DecimalFloatingPoint
  public init(bitPattern bits: UInt, bidEncoding: Bool) {
    self.bid = ID(UInt32(bits)) // FIXME: -
  }
  
  public init(sign: FloatingPointSign, exponentBitPattern: UInt,
              significandDigits: [UInt8]) {
    self = Decimal32(0) // FIXME: -
  }
  
  public init(sign: FloatingPointSign, exponentBitPattern: Self.RawExponent,
       significantBitPattern: Self.BitPattern) {
    self = Decimal32(0) // FIXME: -
  }
  
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var exponentMaximum: Int { ID.maximumExponent }
  public static var exponentBias: Int { ID.exponentBias }
  public static var significandMaxDigitCount: Int { ID.numberOfDigits }
  
  // MARK: - Instance properties and attributes
  
  public var bitPattern: UInt32 { bid.data }
  
  public var exponentBitPattern: UInt {
    0 // FIXME: -
  }
  
  public var significandDigits: [UInt8] {
    [] // FIXME: -
  }
  
  public var significandDigitCount: Int {
    0 // FIXME: -
  }
  
  public var decade: Decimal32 {
    Decimal32(0) // FIXME: -
  }
  
  public var isBIDFormat: Bool { true }
  
  public func unpack() -> (sign: FloatingPointSign, exp: Int, coeff: UInt, valid: Bool) {
    let x = bid.unpack()
    return (x.sign, x.exponent, x.mantissa, x.valid)
  }
  
}

