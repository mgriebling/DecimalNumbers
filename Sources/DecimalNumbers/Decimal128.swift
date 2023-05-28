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



/// Definition of the data storage for the Decimal128 floating-point data type.
/// the `IntDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal128 sign, exponent, and
/// mantissa fields. By specifying some key bit positions, it is possible
/// to completely define many of the Decimal128 operations. The `data` word
/// holds all 128 bits of the Decimal128 data type.
@frozen public struct IntegerDecimal128 : IntDecimal {
    
  public typealias RawData = UInt128
  public typealias Mantissa = UInt128
  
  public var data: RawData = 0
  
  public init(_ word: RawData) {
    self.data = word
  }
  
  public init(sign:Sign = .plus, exponent:Int=0, mantissa:Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  6176 }
  public static var maximumExponent: Int { 12287 } // unbiased
  public static var minimumExponent: Int {     0 } // unbiased
  public static var maximumDigits:   Int {    34 }
  public static var exponentBits:    Int {    14 }
  
  public static var largestNumber: Mantissa {
    let x = StaticBigInt(9_999_999_999_999_999_999_999_999_999_999_999)
    return Mantissa(integerLiteral: x)
  }
  
  // Two mantissa sizes must be supported
  public static var largeMantissaBits: IntRange { 0...112 }
  public static var smallMantissaBits: IntRange { 0...110 }
}

/// Implementation of the 128-bit Decimal128 floating-point operations from
/// IEEE STD 754-2000 for Floating-Point Arithmetic.
///
/// The IEEE Standard 754-2008 for Floating-Point Arithmetic supports two
/// encoding formats: the decimal encoding format, and the binary encoding
/// format. The Intel(R) Decimal Floating-Point Math Library supports primarily
/// the binary encoding format for decimal floating-point values, but the
/// decimal encoding format is supported too in the library, by means of
/// conversion functions between the two encoding formats.
@frozen public struct Decimal128 : Codable, Hashable {
  public typealias ID = IntegerDecimal128
  var bid: ID = ID.zero(.plus)
  
  public init(bid: UInt128) { self.bid.data = bid }
  public init(bid: ID)      { self.bid = bid }
  
  public init?(_ s: String) {
    if let n: ID = numberFromString(s, round: Self.rounding) { bid = n }
    return nil
  }
}

extension Decimal128 : AdditiveArithmetic {
  public static func - (lhs: Self, rhs: Self) -> Self {
    var addIn = rhs
    if !rhs.isNaN { addIn.negate() }
    return lhs + addIn
  }
  
  public mutating func negate() { self.bid.data.toggle(bit: ID.signBit) }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.add(lhs.bid, rhs.bid, rounding: rounding))
  }
  
  public static var zero: Self { Self(bid: ID.zero(.plus)) }
}

extension Decimal128 : Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    ID.equals(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal128 : Comparable {
  public static func < (lhs: Self, rhs: Self) -> Bool {
    ID.lessThan(lhs: lhs.bid, rhs: rhs.bid)
  }
  
  public static func >= (lhs: Self, rhs: Self) -> Bool {
    ID.greaterOrEqual(lhs: lhs.bid, rhs: rhs.bid)
  }
  
  public static func > (lhs: Self, rhs: Self) -> Bool {
    ID.greaterThan(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal128 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}

extension Decimal128 : ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self.init(bid: ID.bid(from: value, Self.rounding))
  }
}

extension Decimal128 : ExpressibleByIntegerLiteral {
  public init(integerLiteral value: IntegerLiteralType) {
    if IntegerLiteralType.isSigned {
      let x = Int(value).magnitude
      bid = ID.bid(from: UInt64(x), Self.rounding)
      if value.signum() < 0 { self.negate() }
    } else {
      bid = ID.bid(from: UInt64(value), Self.rounding)
    }
  }
}

extension Decimal128 : ExpressibleByStringLiteral {
  public init(stringLiteral value: StringLiteralType) {
    bid = numberFromString(value, round: Self.rounding) ?? Self.zero.bid
  }
}

extension Decimal128 : Strideable {
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
}

extension Decimal128 : FloatingPoint {
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for FloatingPoint
  
  public init(sign: Sign, exponent: Int, significand: Self) {
    self.bid = ID(sign: sign, exponent: exponent+Self.exponentBias,
                    mantissa: significand.bid.unpack().mantissa)
  }
  
  public mutating func round(_ rule: Rounding) {
    self.bid = ID.round(self.bid, rule)
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  @inlinable public static var exponentBitCount: Int      {ID.exponentBits}
  @inlinable public static var exponentBias: Int          {ID.exponentBias}
  @inlinable public static var significandDigitCount: Int {ID.maximumDigits}
  
  @inlinable public static var nan: Self          {Self(bid:ID.nan(.plus,0))}
  @inlinable public static var signalingNaN: Self {Self(bid:ID.snan)}
  @inlinable public static var infinity: Self     {Self(bid:ID.infinite())}
  
  @inlinable public static var greatestFiniteMagnitude: Self {
    Self(bid:ID(exponent:ID.maximumExponent, mantissa:ID.largestNumber))
  }
  
  @inlinable public static var leastNormalMagnitude: Self {
    Self(bid:ID(exponent:ID.minimumExponent, mantissa:ID.largestNumber))
  }
  
  @inlinable public static var leastNonzeroMagnitude: Self {
    Self(bid: ID(exponent: ID.minimumExponent, mantissa: 1))
  }
  
  @inlinable public static var pi: Self {
    let ip = StaticBigInt(3_141_592_653_589_793_238_462_643_383_279_503)
    return Self(bid: ID(exponent: ID.exponentBias-ID.maximumDigits+1,
                        mantissa: ID.Mantissa(integerLiteral:ip)))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var ulp: Self            { nextUp - self }
  public var nextUp: Self         { Self(bid: ID.nextup(self.bid)) }
  public var sign: Sign           { bid.sign }
  public var isNormal: Bool       { bid.isNormal }
  public var isSubnormal: Bool    { bid.isSubnormal }
  public var isFinite: Bool       { bid.isFinite }
  public var isZero: Bool         { bid.isZero }
  public var isInfinite: Bool     { bid.isInfinite && !bid.isNaN }
  public var isNaN: Bool          { bid.isNaN }
  public var isSignalingNaN: Bool { bid.isSNaN }
  public var isCanonical: Bool    { bid.isCanonical }
  
  public var exponent: Int {
    bid.exponent - ID.exponentBias + ID.maximumDigits - 1
  }
  
  public var significand: Self {
    let (_, _, man, valid) = bid.unpack()
    if !valid { return self }
    return Self(bid: ID(exponent: Int(exponentBitPattern), mantissa: man))
  }

  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Floating-point basic operations
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    lhs // FIXME: -
  }
  
  public static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    lhs // FIXME: -
  }
  
  public static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }
  
  public mutating func formRemainder(dividingBy other: Self) {
    // FIXME: -
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Self) {
    // FIXME: -
  }
  
  public mutating func formSquareRoot() {
    bid = ID.sqrt(self.bid, Self.rounding)
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    self += lhs * rhs // FIXME: -
  }
  
  public func isEqual(to other: Self) -> Bool  { self == other }
  public func isLess(than other: Self) -> Bool { self < other }
  
  public func isLessThanOrEqualTo(_ other: Self) -> Bool {
    isEqual(to: other) || isLess(than: other)
  }
  
  public var magnitude: Self {
    var data = bid.data; data.clear(bit: ID.signBit)
    return Self(bid: data)
  }
}

extension Decimal128 : DecimalFloatingPoint {
  public typealias RawExponent = UInt
  public typealias RawSignificand = UInt128
  
  public static var rounding = Rounding.toNearestOrEven

  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for DecimalFloatingPoint
  public init(bitPattern bits: RawSignificand, bidEncoding: Bool) {
    if bidEncoding {
      bid.data = bits
    } else {
      // convert from dpd to bid
      bid = ID(dpd: ID.RawData(bits))
    }
  }
  
  public init(sign: Sign, exponentBitPattern: RawExponent,
              significandBitPattern significantBitPattern: RawSignificand) {
    bid = ID(sign: sign, exponent: Int(exponentBitPattern),
             mantissa: ID.Mantissa(significantBitPattern))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var significandBitPattern: UInt128 { UInt128(bid.mantissa) }
  public var exponentBitPattern: UInt       { UInt(bid.exponent) }
  public var dpd: UInt128                   { bid.dpd }
  public var int: Int64                     { bid.int(Self.rounding) }
  public var uint: UInt64                   { bid.uint(Self.rounding) }
  
  public var significandDigitCount: Int {
    guard bid.isValid else { return -1 }
    return ID.digitsIn(bid.mantissa)
  }
  
  public var decade: Self {
    guard bid.isValid else { return self } // For infinity, Nan, sNaN
    return Self(bid: ID(exponent: bid.exponent, mantissa: 1))
  }
}
