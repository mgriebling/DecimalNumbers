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
  
  public init(_ word: RawDataFields) { self.data = word }
  
  public init(sign:FloatingPointSign = .plus, exponent:Int = 0,
              mantissa:Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  101 }
  public static var maximumExponent: Int {  191 }
  public static var minimumExponent: Int {    0 }
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
}

extension Decimal32 : AdditiveArithmetic {
  public static func - (lhs: Self, rhs: Self) -> Self {
    var addIn = rhs
    addIn.negate()
    return lhs + addIn
  }
  
  public mutating func negate() {
    bid.sign = bid.sign == .minus ? FloatingPointSign.plus : .minus
  }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.add(lhs.bid, rhs.bid, rounding: .toNearestOrEven))
  }
  
  public static var zero: Self { Self(bid: ID.zero(.plus)) }
}

extension Decimal32 : Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    ID.equals(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal32 : Comparable {
  public static func < (lhs: Self, rhs: Self) -> Bool {
    ID.lessThan(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal32 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}

extension Decimal32 : ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self.init(bid: ID.bid(from: value, Self.rounding))
  }
}

extension Decimal32 : ExpressibleByIntegerLiteral {
  public init(integerLiteral value: IntegerLiteralType) {
    
  }
}

extension Decimal32 : ExpressibleByStringLiteral {
  public init(stringLiteral value: StringLiteralType) {
    bid = numberFromString(value, round: Self.rounding) ?? Self.zero.bid
  }
}

extension Decimal32 : Strideable {
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
}

extension Decimal32 : FloatingPoint {
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for FloatingPoint
  
  public init(sign: FloatingPointSign, exponent: Int, significand: Self) {
    let (_, _, mantissa, _) = significand.bid.unpack()
    bid = ID(sign: sign, exponent: exponent, mantissa: mantissa)
  }
    
  public mutating func round(_ rule: FloatingPointRoundingRule) {
    // FIXME: - Round
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var rounding = FloatingPointRoundingRule.toNearestOrEven
  
  @inlinable public static var exponentBitCount: Int      { ID.exponentBits }
  @inlinable public static var exponentBias: Int          { ID.exponentBias }
  @inlinable public static var significandDigitCount: Int { ID.numberOfDigits }
  
  @inlinable public static var nan: Self          { Self(bid:ID.nan(.plus,0)) }
  @inlinable public static var signalingNaN: Self { Self(bid:ID.snan) }
  @inlinable public static var infinity: Self     { Self(bid:ID.infinite()) }
  
  @inlinable public static var greatestFiniteMagnitude: Self {
    Self(bid: ID(exponent: ID.maximumExponent, mantissa: ID.largestNumber))
  }
  
  @inlinable public static var leastNormalMagnitude: Self {
    Self(bid: ID(exponent: ID.minimumExponent, mantissa: ID.largestNumber))
  }
  
  @inlinable public static var leastNonzeroMagnitude: Self {
    Self(bid: ID(exponent: ID.minimumExponent, mantissa: 1))
  }
  
  @inlinable public static var pi: Self {
    Self(bid: ID(exponent: -ID.numberOfDigits+1, mantissa: 3141593))
  }
    
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
    // FIXME: -
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    self += lhs * rhs // FIXME: -
  }
  
  public var nextUp: Self {
    Self(0) // FIXME: -
  }
  
  public func isEqual(to other: Self) -> Bool  { self == other }
  public func isLess(than other: Self) -> Bool { self < other }
  
  public func isLessThanOrEqualTo(_ other: Self) -> Bool {
    isEqual(to: other) || isLess(than: other)
  }
  
  public var magnitude: Self {
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
    bid = ID(sign: sign, exponent: Int(exponentBitPattern),
             mantissa: ID.Mantissa(significantBitPattern))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var significandBitPattern: UInt32 { bid.data }
  public var exponentBitPattern: UInt      { UInt(bid.exponent) }
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

