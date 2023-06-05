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
/// the `IntDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal32 sign, exponent, and
/// significand fields.  By specifying some key bit positions, it is possible
/// to completely define many of the Decimal32 operations.  The `data` word
/// holds all 32 bits of the Decimal32 data type.
struct IntDecimal32 : IntDecimal {
  typealias RawData = UInt32
  typealias RawBitPattern = UInt
  
  var data: RawData = 0
  
  init(_ word: RawData) { self.data = word }
  
  init(sign:Sign = .plus, expBitPattern:Int=0, sigBitPattern:RawBitPattern) {
    self.sign = sign
    self.set(exponent: expBitPattern, sigBitPattern: sigBitPattern)
  }
  
  // Define the fields and required parameters
  static var exponentBias:      Int {  101 }
  static var maxEncodedExponent: Int {  191 }
  static var maximumDigits:     Int {    7 }
  static var exponentBits:      Int {    8 }
  
  static var largestNumber: RawBitPattern { 9_999_999 }
}

/// Implementation of the 32-bit Decimal32 floating-point operations from
/// IEEE STD 754-2008 for Floating-Point Arithmetic.
///
/// The IEEE Standard 754-2008 for Floating-Point Arithmetic supports two
/// encoding formats: the decimal encoding format, and the binary encoding
/// format. The Intel(R) Decimal Floating-Point Math Library supports primarily
/// the binary encoding format for decimal floating-point values, but the
/// decimal encoding format is supported too in the library, by means of
/// conversion functions between the two encoding formats.
public struct Decimal32 : Codable, Hashable {
  typealias ID = IntDecimal32
  
  public typealias RawExponent = UInt
  public typealias RawSignificand = UInt32
  
  var bid = ID.zero(.plus)
  
  public init(bid: UInt32) { self.bid.data = bid }
  init(bid: ID)            { self.bid = bid }
}

extension Decimal32 : AdditiveArithmetic {
  public static func - (lhs: Self, rhs: Self) -> Self {
    var addIn = rhs
    if !rhs.isNaN { addIn.negate() }
    return lhs + addIn
  }
  
  public mutating func negate() { self.bid.data.toggle(bit: ID.signBit) }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.add(lhs.bid, rhs.bid, rounding: ID.rounding))
  }
  
  public static var zero: Self { Self(bid: ID.zero()) }
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
  
  public static func >= (lhs: Self, rhs: Self) -> Bool {
    ID.greaterOrEqual(lhs: lhs.bid, rhs: rhs.bid)
  }
  
  public static func > (lhs: Self, rhs: Self) -> Bool {
    ID.greaterThan(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal32 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}

extension Decimal32 : ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self.init(bid: ID.bid(from: value, ID.rounding))
  }
}

extension Decimal32 : ExpressibleByIntegerLiteral {
  public init(integerLiteral value: IntegerLiteralType) {
    if IntegerLiteralType.isSigned {
      let x = Int(value).magnitude
      bid = ID.bid(from: UInt64(x), ID.rounding)
      if value.signum() < 0 { self.negate() }
    } else {
      bid = ID.bid(from: UInt64(value), ID.rounding)
    }
  }
}

extension Decimal32 : ExpressibleByStringLiteral {
  public init(stringLiteral value: StringLiteralType) {
    self.init(stringLiteral: value, round: .toNearestOrEven)
  }
  
  public init(stringLiteral value: StringLiteralType, round: Rounding) {
    bid = numberFromString(value, round: round) ?? Self.zero.bid
  }
}

extension Decimal32 : Strideable {
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
}

extension Decimal32 : FloatingPoint {
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for FloatingPoint
  
  public init(sign: Sign, exponent: Int, significand: Self) {
    self.bid = ID(sign: sign, expBitPattern: exponent+Self.exponentBias,
                    sigBitPattern: significand.bid.unpack().sigBits)
  }
  
  public mutating func round(_ rule: Rounding) {
    self.bid = ID.round(self.bid, rule)
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var exponentBitCount: Int      {ID.exponentBits}
  public static var exponentBias: Int          {ID.exponentBias}
  public static var significandDigitCount: Int {ID.maximumDigits}
  
  public static var nan: Self                  { Self(bid:ID.nan()) }
  public static var signalingNaN: Self         { Self(bid:ID.snan) }
  public static var infinity: Self             { Self(bid:ID.infinite()) }
  
  public static var greatestFiniteMagnitude: Self {
    Self(bid:ID(expBitPattern:ID.maxEncodedExponent, sigBitPattern:ID.largestNumber))
  }
  
  public static var leastNormalMagnitude: Self {
    Self(bid:ID(expBitPattern:ID.minEncodedExponent, sigBitPattern:ID.largestNumber))
  }
  
  public static var leastNonzeroMagnitude: Self {
    Self(bid: ID(expBitPattern: ID.minEncodedExponent, sigBitPattern: 1))
  }
  
  public static var pi: Self {
    Self(bid: ID(expBitPattern: ID.exponentBias-ID.maximumDigits+1,
                 sigBitPattern: 3_141_593))
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
  public var exponent: Int        { bid.expBitPattern - ID.exponentBias }
  
  public var significand: Self {
    let (_, _, man, valid) = bid.unpack()
    if !valid { return self }
    return Self(bid: ID(expBitPattern: Int(exponentBitPattern), sigBitPattern: man))
  }

  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Floating-point basic operations
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.mul(lhs.bid, rhs.bid, ID.rounding))
  }
  
  public static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.div(lhs.bid, rhs.bid, ID.rounding))
  }
  
  public static func /= (lhs: inout Self, rhs: Self) { lhs = lhs / rhs }
  
  public mutating func formRemainder(dividingBy other: Self) {
    bid = ID.rem(self.bid, other.bid)
  }
  
  public mutating func formTruncatingRemainder(dividingBy other: Self) {
    let q = (self/other).rounded(.towardZero)
    self -= q * other
  }
  
  public mutating func formSquareRoot() {
    bid = ID.sqrt(self.bid, ID.rounding)
  }
  
  public mutating func formSquareRoot(round: Rounding) {
    bid = ID.sqrt(self.bid, round)
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    bid = ID.fma(lhs.bid, rhs.bid, self.bid, ID.rounding)
  }
  
  public func isEqual(to other: Self) -> Bool  { self == other }
  public func isLess(than other: Self) -> Bool { self < other }
  
  public func isLessThanOrEqualTo(_ other: Self) -> Bool {
    isEqual(to: other) || isLess(than: other)
  }
  
  public var magnitude: Self { Self(bid: bid.data.clearing(bit: ID.signBit)) }
}

extension Decimal32 : DecimalFloatingPoint {
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for DecimalFloatingPoint
  
  public init(bidBitPattern: RawSignificand) {
    bid.data = bidBitPattern
  }
  
  public init(dpdBitPattern: RawSignificand) {
    bid = ID(dpd: ID.RawData(dpdBitPattern))
  }
  
  public init(sign: Sign, exponentBitPattern: RawExponent,
              significandBitPattern: RawSignificand) {
    bid = ID(sign: sign, expBitPattern: Int(exponentBitPattern),
             sigBitPattern: ID.RawBitPattern(significandBitPattern))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  public var significandBitPattern: UInt32 { UInt32(bid.sigBitPattern) }
  public var exponentBitPattern: UInt      { UInt(bid.expBitPattern) }
  public var bidBitPattern: UInt32         { bid.data }
  public var dpdBitPattern: UInt32         { bid.dpd }
  
//  public var int: Int64                    { bid.int(ID.rounding) }
//  public var uint: UInt64                  { bid.uint(ID.rounding) }
  
  public func double(round:Rounding) -> Double { bid.double(round) }
  
  public var significandDigitCount: Int {
    guard bid.isValid else { return -1 }
    return ID.digitsIn(bid.sigBitPattern)
  }
  
  public var decade: Self {
    guard bid.isValid else { return self } // For infinity, Nan, sNaN
    return Self(bid: ID(expBitPattern: bid.expBitPattern, sigBitPattern: 1))
  }
}
