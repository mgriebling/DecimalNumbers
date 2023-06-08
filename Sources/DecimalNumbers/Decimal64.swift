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

/// Definition of the data storage for the Decimal64 floating-point data type.
/// the `IntDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal64 sign, exponent, and
/// significand fields.  By specifying some key bit positions, it is possible
/// to completely define many of the Decimal64 operations.  The `data` word
/// holds all 64 bits of the Decimal64 data type.
struct IntDecimal64 : IntDecimal {
  typealias RawData = UInt64
  typealias RawBitPattern = UInt64
  
  var data: RawData = 0
  
  init(_ word: RawData) { self.data = word }
  
  init(sign:Sign = .plus, expBitPattern:Int=0, sigBitPattern:RawBitPattern) {
    self.sign = sign
    self.set(exponent: expBitPattern, sigBitPattern: sigBitPattern)
  }
  
  // Define the fields and required parameters
  static var exponentBias:       Int {  398 }
  static var maxEncodedExponent: Int {  767 } 
  static var maximumDigits:      Int {   16 }
  static var exponentBits:       Int {   10 }
  
  static var largestNumber: RawBitPattern { 9_999_999_999_999_999 }
}

/// Implementation of the 64-bit Decimal64 floating-point operations from
/// IEEE STD 754-2000 for Floating-Point Arithmetic.
///
/// The IEEE Standard 754-2008 for Floating-Point Arithmetic supports two
/// encoding formats: the decimal encoding format, and the binary encoding
/// format. The Intel(R) Decimal Floating-Point Math Library supports primarily
/// the binary encoding format for decimal floating-point values, but the
/// decimal encoding format is supported too in the library, by means of
/// conversion functions between the two encoding formats.
public struct Decimal64 : Codable, Hashable {
  typealias ID = IntDecimal64
  var bid: ID = ID.zero(.plus)
  
  public init(bid: UInt64) { self.bid.data = bid }
  init(bid: ID)            { self.bid = bid }
}

extension Decimal64 : AdditiveArithmetic {
  public static func - (lhs: Self, rhs: Self) -> Self {
    var addIn = rhs
    if !rhs.isNaN { addIn.negate() }
    return lhs + addIn
  }
  
  public mutating func negate() { self.bid.data.toggle(bit: ID.signBit) }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.add(lhs.bid, rhs.bid, rounding: ID.rounding))
  }
  
  public static var zero: Self { Self(bid: ID.zero(.plus)) }
}

extension Decimal64 : Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    ID.equals(lhs: lhs.bid, rhs: rhs.bid)
  }
}

extension Decimal64 : Comparable {
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

extension Decimal64 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}

extension Decimal64 : ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self.init(bid: ID.bid(from: value, ID.rounding))
  }
}

extension Decimal64 : ExpressibleByIntegerLiteral {
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

extension Decimal64 : ExpressibleByStringLiteral {
  public init(stringLiteral value: StringLiteralType) {
    bid = numberFromString(value, round: ID.rounding) ?? Self.zero.bid
  }
}

extension Decimal64 : Strideable {
  public func distance(to other: Self) -> Self { other - self }
  public func advanced(by n: Self) -> Self { self + n }
}

extension Decimal64 : FloatingPoint {
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Initializers for FloatingPoint
  
  public init(sign: Sign, exponent: Int, significand: Self) {
    self.bid = ID(sign: sign, expBitPattern: exponent+ID.exponentBias,
                    sigBitPattern: significand.bid.unpack().sigBits)
  }
  
  public mutating func round(_ rule: Rounding) {
    self.bid = ID.round(self.bid, rule)
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var exponentBitCount: Int      {ID.exponentBits}
//  public static var exponentBias: Int          {ID.exponentBias}
  public static var significandDigitCount: Int {ID.maximumDigits}
  public static var nan: Self                  { Self(bid:ID.nan()) }
  public static var signalingNaN: Self         { Self(bid:ID.snan) }
  public static var infinity: Self             { Self(bid:ID.infinite()) }
  
  public static var greatestFiniteMagnitude: Self {
    Self(bid:ID(expBitPattern:ID.maxEncodedExponent,
                sigBitPattern:ID.largestNumber))
  }
  
  public static var leastNormalMagnitude: Self {
    Self(bid:ID(expBitPattern:ID.minEncodedExponent,
                sigBitPattern:ID.largestNumber))
  }
  
  public static var leastNonzeroMagnitude: Self {
    Self(bid: ID(expBitPattern: ID.minEncodedExponent, sigBitPattern: 1))
  }
  
  public static var pi: Self {
    Self(bid: ID(expBitPattern: ID.exponentBias-ID.maximumDigits+1,
                   sigBitPattern: 3_141_592_653_589_793))
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
    return Self(bid: ID(expBitPattern: Int(exponentBitPattern),
                        sigBitPattern: man))
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
    bid = ID.sqrt(self.bid, ID.rounding)
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

extension Decimal64 : DecimalFloatingPoint {
  public typealias RawExponent = UInt
  public typealias RawSignificand = UInt64
  
  // public static var rounding = Rounding.toNearestOrEven

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
  
  public var significandBitPattern: UInt64 { UInt64(bid.sigBitPattern) }
  public var exponentBitPattern: UInt      { UInt(bid.expBitPattern) }
  public var bidBitPattern: UInt64         { bid.data }
  public var dpdBitPattern: UInt64         { bid.dpd }
  
  public var significandDigitCount: Int {
    guard bid.isValid else { return -1 }
    return _digitsIn(bid.sigBitPattern)
  }
  
  public var decade: Self {
    guard bid.isValid else { return self } // For infinity, Nan, sNaN
    return Self(bid: ID(expBitPattern: bid.expBitPattern, sigBitPattern: 1))
  }
}
