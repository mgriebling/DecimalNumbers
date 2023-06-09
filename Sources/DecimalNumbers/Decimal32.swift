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
  typealias RawSignificand = UInt32
  typealias RawData = UInt32
  typealias RawBitPattern = UInt
  
  var data: RawData = 0
  
  init(_ word: RawData) { self.data = word }
  
  init(sign:Sign = .plus, expBitPattern:Int=0, sigBitPattern:RawBitPattern) {
    self.sign = sign
    self.set(exponent: expBitPattern, sigBitPattern: sigBitPattern)
  }
  
  // Define the fields and required parameters
  static var exponentBias:       Int {  101 }
  static var maxEncodedExponent: Int {  191 }
  static var maximumDigits:      Int {    7 }
  static var exponentBits:       Int {    8 }
  
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
  
  // Internal data store for the binary integer decimal encoded number.
  // The internal representation is always binary integer decimal.
  var bid = ID.zero(.plus)
  
  // Raw data initializer -- only for internal use.
  public init(bid: UInt32) { self.bid = ID(bid) }
  init(bid: ID)            { self.bid = bid }
  
  /// Creates a NaN ("not a number") value with the specified payload.
  ///
  /// NaN values compare not equal to every value, including themselves. Most
  /// operations with a NaN operand produce a NaN result. Don't use the
  /// equal-to operator (`==`) to test whether a value is NaN. Instead, use
  /// the value's `isNaN` property.
  ///
  ///     let x = Decimal32(nan: 0, signaling: false)
  ///     print(x == .nan)
  ///     // Prints "false"
  ///     print(x.isNaN)
  ///     // Prints "true"
  ///
  /// - Parameters:
  ///   - payload: The payload to use for the new NaN value.
  ///   - signaling: Pass `true` to create a signaling NaN or `false` to create
  ///     a quiet NaN.
  public init(nan payload: RawSignificand, signaling: Bool) {
    self.bid = ID.init(nan: payload, signaling: signaling)
  }
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
    self.bid = ID(sign: sign, expBitPattern: exponent+ID.exponentBias,
                    sigBitPattern: significand.bid.unpack().sigBits)
  }
  
  public mutating func round(_ rule: Rounding) {
    self.bid = ID.round(self.bid, rule)
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - DecimalFloatingPoint properties and attributes
  
  public static var exponentBitCount: Int      { ID.exponentBits }
  public static var significandDigitCount: Int { ID.maximumDigits }
  public static var nan: Self                  { Self(nan:0, signaling:false) }
  public static var signalingNaN: Self         { Self(nan:0, signaling:true) }
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
    return Self(bid: ID(expBitPattern: Int(exponentBitPattern),
                        sigBitPattern: man))
  }

  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Floating-point basic operations
  
  public static func * (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.mul(lhs.bid, rhs.bid, .toNearestOrEven))
  }
  
  public static func *= (lhs: inout Self, rhs: Self) { lhs = lhs * rhs }
  
  public static func / (lhs: Self, rhs: Self) -> Self {
    Self(bid: ID.div(lhs.bid, rhs.bid, .toNearestOrEven))
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
    self.formSquareRoot(round: .toNearestOrEven)
  }
  
  /// Rounding method equivalend of the `formSquareRoot`
  public mutating func formSquareRoot(round: Rounding) {
    bid = ID.sqrt(self.bid, round)
  }
  
  public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
    self.addProduct(lhs, rhs, round: .toNearestOrEven)
  }
  
  /// Rounding method equivalent of the `addProduct`
  public mutating func addProduct(_ lhs: Self, _ rhs: Self, round: Rounding) {
    bid = ID.fma(lhs.bid, rhs.bid, self.bid, round)
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
  
  /// Creates a new instance from the specified sign and bit patterns.
  ///
  /// The values passed as `exponentBitPattern` and `significandBitPattern` are
  /// interpreted in the decimal interchange format defined by the [IEEE 754
  /// specification][spec].
  ///
  /// [spec]: http://ieeexplore.ieee.org/servlet/opac?punumber=4610933
  ///
  /// The `significandBitPattern` are the big-endian, binary integer decimal
  /// digits of the number. For example, the integer number `314` represents a
  /// significand of `314`.
  ///
  /// - Parameters:
  ///   - sign: The sign of the new value.
  ///   - exponentBitPattern: The bit pattern to use for the exponent field of
  ///     the new value.
  ///   - significandBitPattern: Bit pattern to use for the significand field
  ///     of the new value.
  public init(sign: Sign, exponentBitPattern: RawExponent,
              significandBitPattern: RawSignificand) {
    bid = ID(sign: sign, expBitPattern: Int(exponentBitPattern),
             sigBitPattern: ID.RawBitPattern(significandBitPattern))
  }
  
  ///////////////////////////////////////////////////////////////////////////
  // MARK: - Instance properties and attributes
  
  /// The raw encoding of the value's significand field.
  public var significandBitPattern: UInt32 { UInt32(bid.sigBitPattern) }
  
  /// The raw encoding of the value's exponent field.
  ///
  /// This value is unadjusted by the type's exponent bias.
  public var exponentBitPattern: UInt { UInt(bid.expBitPattern) }
  
  //  Conversions to/from binary integer decimal encoding.  These are not part
  //  of the DecimalFloatingPoint prototype because there's no guarantee that
  //  an integer type of the same size actually exists (e.g. Decimal128).
  //
  //  If we want them in a protocol at some future point, that protocol should
  //  be "InterchangeFloatingPoint" or "PortableFloatingPoint" or similar, and
  //  apply to IEEE 754 "interchange types".
  /// The bit pattern of the value's encoding. A `bid` prefix indicates a
  /// binary integer decimal encoding; while a `dpd` prefix indicates a
  /// densely packed decimal encoding.
  ///
  /// The bit patterns are extracted using the `bidBitPattern` and
  /// `dpdBitPattern` accessors. A new decimal floating point number is
  /// created by passing an appropriate bit pattern to the
  /// `init(bidBitPattern:)` and `init(dpdBitPattern:)` initializers.
  /// If incorrect bit encodings are used, there are no guarantees about
  /// the resultant decimal floating point number.
  ///
  /// The bit patterns match the decimal interchange format defined by the
  /// [IEEE 754 specification][spec].
  ///
  /// For example, a Decimal32 number has been created with the value "1000.3".
  /// Using the `bidBitPattern` accessor, a 32-bit unsigned integer encoded
  /// value of `0x32002713` is returned.  The `dpdBitPattern` returns the
  /// 32-bit unsigned integer encoded value of `0x22404003`. Passing these
  /// numbers to the appropriate initializer recreates the original value
  /// "1000.3".
  ///
  /// [spec]: http://ieeexplore.ieee.org/servlet/opac?punumber=4610933
  public var bidBitPattern: RawSignificand { bid.data }
  public var dpdBitPattern: RawSignificand { bid.dpd }
  
  public init(bidBitPattern: RawSignificand) {
    bid.data = bidBitPattern
  }
  
  public init(dpdBitPattern: RawSignificand) {
    bid = ID(dpd: ID.RawData(dpdBitPattern))
  }
  
  public var significandDigitCount: Int {
    guard bid.isValid else { return -1 }
    return _digitsIn(bid.sigBitPattern)
  }
  
  /// The floating-point value with the same sign and exponent as this value,
  /// but with a significand of 1.0.
  ///
  /// A *decade* is a set of decimal floating-point values that all have the
  /// same sign and exponent. The `decade` property is a member of the same
  /// decade as this value, but with a unit significand.
  ///
  /// In this example, `x` has a value of `21.5`, which is stored as
  /// `2.15 * 10**1`, where `**` is exponentiation. Therefore, `x.decade` is
  /// equal to `1.0 * 10**1`, or `10.0`.
  ///```
  /// let x = 21.5
  /// // x.significand == 2.15
  /// // x.exponent == 1
  ///
  /// let y = x.decade
  /// // y == 10.0
  /// // y.significand == 1.0
  /// // y.exponent == 1
  ///```
  public var decade: Self {
    guard bid.isValid else { return self } // For infinity, Nan, sNaN
    return Self(bid: ID(expBitPattern: bid.expBitPattern, sigBitPattern: 1))
  }
}
