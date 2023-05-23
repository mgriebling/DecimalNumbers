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
/// the `IntegerDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal128 sign, exponent, and
/// mantissa fields. By specifying some key bit positions, it is possible
/// to completely define many of the Decimal128 operations. The `data` word
/// holds all 128 bits of the Decimal128 data type.
public struct IntegerDecimal128 : IntegerDecimal {
    
  public typealias RawDataFields = UInt128
  public typealias Mantissa = UInt128
  
  public var data: RawDataFields = 0
  
  public init(_ word: RawDataFields) {
    self.data = word
  }
  
  public init(sign:FloatingPointSign = .plus, exponent:Int = 0,
              mantissa:Mantissa) {
    self.init(sign: sign, exponent: exponent, mantissa: mantissa, round: 0)
  }
  
  public init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa,
              round: Int) {
    
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  6176 }
  public static var maximumExponent: Int {  6111 } // unbiased
  public static var minimumExponent: Int { -6176 } // unbiased
  public static var numberOfDigits:  Int {    34 }
  public static var exponentBits:    Int {    14 }
  
  // Awkward way of using StaticBigInts — fix me.
  public static var largestNumber: Mantissa {
    if#available(macOS 13.3,iOS 16.4,macCatalyst 16.4,tvOS 16.4,watchOS 9.4,*){
      let x : StaticBigInt = 9_999_999_999_999_999_999_999_999_999_999_999
      return Mantissa(integerLiteral: x)
    } else {
      // Fallback on earlier versions - same number in hexadecimal
      return Mantissa(high: 0x1_ED09_BEAD_87C0, low: 0x378D_8E63_FFFF_FFFF)
    }
  }
  
  // Two mantissa sizes must be supported
  public static var exponentLMBits:    ClosedRange<Int> { 113...126 }
  public static var largeMantissaBits: ClosedRange<Int> {   0...112 }
  
  public static var exponentSMBits:    ClosedRange<Int> { 111...124 }
  public static var smallMantissaBits: ClosedRange<Int> {   0...110 }
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
public struct Decimal128 {
  
  public typealias ID = IntegerDecimal128
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
    return self
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
    let p: ID.Mantissa
    if#available(macOS 13.3,iOS 16.4,macCatalyst 16.4,tvOS 16.4,watchOS 9.4,*){
      let x : StaticBigInt = 3_141_592_653_589_793_238_462_643_383_279_503
      p = ID.Mantissa(integerLiteral: x)
    } else {
      // Fallback on earlier versions - same number in hexadecimal
      p = ID.Mantissa(high: 0x9AE4_7957_96A7, low: 0xBABE_5564_E6F3_9F8F)
    }
    return Self(bid: ID(exponent: -ID.numberOfDigits+1, mantissa: p))
  }
}

extension Decimal128 : AdditiveArithmetic {
  public static func - (lhs: Self, rhs: Self) -> Self {
    var addIn = rhs
    addIn.negate()
    return lhs + addIn
  }
  
  public mutating func negate() {
    bid.sign = bid.sign == .minus ? FloatingPointSign.plus : .minus
  }
  
  public static func + (lhs: Self, rhs: Self) -> Self {
    lhs.add(rhs, rounding: .toNearestOrEven)
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
}

extension Decimal128 : CustomStringConvertible {
  public var description: String {
    string(from: bid)
  }
}
