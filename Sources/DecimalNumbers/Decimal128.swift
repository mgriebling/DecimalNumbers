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

extension UInt128  {
  @available(macOS 13.3, iOS 16.4, macCatalyst 16.4, tvOS 16.4, watchOS 9.4, *)
  init(bigInt: StaticBigInt) {
    precondition(bigInt.signum() >= 0, "UInt128 literal cannot be negative")
    precondition(bigInt.bitWidth <= Self.bitWidth,
                 "\(bigInt.bitWidth)-bit literal too large for UInt128")
    precondition(UInt.bitWidth == 64, "Expecting 64-bit UInt")
    self.init(high: High(bigInt[1]), low: Low(bigInt[0]))
  }
}

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
    
  public init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  6176 }
  public static var maximumExponent: Int {  6111 } // unbiased
  public static var minimumExponent: Int { -6176 } // unbiased
  public static var numberOfDigits:  Int {    34 }
  public static var exponentBits:    Int {    14 }
  
  // Awkward way of using StaticBigInts — fix me.
  public static var largestNumber: Mantissa {
    if #available(macOS 13.3, iOS 16.4, macCatalyst 16.4, tvOS 16.4,
                  watchOS 9.4, *) {
      return Mantissa(bigInt: 9_999_999_999_999_999_999_999_999_999_999_999)
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
  
  var x: IntegerDecimal128
  
  func add(_ y: Self) -> Self {
    if self.x.isInfinite { return self }
    if y.x.isInfinite { return y }
    return self
  }
  
}
