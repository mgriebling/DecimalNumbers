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
/// the `IntegerDecimal` protocol defines many supporting operations
/// including packing and unpacking of the Decimal64 sign, exponent, and
/// mantissa fields.  By specifying some key bit positions, it is possible
/// to completely define many of the Decimal64 operations.  The `data` word
/// holds all 64 bits of the Decimal64 data type.
public struct IntegerDecimal64 : IntegerDecimal {

  public typealias RawDataFields = UInt64
  public typealias Mantissa = UInt64
  
  public var data: RawDataFields = 0
  
  public init(_ word: RawDataFields) {
    self.data = word
  }
  
  public init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  public static var exponentBias:    Int {  398 }
  public static var maximumExponent: Int {  369 } // unbiased
  public static var minimumExponent: Int { -398 } // unbiased
  public static var numberOfDigits:  Int {   16 }
  public static var exponentBits:    Int {   10 }
  
  public static var largestNumber: Mantissa { 9_999_999_999_999_999 }
  
  // Two mantissa sizes must be supported
  public static var exponentLMBits:    ClosedRange<Int> { 53...62 }
  public static var largeMantissaBits: ClosedRange<Int> { 0...52 }
  
  public static var exponentSMBits:    ClosedRange<Int> { 51...60 }
  public static var smallMantissaBits: ClosedRange<Int> { 0...50 }
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
public struct Decimal64 {
  
  var x: IntegerDecimal64
  
  func add(_ y: Decimal64) -> Decimal64 {
    if self.x.isInfinite { return self }
    if y.x.isInfinite { return y }
    return self
  }
  
}
