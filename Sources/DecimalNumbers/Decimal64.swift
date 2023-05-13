//
//  Decimal64.swift
//  
//
//  Created by Mike Griebling on 12.05.2023.
//

/// Define the data fields for the Decimal32 storage type
struct IntegerDecimal64 : IntegerDecimal {
  typealias RawDataFields = UInt64
  typealias Mantissa = UInt64
  
  var data: RawDataFields = 0
  
  init(_ word: RawDataFields) {
    self.data = word
  }
  
  init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  static var exponentBias:    Int {  398 }
  static var maximumExponent: Int {  384 } // unbiased & normal
  static var minimumExponent: Int { -383 } // unbiased & normal
  static var numberOfDigits:  Int {   16 }
  
  static var largestNumber: Mantissa { 9_999_999_999_999_999 }
  
  // Two mantissa sizes must be supported
  static var exponentLMBits:    ClosedRange<Int> { 53...60 }
  static var largeMantissaBits: ClosedRange<Int> { 0...52 }
  
  static var exponentSMBits:    ClosedRange<Int> { 51...58 }
  static var smallMantissaBits: ClosedRange<Int> { 0...50 }
}

/// Implementation of the Decimal32 data type
public struct Decimal64 {
  
  var x: IntegerDecimal64
  
  func add(_ y: Decimal64) -> Decimal64 {
    if self.x.isInfinite { return self }
    if y.x.isInfinite { return y }
    return self
  }
  
}
