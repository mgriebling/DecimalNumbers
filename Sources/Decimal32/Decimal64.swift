//
//  Decimal64.swift
//  
//
//  Created by Mike Griebling on 12.05.2023.
//

/// Define the data fields for the Decimal32 storage type
struct IntegerDecimal64 : IntegerDecimalField {
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
  var exponentBias:    Int {  398 }
  var maximumExponent: Int {  384 } // unbiased & normal
  var minimumExponent: Int { -383 } // unbiased & normal
  var numberOfDigits:  Int {   16 }
  
  var largestNumber: Mantissa { 9_999_999_999_999_999 }
  
  // Two mantissa sizes must be supported
  var exponentLMBits:    ClosedRange<Int> { 53...60 }
  var largeMantissaBits: ClosedRange<Int> { 0...52 }
  
  var exponentSMBits:    ClosedRange<Int> { 51...58 }
  var smallMantissaBits: ClosedRange<Int> { 0...50 }
}

/// Implementation of the Decimal32 data type
public struct Decimal64 {
  
  var x: IntegerDecimal64
  
  func add(_ y: Decimal64) -> Decimal64 {
    if self.x.isInfinite { return self }
    if y.x.isInfinite { return y }
    let signx = self.x.sign
    let signy = y.x.sign
    return self
  }
  
}
