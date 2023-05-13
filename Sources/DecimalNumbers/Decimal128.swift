//
//  Decimal128.swift
//  
//
//  Created by Mike Griebling on 12.05.2023.
//

import UInt128

/// Define the data fields for the Decimal32 storage type
struct IntegerDecimal128 : IntegerDecimal {
  typealias RawDataFields = UInt128
  typealias Mantissa = UInt128
  
  var data: RawDataFields = 0
  
  init(_ word: RawDataFields) {
    self.data = word
  }
  
  init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  static var exponentBias:    Int {  6176 }
  static var maximumExponent: Int {  6144 } // unbiased & normal
  static var minimumExponent: Int { -6143 } // unbiased & normal
  static var numberOfDigits:  Int {    34 }
  
  static var largestNumber: Mantissa {
    UInt128("9999999999999999999999999999999999")!
  }
  
  // Two mantissa sizes must be supported
  static var exponentLMBits:    ClosedRange<Int> { 113...124 }
  static var largeMantissaBits: ClosedRange<Int> {   0...112 }
  
  static var exponentSMBits:    ClosedRange<Int> { 111...122 }
  static var smallMantissaBits: ClosedRange<Int> {   0...110 }
}

/// Implementation of the Decimal32 data type
public struct Decimal128 {
  
  var x: IntegerDecimal128
  
  func add(_ y: Self) -> Self {
    if self.x.isInfinite { return self }
    if y.x.isInfinite { return y }
    return self
  }
  
}
