//
//  Decimal.swift
//  
//
//  Created by Mike Griebling on 12.05.2023.
//

/// Define the data fields for the Decimal32 storage type
struct IntegerDecimal32 : IntegerDecimalField {
  typealias RawDataFields = UInt32
  typealias Mantissa = UInt
  
  var data: RawDataFields = 0
  
  init(_ word: RawDataFields) {
    self.data = word
  }
  
  init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa) {
    self.sign = sign
    self.set(exponent: exponent, mantissa: mantissa)
  }
  
  // Define the fields and required parameters
  var exponentBias:    Int { 101 }
  var maximumExponent: Int {  96 } // unbiased & normal
  var minimumExponent: Int { -95 } // unbiased & normal
  var numberOfDigits:  Int {   7 }
  
  var largestNumber: Mantissa { 9_999_999 }
  
  // Two mantissa sizes must be supported
  var exponentLMBits:    ClosedRange<Int> { 23...30 }
  var largeMantissaBits: ClosedRange<Int> { 0...22 }
  
  var exponentSMBits:    ClosedRange<Int> { 21...28 }
  var smallMantissaBits: ClosedRange<Int> { 0...20 }
}

/// Implementation of the Decimal32 data type
public struct Decimal32 {
  
  var x: IntegerDecimal32
  
  func add(_ y: Self) -> Self {
    let (signX, exponentX, mantissaX, validX) = self.x.unpack()
    let (signY, exponentY, mantissaY, validY) = y.x.unpack()
    if validX && validY {

    }
    
    // Deal with illegal numbers
    if !validX { return self }
    if !validY { return y }

    return self
  }
  
  func test() {
    
  }
  
}
