/**
Copyright © 2023 Computer Inspirations. All rights reserved.
Portions are Copyright (c) 2014 - 2021 Apple Inc. and the
Swift project authors

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

///
/// Groups together algorithms that can be used by all Decimalxx variants
///

// MARK: - Generic Integer Decimal Field Type

protocol IntegerDecimalField {
  
  associatedtype RawDataFields : UnsignedInteger & FixedWidthInteger
  associatedtype Mantissa : UnsignedInteger
  
  var data: RawDataFields { get set }
  
  //////////////////////////////////////////////////////////////////
  /// Initializers
  
  /// Initialize with a raw data word
  init(_ word: RawDataFields)
  
  /// Initialize with sign, exponent, and mantissa
  init(sign: FloatingPointSign, exponent: Int, mantissa: Mantissa)
  
  //////////////////////////////////////////////////////////////////
  /// Essential data to extract or update from the fields
  
  /// Sign of the number
  var sign: FloatingPointSign { get set }
  
  /// Unbiased signed exponent of the number
  var exponent: Int { get }
  
  /// Unsigned mantissa integer
  var mantissa: Mantissa { get }
  
  /// Setting requires both the mantissa and exponent so that a
  /// decision can be made on whether the mantissa is small or large.
  mutating func set(exponent: Int, mantissa: Mantissa)
  
  //////////////////////////////////////////////////////////////////
  /// Decimal number definitions
  var signBit: ClosedRange<Int> { get }
  var specialBits: ClosedRange<Int> { get }
  
  var exponentBias: Int    { get }
  var maximumExponent: Int { get } // unbiased & normal
  var minimumExponent: Int { get } // unbiased & normal
  var numberOfDigits:  Int { get }
  
  var largestNumber: Mantissa { get }
  
  // For large mantissa
  var exponentLMBits: ClosedRange<Int> { get }
  var largeMantissaBits: ClosedRange<Int> { get }
  
  // For small mantissa
  var exponentSMBits: ClosedRange<Int> { get }
  var smallMantissaBits: ClosedRange<Int> { get }
}

///
/// Free functionality when complying with IntegerDecimalField
extension IntegerDecimalField {
  
  var highMantissaBit: Int { 1 << (smallMantissaBits.upperBound+3) }
  
  /// These bit fields can be predetermined just from the size of
  /// the number type `RawDataFields` `bitWidth`
  var maxBit: Int { RawDataFields.bitWidth - 1 }
  
  var signBit:     ClosedRange<Int> { maxBit ... maxBit }
  var specialBits: ClosedRange<Int> { maxBit-2 ... maxBit-1 }
  
  func getBits(_ range: ClosedRange<Int>) -> Int {
    guard data.bitWidth > range.upperBound else { return 0 }
    let mask = (RawDataFields(1) << (range.upperBound - range.lowerBound)) - 1
    let sdata = (data >> range.lowerBound) & mask
    return Int(sdata)
  }
  
  mutating func setBits(_ range: ClosedRange<Int>, bits: Int) {
    guard data.bitWidth > range.upperBound else { return }
    let width = range.upperBound - range.lowerBound
    let mask = (RawDataFields(1) << width) - 1
    let mbits = RawDataFields(bits) & mask    // limit `bits` size
    let smask = ~(mask << range.lowerBound)   // inverted mask
    data = (data & smask) | (mbits << range.lowerBound)
  }
  
  var sign: FloatingPointSign {
    get {
      let s = getBits(signBit)
      return s == 0 ? .plus : .minus
    }
    set {
      setBits(signBit, bits: newValue == .minus ? 1 : 0)
    }
  }
  
  var exponent: Int {
    if isSmallMantissa {
      return getBits(exponentSMBits) - exponentBias
    } else {
      return getBits(exponentLMBits) - exponentBias
    }
  }
  
  var mantissa: Mantissa {
    if isSmallMantissa {
      return Mantissa(getBits(smallMantissaBits) + highMantissaBit)
    } else {
      return Mantissa(getBits(largeMantissaBits))
    }
  }
  
  mutating func set(exponent: Int, mantissa: Mantissa) {
    if mantissa > highMantissaBit {
      // small mantissa
      setBits(exponentSMBits, bits: exponent + exponentBias)
      setBits(smallMantissaBits, bits: Int(mantissa) - highMantissaBit)
    } else {
      // large mantissa
      setBits(exponentLMBits, bits: exponent + exponentBias)
      setBits(largeMantissaBits, bits: Int(mantissa))
    }
  }
  
  /// Return `self's` pieces all at once
  func unpack() ->
    (sign: FloatingPointSign, exponent: Int, mantissa: Mantissa, valid: Bool) {
      let exponent: Int, mantissa: Mantissa
      if isSmallMantissa {
        // small mantissa
        exponent = getBits(exponentSMBits) - exponentBias
        mantissa = Mantissa(getBits(smallMantissaBits) + highMantissaBit)
      } else {
        // large mantissa
        exponent = getBits(exponentLMBits) - exponentBias
        mantissa = Mantissa(getBits(largeMantissaBits))
      }
      return (self.sign, exponent, mantissa, self.isValid)
  }
  
  /// Handy routines for testing different aspects of the number
  
  var isSmallMantissa: Bool {
    let range = specialBits.lowerBound...specialBits.upperBound
    return isSpecial && getBits(range) != 0b11
  }
  
  var isSpecial: Bool {
    getBits(specialBits) == 0b11
  }
  
  var isInfinite: Bool {
    let range = signBit.lowerBound-5...signBit.lowerBound-1
    return getBits(range) == 0x1_1110
  }
  
  var isNaN: Bool {
    let range = signBit.lowerBound-6...signBit.lowerBound-1
    return getBits(range) == 0x1_1111_0
  }
  
  var isSNaN: Bool {
    let range = signBit.lowerBound-6...signBit.lowerBound-1
    return getBits(range) == 0x1_1111_1
  }
  
  var isValid: Bool {
    let range = signBit.lowerBound-4...signBit.lowerBound-1
    return getBits(range) != 0x1_111
  }
  
  var isZero: Bool {
    guard isValid else { return false }
    return mantissa == 0 || mantissa > largestNumber
  }
  
}

// MARK: - Extended UInt Definitions
// Use these for now
struct UInt512 { var w = [UInt64](repeating: 0, count: 8) }
struct UInt384 { var w = [UInt64](repeating: 0, count: 6) }
struct UInt256 { var w = [UInt64](repeating: 0, count: 4) }
struct UInt192 { var w = [UInt64](repeating: 0, count: 3) }

// MARK: - Status and Rounding Type Definitions

public typealias Rounding = FloatingPointRoundingRule
let BID_ROUNDING_UP = Rounding.up
let BID_ROUNDING_DOWN = Rounding.down
let BID_ROUNDING_TO_ZERO = Rounding.towardZero
let BID_ROUNDING_TO_NEAREST = Rounding.toNearestOrEven
let BID_ROUNDING_TIES_AWAY = Rounding.toNearestOrAwayFromZero

public struct Status: OptionSet, CustomStringConvertible {
    public let rawValue: Int32
    
    /* IEEE extended flags only */
    private static let DEC_Conversion_syntax    = 0x00000001
    private static let DEC_Division_by_zero     = 0x00000002
    private static let DEC_Division_impossible  = 0x00000004
    private static let DEC_Division_undefined   = 0x00000008
    private static let DEC_Insufficient_storage = 0x00000010 /* [when malloc fails]  */
    private static let DEC_Inexact              = 0x00000020
    private static let DEC_Invalid_context      = 0x00000040
    private static let DEC_Invalid_operation    = 0x00000080
    private static let DEC_Lost_digits          = 0x00000100
    private static let DEC_Overflow             = 0x00000200
    private static let DEC_Clamped              = 0x00000400
    private static let DEC_Rounded              = 0x00000800
    private static let DEC_Subnormal            = 0x00001000
    private static let DEC_Underflow            = 0x00002000
    
    public static let conversionSyntax    = Status(rawValue: Int32(DEC_Conversion_syntax))
    public static let divisionByZero      = Status(rawValue: Int32(DEC_Division_by_zero))
    public static let divisionImpossible  = Status(rawValue: Int32(DEC_Division_impossible))
    public static let divisionUndefined   = Status(rawValue: Int32(DEC_Division_undefined))
    public static let insufficientStorage = Status(rawValue: Int32(DEC_Insufficient_storage))
    public static let inexact             = Status(rawValue: Int32(DEC_Inexact))
    public static let invalidContext      = Status(rawValue: Int32(DEC_Invalid_context))
    public static let lostDigits          = Status(rawValue: Int32(DEC_Lost_digits))
    public static let invalidOperation    = Status(rawValue: Int32(DEC_Invalid_operation))
    public static let overflow            = Status(rawValue: Int32(DEC_Overflow))
    public static let clamped             = Status(rawValue: Int32(DEC_Clamped))
    public static let rounded             = Status(rawValue: Int32(DEC_Rounded))
    public static let subnormal           = Status(rawValue: Int32(DEC_Subnormal))
    public static let underflow           = Status(rawValue: Int32(DEC_Underflow))
    public static let clearFlags          = Status([])
    
    public static let errorFlags = Status(rawValue: Int32(DEC_Division_by_zero | DEC_Overflow |
        DEC_Underflow | DEC_Conversion_syntax | DEC_Division_impossible |
        DEC_Division_undefined | DEC_Insufficient_storage | DEC_Invalid_context | DEC_Invalid_operation))
    public static let informationFlags = Status(rawValue: Int32(DEC_Clamped | DEC_Rounded |
        DEC_Inexact | DEC_Lost_digits))
    
    public init(rawValue: Int32) { self.rawValue = rawValue }
    
    public var hasError: Bool { !Status.errorFlags.intersection(self).isEmpty }
    public var hasInfo: Bool { !Status.informationFlags.intersection(self).isEmpty }
    
    public var description: String {
        var str = ""
        if self.contains(.conversionSyntax)    { str += "Conversion syntax, "}
        if self.contains(.divisionByZero)      { str += "Division by zero, " }
        if self.contains(.divisionImpossible)  { str += "Division impossible, "}
        if self.contains(.divisionUndefined)   { str += "Division undefined, "}
        if self.contains(.insufficientStorage) { str += "Insufficient storage, " }
        if self.contains(.inexact)             { str += "Inexact number, " }
        if self.contains(.invalidContext)      { str += "Invalid context, " }
        if self.contains(.invalidOperation)    { str += "Invalid operation, " }
        if self.contains(.lostDigits)          { str += "Lost digits, " }
        if self.contains(.overflow)            { str += "Overflow, " }
        if self.contains(.clamped)             { str += "Clamped, " }
        if self.contains(.rounded)             { str += "Rounded, " }
        if self.contains(.subnormal)           { str += "Subnormal, " }
        if self.contains(.underflow)           { str += "Underflow, " }
        if str.hasSuffix(", ") { str.removeLast(2) }
        return str
    }
}

// MARK: - Common Utility Functions

internal func addDecimalPointAndExponent(_ ps:String, _ exponent:Int,
                                         _ maxDigits:Int) -> String {
  var digits = ps.count
  var ps = ps
  var exponent_x = exponent
  if exponent_x == 0 {
    ps.insert(".", at: ps.index(ps.startIndex, offsetBy: exponent_x+1))
  } else if abs(exponent_x) > maxDigits {
    if ps.count > 1 {
      ps.insert(".", at: ps.index(after: ps.startIndex))
    }
    ps += "e"
    if exponent_x < 0 {
      ps += "-"
      exponent_x = -exponent_x
    } else {
      ps += "+"
    }
    ps += String(exponent_x)
  } else if digits <= exponent_x {
    // format the number without an exponent
    while digits <= exponent_x {
      // pad the number with zeros
      ps += "0"; digits += 1
    }
  } else if exponent_x < 0 {
    while exponent_x < -1 {
      // insert leading zeros
      ps = "0" + ps; exponent_x += 1
    }
    ps = "0." + ps
  } else {
    // insert the decimal point
    ps.insert(".", at: ps.index(ps.startIndex, offsetBy: exponent_x+1))
    if ps.hasSuffix(".") { ps.removeLast() }
  }
  return ps
}

// Rounding boundaries table, indexed by
// 4 * rounding_mode + 2 * sign + lsb of truncation
// We round up if the round/sticky data is strictly > this boundary
//
// NB: This depends on the particular values of the rounding mode
// numbers, which are supposed to be defined as here:
//
// #define BID_ROUNDING_TO_NEAREST     0x00000
// #define BID_ROUNDING_DOWN           0x00001
// #define BID_ROUNDING_UP             0x00002
// #define BID_ROUNDING_TO_ZERO        0x00003
// #define BID_ROUNDING_TIES_AWAY      0x00004
//
// Some of the shortcuts below in "underflow after rounding" also use
// the concrete values.
//
// So we add a directive here to double-check that this is the case
internal func roundboundIndex(_ round:Rounding, _ negative:Bool=false,
                            _ lsb:Int=0) -> Int {
  var index = (lsb & 1) + (negative ? 2 : 0)
  switch round {
    case BID_ROUNDING_TO_NEAREST: index += 0
    case BID_ROUNDING_DOWN: index += 4
    case BID_ROUNDING_UP: index += 8
    case BID_ROUNDING_TO_ZERO: index += 12
    default: index += 16
  }
  return index
}

// MARK: - Generic String Conversion functions

/// Converts a decimal floating point number `x` into a string
internal func string<T:DecimalFloatingPoint>(from x: T) -> String {
  // unpack arguments, check for NaN or Infinity
  let (negative, exp, coeff, valid) = x.unpack()
  let s = negative ? "-" : ""
  if valid {
    // x is not special
    let ps = String(coeff)
    let exponent_x = Int(exp) - T.exponentBias + (ps.count - 1)
    return s + addDecimalPointAndExponent(ps, exponent_x,
                                           T.significandMaxDigitCount)
  } else {
    // x is Inf. or NaN or 0
    var ps = s
    if x.isNaN {
      if x.isSignalingNaN { ps.append("S") }
      ps.append("NaN")
      return ps
    }
    if x.isInfinite {
      ps.append("Inf")
      return ps
    }
    ps.append("0")
    return ps
  }
}

/// Converts a decimal number string of the form:
/// `[+|-] digit {digit} [. digit {digit}] [e [+|-] digit {digit} ]`
/// to a Decimal<n> number
internal func numberFromString<T:DecimalFloatingPoint>(_ s: String) -> T? {
  // keep consistent character case for "infinity", "nan", etc.
  var ps = s.lowercased()
  
  // remove leading whitespace characters
  while ps.hasPrefix(" ") { ps.removeFirst() }
  
  // get first non-whitespace character
  var c = ps.isEmpty ? "\0" : ps.removeFirst()
  
  // detect special cases (INF or NaN)
  if c == "\0" || (c != "." && c != "-" && c != "+" && (c < "0" || c > "9")){
    // Infinity?
    if c == "i" && (ps.hasPrefix("nfinity") || ps.hasPrefix("nf")) {
      return T.infinity
    }
    // return sNaN
    if c == "s" && ps.hasPrefix("nan") {
      // case insensitive check for snan
      return T.signalingNaN
    } else {
      // return qNaN & any coefficient
      let coeff = Int(ps.dropFirst(2)) ?? 0 // drop "AN"
      return T.nan(with: coeff)
    }
  }
  
  // detect +INF or -INF
  if ps.hasPrefix("infinity") || ps.hasPrefix("inf") {
    if c == "+" {
      return T.infinity
    } else if c == "-" {
      return -T.infinity
    } else {
      return T.nan
    }
  }
  
  // if +sNaN, +SNaN, -sNaN, or -SNaN
  if ps.hasPrefix("snan") {
    if c == "-" {
      return -T.signalingNaN
    } else {
      return T.signalingNaN
    }
  }
  
  // determine sign
  var isNegative = false
  if c == "-" {
    isNegative = true
  }
  
  // get next character if leading +/- sign
  if c == "-" || c == "+" {
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  // if c isn"t a decimal point or a decimal digit, return NaN
  if c != "." && (c < "0" || c > "9") {
    // return NaN
    return isNegative ? -T.nan : T.nan
  }
  
  var rdx_pt_enc = false
  var right_radix_leading_zeros = 0
  var coefficient_x = 0
  
  // detect zero (and eliminate/ignore leading zeros)
  if c == "0" || c == "." {
    if c == "." {
      rdx_pt_enc = true
      c = ps.isEmpty ? "\0" : ps.removeFirst()
    }
    // if all numbers are zeros (with possibly 1 radix point, the number
    // is zero
    // should catch cases such as: 000.0
    while c == "0" {
      c = ps.isEmpty ? "\0" : ps.removeFirst()
      // for numbers such as 0.0000000000000000000000000000000000001001,
      // we want to count the leading zeros
      if rdx_pt_enc {
        right_radix_leading_zeros+=1
      }
      // if this character is a radix point, make sure we haven't already
      // encountered one
      if c == "." {
        if !rdx_pt_enc {
          rdx_pt_enc = true
          // if this is the first radix point, and the next character is
          // NULL, we have a zero
          if ps.isEmpty {
            right_radix_leading_zeros = T.exponentBias -
            right_radix_leading_zeros
            if right_radix_leading_zeros < 0 {
              right_radix_leading_zeros = 0
            }
            let bits = T.BitPattern(right_radix_leading_zeros) << 23
            return isNegative ? -T(bitPattern: bits, bidEncoding: true) :
            T(bitPattern: bits, bidEncoding: true)
          }
          c = ps.isEmpty ? "\0" : ps.removeFirst()
        } else {
          // if 2 radix points, return NaN
          return isNegative ? -T.nan : T.nan
        }
      } else if ps.isEmpty {
        right_radix_leading_zeros = T.exponentBias -
        right_radix_leading_zeros
        if right_radix_leading_zeros < 0 {
          right_radix_leading_zeros = 0
        }
        let bits = T.BitPattern(right_radix_leading_zeros) << 23
        return isNegative ? -T(bitPattern: bits, bidEncoding: true) :
        T(bitPattern: bits, bidEncoding: true)
      }
    }
  }
  
  var ndigits = 0
  var dec_expon_scale = 0
  var midpoint = 0
  var rounded_up = 0
  var add_expon = 0
  var rounded = 0
  while (c >= "0" && c <= "9") || c == "." {
    if c == "." {
      if rdx_pt_enc {
        // return NaN
        return isNegative ? -T.nan : T.nan
      }
      rdx_pt_enc = true
      c = ps.isEmpty ? "\0" : ps.removeFirst()
      continue
    }
    if rdx_pt_enc { dec_expon_scale += 1 }
    
    ndigits+=1
    if ndigits <= 7 {
      coefficient_x = (coefficient_x << 1) + (coefficient_x << 3);
      coefficient_x += c.wholeNumberValue ?? 0
    } else if ndigits == 8 {
      // coefficient rounding
      switch T.rounding {
        case BID_ROUNDING_TO_NEAREST:
          midpoint = (c == "5" && (coefficient_x & 1 == 0)) ? 1 : 0;
          // if coefficient is even and c is 5, prepare to round up if
          // subsequent digit is nonzero
          // if str[MAXDIG+1] > 5, we MUST round up
          // if str[MAXDIG+1] == 5 and coefficient is ODD, ROUND UP!
          if c > "5" || (c == "5" && (coefficient_x & 1) != 0) {
            coefficient_x+=1
            rounded_up = 1
          }
        case BID_ROUNDING_DOWN:
          if isNegative { coefficient_x+=1; rounded_up=1 }
        case BID_ROUNDING_UP:
          if !isNegative { coefficient_x+=1; rounded_up=1 }
        case BID_ROUNDING_TIES_AWAY:
          if c >= "5" { coefficient_x+=1; rounded_up=1 }
        default: break
      }
      if coefficient_x == 10000000 {
        coefficient_x = 1000000
        add_expon = 1;
      }
      if c > "0" {
        rounded = 1;
      }
      add_expon += 1;
    } else { // ndigits > 8
      add_expon+=1
      if midpoint != 0 && c > "0" {
        coefficient_x+=1
        midpoint = 0;
        rounded_up = 1;
      }
      if c > "0" {
        rounded = 1;
      }
    }
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  add_expon -= dec_expon_scale + Int(right_radix_leading_zeros)
  
  if c == "\0" {
    if rounded != 0 {
      T.state.insert(.inexact)
    }
    return T(isNegative: isNegative, exponent: add_expon + T.exponentBias,
             mantissa: coefficient_x, round: 0)
  }
  
  if c != "e" {
    // return NaN
    return isNegative ? -T.nan : T.nan
  }
  c = ps.isEmpty ? "\0" : ps.removeFirst()
  let sgn_expon = (c == "-") ? 1 : 0
  var expon_x = 0
  if c == "-" || c == "+" {
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  if c == "\0" || c < "0" || c > "9" {
    // return NaN
    return isNegative ? -T.nan : T.nan
  }
  
  while (c >= "0") && (c <= "9") {
    if expon_x < (1<<20) {
      expon_x = (expon_x << 1) + (expon_x << 3)
      expon_x += c.wholeNumberValue ?? 0
    }
    c = ps.isEmpty ? "\0" : ps.removeFirst()
  }
  
  if c != "\0" {
    // return NaN
    return isNegative ? -T.nan : T.nan
  }
  
  if rounded != 0 {
    T.state.insert(.inexact)
  }
  
  if sgn_expon != 0 {
    expon_x = -expon_x
  }
  
  expon_x += add_expon + T.exponentBias
  
  if expon_x < 0 {
    if rounded_up != 0 {
      coefficient_x-=1
    }
  }
  return T(isNegative: isNegative, exponent: expon_x, mantissa: coefficient_x,
           round: rounded)
}

/// Returns x^exp where x = *num*.
/// - Precondition: x ≥ 0, exp ≥ 0
internal func power<T:FixedWidthInteger>(_ num:T, to exp: Int) -> T {
  // Zero raised to anything except zero is zero (provided exponent is valid)
  guard exp >= 0 else { return T.max }
  if num == 0 { return exp == 0 ? 1 : 0 }
  var z = num
  var y : T = 1
  var n = abs(exp)
  while true {
    if !n.isMultiple(of: 2) { y *= z }
    n >>= 1
    if n == 0 { break }
    z *= z
  }
  return y
}

