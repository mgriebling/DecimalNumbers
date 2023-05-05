//
//  Common.swift
//  
//
//  Created by Mike Griebling on 05.05.2023.
//

///
/// Groups together algorithms that can be used by all Decimalxx variants

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

