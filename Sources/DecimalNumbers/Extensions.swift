//
//  Extensions.swift
//  
//
//  Created by Mike Griebling on 21.05.2023.
//

import UInt128

extension UInt128 {
  public init(w: [UInt64]) {
    self.init((UInt128.High(w[1]), UInt128.Low(w[0])))
  }
}

extension FixedWidthInteger {
  @_semantics("optimize.sil.specialize.generic.partial.never")
  public // @testable
  static func _convert<Source: DecimalFloatingPoint>(
    from source: Source
  ) -> (value: Self?, exact: Bool) {
    guard _fastPath(!source.isZero) else { return (0, true) }
    guard _fastPath(source.isFinite) else { return (nil, false) }
    guard Self.isSigned || source > -1 else { return (nil, false) }
    let exponent = Int(source.exponent)
    let destMaxDigits = _decimalLogarithm(Self.max)
    let digitWidth = source.significandDigitCount // exact digit width
    if _slowPath(digitWidth+exponent <= destMaxDigits) { return (nil, false) }
    let isExact = exponent >= 0
    let bitPattern = Self.Magnitude(source.significandBitPattern) // all digits
    
    // Determine the actual number of integral significand digits.
    // We can ignore any fraction since we are rounding to zero
    let shift = Int(exponent)
    // Use `Self.Magnitude` to prevent sign extension if `shift < 0`.
    let shiftedDigits = shift >= 0
        ? bitPattern * power(10, to:shift)
        : bitPattern / power(10, to:-shift)
    if _slowPath(Self.isSigned && Self.bitWidth &- 1 == exponent) {
      return source < 0 && shiftedDigits == 0
        ? (Self.min, isExact)
        : (nil, false)
    }
    let magnitude = shiftedDigits
    return (
      Self.isSigned && source < 0 ? 0 &- Self(magnitude) : Self(magnitude),
      isExact)
  }
  
  /// Creates an integer from the given floating-point value, rounding toward
  /// zero. Any fractional part of the value passed as `source` is removed.
  ///
  ///     let x = Int(21.5)
  ///     // x == 21
  ///     let y = Int(-21.5)
  ///     // y == -21
  ///
  /// If `source` is outside the bounds of this type after rounding toward
  /// zero, a runtime error may occur.
  ///
  ///     let z = UInt(-21.5)
  ///     // Error: ...outside the representable range
  ///
  /// - Parameter source: A floating-point value to convert to an integer.
  ///   `source` must be representable in this type after rounding toward
  ///   zero.
  @inlinable
  @_semantics("optimize.sil.specialize.generic.partial.never")
  @inline(__always)
  public init<T: DecimalFloatingPoint>(_ source: T) {
    guard let value = Self._convert(from: source).value else {
      fatalError("""
        \(T.self) value cannot be converted to \(Self.self) because it is \
        outside the representable range
        """)
    }
    self = value
  }
  
  /// Creates an integer from the given floating-point value, if it can be
  /// represented exactly.
  ///
  /// If the value passed as `source` is not representable exactly, the result
  /// is `nil`. In the following example, the constant `x` is successfully
  /// created from a value of `21.0`, while the attempt to initialize the
  /// constant `y` from `21.5` fails:
  ///
  ///     let x = Int(exactly: 21.0)
  ///     // x == Optional(21)
  ///     let y = Int(exactly: 21.5)
  ///     // y == nil
  ///
  /// - Parameter source: A floating-point value to convert to an integer.
  @_semantics("optimize.sil.specialize.generic.partial.never")
  @inlinable
  public init?<T: DecimalFloatingPoint>(exactly source: T) {
    let (temporary, exact) = Self._convert(from: source)
    guard exact, let value = temporary else {
      return nil
    }
    self = value
  }
}

/// Defines bit-related operations such as setting/getting bits of a number
extension FixedWidthInteger {
  private func mask(_ size: Int) -> Self { (Self(1) << size) - 1 }
  
  /// Returns the bits in the `range` of the current number where
  /// `range.lowerBound` ≥ 0 and the `range.upperBound` < Self.bitWidth
  public func get(range: IntRange) -> Self {
    precondition(range.lowerBound >= 0 && range.upperBound < Self.bitWidth)
    return (self >> range.lowerBound) & mask(range.count)
  }
  
  public func getInt(range: IntRange) -> Int {
    precondition(range.lowerBound >= 0 && range.upperBound < Self.bitWidth)
    precondition(range.count <= Int.bitWidth)
    return Int((self >> range.lowerBound) & mask(range.count))
  }
  
  /// Returns the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public func get(bit n: Int) -> Int {
    precondition(n >= 0 && n < Self.bitWidth)
    return ((1 << n) & self) == 0 ? 0 : 1
  }
  
  /// Logically inverts the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func toggle(bit n: Int) {
    precondition(n >= 0 && n < Self.bitWidth)
    self ^= 1 << n
  }
  
  /// Non-mutating version of the `toggle(bit:)` method.
  public func toggling(bit n: Int) -> Self {
    precondition(n >= 0 && n < Self.bitWidth)
    return self ^ (1 << n)
  }
  
  /// Sets to `0` the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func clear(bit n: Int) {
    precondition(n >= 0 && n < Self.bitWidth)
    self &= ~(1 << n)
  }
  
  /// Non-mutating version of the `clear(bit:)` method
  public func clearing(bit n: Int) -> Self {
    precondition(n >= 0 && n < Self.bitWidth)
    return self & ~(1 << n)
  }
  
  /// Sets to `1` the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func set(bit n: Int) {
    precondition(n >= 0 && n < Self.bitWidth)
    self |= 1 << n
  }
  
  /// Non-mutating version of the `set(bit:)` method.
  public func setting(bit n: Int) -> Self {
    precondition(n >= 0 && n < Self.bitWidth)
    return self | (1 << n)
  }
  
  /// Replaces the `n`th bit of the current number with `value` where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func set(bit n: Int, with value: Int) {
    value.isMultiple(of: 2) ? self.clear(bit: n) : self.set(bit: n)
  }
  
  /// Non-mutating version of the `set(bit:value:)` method.
  public func setting(bit n: Int, with value: Int) -> Self {
    value.isMultiple(of: 2) ? self.clearing(bit: n) : self.setting(bit: n)
  }
  
  /// Sets to `0` the bits in the `range` of the current number where
  /// `range.lowerBound` ≥ 0 and the `range.upperBound` < Self.bitWidth
  public mutating func clear(range: IntRange) {
    precondition(range.lowerBound >= 0 && range.upperBound < Self.bitWidth)
    self &= ~(mask(range.count) << range.lowerBound)
  }
  
  /// Nonmutating version of the `clear(range:)` method.
  public func clearing(range: IntRange) -> Self {
    precondition(range.lowerBound >= 0 && range.upperBound < Self.bitWidth)
    return self & ~(mask(range.count) << range.lowerBound)
  }
  
  /// Replaces the bits in the `range` of the current number where
  /// `range.lowerBound` ≥ 0 and the `range.upperBound` < Self.bitWidth
  public mutating func set<T:FixedWidthInteger>(range:IntRange, with value:T) {
    self.clear(range: range)
    self |= (Self(value) & mask(range.count)) << range.lowerBound
  }
  
  /// Nonmutating version of the `set(range:)` method.
  public func setting<T:FixedWidthInteger>(range:IntRange,with value:T)->Self {
    let x = self.clearing(range: range)
    return x | (Self(value) & mask(range.count)) << range.lowerBound
  }
}
