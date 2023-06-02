//
//  Extensions.swift
//  
//
//  Created by Mike Griebling on 21.05.2023.
//

import UInt128

extension UInt128 {
  typealias IntegerLiteralType = StaticBigInt
  public init(integerLiteral value: StaticBigInt) {
    precondition(value.signum() >= 0, "UInt128 literal cannot be negative")
    precondition(value.bitWidth <= Self.bitWidth+1,
                 "\(value.bitWidth)-bit literal too large for UInt128")
    precondition(Low.bitWidth == 64, "Expecting 64-bit UInt")
    self.init(high: High(value[1]), low: Low(value[0]))
  }
}

extension UInt128 {
  public init(w: [UInt64]) {
    let high = UInt128.High(w[1])
    let low  = UInt128.Low(w[0])
    self.init(high: high, low: low)
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
