//
//  Extensions.swift
//  
//
//  Created by Mike Griebling on 21.05.2023.
//

import UInt128

extension UInt128 {
  @available(macOS 13.3, iOS 16.4, macCatalyst 16.4, tvOS 16.4, watchOS 9.4, *)
  public init(integerLiteral value: StaticBigInt) {
    precondition(value.signum() >= 0, "UInt128 literal cannot be negative")
    precondition(value.bitWidth <= Self.bitWidth,
                 "\(value.bitWidth)-bit literal too large for UInt128")
    precondition(Low.bitWidth == 64, "Expecting 64-bit UInt")
    self.init(high: High(value[1]), low: Low(value[0]))
  }
  
  @available(macOS 10, *)
  public init(integerLiteral value: IntegerLiteralType) {
    self.init(high: 0, low: Low(value))
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
  public func get(range: ClosedRange<Int>) -> Int {
    precondition(range.lowerBound >= 0 && range.upperBound < Self.bitWidth)
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
  
  /// Sets to `0` the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func clear(bit n: Int) {
    precondition(n >= 0 && n < Self.bitWidth)
    self &= ~(1 << n)
  }
  
  /// Sets to `1` the `n`th bit of the current number where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func set(bit n: Int) {
    precondition(n >= 0 && n < Self.bitWidth)
    self |= 1 << n
  }
  
  /// Replaces the `n`th bit of the current number with `value` where
  /// 0 ≤ `n` < Self.bitWidth
  public mutating func set(bit n: Int, with value: Int) {
    value.isMultiple(of: 2) ? self.clear(bit: n) : self.set(bit: n)
  }
  
  /// Sets to `0` the bits in the `range` of the current number where
  /// `range.lowerBound` ≥ 0 and the `range.upperBound` < Self.bitWidth
  public mutating func clear(range: ClosedRange<Int>) {
    precondition(range.lowerBound >= 0 && range.upperBound < Int.bitWidth)
    self &= ~(mask(range.count) << range.lowerBound)
  }
  
  /// Replaces the bits in the `range` of the current number where
  /// `range.lowerBound` ≥ 0 and the `range.upperBound` < Self.bitWidth
  public mutating func set(range: ClosedRange<Int>, with value: Int) {
    self.clear(range: range)
    self |= (Self(value) & mask(range.count)) << range.lowerBound
  }
}
