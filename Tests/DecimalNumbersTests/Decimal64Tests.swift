//
//  Decimal64Tests.swift
//  
//
//  Created by Mike Griebling on 31.05.2023.
//

import XCTest
@testable import DecimalNumbers
@testable import UInt128

final class Decimal64Tests: XCTestCase {
  
  func testGeneric() {
    typealias ID64 = IntDecimal64
    
    // Sanity check that the masks were generated correctly from bit defns
    XCTAssert(ID64.signBit == ID64.signBit)
    XCTAssert(UInt64(1) << ID64.signBit == UInt64(0x8000_0000_0000_0000))
    XCTAssert(Decimal64.nan.bid.data == 0x7c00_0000_0000_0000)
    XCTAssert(Decimal64.signalingNaN.bid.data == 0x7e00_0000_0000_0000)
    XCTAssert(Decimal64.infinity.bid.data == 0x7800_0000_0000_0000)
    XCTAssert(Decimal64.radix == 10)
    
    // back to default rounding mode
    // Decimal64.rounding = .toNearestOrEven
    let s = "123456789012345678901234567890"
    let y1 = Decimal64(stringLiteral: s)
    XCTAssert(y1.description == "1.234567890123457e+29")
    print("\(s) -> \(y1)")
    
    let y = Decimal64(stringLiteral: "234.5")
    XCTAssert(y.description == "234.5")
    let x = Decimal64(stringLiteral: "345.5")
    XCTAssert(x.description == "345.5")
    
    let n = UInt64(0xA2300000000003D0)
    var a = Decimal64(bitPattern: Decimal64.RawSignificand(n), encoding: .dpd)
    XCTAssert(a.description == "-7.50")
    print(a, a.bitPattern(.dpd) == n ? "a = n" : "a != n")
    XCTAssert(a.bitPattern(.dpd) == n)
    
    print("\(x) -> digits = \(x.significandDigitCount), " +
          "bcd = \(x.significandBitPattern)")
    XCTAssert(x.significandDigitCount == 4 && x.significandBitPattern == 3455)
    print("\(y) -> digits = \(y.significandDigitCount), " +
          "bcd = \(y.significandBitPattern)")
    XCTAssert(y.significandDigitCount == 4 && y.significandBitPattern == 2345)
    
    print(x, y, x*y, y/x, Int(x), Int(y), x.decade, y.decade)
    print(x.significand, x.exponent, y.significand, y.exponent)
    var b = Decimal64.leastNormalMagnitude
    print(Decimal64.greatestFiniteMagnitude,b,Decimal64.leastNonzeroMagnitude)

    let exponentBias = Decimal64.zero.exponentBitPattern
    let maxDigits = Decimal64.significandDigitCount
    XCTAssert(maxDigits == 16)
    XCTAssert(exponentBias == 398)
    print(Decimal64.greatestFiniteMagnitude.exponent)
    print(Decimal64.leastNormalMagnitude.exponent)
    XCTAssert(Decimal64.greatestFiniteMagnitude.exponent == 384 - maxDigits + 1)
    XCTAssert(Decimal64.leastNormalMagnitude.exponent == -383 - maxDigits + 1)
    
    let x5 = Decimal64("1000.3")
    print(String(x5.bitPattern(.bid), radix: 16), x5)
    XCTAssert(x5.bitPattern(.bid) == 0x31a0000000002713)
    XCTAssert(x5.bitPattern(.dpd) == 0x2234000000004003)
    print(String(x5.bitPattern(.dpd), radix: 16))
    
    a = "-21.5"; b = "305.15"
    let c = Decimal64(signOf: a, magnitudeOf: b)
    print(c); XCTAssert((-b) == c)
    
    a = Decimal64(sign: .plus, exponentBitPattern:UInt(exponentBias),
                  significandBitPattern: 1234)
    print(a); XCTAssert(a.description == "1234")
    
    a = Decimal64.random(in: 1..<1000)
    print(a); XCTAssert(a >= 1 && a < 1000)
    
    var numbers : [Decimal64] = [2.5, 21.25, 3.0, .nan, -9.5]
    let ordered : [Decimal64] = [-9.5, 2.5, 3.0, 21.25, .nan]
    numbers.sort { !$1.isTotallyOrdered(belowOrEqualTo: $0) }
    print(numbers)
    XCTAssert(ordered.description == numbers.description)
    
    print("Decimal64.zero =", Decimal64.zero)
    XCTAssert(Decimal64.zero.description == "0")
    print("Decimal64.pi =", Decimal64.pi)
    XCTAssert(Decimal64.pi.description == "3.141592653589793")
    print("Decimal64.nan =", Decimal64.nan)
    XCTAssert(Decimal64.nan.description == "NaN")
    print("Decimal64.signalingNaN =", Decimal64.signalingNaN)
    XCTAssert(Decimal64.signalingNaN.description == "SNaN")
    print("Decimal64.Infinity =", Decimal64.infinity)
    XCTAssert(Decimal64.infinity.description == "Inf")
    
    var a1 = Decimal64(8.625); let b1 = Decimal64(0.75)
    let rem = a1.remainder(dividingBy: b1)
    print("\(a1).formRemainder(dividingBy: \(b1) = ", rem)
    XCTAssert(rem == Decimal64(-0.375))
    a1 = Decimal64(8.625)
    let q = (a1/b1).rounded(.towardZero); print(q)
    a1 = a1 - q * b1
    print("\(a1)")

    // Equivalent to the C 'round' function:
    let w = Decimal64(6.5)
    print(w.rounded(.toNearestOrAwayFromZero))
    XCTAssert(w.rounded(.toNearestOrAwayFromZero) == Decimal64(7)) // w = 7.0

    // Equivalent to the C 'trunc' function:
    print(w.rounded(.towardZero))
    XCTAssert(w.rounded(.towardZero) == Decimal64(6)) // x = 6.0

    // Equivalent to the C 'ceil' function:
    print(w.rounded(.up))
    XCTAssert(w.rounded(.up) == Decimal64(7)) // w = 7.0

    // Equivalent to the C 'floor' function:
    print(w.rounded(.down))
    XCTAssert(w.rounded(.down) == Decimal64(6)) // x = 6.0
  }
  
  func testEncodingDecimal64() {
    // Test encoding for Decimal64 strings and integers
    var testNumber = 0
    
    func test(_ value: String, result: String) {
      testNumber += 1
      if let n = Decimal64(value) {
        print("Test \(testNumber): \"\(value)\" [\(n)] = \(result.lowercased()) - \(n.floatingPointClass.description)")
        var nstr = String(n.bitPattern(.dpd), radix:16)
        nstr = "".padding(toLength: result.count-nstr.count, withPad: "0", startingAt: 0) + nstr

        XCTAssertEqual(nstr, result.lowercased())
      } else {
        XCTAssert(false, "Failed to convert '\(value)'")
      }
    }
    
    func test(_ value: Int, result : String) {
      testNumber += 1
      let n = Decimal64(value)
      print("Test \(testNumber): \(value) [\(n)] = \(result.lowercased()) - \(n.floatingPointClass.description)")
      XCTAssertEqual(String(n.bitPattern(.dpd), radix:16), result.lowercased())
    }
    
    /// Check min/max values
    XCTAssertEqual(Decimal64.greatestFiniteMagnitude.description, "9.999999999999999e+384")
    XCTAssertEqual(Decimal64.leastNonzeroMagnitude.description,   "1e-398")
    XCTAssertEqual(Decimal64.leastNormalMagnitude.description,    "9.999999999999999e-383")
    
    /// Verify various string and integer encodings
    test("-7.50",             result: "A2300000000003D0")
    test("-7.50E+3",          result: "A23c0000000003D0")
    test("-750",              result: "A2380000000003D0")
    test("-75.0",             result: "A2340000000003D0")
    test("-0.750",            result: "A22C0000000003D0")
    test("-0.0750",           result: "A2280000000003D0")
    test("-0.000750",         result: "A2200000000003D0")
    test("-0.00000750",       result: "A2180000000003D0")
    test("-7.50E-7",          result: "A2140000000003D0")
    
    // Normality
    test(1234567890123456,    result: "263934b9c1e28e56")
    test(-1234567890123456,   result: "a63934b9c1e28e56")
    test("1234.567890123456", result: "260934b9c1e28e56")
    test(1111111111111111,    result: "2638912449124491")
    test(9999999999999999,    result: "6e38ff3fcff3fcff")
    
    // Nmax and similar
    test("9999999999999999E+369",  result: "77fcff3fcff3fcff")
    test("9.999999999999999E+384", result: "77fcff3fcff3fcff")
    test("1.234567890123456E+384", result: "47fd34b9c1e28e56")
    // fold-downs (more below)
    test("1.23E+384",           result: "47fd300000000000") // Clamped
    test("1E+384",              result: "47fc000000000000") // Clamped
    test(12345,                 result: "22380000000049c5")
    test(1234,                  result: "2238000000000534")
    test(123,                   result: "22380000000000a3")
    test(12,                    result: "2238000000000012")
    test(1,                     result: "2238000000000001")
    test("1.23",                result: "22300000000000a3")
    test("123.45",              result: "22300000000049c5")
    
    // Nmin and below
    test("1E-383"                , result: "003c000000000001")
    test("1.000000000000000E-383", result: "0400000000000000")
    test("1.000000000000001E-383", result: "0400000000000001")
    
    test("0.100000000000000E-383", result: "0000800000000000") // Subnormal
    test("0.000000000000010E-383", result: "0000000000000010") // Subnormal
    test("0.00000000000001E-383",  result: "0004000000000001") // Subnormal
    test("0.000000000000001E-383", result: "0000000000000001") // Subnormal
    // next is smallest all-nines
    test("9999999999999999E-398",  result: "6400ff3fcff3fcff")
    // and a problematic divide result
    test("1.111111111111111E-383", result: "0400912449124491")
    
    // forties
    test(40,        result: "2238000000000040")
    test("39.99",   result: "2230000000000cff")
    
    // underflows cannot be tested as all LHS exact
    
    // Same again, negatives
    // Nmax and similar
    test("-9.999999999999999E+384", result: "f7fcff3fcff3fcff")
    test("-1.234567890123456E+384", result: "c7fd34b9c1e28e56")
    // fold-downs (more below)
    test("-1.23E+384",              result: "c7fd300000000000") // Clamped
    test("-1E+384",                 result: "c7fc000000000000") // Clamped
    
    // overflows
    test(-12345,    result: "a2380000000049c5")
    test(-1234,     result: "a238000000000534")
    test(-123,      result: "a2380000000000a3")
    test(-12,       result: "a238000000000012")
    test(-1,        result: "a238000000000001")
    test("-1.23",   result: "a2300000000000a3")
    test("-123.45", result: "a2300000000049c5")
    
    // Nmin and below
    test("-1E-383",                 result: "803c000000000001")
    test("-1.000000000000000E-383", result: "8400000000000000")
    test("-1.000000000000001E-383", result: "8400000000000001")
    
    test("-0.100000000000000E-383", result: "8000800000000000") //        Subnormal
    test("-0.000000000000010E-383", result: "8000000000000010") //        Subnormal
    test("-0.00000000000001E-383",  result: "8004000000000001") //        Subnormal
    test("-0.000000000000001E-383", result: "8000000000000001") //        Subnormal
    // next is smallest all-nines
    test("-9999999999999999E-398",  result: "e400ff3fcff3fcff")
    // and a tricky subnormal
    test("1.11111111111524E-384",   result: "00009124491246a4") //       Subnormal
    
    // near-underflows
    test("-1e-398",   result: "8000000000000001") //   Subnormal
    test("-1.0e-398", result: "8000000000000001") //   Subnormal Rounded
    
    // zeros
    test("0E-500", result: "0000000000000000") //   Clamped
    test("0E-400", result: "0000000000000000") //   Clamped
    test("0E-398", result: "0000000000000000")
    test("0.000000000000000E-383", result: "0000000000000000")
    test("0E-2",   result: "2230000000000000")
    test("0",      result: "2238000000000000")
    test("0E+3",   result: "2244000000000000")
    test("0E+369", result: "43fc000000000000")
    // clamped zeros...
    test("0E+370", result: "43fc000000000000") //   Clamped
    test("0E+384", result: "43fc000000000000") //   Clamped
    test("0E+400", result: "43fc000000000000") //   Clamped
    test("0E+500", result: "43fc000000000000") //   Clamped
    
    // negative zeros
    test("-0E-400", result: "8000000000000000") //   Clamped
    test("-0E-400", result: "8000000000000000") //   Clamped
    test("-0E-398", result: "8000000000000000")
    test("-0.000000000000000E-383", result: "8000000000000000")
    test("-0E-2",   result: "a230000000000000")
    test("-0",      result: "a238000000000000")
    test("-0E+3",   result: "a244000000000000")
    test("-0E+369", result: "c3fc000000000000")
    // clamped zeros...
    test("-0E+370", result: "c3fc000000000000") //   Clamped
    test("-0E+384", result: "c3fc000000000000") //   Clamped
    test("-0E+400", result: "c3fc000000000000") //   Clamped
    test("-0E+500", result: "c3fc000000000000") //   Clamped
    
    // exponents
    test("7E+9",    result: "225c000000000007")
    test("7E+99",   result: "23c4000000000007")
    
    // diagnostic NaNs
    test("NaN",       result: "7c00000000000000")
    test("NaN0",      result: "7c00000000000000")
    test("NaN1",      result: "7c00000000000001")
    test("NaN12",     result: "7c00000000000012")
    test("NaN79",     result: "7c00000000000079")
    test("NaN12345",  result: "7c000000000049c5")
    test("NaN123456", result: "7c00000000028e56")
    test("NaN799799", result: "7c000000000f7fdf")
    test("NaN799799799799799", result: "7c03dff7fdff7fdf")
    test("NaN999999999999999", result: "7c00ff3fcff3fcff")
    
    // fold-down full sequence
    test("1E+384", result: "47fc000000000000") //  Clamped
    test("1E+383", result: "43fc800000000000") //  Clamped
    test("1E+382", result: "43fc100000000000") //  Clamped
    test("1E+381", result: "43fc010000000000") //  Clamped
    test("1E+380", result: "43fc002000000000") //  Clamped
    test("1E+379", result: "43fc000400000000") //  Clamped
    test("1E+378", result: "43fc000040000000") //  Clamped
    test("1E+377", result: "43fc000008000000") //  Clamped
    test("1E+376", result: "43fc000001000000") //  Clamped
    test("1E+375", result: "43fc000000100000") //  Clamped
    test("1E+374", result: "43fc000000020000") //  Clamped
    test("1E+373", result: "43fc000000004000") //  Clamped
    test("1E+372", result: "43fc000000000400") //  Clamped
    test("1E+371", result: "43fc000000000080") //  Clamped
    test("1E+370", result: "43fc000000000010") //  Clamped
    test("1E+369", result: "43fc000000000001")
    test("1E+368", result: "43f8000000000001")
    
    // same with 9s
    test("9E+384", result: "77fc000000000000") //  Clamped
    test("9E+383", result: "43fc8c0000000000") //  Clamped
    test("9E+382", result: "43fc1a0000000000") //  Clamped
    test("9E+381", result: "43fc090000000000") //  Clamped
    test("9E+380", result: "43fc002300000000") //  Clamped
    test("9E+379", result: "43fc000680000000") //  Clamped
    test("9E+378", result: "43fc000240000000") //  Clamped
    test("9E+377", result: "43fc000008c00000") //  Clamped
    test("9E+376", result: "43fc000001a00000") //  Clamped
    test("9E+375", result: "43fc000000900000") //  Clamped
    test("9E+374", result: "43fc000000023000") //  Clamped
    test("9E+373", result: "43fc000000006800") //  Clamped
    test("9E+372", result: "43fc000000002400") //  Clamped
    test("9E+371", result: "43fc00000000008c") //  Clamped
    test("9E+370", result: "43fc00000000001a") //  Clamped
    test("9E+369", result: "43fc000000000009")
    test("9E+368", result: "43f8000000000009")
    
    // values around [u]int32 edges (zeros done earlier)
    test(-2147483646, result: "a23800008c78af46")
    test(-2147483647, result: "a23800008c78af47")
    test(-2147483648, result: "a23800008c78af48")
    test(-2147483649, result: "a23800008c78af49")
    test(2147483646,  result: "223800008c78af46")
    test(2147483647,  result: "223800008c78af47")
    test(2147483648,  result: "223800008c78af48")
    test(2147483649,  result: "223800008c78af49")
    test(4294967294,  result: "2238000115afb55a")
    test(4294967295,  result: "2238000115afb55b")
    test(4294967296,  result: "2238000115afb57a")
    test(4294967297,  result: "2238000115afb57b")
  }
  
  struct test {

      let input: UInt64
      let result: String

      init(_ input: UInt64, _ result: String) {
          self.input = input
          self.result = result
      }
  }
  
  let tests1: [test] = [
      test(0xA2300000000003D0, "-7.50"),
      test(0xA23c0000000003D0, "-7500"),
      test(0xA2380000000003D0, "-750"),
      test(0xA2340000000003D0, "-75.0"),
      test(0xA22c0000000003D0, "-0.750"),
      test(0xA2280000000003D0, "-0.0750"),
      test(0xA2200000000003D0, "-0.000750"),
      test(0xA2180000000003D0, "-0.00000750"),
      test(0xA2140000000003D0, "-7.50e-7"),
      test(0x263934b9c1e28e56, "1234567890123456"),
      test(0xa63934b9c1e28e56, "-1234567890123456"),
      test(0x260934b9c1e28e56, "1234.567890123456"),
      test(0x2638912449124491, "1111111111111111"),
      test(0x6e38ff3fcff3fcff, "9999999999999999"),
      test(0x77fcff3fcff3fcff, "9.999999999999999e+384"),
      test(0x47fd34b9c1e28e56, "1.234567890123456e+384"),
      test(0x47fd300000000000, "1.230000000000000e+384"),
      test(0x47fc000000000000, "1.000000000000000e+384"),
      test(0x22380000000049c5, "12345"),
      test(0x2238000000000534, "1234"),
      test(0x22380000000000a3, "123"),
      test(0x2238000000000012, "12"),
      test(0x2238000000000001, "1"),
      test(0x22300000000000a3, "1.23"),
      test(0x22300000000049c5, "123.45"),
      test(0x003c000000000001, "1e-383"),
      test(0x0400000000000000, "1.000000000000000e-383"),
      test(0x0400000000000001, "1.000000000000001e-383"),
      test(0x0000800000000000, "1.00000000000000e-384"),
      test(0x0000000000000010, "1.0e-397"),
      test(0x0004000000000001, "1e-397"),
      test(0x0000000000000001, "1e-398"),
      test(0x6400ff3fcff3fcff, "9.999999999999999e-383"),
      test(0x0400912449124491, "1.111111111111111e-383"),
  ]

  func test1() throws {
      for t in tests1 {
        XCTAssertEqual(Decimal64(bitPattern: t.input, encoding: .dpd), Decimal64(stringLiteral: t.result))
        XCTAssertEqual(Decimal64(bitPattern: t.input, encoding: .dpd).description, t.result)
      }
  }
  

  static func U64(_ x: String) -> UInt64 {
      assert(x.count == 16)
      return UInt64(x, radix: 16)!
  }

  let tests1a: [test] = [
      test(U64("A2300000000003D0"), "-7.50"),
      test(U64("A23c0000000003D0"), "-7500"),
      test(U64("A2380000000003D0"), "-750"),
      test(U64("A2340000000003D0"), "-75.0"),
      test(U64("A22C0000000003D0"), "-0.750"),
      test(U64("A2280000000003D0"), "-0.0750"),
      test(U64("A2200000000003D0"), "-0.000750"),
      test(U64("A2180000000003D0"), "-0.00000750"),
      test(U64("A2140000000003D0"), "-7.50e-7"),
  ]

  func test2() {
    for t in tests1a {
      XCTAssertEqual(Decimal64(bitPattern:t.input, encoding: .dpd), Decimal64(stringLiteral: t.result))
      XCTAssertEqual(Decimal64(bitPattern:t.input, encoding: .dpd).description, t.result)
      XCTAssertFalse(Decimal64(bitPattern:t.input, encoding: .dpd).isNaN)
    }
  }

  let tests2a: [test] = [
      test(U64("223800000000006e"), "888"),
      test(U64("223800000000016e"), "888"),
      test(U64("223800000000026e"), "888"),
      test(U64("223800000000036e"), "888"),
      test(U64("223800000000006f"), "889"),
      test(U64("223800000000016f"), "889"),
      test(U64("223800000000026f"), "889"),
      test(U64("223800000000036f"), "889"),
  ]

  func test3() {
      for t in tests2a {
        XCTAssertEqual(Decimal64(bitPattern:t.input, encoding: .dpd), Decimal64(stringLiteral: t.result))
        XCTAssertEqual(Decimal64(bitPattern:t.input, encoding: .dpd).description, t.result)
        XCTAssertFalse(Decimal64(bitPattern:t.input, encoding: .dpd).isNaN)
      }
  }

  let tests3a: [test] = [
      test(U64("7800000000000000"), ""),
      test(U64("7900000000000000"), ""),
      test(U64("7a00000000000000"), ""),
      test(U64("7b00000000000000"), ""),
      test(U64("7c00000000000000"), ""),
      test(U64("7d00000000000000"), ""),
      test(U64("7e00000000000000"), ""),
      test(U64("7f00000000000000"), ""),
  ]

  func test4() {
      for t in tests3a {
        let bd = Decimal64(bitPattern:t.input, encoding: .dpd)
        XCTAssertTrue(bd.isInfinite || bd.isNaN)
      }
  }
  
  func testDecimal64() throws {
    let s = "123456789012345678"
    let y1 = Decimal64(stringLiteral: s)
    XCTAssert(y1.description == "1.234567890123457e+17")
    print("\(s) -> \(y1)")
    
    print("Decimal64.zero =", Decimal64.zero)
    XCTAssert(Decimal64.zero.description == "0")
    print("Decimal64.pi =", Decimal64.pi)
    XCTAssert(Decimal64.pi.description == "3.141592653589793")
    print("Decimal64.nan =", Decimal64.nan)
    XCTAssert(Decimal64.nan.description == "NaN")
    print("Decimal64.signalingNaN =", Decimal64.signalingNaN)
    XCTAssert(Decimal64.signalingNaN.description == "SNaN")
    print("Decimal64.infinity =", Decimal64.infinity)
    XCTAssert(Decimal64.infinity.description == "Inf")
    
    // basic DPD to BID conversion and BID to DPD
    let n = UInt64(0xA2300000000003D0)
    let a = Decimal64(bitPattern: n, encoding: .dpd)
    XCTAssert(a.description == "-7.50")
    print(a, a.bitPattern(.dpd) == n ? "a = n" : "a != n")
    XCTAssert(a.bitPattern(.dpd) == n)
    
    let d32 = Decimal64("1000.1234"); print(d32, String(d32.bid.data,radix:16))
    let d64 = Decimal64(d32); print(d64, String(d64.bid.data,radix:16))
    let d128 = Decimal128(d32); print(d128, String(d128.bid.data,radix:16))
    let d128a = Decimal128(d64); print(d128a, String(d128a.bid.data,radix:16))
  }
  
}
