//
//  Decimal128Tests.swift
//
//
//  Created by Mike Griebling on 31.05.2023.
//

import XCTest
@testable import DecimalNumbers
@testable import UInt128

final class Decimal128Tests: XCTestCase {
  
  func testGeneric() {
    typealias ID128 = IntDecimal128
    
    // Sanity check that the masks were generated correctly from bit defns
    XCTAssert(ID128.signBit == ID128.signBit)
    XCTAssert(UInt128(1) << ID128.signBit == 0x8000_0000_0000_0000_0000_0000_0000_0000)
    XCTAssert(Decimal128.nan.bid.data == 0x7c00_0000_0000_0000_0000_0000_0000_0000)
    XCTAssert(Decimal128.signalingNaN.bid.data == 0x7e00_0000_0000_0000_0000_0000_0000_0000)
    XCTAssert(Decimal128.infinity.bid.data == 0x7800_0000_0000_0000_0000_0000_0000_0000)
    XCTAssert(Decimal128.radix == 10)
    
    // back to default rounding mode
    // Decimal128.rounding = .toNearestOrEven
    let s = "12345678901234567890123456789012345678901234567890"
    let y1 = Decimal128(stringLiteral: s)
    XCTAssert(y1.description == "1.234567890123456789012345678901235e+49")
    print("\(s) -> \(y1)")
    
    let y = Decimal128(stringLiteral: "234.5")
    XCTAssert(y.description == "234.5")
    let x = Decimal128(stringLiteral: "345.5")
    XCTAssert(x.description == "345.5")
    
    let n = UInt128(0xa207_8000_0000_0000_0000_0000_0000_03d0) // DPD
    var a = Decimal128(dpdBitPattern: Decimal128.RawSignificand(n))
    XCTAssert(a.description == "-7.50")
    print(a, a.dpdBitPattern == n ? "a = n" : "a != n")
    XCTAssert(a.dpdBitPattern == n)
    
    print("\(x) -> digits = \(x.significandDigitCount), " +
          "bcd = \(x.significandBitPattern)")
    XCTAssert(x.significandDigitCount == 4 && x.significandBitPattern == 3455)
    print("\(y) -> digits = \(y.significandDigitCount), " +
          "bcd = \(y.significandBitPattern)")
    XCTAssert(y.significandDigitCount == 4 && y.significandBitPattern == 2345)
    
    print(x, y, x*y, y/x, Int(x), Int(y), x.decade, y.decade)
    print(x.significand, x.exponent, y.significand, y.exponent)
    var b = Decimal128.leastNormalMagnitude
    print(Decimal128.greatestFiniteMagnitude,b,Decimal128.leastNonzeroMagnitude)

    let exponentBias = Decimal128.zero.exponentBitPattern
    let maxDigits = Decimal128.significandDigitCount
    XCTAssert(maxDigits == 34)
    XCTAssert(exponentBias == 6176)
    print(Decimal128.greatestFiniteMagnitude.exponent)
    print(Decimal128.leastNormalMagnitude.exponent)
    XCTAssert(Decimal128.greatestFiniteMagnitude.exponent == 6144 - maxDigits + 1)
    XCTAssert(Decimal128.leastNormalMagnitude.exponent == -6143 - maxDigits + 1)
    
    let x5 = Decimal128("1000.3")
    print(String(x5.bidBitPattern, radix: 16), x5)
    XCTAssert(x5.bidBitPattern == 0x303e0000_00000000_00000000_00002713)
    XCTAssert(x5.dpdBitPattern == 0x2207c000_00000000_00000000_00004003)
    print(String(x5.dpdBitPattern, radix: 16))
    
    a = "-21.5"; b = "305.15"
    let c = Decimal128(signOf: a, magnitudeOf: b)
    print(c); XCTAssert((-b) == c)
    
    a = Decimal128(sign: .plus, exponentBitPattern:UInt(exponentBias),
                  significandBitPattern: 1234)
    print(a); XCTAssert(a.description == "1234")
    
    a = Decimal128.random(in: 1..<1000)
    print(a); XCTAssert(a >= 1 && a < 1000)
    
    var numbers : [Decimal128] = [2.5, 21.25, 3.0, .nan, -9.5]
    let ordered : [Decimal128] = [-9.5, 2.5, 3.0, 21.25, .nan]
    numbers.sort { !$1.isTotallyOrdered(belowOrEqualTo: $0) }
    print(numbers)
    XCTAssert(ordered.description == numbers.description)
    
    print("Decimal128.zero =", Decimal128.zero)
    XCTAssert(Decimal128.zero.description == "0")
    print("Decimal128.pi =", Decimal128.pi)
    XCTAssert(Decimal128.pi.description == "3.141592653589793238462643383279503")
    print("Decimal128.nan =", Decimal128.nan)
    XCTAssert(Decimal128.nan.description == "NaN")
    print("Decimal128.signalingNaN =", Decimal128.signalingNaN)
    XCTAssert(Decimal128.signalingNaN.description == "SNaN")
    print("Decimal128.Infinity =", Decimal128.infinity)
    XCTAssert(Decimal128.infinity.description == "Inf")
    
    var a1 = Decimal128(8.625); let b1 = Decimal128(0.75)
    let rem = a1.remainder(dividingBy: b1)
    print("\(a1).formRemainder(dividingBy: \(b1) = ", rem)
    XCTAssert(rem == Decimal128(-0.375))
    a1 = Decimal128(8.625)
    let q = (a1/b1).rounded(.towardZero); print(q)
    a1 = a1 - q * b1
    print("\(a1)")

    // Equivalent to the C 'round' function:
    let w = Decimal128(6.5)
    print(w.rounded(.toNearestOrAwayFromZero))
    XCTAssert(w.rounded(.toNearestOrAwayFromZero) == Decimal128(7)) // w = 7.0

    // Equivalent to the C 'trunc' function:
    print(w.rounded(.towardZero))
    XCTAssert(w.rounded(.towardZero) == Decimal128(6)) // x = 6.0

    // Equivalent to the C 'ceil' function:
    print(w.rounded(.up))
    XCTAssert(w.rounded(.up) == Decimal128(7)) // w = 7.0

    // Equivalent to the C 'floor' function:
    print(w.rounded(.down))
    XCTAssert(w.rounded(.down) == Decimal128(6)) // x = 6.0
  }
  
  func testEncodingDecimal128() {
    // Test encoding for Decimal64 strings and integers
    var testNumber = 0
    
    func test(_ value: String, result: String) {
      testNumber += 1
      if testNumber == 2 {
        print()
      }
      if let n = Decimal128(value) {
        var nstr = String(n.dpdBitPattern, radix:16)
        nstr = "".padding(toLength: result.count-nstr.count, withPad: "0", startingAt: 0) + nstr
        print("Test \(testNumber): \"\(value)\" [\(n)] = \(result.lowercased()) - \(n.floatingPointClass.description)")

        XCTAssertEqual(nstr, result.lowercased())
      } else {
        XCTAssert(false, "Failed to convert '\(value)'")
      }
    }
    
    func test(_ value: Int, result : String) {
      testNumber += 1
      let n = Decimal128(value)
      print("Test \(testNumber): \(value) [\(n)] = \(result.lowercased()) - \(n.floatingPointClass.description)")
      XCTAssertEqual(String(n.dpdBitPattern, radix:16), result.lowercased())
    }
    
    /// Check min/max values
    XCTAssertEqual(Decimal128.greatestFiniteMagnitude.description, "9.999999999999999999999999999999999e+6144")
    XCTAssertEqual(Decimal128.leastNonzeroMagnitude.description,   "1e-6176")
    XCTAssertEqual(Decimal128.leastNormalMagnitude.description,    "9.999999999999999999999999999999999e-6143")
    
    /// Verify various string and integer encodings
    // General testcases
    // (mostly derived from the Strawman 4 document and examples)
    test("-7.50",       result: "A20780000000000000000000000003D0")
    // derivative canonical plain strings
    test("-7.50e+3",    result: "A20840000000000000000000000003D0")
    test(-750,          result: "A20800000000000000000000000003D0")
    test("-75.0",       result: "A207c0000000000000000000000003D0")
    test("-0.750",      result: "A20740000000000000000000000003D0")
    test("-0.0750",     result: "A20700000000000000000000000003D0")
    test("-0.000750",   result: "A20680000000000000000000000003D0")
    test("-0.00000750", result: "A20600000000000000000000000003D0")
    test("-7.50e-7",    result: "A205c0000000000000000000000003D0")
    
    // Normality
    test("1234567890123456789012345678901234",  result: "2608134b9c1e28e56f3c127177823534")
    test("-1234567890123456789012345678901234", result: "a608134b9c1e28e56f3c127177823534")
    test("1111111111111111111111111111111111",  result: "26080912449124491244912449124491")
    
    // Nmax and similar
    test("9.999999999999999999999999999999999e+6144", result: "77ffcff3fcff3fcff3fcff3fcff3fcff")
    test("1.234567890123456789012345678901234e+6144", result: "47ffd34b9c1e28e56f3c127177823534")
    // fold-downs (more below)
    test("1.23e+6144", result: "47ffd300000000000000000000000000")
    test("1e+6144",    result: "47ffc000000000000000000000000000")
    
    test(12345,    result: "220800000000000000000000000049c5")
    test(1234,     result: "22080000000000000000000000000534")
    test(123,      result: "220800000000000000000000000000a3")
    test(12,       result: "22080000000000000000000000000012")
    test(1,        result: "22080000000000000000000000000001")
    test("1.23",   result: "220780000000000000000000000000a3")
    test("123.45", result: "220780000000000000000000000049c5")
    
    // Nmin and below
    test("1e-6143", result: "00084000000000000000000000000001")
    test("1.000000000000000000000000000000000E-6143", result: "04000000000000000000000000000000")
    test("1.000000000000000000000000000000001E-6143", result: "04000000000000000000000000000001")
    test("0.100000000000000000000000000000000E-6143", result: "00000800000000000000000000000000")
    test("0.000000000000000000000000000000010E-6143", result: "00000000000000000000000000000010")
    test("0.00000000000000000000000000000001E-6143",  result: "00004000000000000000000000000001")
    test("0.000000000000000000000000000000001E-6143", result: "00000000000000000000000000000001")
    
    // underflows cannot be tested for simple copies, check edge cases
    test("1e-6176", result: "00000000000000000000000000000001")
    test("999999999999999999999999999999999e-6176", result: "00000ff3fcff3fcff3fcff3fcff3fcff")
    
    // same again, negatives
    // Nmax and similar
    test("-9.999999999999999999999999999999999e+6144", result: "f7ffcff3fcff3fcff3fcff3fcff3fcff")
    test("-1.234567890123456789012345678901234e+6144", result: "c7ffd34b9c1e28e56f3c127177823534")
    // fold-downs (more below)
    test("-1.23e+6144", result: "c7ffd300000000000000000000000000")
    test("-1e+6144",    result: "c7ffc000000000000000000000000000")
    
    test(-12345,        result: "a20800000000000000000000000049c5")
    test(-1234,         result: "a2080000000000000000000000000534")
    test(-123,          result: "a20800000000000000000000000000a3")
    test(-12,           result: "a2080000000000000000000000000012")
    test(-1,            result: "a2080000000000000000000000000001")
    test("-1.23",       result: "a20780000000000000000000000000a3")
    test("-123.45",     result: "a20780000000000000000000000049c5")
    
    // Nmin and below
    test("-1e-6143", result: "80084000000000000000000000000001")
    test("-1.000000000000000000000000000000000e-6143", result: "84000000000000000000000000000000")
    test("-1.000000000000000000000000000000001e-6143", result: "84000000000000000000000000000001")
    
    test("-0.100000000000000000000000000000000e-6143", result: "80000800000000000000000000000000")
    test("-0.000000000000000000000000000000010e-6143", result: "80000000000000000000000000000010")
    test("-0.00000000000000000000000000000001e-6143",  result: "80004000000000000000000000000001")
    test("-0.000000000000000000000000000000001e-6143", result: "80000000000000000000000000000001")
    
    // underflow edge cases
    test("-1e-6176", result: "80000000000000000000000000000001")
    test("-999999999999999999999999999999999e-6176", result: "80000ff3fcff3fcff3fcff3fcff3fcff")
    // zeros
    test("0E-8000", result: "00000000000000000000000000000000")
    test("0E-6177", result: "00000000000000000000000000000000")
    test("0E-6176", result: "00000000000000000000000000000000")
    test("0.000000000000000000000000000000000E-6143", result: "00000000000000000000000000000000")
    test("0E-2",    result: "22078000000000000000000000000000")
    test("0",       result: "22080000000000000000000000000000")
    test("0E+3",    result: "2208c000000000000000000000000000")
    test("0E+6111", result: "43ffc000000000000000000000000000")
    // clamped zeros...
    test("0E+6112", result: "43ffc000000000000000000000000000")
    test("0E+6144", result: "43ffc000000000000000000000000000")
    test("0E+8000", result: "43ffc000000000000000000000000000")
    // negative zeros
    test("-0E-8000", result: "80000000000000000000000000000000")
    test("-0E-6177", result: "80000000000000000000000000000000")
    test("-0E-6176", result: "80000000000000000000000000000000")
    test("-0.000000000000000000000000000000000E-6143", result: "80000000000000000000000000000000")
    test("-0E-2",    result: "a2078000000000000000000000000000")
    test("-0",       result: "a2080000000000000000000000000000")
    test("-0E+3",    result: "a208c000000000000000000000000000")
    test("-0E+6111", result: "c3ffc000000000000000000000000000")
    // clamped zeros...
    test("-0E+6112", result: "c3ffc000000000000000000000000000")
    test("-0E+6144", result: "c3ffc000000000000000000000000000")
    test("-0E+8000", result: "c3ffc000000000000000000000000000")
    
    // exponent lengths
    test("7",        result: "22080000000000000000000000000007")
    test("7E+9",     result: "220a4000000000000000000000000007")
    test("7E+99",    result: "2220c000000000000000000000000007")
    test("7E+999",   result: "2301c000000000000000000000000007")
    test("7E+5999",  result: "43e3c000000000000000000000000007")
    
    // Specials
    test("Infinity",  result: "78000000000000000000000000000000")
    test("NaN",       result: "7c000000000000000000000000000000")
    test("-Infinity", result: "f8000000000000000000000000000000")
    test("-NaN",      result: "fc000000000000000000000000000000")
    
    test("NaN",       result: "7c000000000000000000000000000000")
    test("NaN0",      result: "7c000000000000000000000000000000")
    test("NaN1",      result: "7c000000000000000000000000000001")
    test("NaN12",     result: "7c000000000000000000000000000012")
    test("NaN79",     result: "7c000000000000000000000000000079")
    test("NaN12345",  result: "7c0000000000000000000000000049c5")
    test("NaN123456", result: "7c000000000000000000000000028e56")
    test("NaN799799", result: "7c0000000000000000000000000f7fdf")
    test("NaN799799799799799799799799799799799", result: "7c003dff7fdff7fdff7fdff7fdff7fdf")
    test("NaN999999999999999999999999999999999", result: "7c000ff3fcff3fcff3fcff3fcff3fcff")
    test("9999999999999999999999999999999999",   result: "6e080ff3fcff3fcff3fcff3fcff3fcff")
    
    // fold-down full sequence
    test("1E+6144", result: "47ffc000000000000000000000000000")
    test("1E+6143", result: "43ffc800000000000000000000000000")
    test("1E+6142", result: "43ffc100000000000000000000000000")
    test("1E+6141", result: "43ffc010000000000000000000000000")
    test("1E+6140", result: "43ffc002000000000000000000000000")
    test("1E+6139", result: "43ffc000400000000000000000000000")
    test("1E+6138", result: "43ffc000040000000000000000000000")
    test("1E+6137", result: "43ffc000008000000000000000000000")
    test("1E+6136", result: "43ffc000001000000000000000000000")
    test("1E+6135", result: "43ffc000000100000000000000000000")
    test("1E+6134", result: "43ffc000000020000000000000000000")
    test("1E+6133", result: "43ffc000000004000000000000000000")
    test("1E+6132", result: "43ffc000000000400000000000000000")
    test("1E+6131", result: "43ffc000000000080000000000000000")
    test("1E+6130", result: "43ffc000000000010000000000000000")
    test("1E+6129", result: "43ffc000000000001000000000000000")
    test("1E+6128", result: "43ffc000000000000200000000000000")
    test("1E+6127", result: "43ffc000000000000040000000000000")
    test("1E+6126", result: "43ffc000000000000004000000000000")
    test("1E+6125", result: "43ffc000000000000000800000000000")
    test("1E+6124", result: "43ffc000000000000000100000000000")
    test("1E+6123", result: "43ffc000000000000000010000000000")
    test("1E+6122", result: "43ffc000000000000000002000000000")
    test("1E+6121", result: "43ffc000000000000000000400000000")
    test("1E+6120", result: "43ffc000000000000000000040000000")
    test("1E+6119", result: "43ffc000000000000000000008000000")
    test("1E+6118", result: "43ffc000000000000000000001000000")
    test("1E+6117", result: "43ffc000000000000000000000100000")
    test("1E+6116", result: "43ffc000000000000000000000020000")
    test("1E+6115", result: "43ffc000000000000000000000004000")
    test("1E+6114", result: "43ffc000000000000000000000000400")
    test("1E+6113", result: "43ffc000000000000000000000000080")
    test("1E+6112", result: "43ffc000000000000000000000000010")
    test("1E+6111", result: "43ffc000000000000000000000000001")
    test("1E+6110", result: "43ff8000000000000000000000000001")
    
    // Miscellaneous (testers' queries, etc.)
    test(30000,  result: "2208000000000000000000000000c000")
    test(890000, result: "22080000000000000000000000007800")
    
    // d [u]int32 edges (zeros done earlier)
    test(-2147483646, result: "a208000000000000000000008c78af46")
    test(-2147483647, result: "a208000000000000000000008c78af47")
    test(-2147483648, result: "a208000000000000000000008c78af48")
    test(-2147483649, result: "a208000000000000000000008c78af49")
    test(2147483646,  result: "2208000000000000000000008c78af46")
    test(2147483647,  result: "2208000000000000000000008c78af47")
    test(2147483648,  result: "2208000000000000000000008c78af48")
    test(2147483649,  result: "2208000000000000000000008c78af49")
    test(4294967294,  result: "22080000000000000000000115afb55a")
    test(4294967295,  result: "22080000000000000000000115afb55b")
    test(4294967296,  result: "22080000000000000000000115afb57a")
    test(4294967297,  result: "22080000000000000000000115afb57b")
  }
  
}
