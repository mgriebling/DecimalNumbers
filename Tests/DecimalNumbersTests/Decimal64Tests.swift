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
        XCTAssertEqual(Decimal64(dpdBitPattern: t.input), Decimal64(stringLiteral: t.result))
        XCTAssertEqual(Decimal64(dpdBitPattern: t.input).description, t.result)
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
      XCTAssertEqual(Decimal64(dpdBitPattern:t.input), Decimal64(stringLiteral: t.result))
      XCTAssertEqual(Decimal64(dpdBitPattern:t.input).description, t.result)
      XCTAssertFalse(Decimal64(dpdBitPattern:t.input).isNaN)
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
        XCTAssertEqual(Decimal64(dpdBitPattern:t.input), Decimal64(stringLiteral: t.result))
        XCTAssertEqual(Decimal64(dpdBitPattern:t.input).description, t.result)
        XCTAssertFalse(Decimal64(dpdBitPattern:t.input).isNaN)
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
        let bd = Decimal64(dpdBitPattern:t.input)
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
    let a = Decimal64(dpdBitPattern: n)
    XCTAssert(a.description == "-7.50")
    print(a, a.dpdBitPattern == n ? "a = n" : "a != n")
    XCTAssert(a.dpdBitPattern == n)
    
    let d32 = Decimal32("1000.1234"); print(d32, String(d32.bid.data,radix:16))
    let d64 = Decimal64(d32); print(d64, String(d64.bid.data,radix:16))
    let d128 = Decimal128(d32); print(d128, String(d128.bid.data,radix:16))
    let d128a = Decimal128(d64); print(d128a, String(d128a.bid.data,radix:16))
  }
  
}
