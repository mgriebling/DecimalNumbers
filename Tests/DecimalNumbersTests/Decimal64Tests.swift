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
    let a = Decimal64(bitPattern: n, bidEncoding: false)
    XCTAssert(a.description == "-7.50")
    print(a, a.dpd == n ? "a = n" : "a != n"); XCTAssert(a.dpd == n)
  }
  
}
