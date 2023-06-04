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
  
  func testDecimal128() throws {
    let s = "12345678901234567890.12345678901234567890"
    let y1 = Decimal128(stringLiteral: s);
    XCTAssert(y1.description == "12345678901234567890.12345678901235")
    print("\(s) -> \(y1)")
    
    print("Decimal128.zero =", Decimal128.zero)
    XCTAssert(Decimal128.zero.description == "0")
    print("Decimal128.pi =", Decimal128.pi)
    XCTAssert(Decimal128.pi.description=="3.141592653589793238462643383279503")
    print("Decimal128.nan =", String(Decimal128.nan.bid.data, radix: 16))
    XCTAssert(Decimal128.nan.description == "NaN")
    print("Decimal128.signalingNaN =", Decimal128.signalingNaN)
    XCTAssert(Decimal128.signalingNaN.description == "SNaN")
    print("Decimal128.infinity =", Decimal128.infinity)
    XCTAssert(Decimal128.infinity.description == "Inf")
    
    // basic DPD to BID conversion and BID to DPD
    let dpd = StaticBigInt(0xA207_8000_0000_0000_0000_0000_0000_03D0)
    let n = UInt128(integerLiteral: dpd)
    let a = Decimal128(dpdBitPattern: n)
    XCTAssert(a.description == "-7.50")
    print(a, a.dpdBitPattern == n ? "a = n" : "a != n")
    XCTAssert(a.dpdBitPattern == n)
  }
  
}
