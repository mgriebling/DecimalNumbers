// swift-tools-version: 5.8
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "DecimalNumbers",
    platforms: [
        .macOS("13.3"), .iOS("16.4"), .macCatalyst(.v16), .tvOS("16.4"),
        .watchOS("9.4")
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "DecimalNumbers",
            targets: ["DecimalNumbers"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        .package(url: "https://github.com/mgriebling/UInt128.git", from: "3.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "DecimalNumbers",
            dependencies: ["UInt128"]),
        .testTarget(
            name: "DecimalNumbersTests",
            dependencies: ["DecimalNumbers"]),
    ]
)
