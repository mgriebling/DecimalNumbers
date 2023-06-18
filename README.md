# Decimal Number Libraries

A Swift implementation based on the Intel Corp Decimal Floating-Point Math 
Library v2.2. This library uses Binary Integer Decimal (BID) encoded numbers 
in its implementation of the Decimal32, Decimal64, and Decimal128 numbers. 
Conversions to/from Densely Packed Decimal (DPD) encoded numbers are provided.

This library is preliminary and has not been completely tested.  Please feel free
to report any issues.  The Decimal32 implementation should be working and is tested.

# DecimalFloatingPoint Protocol

A new `DecimalFloatingPoint` protocol is included with this library with similar 
requirements to the `BinaryFloatingPoint` protocol except with specific requirements
for decimal floating point types.  Decimal32, Decimal64, and
Decimal128 are compliant to both this protocol and the general-purpose `FloatingPoint`
protocol.

# Dependencies

This library requires a UInt128 implementation to be present. The included UInt128 is
derived from one available in the Swift runtime (but still unreleased). Note: It also
contains an Int128 implementation which is not used but available.

**NOTE**

Not complete. Decimal32 has been tested and is working.  Decimal64 and Decimal128 still need testing and work.
