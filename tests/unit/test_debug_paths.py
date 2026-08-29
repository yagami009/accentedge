"""Debug module paths."""

import sys

print("sys.path:")
for p in sys.path:
    print(" ", p)

import accentedge_lab.models.streaming_ac.paper_style as m
print("\npaper_style module file:", m.__file__)

import accentedge_lab.models.streaming_ac.low_lookahead as low
print("low_lookahead module file:", low.__file__)

print("\npaper_style ContentProsodyEncoder:")
print(m.ContentProsodyEncoder)

print("\nlow_lookahead LowLookaheadContentProsodyEncoder:")
print(low.LowLookaheadContentProsodyEncoder)

print("\nisinstance check:")
print("is subclass:", issubclass(low.LowLookaheadContentProsodyEncoder, m.ContentProsodyEncoder))
