#pragma once

#define TB_CUSTOM_POP_COUNT(value) __builtin_popcountll(value)
#define TB_CUSTOM_LSB(value) __builtin_ctzll(value)
#define TB_NO_HELPER_API

#define TB_VALUE_PAWN 100
#define TB_VALUE_MATE 32000
#define TB_VALUE_INFINITE 32001
#define TB_VALUE_DRAW 0
#define TB_MAX_MATE_PLY 120
