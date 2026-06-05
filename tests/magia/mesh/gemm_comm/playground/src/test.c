#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "gemm_utils.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
#include "utils/l1_fifo.h"

#define WAIT_MODE            WFE
#define abs_threshold_millis 8 /* 0.008 expressed as integer millis */

#define GEMM1_N_TILES        1
// #define GEMM2_N_TILES        6
// #define GEMM3_N_TILES        3
// #define GEMM4_N_TILES        6

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0};
// static const uint32_t gemm2_tiles[GEMM2_N_TILES] = {4, 5, 8, 9, 12, 13};
// static const uint32_t gemm3_tiles[GEMM3_N_TILES] = {1, 2, 3};
// static const uint32_t gemm4_tiles[GEMM4_N_TILES] = {6, 7, 10, 11, 14, 15};

#ifndef FIFO_N_CHUNKS
#define FIFO_N_CHUNKS 5
#endif
#define FIFO_BATCH_FRAC (1.0f / FIFO_N_CHUNKS)

static int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

/* Zero `n_halfwords` fp16 elements starting at L1 address `base` using 32-bit
 * stores. Assumes `base` is 4-byte aligned and `n_halfwords` is even, which
 * holds for the ping-pong chunks below (chunk_rows * DIM_C with even DIM_C). */
static inline void l1_zero_fp16(uint32_t base, uint32_t n_halfwords)
{
    volatile uint32_t *p = (volatile uint32_t *)base;
    uint32_t n           = n_halfwords >> 1;
    for (uint32_t i = 0; i < n; i++)
        p[i] = 0;
}

/* Compare `n_elems` fp16 elements at L1 address `base` against the golden
 * `m2_inp` array in L2. The sandbox transfers are pure DMA copies, so the
 * data must be bit-identical; compare the raw 16-bit patterns directly.
 *
 * Each volatile load is materialized into a uint16_t local *before* the
 * comparison. This is load-bearing: the PULP GCC backend loads a
 * `volatile uint16_t` with the signed `p.lh` instruction, and when the
 * result feeds an arithmetic compare directly it skips the zero-extension,
 * so every fp16 with bit 15 set (all negative values) is sign-extended and
 * falsely reported as a mismatch. Assigning to a uint16_t local forces the
 * `p.exthz` zero-extension and makes the compare correct. */
static uint32_t check_against_m2(const char *label, uint32_t base, uint32_t n_elems)
{
    volatile uint16_t *got = (volatile uint16_t *)base;
    const uint16_t *exp    = (const uint16_t *)m2_inp;
    uint32_t errs          = 0;
    for (uint32_t i = 0; i < n_elems; i++) {
        uint16_t g = got[i];
        uint16_t e = exp[i];
        if (g != e) {
            errs++;
            if (errs <= 8)
                printf("  %s: m2[%u] mismatch: got=0x%04x exp=0x%04x\n",
                       label,
                       i,
                       g,
                       e);
        }
    }
    return errs;
}

int main(void)
{
    /* ~~~~~~~~~~~~~~~~~~~~ 0. Initialization ~~~~~~~~~~~~~~~~~~~~ */
    uint32_t hartid       = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    /* Init iDMA */
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    /* Init RedMulE */
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    /* Init FractalSync */
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

/* Init Event Unit */
#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
#endif

    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    // int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    // int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    // int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ------------------------------------------------------------------ */
    /* GEMM1: R1 = M1 @ M2                                                */
    /* ------------------------------------------------------------------ */
    // if (gemm1_idx >= 0) {
    //     // Compute chunks
    //     uint32_t chunk_rows  = DIM_A * FIFO_BATCH_FRAC;
    //     uint32_t chunk_bytes = chunk_rows * DIM_B * 2;

    //     /* Define workspace: [M2 | M1_pp[0] | M1_pp[1] | R1_pp[0] | R1_pp[1]] */
    //     uint32_t obi_m2    = l1_tile_base;
    //     uint32_t obi_m1[2] = {obi_m2 + DIM_B * DIM_C * 2, obi_m2 + DIM_B * DIM_C * 2 +
    //     chunk_bytes}; uint32_t obi_r1[2] = {obi_m1[1] + chunk_bytes,
    //                           obi_m1[1] + chunk_bytes + chunk_rows * DIM_C * 2};

    //     // Load full M2 from L2
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);

    //     /* redmule_gemm accumulates (Y = X*W + Y); zero both R1 ping-pong
    //      * slots once up front, then re-zero each slot inside the loop after
    //      * its previous DMA-out completes. */
    //     l1_zero_fp16(obi_r1[0], chunk_rows * DIM_C);
    //     l1_zero_fp16(obi_r1[1], chunk_rows * DIM_C);

    //     // Wait for M2 to be fully loaded before starting the ping-pong
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    //     // Prime the pipeline: load M1_pp[0]
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m1_inp, obi_m1[0], chunk_bytes);

    //     for (uint32_t i = 0; i < FIFO_N_CHUNKS; i++) {
    //         // Re-zero the slot we're about to reuse (slot (i-2)%2 == i%2)
    //         if (i >= 2) {
    //             l1_zero_fp16(obi_r1[i % 2], chunk_rows * DIM_C);
    //         }

    //         // Wait for M1 chunk i to arrive in L1
    //         eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    //         // Compute R1 chunk i
    //         redmule_gemm(
    //             &redmule_ctrl, obi_m1[i % 2], obi_m2, obi_r1[i % 2], chunk_rows, DIM_B, DIM_C);

    //         // Load M1 chunk from L2
    //         if (i + 1 < FIFO_N_CHUNKS) {
    //             idma_memcpy_1d(&idma_ctrl,
    //                            0,
    //                            (uint32_t)m1_inp + (i + 1) * chunk_bytes,
    //                            obi_m1[(i + 1) % 2],
    //                            chunk_bytes);
    //         }

    //         // Wait for RedMulE to finish writing R1[i%2] before DMAing it out
    //         eu_redmule_wait(&eu_ctrl, WAIT_MODE);

    //         if (i > 0) {
    //             // Wait for R1 chunk i-1 to be sent back to L2 (except for the first iteration)
    //             eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    //         }

    //         // Store R1 chunk back to L2 output buffer
    //         idma_memcpy_1d(&idma_ctrl,
    //                        1,
    //                        (uint32_t)r1_out + i * chunk_rows * DIM_C * 2,
    //                        obi_r1[i % 2],
    //                        chunk_rows * DIM_C * 2);
    //     }

    //     eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    // }

    /* ~~~~~~~~~~~~~~~~~~~~ Transfer Tests (one tile at a time) ~~~~~~~~~~~~~~~~~~~~ */

    uint32_t m2_size = (uint32_t)(DIM_B * DIM_C * 2);

/* peer[i]: tile i transfers to/from this tile in the L1<->L1 steps.
 * Derangement of {0..15}: no tile maps to itself. */
    static const uint32_t peer[16] = {7, 13, 9, 14, 11, 0, 3, 15, 1, 6, 4, 12, 5, 8, 2, 10};

    for (uint32_t t = 0; t < 1; t++) {
        /* Transfer 1: L2 → own L1 */
        if (hartid == t) {
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, l1_tile_base, m2_size);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        /* Transfer 2: own L1 → peer L1 (push) */
        if (hartid == t) {
            idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(peer[t]), l1_tile_base, m2_size);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        /* Transfer 3: peer L1 → own L1 (pull) */
        if (hartid == t) {
            idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(peer[t]), l1_tile_base, m2_size);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        }
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        /* Transfer 4: own L1 → L2 */
        if (hartid == t) {
            idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)m2_inp, l1_tile_base, m2_size);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    }

    return 0;
}
