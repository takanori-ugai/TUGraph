package jp.live.ugai.tugraph

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Test class for adaptiveHyperComplExDims function in Commons.kt.
 *
 * Tests the adaptive dimension allocation for HyperComplEx based on graph size,
 * verifying correct interpolation, boundary conditions, and constraint satisfaction.
 */
class AdaptiveHyperComplExDimsTest {
    @Test
    fun testAdaptiveHyperComplExDimsSmallGraph() {
        val numEntities = 1000L
        val baseDim = 200L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")
        assertTrue(dims.dH > 0, "Hyperbolic dimension should be positive")
        assertTrue(dims.dC > 0, "Complex dimension should be positive")
        assertTrue(dims.dE > 0, "Euclidean dimension should be positive")
    }

    @Test
    fun testAdaptiveHyperComplExDimsLargeGraph() {
        val numEntities = 100000000L
        val baseDim = 200L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")

        // For large graphs, hyperbolic dimension should be dominant
        assertTrue(
            dims.dH >= dims.dC && dims.dH >= dims.dE,
            "Hyperbolic dimension should be largest for large graphs",
        )
    }

    @Test
    fun testAdaptiveHyperComplExDimsMediumGraph() {
        val numEntities = 10000L
        val baseDim = 200L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")

        // All dimensions should be reasonably balanced
        assertTrue(dims.dH >= 10, "Hyperbolic dimension should be at least 10")
        assertTrue(dims.dC >= 10, "Complex dimension should be at least 10")
        assertTrue(dims.dE >= 10, "Euclidean dimension should be at least 10")
    }

    @Test
    fun testAdaptiveHyperComplExDimsMinimumBaseDim() {
        val numEntities = 1000L
        val baseDim = 3L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")
        assertTrue(dims.dH >= 1, "Hyperbolic dimension should be at least 1")
        assertTrue(dims.dC >= 1, "Complex dimension should be at least 1")
        assertTrue(dims.dE >= 1, "Euclidean dimension should be at least 1")
    }

    @Test
    fun testAdaptiveHyperComplExDimsInvalidBaseDim() {
        val numEntities = 1000L
        val baseDim = 2L

        assertThrows<IllegalArgumentException> {
            adaptiveHyperComplExDims(numEntities, baseDim)
        }
    }

    @Test
    fun testAdaptiveHyperComplExDimsZeroEntities() {
        val numEntities = 0L
        val baseDim = 200L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        // Should handle edge case gracefully
        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")
    }

    @Test
    fun testAdaptiveHyperComplExDimsOneEntity() {
        val numEntities = 1L
        val baseDim = 200L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")
    }

    @Test
    fun testAdaptiveHyperComplExDimsInterpolation() {
        val baseDim = 200L

        // Test at interpolation boundaries
        val dims1k = adaptiveHyperComplExDims(1000L, baseDim)
        val dims100k = adaptiveHyperComplExDims(100000L, baseDim)
        val dims10M = adaptiveHyperComplExDims(10000000L, baseDim)

        // Verify that hyperbolic dimension increases with graph size
        assertTrue(
            dims1k.dH <= dims100k.dH && dims100k.dH <= dims10M.dH,
            "Hyperbolic dimension should generally increase with graph size",
        )

        // Verify all maintain budget
        assertEquals(baseDim, dims1k.dH + dims1k.dC + dims1k.dE)
        assertEquals(baseDim, dims100k.dH + dims100k.dC + dims100k.dE)
        assertEquals(baseDim, dims10M.dH + dims10M.dC + dims10M.dE)
    }

    @Test
    fun testAdaptiveHyperComplExDimsDifferentBaseDims() {
        val numEntities = 10000L

        val dims100 = adaptiveHyperComplExDims(numEntities, 100L)
        val dims200 = adaptiveHyperComplExDims(numEntities, 200L)
        val dims400 = adaptiveHyperComplExDims(numEntities, 400L)

        // Verify total matches baseDim
        assertEquals(100L, dims100.dH + dims100.dC + dims100.dE)
        assertEquals(200L, dims200.dH + dims200.dC + dims200.dE)
        assertEquals(400L, dims400.dH + dims400.dC + dims400.dE)

        // Verify proportions are similar
        val ratio100H = dims100.dH.toFloat() / 100f
        val ratio200H = dims200.dH.toFloat() / 200f
        val ratio400H = dims400.dH.toFloat() / 400f

        // Ratios should be approximately equal (within 10% tolerance)
        assertTrue(
            kotlin.math.abs(ratio100H - ratio200H) < 0.1f,
            "Proportions should be similar across different base dimensions",
        )
        assertTrue(
            kotlin.math.abs(ratio200H - ratio400H) < 0.1f,
            "Proportions should be similar across different base dimensions",
        )
    }

    @Test
    fun testAdaptiveHyperComplExDimsVeryLargeBaseDim() {
        val numEntities = 10000L
        val baseDim = 1000L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")
        assertTrue(dims.dH > 0, "All dimensions should be positive")
        assertTrue(dims.dC > 0, "All dimensions should be positive")
        assertTrue(dims.dE > 0, "All dimensions should be positive")
    }

    @Test
    fun testAdaptiveHyperComplExDimsConsistency() {
        val numEntities = 50000L
        val baseDim = 200L

        // Call multiple times with same parameters
        val dims1 = adaptiveHyperComplExDims(numEntities, baseDim)
        val dims2 = adaptiveHyperComplExDims(numEntities, baseDim)
        val dims3 = adaptiveHyperComplExDims(numEntities, baseDim)

        // Should produce consistent results
        assertEquals(dims1.dH, dims2.dH, "Results should be deterministic")
        assertEquals(dims1.dC, dims2.dC, "Results should be deterministic")
        assertEquals(dims1.dE, dims2.dE, "Results should be deterministic")

        assertEquals(dims2.dH, dims3.dH, "Results should be deterministic")
        assertEquals(dims2.dC, dims3.dC, "Results should be deterministic")
        assertEquals(dims2.dE, dims3.dE, "Results should be deterministic")
    }

    @Test
    fun testAdaptiveHyperComplExDimsMonotonicity() {
        val baseDim = 200L

        // Test that hyperbolic dimension increases monotonically with graph size
        val sizes = listOf(1000L, 10000L, 100000L, 1000000L, 10000000L)
        val dims = sizes.map { adaptiveHyperComplExDims(it, baseDim) }

        for (i in 1 until dims.size) {
            assertTrue(
                dims[i].dH >= dims[i - 1].dH,
                "Hyperbolic dimension should increase or stay same with larger graphs",
            )
        }
    }

    @Test
    fun testAdaptiveHyperComplExDimsDataClass() {
        val dims = HyperComplExDims(100L, 50L, 50L)

        assertEquals(100L, dims.dH)
        assertEquals(50L, dims.dC)
        assertEquals(50L, dims.dE)

        // Test data class properties
        val dims2 = HyperComplExDims(100L, 50L, 50L)
        assertEquals(dims, dims2, "Data classes with same values should be equal")

        val dims3 = dims.copy(dH = 120L)
        assertEquals(120L, dims3.dH)
        assertEquals(50L, dims3.dC)
        assertEquals(50L, dims3.dE)
    }

    @Test
    fun testAdaptiveHyperComplExDimsExtremelySmallGraph() {
        val numEntities = 10L
        val baseDim = 30L

        val dims = adaptiveHyperComplExDims(numEntities, baseDim)

        assertNotNull(dims)
        assertEquals(baseDim, dims.dH + dims.dC + dims.dE, "Total dimensions should equal baseDim")

        // For extremely small graphs, dimensions should favor complex and Euclidean
        assertTrue(dims.dH > 0 && dims.dC > 0 && dims.dE > 0, "All dimensions should be positive")
    }

    @Test
    fun testAdaptiveHyperComplExDimsRoundingCorrectness() {
        val baseDim = 100L

        // Test various graph sizes to ensure rounding doesn't break the budget
        for (numEntities in listOf(100L, 500L, 1000L, 5000L, 10000L, 50000L, 100000L)) {
            val dims = adaptiveHyperComplExDims(numEntities, baseDim)
            assertEquals(
                baseDim,
                dims.dH + dims.dC + dims.dE,
                "Total dimensions should exactly equal baseDim for $numEntities entities",
            )
        }
    }

    private fun assertNotNull(value: Any?) {
        assertTrue(value != null, "Value should not be null")
    }
}