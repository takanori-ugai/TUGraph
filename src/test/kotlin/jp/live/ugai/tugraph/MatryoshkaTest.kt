package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Comprehensive test class for Matryoshka utility.
 *
 * Tests Matryoshka nested embeddings including initialization, slicing,
 * dot product scores, dimension validation, and edge cases.
 */
class MatryoshkaTest {
    private lateinit var manager: NDManager

    @BeforeEach
    fun setUp() {
        manager = NDManager.newBaseManager()
    }

    @AfterEach
    fun tearDown() {
        manager.close()
    }

    @Test
    fun testMatryoshkaInitialization() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        assertNotNull(matryoshka)
        assertEquals(3, matryoshka.dims.size)
        assertEquals(32L, matryoshka.dims[0])
        assertEquals(64L, matryoshka.dims[1])
        assertEquals(128L, matryoshka.dims[2])
    }

    @Test
    fun testMatryoshkaWithEmptyDimensionsThrows() {
        val dims = longArrayOf()

        assertThrows<IllegalArgumentException> {
            Matryoshka(dims)
        }
    }

    @Test
    fun testMatryoshkaWithNonIncreasingDimensionsThrows() {
        val dims = longArrayOf(32, 64, 64) // Not strictly increasing

        assertThrows<IllegalArgumentException> {
            Matryoshka(dims)
        }
    }

    @Test
    fun testMatryoshkaWithDecreasingDimensionsThrows() {
        val dims = longArrayOf(128, 64, 32) // Decreasing

        assertThrows<IllegalArgumentException> {
            Matryoshka(dims)
        }
    }

    @Test
    fun testMatryoshkaSlice() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(10, 128))

        val sliced32 = matryoshka.slice(embedding, 32)
        assertNotNull(sliced32)
        assertEquals(10L, sliced32.shape[0])
        assertEquals(32L, sliced32.shape[1])

        val sliced64 = matryoshka.slice(embedding, 64)
        assertNotNull(sliced64)
        assertEquals(10L, sliced64.shape[0])
        assertEquals(64L, sliced64.shape[1])

        val sliced128 = matryoshka.slice(embedding, 128)
        assertNotNull(sliced128)
        assertEquals(10L, sliced128.shape[0])
        assertEquals(128L, sliced128.shape[1])

        sliced32.close()
        sliced64.close()
        sliced128.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaSliceWithInvalidDimensionThrows() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(10, 128))

        // Test with zero dimension
        assertThrows<IllegalArgumentException> {
            matryoshka.slice(embedding, 0)
        }

        // Test with dimension larger than embedding
        assertThrows<IllegalArgumentException> {
            matryoshka.slice(embedding, 256)
        }

        embedding.close()
    }

    @Test
    fun testMatryoshkaDotScores() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val left = manager.ones(Shape(10, 128))
        val right = manager.ones(Shape(10, 128))

        val scores = matryoshka.dotScores(left, right)

        assertNotNull(scores)
        assertEquals(3, scores.size, "Should produce scores for each configured dimension")

        // Score for dim=32 should be 32 (sum of 32 ones)
        assertEquals(10L, scores[0].shape[0], "Score should have batch dimension")
        val score32Values = scores[0].toFloatArray()
        for (value in score32Values) {
            assertEquals(32.0f, value, 0.01f, "Dot product of ones should equal dimension")
        }

        // Score for dim=64 should be 64
        val score64Values = scores[1].toFloatArray()
        for (value in score64Values) {
            assertEquals(64.0f, value, 0.01f)
        }

        // Score for dim=128 should be 128
        val score128Values = scores[2].toFloatArray()
        for (value in score128Values) {
            assertEquals(128.0f, value, 0.01f)
        }

        scores.forEach { it.close() }
        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaDotScoresWithDifferentShapesThrows() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val left = manager.ones(Shape(10, 128))
        val right = manager.ones(Shape(10, 64)) // Different embedding dimension

        assertThrows<IllegalArgumentException> {
            matryoshka.dotScores(left, right)
        }

        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaDotScoresWithInvalidDimensionThrows() {
        val dims = longArrayOf(32, 64, 256) // 256 > embedding dimension
        val matryoshka = Matryoshka(dims)

        val left = manager.ones(Shape(10, 128))
        val right = manager.ones(Shape(10, 128))

        assertThrows<IllegalArgumentException> {
            matryoshka.dotScores(left, right)
        }

        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaWithSingleDimension() {
        val dims = longArrayOf(64)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(5, 128))

        val sliced = matryoshka.slice(embedding, 64)
        assertNotNull(sliced)
        assertEquals(5L, sliced.shape[0])
        assertEquals(64L, sliced.shape[1])

        sliced.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaSlicePreservesContent() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        // Create embedding with specific pattern
        val embedding = manager.arange(0, 128, 1, ai.djl.ndarray.types.DataType.FLOAT32).reshape(1, 128)

        val sliced32 = matryoshka.slice(embedding, 32)
        val sliced32Values = sliced32.toFloatArray()

        // Verify first 32 values are preserved
        for (i in 0 until 32) {
            assertEquals(i.toFloat(), sliced32Values[i], 0.01f, "Sliced values should match original")
        }

        sliced32.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaDotScoresWithZeroVectors() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val left = manager.zeros(Shape(10, 128))
        val right = manager.zeros(Shape(10, 128))

        val scores = matryoshka.dotScores(left, right)

        assertNotNull(scores)
        assertEquals(3, scores.size)

        // All scores should be zero
        for (score in scores) {
            val values = score.toFloatArray()
            for (value in values) {
                assertEquals(0.0f, value, 0.01f, "Dot product of zeros should be zero")
            }
        }

        scores.forEach { it.close() }
        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaDotScoresWithOrthogonalVectors() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val left = manager.zeros(Shape(1, 128))
        val right = manager.zeros(Shape(1, 128))

        // Set first half to 1 in left, second half to 1 in right
        for (i in 0 until 64) {
            left.set(
                ai.djl.ndarray.index
                    .NDIndex("0, $i"),
                1.0f,
            )
        }
        for (i in 64 until 128) {
            right.set(
                ai.djl.ndarray.index
                    .NDIndex("0, $i"),
                1.0f,
            )
        }

        val scores = matryoshka.dotScores(left, right)

        assertNotNull(scores)
        assertEquals(3, scores.size)

        // For dim=32, left has ones in first 32, right has zeros in first 32, so dot product = 0
        assertEquals(0.0f, scores[0].toFloatArray()[0], 0.01f)

        // For dim=64, left has first 64 as 1, right has first 64 as 0, so dot product = 0
        assertEquals(0.0f, scores[1].toFloatArray()[0], 0.01f)

        // For dim=128, left has 64 ones in first half, right has 64 ones in second half, so dot product = 0
        assertEquals(0.0f, scores[2].toFloatArray()[0], 0.01f)

        scores.forEach { it.close() }
        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaWithLargeBatch() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val batchSize = 100L
        val left = manager.ones(Shape(batchSize, 128))
        val right = manager.ones(Shape(batchSize, 128))

        val scores = matryoshka.dotScores(left, right)

        assertNotNull(scores)
        assertEquals(3, scores.size)

        for (i in scores.indices) {
            assertEquals(batchSize, scores[i].shape[0], "Score should maintain batch dimension")
        }

        scores.forEach { it.close() }
        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaSliceViewNotCopy() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(1, 128))

        val sliced = matryoshka.slice(embedding, 64)

        // Sliced should be a view, not a copy
        assertNotNull(sliced)
        assertEquals(64L, sliced.shape[1])

        sliced.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaWithNonPowerOfTwoDimensions() {
        val dims = longArrayOf(20, 50, 100)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(10, 100))

        val sliced20 = matryoshka.slice(embedding, 20)
        assertEquals(20L, sliced20.shape[1])

        val sliced50 = matryoshka.slice(embedding, 50)
        assertEquals(50L, sliced50.shape[1])

        val sliced100 = matryoshka.slice(embedding, 100)
        assertEquals(100L, sliced100.shape[1])

        sliced20.close()
        sliced50.close()
        sliced100.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaDotScoresSymmetry() {
        val dims = longArrayOf(32, 64, 128)
        val matryoshka = Matryoshka(dims)

        val left = manager.randomUniform(0f, 1f, Shape(5, 128))
        val right = manager.randomUniform(0f, 1f, Shape(5, 128))

        val scoresLR = matryoshka.dotScores(left, right)
        val scoresRL = matryoshka.dotScores(right, left)

        // Dot product is commutative: left · right = right · left
        for (i in scoresLR.indices) {
            val valuesLR = scoresLR[i].toFloatArray()
            val valuesRL = scoresRL[i].toFloatArray()

            for (j in valuesLR.indices) {
                assertEquals(valuesLR[j], valuesRL[j], 1e-4f, "Dot product should be symmetric")
            }
        }

        scoresLR.forEach { it.close() }
        scoresRL.forEach { it.close() }
        left.close()
        right.close()
    }

    @Test
    fun testMatryoshkaWithMinimalDimensions() {
        val dims = longArrayOf(1, 2)
        val matryoshka = Matryoshka(dims)

        val embedding = manager.ones(Shape(5, 10))

        val sliced1 = matryoshka.slice(embedding, 1)
        assertEquals(1L, sliced1.shape[1])

        val sliced2 = matryoshka.slice(embedding, 2)
        assertEquals(2L, sliced2.shape[1])

        sliced1.close()
        sliced2.close()
        embedding.close()
    }

    @Test
    fun testMatryoshkaDotScoresMultipleBatches() {
        val dims = longArrayOf(16, 32, 64)
        val matryoshka = Matryoshka(dims)

        val left =
            manager
                .create(floatArrayOf(1f, 2f, 3f, 4f), Shape(1, 4))
                .repeat(0, 3)
                .repeat(1, 16)
        val right =
            manager
                .create(floatArrayOf(2f, 3f, 4f, 5f), Shape(1, 4))
                .repeat(0, 3)
                .repeat(1, 16)

        val scores = matryoshka.dotScores(left, right)

        assertNotNull(scores)
        assertEquals(3, scores.size)

        // Each score should have batch size 3
        for (score in scores) {
            assertEquals(3L, score.shape[0])
        }

        scores.forEach { it.close() }
        left.close()
        right.close()
    }
}
