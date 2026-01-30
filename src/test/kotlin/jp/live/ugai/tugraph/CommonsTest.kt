package jp.live.ugai.tugraph

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Test class for Commons constants and configuration values.
 *
 * Verifies that all hyperparameters and configuration constants are set to valid values
 * and within reasonable ranges for knowledge graph embedding training.
 */
class CommonsTest {
    @Test
    fun learningRateShouldBePositive() {
        assertTrue(LEARNING_RATE > 0.0f, "Learning rate must be positive")
    }

    @Test
    fun learningRateShouldBeReasonable() {
        assertTrue(LEARNING_RATE <= 1.0f, "Learning rate should typically be <= 1.0")
        assertTrue(LEARNING_RATE >= 1e-6f, "Learning rate should be >= 1e-6 to make progress")
    }

    @Test
    fun dimensionShouldBePositive() {
        assertTrue(DIMENSION > 0, "Embedding dimension must be positive")
    }

    @Test
    fun dimensionShouldBeReasonable() {
        assertTrue(DIMENSION >= 10, "Embedding dimension should be at least 10 for meaningful representations")
        assertTrue(DIMENSION <= 10000, "Embedding dimension should be <= 10000 for practical memory constraints")
    }

    @Test
    fun nepochShouldBePositive() {
        assertTrue(NEPOCH > 0, "Number of epochs must be positive")
    }

    @Test
    fun nepochShouldBeReasonable() {
        assertTrue(NEPOCH >= 1, "At least one epoch needed for training")
        assertTrue(NEPOCH <= 10000, "Epochs should be reasonable for practical training")
    }

    @Test
    fun batchSizeShouldBePositive() {
        assertTrue(BATCH_SIZE > 0, "Batch size must be positive")
    }

    @Test
    fun batchSizeShouldBeReasonable() {
        assertTrue(BATCH_SIZE >= 1, "Batch size should be at least 1")
        assertTrue(BATCH_SIZE <= 100000, "Batch size should be reasonable for memory constraints")
    }

    @Test
    fun negativeResampleCapShouldBePositive() {
        assertTrue(NEGATIVE_RESAMPLE_CAP > 0, "Negative resample cap must be positive")
    }

    @Test
    fun negativeResampleCapShouldBeReasonable() {
        assertTrue(NEGATIVE_RESAMPLE_CAP >= 1, "At least 1 resample attempt needed")
        assertTrue(NEGATIVE_RESAMPLE_CAP <= 1000, "Resample cap should be reasonable to avoid infinite loops")
    }

    @Test
    fun resultEvalBatchSizeShouldBePositive() {
        assertTrue(RESULT_EVAL_BATCH_SIZE > 0, "Result eval batch size must be positive")
    }

    @Test
    fun resultEvalEntityChunkSizeShouldBePositive() {
        assertTrue(RESULT_EVAL_ENTITY_CHUNK_SIZE > 0, "Entity chunk size must be positive")
    }

    @Test
    fun resultEvalEntityChunkSizeShouldBeReasonable() {
        assertTrue(
            RESULT_EVAL_ENTITY_CHUNK_SIZE >= 100,
            "Entity chunk size should be at least 100 for efficiency",
        )
        assertTrue(
            RESULT_EVAL_ENTITY_CHUNK_SIZE <= 1000000,
            "Entity chunk size should be reasonable for memory",
        )
    }

    @Test
    fun evalEveryShouldBePositive() {
        assertTrue(EVAL_EVERY > 0, "Eval frequency must be positive")
    }

    @Test
    fun evalEveryShouldBeReasonable() {
        assertTrue(EVAL_EVERY >= 1, "Eval frequency should be at least 1")
        assertTrue(EVAL_EVERY <= 1000, "Eval frequency should be reasonable")
    }

    @Test
    fun distMultL2ShouldBeNonNegative() {
        assertTrue(DISTMULT_L2 >= 0.0f, "L2 regularization weight must be non-negative")
    }

    @Test
    fun distMultL2ShouldBeReasonable() {
        assertTrue(DISTMULT_L2 <= 1.0f, "L2 regularization weight should be <= 1.0 typically")
    }

    @Test
    fun complexL2ShouldBeNonNegative() {
        assertTrue(COMPLEX_L2 >= 0.0f, "L2 regularization weight must be non-negative")
    }

    @Test
    fun complexL2ShouldBeReasonable() {
        assertTrue(COMPLEX_L2 <= 1.0f, "L2 regularization weight should be <= 1.0 typically")
    }

    @Test
    fun negativeSamplesShouldBePositive() {
        assertTrue(NEGATIVE_SAMPLES > 0, "Number of negative samples must be positive")
    }

    @Test
    fun negativeSamplesShouldBeReasonable() {
        assertTrue(NEGATIVE_SAMPLES >= 1, "At least 1 negative sample needed")
        assertTrue(NEGATIVE_SAMPLES <= 1000, "Negative samples should be reasonable for efficiency")
    }

    @Test
    fun selfAdversarialTempShouldBeNonNegative() {
        assertTrue(SELF_ADVERSARIAL_TEMP >= 0.0f, "Self-adversarial temperature must be non-negative")
    }

    @Test
    fun selfAdversarialTempShouldBeReasonable() {
        assertTrue(
            SELF_ADVERSARIAL_TEMP <= 10.0f,
            "Self-adversarial temperature should be reasonable (typically <= 10)",
        )
    }

    @Test
    fun tripleShouldBeThree() {
        assertEquals(3L, TRIPLE, "Triple constant should be 3 (head, relation, tail)")
    }

    @Test
    fun commonsClassShouldExist() {
        val commons = Commons()
        assertTrue(commons is Commons, "Commons class should be instantiable")
    }

    @Test
    fun verifyLearningRateValue() {
        assertEquals(0.005f, LEARNING_RATE, 1e-6f, "Learning rate should be 0.005")
    }

    @Test
    fun verifyDimensionValue() {
        assertEquals(200L, DIMENSION, "Dimension should be 200")
    }

    @Test
    fun verifyNepochValue() {
        assertEquals(30, NEPOCH, "Number of epochs should be 30")
    }

    @Test
    fun verifyBatchSizeValue() {
        assertEquals(1000, BATCH_SIZE, "Batch size should be 1000")
    }

    @Test
    fun verifyNegativeResampleCapValue() {
        assertEquals(10, NEGATIVE_RESAMPLE_CAP, "Negative resample cap should be 10")
    }

    @Test
    fun verifyResultEvalBatchSizeValue() {
        assertEquals(8, RESULT_EVAL_BATCH_SIZE, "Result eval batch size should be 8")
    }

    @Test
    fun verifyResultEvalEntityChunkSizeValue() {
        assertEquals(1000, RESULT_EVAL_ENTITY_CHUNK_SIZE, "Result eval entity chunk size should be 1000")
    }

    @Test
    fun verifyEvalEveryValue() {
        assertEquals(5, EVAL_EVERY, "Eval frequency should be 5")
    }

    @Test
    fun verifyDistMultL2Value() {
        assertEquals(1.0e-4f, DISTMULT_L2, 1e-8f, "DistMult L2 should be 1.0e-4")
    }

    @Test
    fun verifyComplExL2Value() {
        assertEquals(1.0e-4f, COMPLEX_L2, 1e-8f, "ComplEx L2 should be 1.0e-4")
    }

    @Test
    fun verifyNegativeSamplesValue() {
        assertEquals(1, NEGATIVE_SAMPLES, "Number of negative samples should be 1")
    }

    @Test
    fun verifySelfAdversarialTempValue() {
        assertEquals(0.0f, SELF_ADVERSARIAL_TEMP, 1e-6f, "Self-adversarial temperature should be 0.0")
    }

    @Test
    fun verifyTripleValue() {
        assertEquals(3L, TRIPLE, "Triple constant should be 3")
    }
}
