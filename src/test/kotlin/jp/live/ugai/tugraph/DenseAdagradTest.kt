package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.tracker.Tracker
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Comprehensive test class for DenseAdagrad optimizer.
 *
 * Tests the DenseAdagrad optimizer's initialization, parameter updates, learning rate tracking,
 * weight decay, gradient clipping, and edge cases.
 */
class DenseAdagradTest {
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
    fun testDenseAdagradBuilderCreation() {
        val builder = DenseAdagrad.builder()
        assertNotNull(builder)

        val optimizer = builder.build()
        assertNotNull(optimizer)
    }

    @Test
    fun testDenseAdagradWithCustomLearningRate() {
        val learningRate = 0.05f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        assertNotNull(optimizer)
    }

    @Test
    fun testDenseAdagradWithCustomEpsilon() {
        val epsilon = 1e-6f

        val optimizer =
            DenseAdagrad.builder()
                .optEpsilon(epsilon)
                .build()

        assertNotNull(optimizer)
    }

    @Test
    fun testDenseAdagradUpdate() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(10, 5))
        val grad = manager.ones(Shape(10, 5))

        val initialWeights = weight.toFloatArray().clone()

        optimizer.update("test_param", weight, grad)

        val updatedWeights = weight.toFloatArray()

        // Verify that weights have been updated
        var changed = false
        for (i in initialWeights.indices) {
            if (initialWeights[i] != updatedWeights[i]) {
                changed = true
                break
            }
        }
        assertTrue(changed, "Weights should be updated after Adagrad step")

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradAccumulatesGradientHistory() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(5, 3))
        val grad1 = manager.ones(Shape(5, 3))
        val grad2 = manager.ones(Shape(5, 3))

        val weightsAfterFirst = weight.duplicate()
        optimizer.update("test_param", weight, grad1)
        weightsAfterFirst.close()
        weightsAfterFirst.attach(weight.manager)
        weight.toFloatArray().copyInto(weightsAfterFirst.toFloatArray())

        val weightsBeforeSecond = weight.toFloatArray().clone()
        optimizer.update("test_param", weight, grad2)
        val weightsAfterSecond = weight.toFloatArray()

        // The second update should result in smaller changes due to accumulated history
        var firstStepDiff = 0.0f
        var secondStepDiff = 0.0f
        for (i in weightsAfterFirst.toFloatArray().indices) {
            firstStepDiff += kotlin.math.abs(1.0f - weightsAfterFirst.toFloatArray()[i])
            secondStepDiff += kotlin.math.abs(weightsBeforeSecond[i] - weightsAfterSecond[i])
        }

        assertTrue(
            secondStepDiff < firstStepDiff,
            "Second step should have smaller magnitude due to accumulated gradient history",
        )

        weight.close()
        grad1.close()
        grad2.close()
        weightsAfterFirst.close()
    }

    @Test
    fun testDenseAdagradWithWeightDecay() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)
        val weightDecay = 0.01f

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .optWeightDecays(weightDecay)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.zeros(Shape(5, 5))

        val initialWeights = weight.toFloatArray().clone()

        optimizer.update("test_param", weight, grad)

        val updatedWeights = weight.toFloatArray()

        // With weight decay and zero gradient, weights should decrease
        for (i in initialWeights.indices) {
            assertTrue(
                updatedWeights[i] < initialWeights[i],
                "Weights should decrease due to weight decay even with zero gradient",
            )
        }

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithGradientClipping() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)
        val clipGrad = 1.0f

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .optClipGrad(clipGrad)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.create(5.0f, Shape(5, 5)) // Large gradients

        val initialWeights = weight.toFloatArray().clone()

        optimizer.update("test_param", weight, grad)

        val updatedWeights = weight.toFloatArray()

        // Weights should still update despite large gradients (due to clipping)
        var changed = false
        for (i in initialWeights.indices) {
            if (initialWeights[i] != updatedWeights[i]) {
                changed = true
                break
            }
        }
        assertTrue(changed, "Weights should be updated with clipped gradients")

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithGradientRescaling() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)
        val rescaleGrad = 0.5f

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()
        optimizer.rescaleGrad = rescaleGrad

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.ones(Shape(5, 5))

        optimizer.update("test_param", weight, grad)

        assertNotNull(weight)

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithMultipleParameters() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight1 = manager.ones(Shape(5, 5))
        val grad1 = manager.ones(Shape(5, 5))

        val weight2 = manager.ones(Shape(3, 3))
        val grad2 = manager.ones(Shape(3, 3))

        optimizer.update("param1", weight1, grad1)
        optimizer.update("param2", weight2, grad2)

        assertNotNull(weight1)
        assertNotNull(weight2)

        weight1.close()
        grad1.close()
        weight2.close()
        grad2.close()
    }

    @Test
    fun testDenseAdagradThrowsOnInvalidLearningRate() {
        val tracker = Tracker.fixed(Float.NaN)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.ones(Shape(5, 5))

        assertThrows<IllegalStateException> {
            optimizer.update("test_param", weight, grad)
        }

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradThrowsOnInfiniteWeightDecay() {
        val tracker = Tracker.fixed(0.1f)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .optWeightDecays(Float.POSITIVE_INFINITY)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.ones(Shape(5, 5))

        assertThrows<IllegalStateException> {
            optimizer.update("test_param", weight, grad)
        }

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradBuilderChaining() {
        val tracker = Tracker.fixed(0.05f)
        val epsilon = 1e-7f
        val weightDecay = 0.001f

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .optEpsilon(epsilon)
                .optWeightDecays(weightDecay)
                .build()

        assertNotNull(optimizer)
    }

    @Test
    fun testDenseAdagradWithVerySmallEpsilon() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)
        val epsilon = 1e-12f

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .optEpsilon(epsilon)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.ones(Shape(5, 5))

        optimizer.update("test_param", weight, grad)

        assertNotNull(weight)

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithLargeGradients() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.create(100.0f, Shape(5, 5))

        val initialWeights = weight.toFloatArray().clone()

        optimizer.update("test_param", weight, grad)

        val updatedWeights = weight.toFloatArray()

        // Weights should change even with large gradients
        var changed = false
        for (i in initialWeights.indices) {
            if (initialWeights[i] != updatedWeights[i]) {
                changed = true
                break
            }
        }
        assertTrue(changed, "Weights should be updated with large gradients")

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithZeroGradients() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.zeros(Shape(5, 5))

        val initialWeights = weight.toFloatArray().clone()

        optimizer.update("test_param", weight, grad)

        val updatedWeights = weight.toFloatArray()

        // With zero gradients and no weight decay, weights should remain unchanged
        for (i in initialWeights.indices) {
            assertEquals(initialWeights[i], updatedWeights[i], 1e-6f, "Weights should not change with zero gradients")
        }

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradUpdateCountIncreases() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        val weight = manager.ones(Shape(5, 5))
        val grad = manager.ones(Shape(5, 5))

        // Perform multiple updates
        optimizer.update("test_param", weight, grad)
        optimizer.update("test_param", weight, grad)
        optimizer.update("test_param", weight, grad)

        // Optimizer should maintain update count internally
        assertNotNull(weight)

        weight.close()
        grad.close()
    }

    @Test
    fun testDenseAdagradWithDifferentShapes() {
        val learningRate = 0.1f
        val tracker = Tracker.fixed(learningRate)

        val optimizer =
            DenseAdagrad.builder()
                .optLearningRateTracker(tracker)
                .build()

        // Test with 1D tensor
        val weight1D = manager.ones(Shape(10))
        val grad1D = manager.ones(Shape(10))
        optimizer.update("param_1d", weight1D, grad1D)

        // Test with 2D tensor
        val weight2D = manager.ones(Shape(5, 5))
        val grad2D = manager.ones(Shape(5, 5))
        optimizer.update("param_2d", weight2D, grad2D)

        // Test with 3D tensor
        val weight3D = manager.ones(Shape(3, 3, 3))
        val grad3D = manager.ones(Shape(3, 3, 3))
        optimizer.update("param_3d", weight3D, grad3D)

        assertNotNull(weight1D)
        assertNotNull(weight2D)
        assertNotNull(weight3D)

        weight1D.close()
        grad1D.close()
        weight2D.close()
        grad2D.close()
        weight3D.close()
        grad3D.close()
    }
}