package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import ai.djl.translate.NoopTranslator
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Comprehensive test class for HyperComplEx knowledge graph embedding model.
 *
 * Tests the HyperComplEx model's initialization, forward pass, multi-space scoring
 * (hyperbolic, complex, and Euclidean), attention mechanism, projection operations,
 * and edge cases including dimension requirements and resource management.
 */
class HyperComplExTest {
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
    fun testHyperComplExModelInitialization() {
        val numEntities = 10L
        val numEdges = 3L
        val dimH = 8L
        val dimC = 8L
        val dimE = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dimH, dimC, dimE).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entitiesH = hyperComplEx.getEntitiesH()
        assertNotNull(entitiesH)
        assertEquals(numEntities, entitiesH.shape[0])
        assertEquals(dimH, entitiesH.shape[1])
        entitiesH.close()

        val entitiesC = hyperComplEx.getEntitiesC()
        assertNotNull(entitiesC)
        assertEquals(numEntities, entitiesC.shape[0])
        assertEquals(dimC * 2, entitiesC.shape[1])
        entitiesC.close()

        val entitiesE = hyperComplEx.getEntitiesE()
        assertNotNull(entitiesE)
        assertEquals(numEntities, entitiesE.shape[0])
        assertEquals(dimE, entitiesE.shape[1])
        entitiesE.close()

        val attention = hyperComplEx.getAttention()
        assertNotNull(attention)
        assertEquals(numEdges, attention.shape[0])
        assertEquals(3L, attention.shape[1], "Attention should have 3 weights per relation (H, C, E)")
    }

    @Test
    fun testHyperComplExConvenienceConstructor() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        // Should use the same dimension for all three spaces
        val entitiesH = hyperComplEx.getEntitiesH()
        assertEquals(dim, entitiesH.shape[1])
        entitiesH.close()

        val entitiesC = hyperComplEx.getEntitiesC()
        assertEquals(dim * 2, entitiesC.shape[1])
        entitiesC.close()

        val entitiesE = hyperComplEx.getEntitiesE()
        assertEquals(dim, entitiesE.shape[1])
        entitiesE.close()
    }

    @Test
    fun testHyperComplExForwardPass() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)
        prediction.close()

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExScoreParts() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val parameterStore = ParameterStore(manager, false)

        val scoreParts = hyperComplEx.scoreParts(testTriple, parameterStore, false)

        assertNotNull(scoreParts.total)
        assertNotNull(scoreParts.hyperbolic)
        assertNotNull(scoreParts.complex)
        assertNotNull(scoreParts.euclidean)

        assertEquals(1, scoreParts.total.shape[0])
        assertEquals(1, scoreParts.hyperbolic.shape[0])
        assertEquals(1, scoreParts.complex.shape[0])
        assertEquals(1, scoreParts.euclidean.shape[0])

        // Verify all scores are finite
        assertTrue(scoreParts.total.toFloatArray()[0].isFinite())
        assertTrue(scoreParts.hyperbolic.toFloatArray()[0].isFinite())
        assertTrue(scoreParts.complex.toFloatArray()[0].isFinite())
        assertTrue(scoreParts.euclidean.toFloatArray()[0].isFinite())

        scoreParts.total.close()
        scoreParts.hyperbolic.close()
        scoreParts.complex.close()
        scoreParts.euclidean.close()

        testTriple.close()
    }

    @Test
    fun testHyperComplExWithMultipleTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriples =
            manager.create(
                longArrayOf(
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    0,
                    3,
                ),
            )
        val prediction = predictor.predict(NDList(testTriples))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)

        val scores = prediction.singletonOrThrow().toFloatArray()
        assertEquals(3, scores.size)
        prediction.close()

        testTriples.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExProjectHyperbolic() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim, curvature = 1.0f).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        // Project hyperbolic embeddings
        hyperComplEx.projectHyperbolic()

        // Verify embeddings are still valid
        val entitiesH = hyperComplEx.getEntitiesH()
        val values = entitiesH.toFloatArray()

        for (value in values) {
            assertTrue(value.isFinite(), "Hyperbolic embeddings should be finite after projection")
        }

        entitiesH.close()
    }

    @Test
    fun testHyperComplExGetOutputShapes() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx = HyperComplEx(numEntities, numEdges, dim)

        // Test with single input
        val inputShapes = arrayOf(Shape(9)) // 3 triples * 3 values each
        val outputShapes = hyperComplEx.getOutputShapes(inputShapes)

        assertEquals(1, outputShapes.size)
        assertEquals(3L, outputShapes[0].size(), "Should output 3 scores for 3 triples")

        // Test with two inputs (positive + negative)
        val inputShapesDouble = arrayOf(Shape(9), Shape(9))
        val outputShapesDouble = hyperComplEx.getOutputShapes(inputShapesDouble)

        assertEquals(2, outputShapesDouble.size)
        assertEquals(3L, outputShapesDouble[0].size())
        assertEquals(3L, outputShapesDouble[1].size())
    }

    @Test
    fun testHyperComplExL2RegLoss() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val parameterStore = ParameterStore(manager, false)
        val device = manager.device

        val regLoss = hyperComplEx.l2RegLoss(parameterStore, device, false)

        assertNotNull(regLoss)
        assertTrue(regLoss.toFloatArray()[0] >= 0.0f, "L2 regularization loss should be non-negative")

        regLoss.close()
    }

    @Test
    fun testHyperComplExMixedPrecision() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(
                numEntities,
                numEdges,
                dim,
                mixedPrecision = true,
                lossScale = 1024.0f,
            ).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        assertEquals(true, hyperComplEx.mixedPrecision)
        assertEquals(1024.0f, hyperComplEx.lossScale)

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        prediction.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExUnscaleGradients() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        // Test unscaling with zero scale (should not crash)
        hyperComplEx.unscaleGradients(0.0f)

        // Test unscaling with normal scale
        hyperComplEx.unscaleGradients(1024.0f)
    }

    @Test
    fun testHyperComplExShardedEmbeddings() {
        val numEntities = 100L
        val numEdges = 10L
        val dim = 8L
        val entityShards = 4
        val relationShards = 2

        val hyperComplEx =
            HyperComplEx(
                numEntities,
                numEdges,
                dim,
                entityShards = entityShards,
                relationShards = relationShards,
            ).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        assertEquals(entityShards, hyperComplEx.entityShards)
        assertEquals(relationShards, hyperComplEx.relationShards)

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(25, 3, 75))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        prediction.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExAttentionMechanism() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val attention = hyperComplEx.getAttention()

        // Attention weights should be initialized
        assertNotNull(attention)
        assertEquals(numEdges, attention.shape[0])
        assertEquals(3L, attention.shape[1])

        val attentionValues = attention.toFloatArray()
        for (value in attentionValues) {
            assertTrue(value.isFinite(), "Attention weights should be finite")
        }
    }

    @Test
    fun testHyperComplExCurvatureParameter() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L
        val curvature = 2.0f

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim, curvature = curvature).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        assertEquals(curvature, hyperComplEx.curvature)
    }

    @Test
    fun testHyperComplExScoreConsistency() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))

        val result1 = predictor.predict(NDList(testTriple))
        val prediction1 = result1.singletonOrThrow().toFloatArray()[0]
        result1.close()

        val result2 = predictor.predict(NDList(testTriple))
        val prediction2 = result2.singletonOrThrow().toFloatArray()[0]
        result2.close()

        assertEquals(prediction1, prediction2, 1e-5f, "Predictions should be consistent")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExWithSmallDimensions() {
        val numEntities = 5L
        val numEdges = 2L
        val dimH = 2L
        val dimC = 2L
        val dimE = 2L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dimH, dimC, dimE).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        prediction.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExResourceCleanup() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExBoundaryEntities() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        // Test with boundary entity indices
        val testTriple1 = manager.create(longArrayOf(0, 0, 0)) // Same head and tail
        val prediction1 = predictor.predict(NDList(testTriple1))
        assertNotNull(prediction1)
        prediction1.close()

        val testTriple2 = manager.create(longArrayOf(numEntities - 1, numEdges - 1, numEntities - 1))
        val prediction2 = predictor.predict(NDList(testTriple2))
        assertNotNull(prediction2)
        prediction2.close()

        testTriple1.close()
        testTriple2.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHyperComplExGetEdgesAllSpaces() {
        val numEntities = 10L
        val numEdges = 3L
        val dimH = 8L
        val dimC = 8L
        val dimE = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dimH, dimC, dimE).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val edgesH = hyperComplEx.getEdgesH()
        assertNotNull(edgesH)
        assertEquals(numEdges, edgesH.shape[0])
        assertEquals(dimH, edgesH.shape[1])
        edgesH.close()

        val edgesC = hyperComplEx.getEdgesC()
        assertNotNull(edgesC)
        assertEquals(numEdges, edgesC.shape[0])
        assertEquals(dimC * 2, edgesC.shape[1])
        edgesC.close()

        val edgesE = hyperComplEx.getEdgesE()
        assertNotNull(edgesE)
        assertEquals(numEdges, edgesE.shape[0])
        assertEquals(dimE, edgesE.shape[1])
        edgesE.close()
    }

    @Test
    fun testHyperComplExParameterStoreAccess() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val parameterStore = ParameterStore(manager, false)
        val device = manager.device

        val entitiesH = hyperComplEx.getEntitiesH(parameterStore, device, false)
        assertNotNull(entitiesH)
        assertEquals(numEntities, entitiesH.shape[0])

        val entitiesC = hyperComplEx.getEntitiesC(parameterStore, device, false)
        assertNotNull(entitiesC)
        assertEquals(numEntities, entitiesC.shape[0])

        val entitiesE = hyperComplEx.getEntitiesE(parameterStore, device, false)
        assertNotNull(entitiesE)
        assertEquals(numEntities, entitiesE.shape[0])

        val edgesH = hyperComplEx.getEdgesH(parameterStore, device, false)
        assertNotNull(edgesH)
        assertEquals(numEdges, edgesH.shape[0])

        val edgesC = hyperComplEx.getEdgesC(parameterStore, device, false)
        assertNotNull(edgesC)
        assertEquals(numEdges, edgesC.shape[0])

        val edgesE = hyperComplEx.getEdgesE(parameterStore, device, false)
        assertNotNull(edgesE)
        assertEquals(numEdges, edgesE.shape[0])

        val attention = hyperComplEx.getAttention(parameterStore, device, false)
        assertNotNull(attention)
        assertEquals(numEdges, attention.shape[0])
    }

    @Test
    fun testHyperComplExForwardInternalWithNegatives() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val hyperComplEx =
            HyperComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("hypercomplEx").also { it.block = hyperComplEx }
        val predictor = model.newPredictor(NoopTranslator())

        // Test with both positive and negative samples
        val positiveTriples = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val negativeTriples = manager.create(longArrayOf(0, 0, 2, 1, 1, 0))

        val prediction = predictor.predict(NDList(positiveTriples, negativeTriples))

        // Should have two outputs (positive scores + negative scores)
        assertEquals(2, prediction.size)

        prediction.close()
        positiveTriples.close()
        negativeTriples.close()
        predictor.close()
        model.close()
    }
}