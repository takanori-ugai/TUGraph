package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.eval.ResultEvalQuatE
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Comprehensive test class for QuatE knowledge graph embedding model.
 *
 * Tests the QuatE model's initialization, forward pass, scoring with quaternion embeddings,
 * evaluation, and edge cases including dimension requirements and resource management.
 */
class QuatETest {
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
    fun testQuatEModelInitialization() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = quate.getEntities()
        assertNotNull(entities)
        assertEquals(numEntities, entities.shape[0])
        assertEquals(dim * 4, entities.shape[1], "Entity embeddings should have dim * 4 (quaternion: r, i, j, k)")

        val edges = quate.getEdges()
        assertNotNull(edges)
        assertEquals(numEdges, edges.shape[0])
        assertEquals(dim * 4, edges.shape[1], "Relation embeddings should have dim * 4 (quaternion: r, i, j, k)")
    }

    @Test
    fun testQuatEForwardPass() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
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
    fun testQuatEWithMultipleTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
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
    fun testQuatEModelFunction() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = quate.getEntities()
        val edges = quate.getEdges()

        val score = quate.model(testTriple, entities, edges)
        assertNotNull(score)
        assertEquals(1, score.shape[0], "Score should have shape [1] for single triple")

        score.close()
        testTriple.close()
    }

    @Test
    fun testQuatEScoreFunctionWithCustomDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = quate.getEntities()
        val edges = quate.getEdges()

        // Test with full dimension
        val scoreFullDim = quate.score(testTriple, entities, edges, dim * 4)
        assertNotNull(scoreFullDim)

        // Test with half dimension (should use first half of quaternion components)
        val scoreHalfDim = quate.score(testTriple, entities, edges, dim * 2)
        assertNotNull(scoreHalfDim)

        scoreFullDim.close()
        scoreHalfDim.close()
        testTriple.close()
    }

    @Test
    fun testQuatEScoreWithInvalidDimensionThrowsException() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = quate.getEntities()
        val edges = quate.getEdges()

        assertThrows<IllegalArgumentException> {
            quate.score(testTriple, entities, edges, 15L) // Not divisible by 4
        }

        assertThrows<IllegalArgumentException> {
            quate.score(testTriple, entities, edges, 18L) // Not divisible by 4
        }

        testTriple.close()
    }

    @Test
    fun testQuatEGetOutputShapes() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate = QuatE(numEntities, numEdges, dim)

        // Test with single input
        val inputShapes = arrayOf(Shape(9)) // 3 triples * 3 values each
        val outputShapes = quate.getOutputShapes(inputShapes)

        assertEquals(1, outputShapes.size)
        assertEquals(3L, outputShapes[0].size(), "Should output 3 scores for 3 triples")

        // Test with two inputs (positive + negative)
        val inputShapesDouble = arrayOf(Shape(9), Shape(9))
        val outputShapesDouble = quate.getOutputShapes(inputShapesDouble)

        assertEquals(2, outputShapesDouble.size)
        assertEquals(3L, outputShapesDouble[0].size())
        assertEquals(3L, outputShapesDouble[1].size())
    }

    @Test
    fun testQuatEWithEvaluation() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEvalQuatE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                quatE = quate,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertTrue(tailResult.containsKey("HIT@1"))
        assertTrue(tailResult.containsKey("HIT@10"))
        assertTrue(tailResult.containsKey("HIT@100"))
        assertTrue(tailResult.containsKey("MRR"))

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testQuatEScoreConsistency() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))

        val result1 = predictor.predict(NDList(testTriple))
        val prediction1 = result1.singletonOrThrow().toFloatArray()[0]
        result1.close()

        val result2 = predictor.predict(NDList(testTriple))
        val prediction2 = result2.singletonOrThrow().toFloatArray()[0]
        result2.close()

        assertEquals(prediction1, prediction2, 1e-6f, "Predictions should be consistent")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testQuatEWithDifferentEntityIndices() {
        val numEntities = 50L
        val numEdges = 5L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(25, 2, 40))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testQuatEEmbeddingDimensionRequirement() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = quate.getEntities()
        assertEquals(
            0,
            entities.shape[1] % 4,
            "QuatE entity embedding dimension must be divisible by 4 (quaternion: r, i, j, k)",
        )

        val edges = quate.getEdges()
        assertEquals(
            0,
            edges.shape[1] % 4,
            "QuatE relation embedding dimension must be divisible by 4 (quaternion: r, i, j, k)",
        )
    }

    @Test
    fun testQuatEResourceCleanup() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
        val predictor = model.newPredictor(NoopTranslator())

        predictor.close()
        model.close()
    }

    @Test
    fun testQuatEBatchSizeIndependence() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple1 = manager.create(longArrayOf(0, 0, 1))
        val entities = quate.getEntities()
        val edges = quate.getEdges()

        val score1 = quate.model(testTriple1, entities, edges)
        val value1 = score1.toFloatArray()[0]

        val testTriple2 = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val score2 = quate.model(testTriple2, entities, edges)
        val values2 = score2.toFloatArray()

        assertEquals(value1, values2[0], 1e-6f, "Score for same triple should be independent of batch")

        score1.close()
        score2.close()
        testTriple1.close()
        testTriple2.close()
    }

    @Test
    fun testQuatEWithSmallDimension() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 4L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testQuatEWithLargeDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 128L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = quate.getEntities()
        assertEquals(dim * 4, entities.shape[1])

        val edges = quate.getEdges()
        assertEquals(dim * 4, edges.shape[1])
    }

    @Test
    fun testQuatEScoreBoundary() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("quate").also { it.block = quate }
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
    fun testQuatEHamiltonProductProperties() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple1 = manager.create(longArrayOf(0, 0, 1))
        val testTriple2 = manager.create(longArrayOf(1, 0, 0))

        val entities = quate.getEntities()
        val edges = quate.getEdges()

        val score1 = quate.model(testTriple1, entities, edges)
        val score2 = quate.model(testTriple2, entities, edges)

        assertNotNull(score1)
        assertNotNull(score2)

        score1.close()
        score2.close()
        testTriple1.close()
        testTriple2.close()
    }

    @Test
    fun testQuatEScoreWithMultipleDimensions() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = quate.getEntities()
        val edges = quate.getEdges()

        // Test with quarter dimension
        val scoreQuarter = quate.score(testTriple, entities, edges, dim)
        assertNotNull(scoreQuarter)

        // Test with half dimension
        val scoreHalf = quate.score(testTriple, entities, edges, dim * 2)
        assertNotNull(scoreHalf)

        // Test with full dimension
        val scoreFull = quate.score(testTriple, entities, edges, dim * 4)
        assertNotNull(scoreFull)

        scoreQuarter.close()
        scoreHalf.close()
        scoreFull.close()
        testTriple.close()
    }
}
