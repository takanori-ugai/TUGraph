package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.translate.NoopTranslator
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/**
 * Comprehensive test class for RotatE knowledge graph embedding model.
 *
 * Tests the RotatE model's initialization, forward pass, scoring, evaluation,
 * and edge cases including dimension requirements and resource management.
 */
class RotatETest {
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
    fun testRotatEModelInitialization() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = rotate.getEntities()
        assertNotNull(entities)
        assertEquals(numEntities, entities.shape[0])
        assertEquals(dim * 2, entities.shape[1], "Entity embeddings should have dim * 2 (real + imaginary)")

        val edges = rotate.getEdges()
        assertNotNull(edges)
        assertEquals(numEdges, edges.shape[0])
        assertEquals(dim, edges.shape[1], "Relation embeddings should have dim (phase)")
    }

    @Test
    fun testRotatEForwardPass() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)

        val score = prediction.singletonOrThrow().toFloatArray()[0]
        assertTrue(score >= 0.0f, "RotatE score should be non-negative (L1 distance)")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEWithMultipleTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
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

        for (score in scores) {
            assertTrue(score >= 0.0f, "All RotatE scores should be non-negative")
        }

        testTriples.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEScoreFunction() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = rotate.getEntities()
        val edges = rotate.getEdges()

        val score = rotate.model(testTriple, entities, edges)
        assertNotNull(score)
        assertEquals(1, score.shape[0], "Score should have shape [1] for single triple")

        val scoreValue = score.toFloatArray()[0]
        assertTrue(scoreValue >= 0.0f, "RotatE score should be non-negative")

        score.close()
        testTriple.close()
    }

    @Test
    fun testRotatEScoreWithCustomDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = rotate.getEntities()
        val edges = rotate.getEdges()

        // Test with full dimension
        val scoreFullDim = rotate.score(testTriple, entities, edges, dim * 2)
        assertNotNull(scoreFullDim)
        assertTrue(scoreFullDim.toFloatArray()[0] >= 0.0f)

        // Test with half dimension (should use first half of embeddings)
        val scoreHalfDim = rotate.score(testTriple, entities, edges, dim)
        assertNotNull(scoreHalfDim)
        assertTrue(scoreHalfDim.toFloatArray()[0] >= 0.0f)

        scoreFullDim.close()
        scoreHalfDim.close()
        testTriple.close()
    }

    @Test
    fun testRotatEScoreWithOddDimensionThrowsException() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = rotate.getEntities()
        val edges = rotate.getEdges()

        assertThrows<IllegalArgumentException> {
            rotate.score(testTriple, entities, edges, 15L) // Odd dimension
        }

        testTriple.close()
    }

    @Test
    fun testRotatEGetOutputShapes() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate = RotatE(numEntities, numEdges, dim)

        // Test with single input
        val inputShapes = arrayOf(Shape(9)) // 3 triples * 3 values each
        val outputShapes = rotate.getOutputShapes(inputShapes)

        assertEquals(1, outputShapes.size)
        assertEquals(3L, outputShapes[0].size(), "Should output 3 scores for 3 triples")

        // Test with two inputs (positive + negative)
        val inputShapesDouble = arrayOf(Shape(9), Shape(9))
        val outputShapesDouble = rotate.getOutputShapes(inputShapesDouble)

        assertEquals(2, outputShapesDouble.size)
        assertEquals(3L, outputShapesDouble[0].size())
        assertEquals(3L, outputShapesDouble[1].size())
    }

    @Test
    fun testRotatEWithEvaluation() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEvalRotatE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = false,
                rotatE = rotate,
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
    fun testRotatEScoreConsistency() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))

        val prediction1 = predictor.predict(NDList(testTriple)).singletonOrThrow().toFloatArray()[0]
        val prediction2 = predictor.predict(NDList(testTriple)).singletonOrThrow().toFloatArray()[0]

        assertEquals(prediction1, prediction2, 1e-6f, "Predictions should be consistent")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEWithDifferentEntityIndices() {
        val numEntities = 50L
        val numEdges = 5L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(25, 2, 40))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        val score = prediction.singletonOrThrow().toFloatArray()[0]
        assertTrue(score >= 0.0f)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEEmbeddingDimensionRequirement() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = rotate.getEntities()
        assertEquals(
            0,
            entities.shape[1] % 2,
            "RotatE entity embedding dimension must be even (real + imaginary)",
        )
    }

    @Test
    fun testRotatERelationPhaseDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val edges = rotate.getEdges()
        assertEquals(
            dim,
            edges.shape[1],
            "RotatE relation embeddings should be phase angles with dimension equal to half entity dimension",
        )
    }

    @Test
    fun testRotatEResourceCleanup() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEBatchSizeIndependence() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple1 = manager.create(longArrayOf(0, 0, 1))
        val entities = rotate.getEntities()
        val edges = rotate.getEdges()

        val score1 = rotate.model(testTriple1, entities, edges)
        val value1 = score1.toFloatArray()[0]

        val testTriple2 = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val score2 = rotate.model(testTriple2, entities, edges)
        val values2 = score2.toFloatArray()

        assertEquals(value1, values2[0], 1e-6f, "Score for same triple should be independent of batch")

        score1.close()
        score2.close()
        testTriple1.close()
        testTriple2.close()
    }

    @Test
    fun testRotatEWithSmallDimension() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 4L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRotatEWithLargeDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 256L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = rotate.getEntities()
        assertEquals(dim * 2, entities.shape[1])

        val edges = rotate.getEdges()
        assertEquals(dim, edges.shape[1])
    }

    @Test
    fun testRotatEScoreBoundary() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("rotate").also { it.block = rotate }
        val predictor = model.newPredictor(NoopTranslator())

        // Test with boundary entity indices
        val testTriple1 = manager.create(longArrayOf(0, 0, 0)) // Same head and tail
        val prediction1 = predictor.predict(NDList(testTriple1))
        assertNotNull(prediction1)

        val testTriple2 = manager.create(longArrayOf(numEntities - 1, numEdges - 1, numEntities - 1))
        val prediction2 = predictor.predict(NDList(testTriple2))
        assertNotNull(prediction2)

        testTriple1.close()
        testTriple2.close()
        predictor.close()
        model.close()
    }
}
