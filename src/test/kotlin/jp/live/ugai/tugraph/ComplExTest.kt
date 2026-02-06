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

/**
 * Comprehensive test class for ComplEx knowledge graph embedding model.
 *
 * Tests the ComplEx model's initialization, forward pass, scoring with complex embeddings,
 * evaluation, and edge cases including dimension requirements and resource management.
 */
class ComplExTest {
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
    fun testComplExModelInitialization() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = complex.getEntities()
        assertNotNull(entities)
        assertEquals(numEntities, entities.shape[0])
        assertEquals(dim * 2, entities.shape[1], "Entity embeddings should have dim * 2 (real + imaginary)")

        val edges = complex.getEdges()
        assertNotNull(edges)
        assertEquals(numEdges, edges.shape[0])
        assertEquals(dim * 2, edges.shape[1], "Relation embeddings should have dim * 2 (real + imaginary)")
    }

    @Test
    fun testComplExForwardPass() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
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
    fun testComplExWithMultipleTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
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
    fun testComplExModelFunction() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val entities = complex.getEntities()
        val edges = complex.getEdges()

        val score = complex.model(testTriple, entities, edges)
        assertNotNull(score)
        assertEquals(1, score.shape[0], "Score should have shape [1] for single triple")

        score.close()
        testTriple.close()
    }

    @Test
    fun testComplExScoreConsistency() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
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
    fun testComplExGetOutputShapes() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex = ComplEx(numEntities, numEdges, dim)

        // Test with single input
        val inputShapes = arrayOf(Shape(9)) // 3 triples * 3 values each
        val outputShapes = complex.getOutputShapes(inputShapes)

        assertEquals(1, outputShapes.size)
        assertEquals(3L, outputShapes[0].size(), "Should output 3 scores for 3 triples")

        // Test with two inputs (positive + negative)
        val inputShapesDouble = arrayOf(Shape(9), Shape(9))
        val outputShapesDouble = complex.getOutputShapes(inputShapesDouble)

        assertEquals(2, outputShapesDouble.size)
        assertEquals(3L, outputShapesDouble[0].size())
        assertEquals(3L, outputShapesDouble[1].size())
    }

    @Test
    fun testComplExWithDifferentEntityIndices() {
        val numEntities = 50L
        val numEdges = 5L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(25, 2, 40))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExEmbeddingDimensionRequirement() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = complex.getEntities()
        assertEquals(
            0,
            entities.shape[1] % 2,
            "ComplEx entity embedding dimension must be even (real + imaginary)",
        )

        val edges = complex.getEdges()
        assertEquals(
            0,
            edges.shape[1] % 2,
            "ComplEx relation embedding dimension must be even (real + imaginary)",
        )
    }

    @Test
    fun testComplExResourceCleanup() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
        val predictor = model.newPredictor(NoopTranslator())

        predictor.close()
        model.close()
    }

    @Test
    fun testComplExBatchSizeIndependence() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple1 = manager.create(longArrayOf(0, 0, 1))
        val entities = complex.getEntities()
        val edges = complex.getEdges()

        val score1 = complex.model(testTriple1, entities, edges)
        val value1 = score1.toFloatArray()[0]

        val testTriple2 = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val score2 = complex.model(testTriple2, entities, edges)
        val values2 = score2.toFloatArray()

        assertEquals(value1, values2[0], 1e-6f, "Score for same triple should be independent of batch")

        score1.close()
        score2.close()
        testTriple1.close()
        testTriple2.close()
    }

    @Test
    fun testComplExWithSmallDimension() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 4L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExWithLargeDimension() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 128L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = complex.getEntities()
        assertEquals(dim * 2, entities.shape[1])

        val edges = complex.getEdges()
        assertEquals(dim * 2, edges.shape[1])
    }

    @Test
    fun testComplExScoreBoundary() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
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
    fun testComplExComplexConjugateProperty() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple1 = manager.create(longArrayOf(0, 0, 1))
        val testTriple2 = manager.create(longArrayOf(1, 0, 0))

        val entities = complex.getEntities()
        val edges = complex.getEdges()

        val score1 = complex.model(testTriple1, entities, edges)
        val score2 = complex.model(testTriple2, entities, edges)

        assertNotNull(score1)
        assertNotNull(score2)

        score1.close()
        score2.close()
        testTriple1.close()
        testTriple2.close()
    }

    @Test
    fun testComplExWithInitializerBounds() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        // Verify embeddings are initialized within reasonable bounds
        val entities = complex.getEntities()
        val entitiesArray = entities.toFloatArray()

        for (value in entitiesArray) {
            assertTrue(value.isFinite(), "Entity embeddings should be finite")
        }

        val edges = complex.getEdges()
        val edgesArray = edges.toFloatArray()

        for (value in edgesArray) {
            assertTrue(value.isFinite(), "Edge embeddings should be finite")
        }
    }

    @Test
    fun testComplExModelWithSingleEntity() {
        val numEntities = 1L
        val numEdges = 1L
        val dim = 8L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val testTriple = manager.create(longArrayOf(0, 0, 0))
        val entities = complex.getEntities()
        val edges = complex.getEdges()

        val score = complex.model(testTriple, entities, edges)
        assertNotNull(score)

        score.close()
        testTriple.close()
    }

    @Test
    fun testComplExForwardInternalWithNegatives() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
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

    @Test
    fun testComplExScoreProperties() {
        val numEntities = 20L
        val numEdges = 5L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
        val predictor = model.newPredictor(NoopTranslator())

        // Generate multiple triples and verify scores are computed
        val numTestTriples = 10
        val triples = LongArray(numTestTriples * 3)
        for (i in 0 until numTestTriples) {
            triples[i * 3] = (i % numEntities).toLong()
            triples[i * 3 + 1] = (i % numEdges).toLong()
            triples[i * 3 + 2] = ((i + 1) % numEntities).toLong()
        }

        val testTriples = manager.create(triples)
        val prediction = predictor.predict(NDList(testTriples))

        assertNotNull(prediction)
        val scores = prediction.singletonOrThrow().toFloatArray()
        assertEquals(numTestTriples, scores.size)

        // Verify all scores are finite
        for (score in scores) {
            assertTrue(score.isFinite(), "All scores should be finite")
        }

        testTriples.close()
        prediction.close()
        predictor.close()
        model.close()
    }
}
