package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

/**
 * Test class for TransR knowledge graph embedding model.
 *
 * Tests initialization, forward pass, entity/relation projections, normalization,
 * and proper resource management for the TransR model.
 */
class TransRTest {
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
    fun testTransRInitialization() {
        val numEnt = 10L
        val numEdge = 5L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        assertEquals(numEnt, transR.numEnt)
        assertEquals(numEdge, transR.numEdge)
        assertEquals(entDim, transR.entDim)
        assertEquals(relDim, transR.relDim)

        val entities = transR.getEntities()
        assertNotNull(entities)
        assertEquals(numEnt, entities.shape[0])
        assertEquals(entDim, entities.shape[1])

        val edges = transR.getEdges()
        assertNotNull(edges)
        assertEquals(numEdge, edges.shape[0])
        assertEquals(relDim, edges.shape[1])

        val matrix = transR.getMatrix()
        assertNotNull(matrix)
        assertEquals(numEdge, matrix.shape[0])
        assertEquals(relDim, matrix.shape[1])
        assertEquals(entDim, matrix.shape[2])
    }

    @Test
    fun testTransRWithSameDimensions() {
        val numEnt = 10L
        val numEdge = 5L
        val dim = 16L

        val transR = TransR(numEnt, numEdge, dim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        assertEquals(dim, transR.entDim)
        assertEquals(dim, transR.relDim)
    }

    @Test
    fun testTransRForwardPass() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()

        val input = manager.create(longArrayOf(0, 0, 1))
        val scores = transR.model(input, entities, edges, matrix)

        assertNotNull(scores)
        assertEquals(1, scores.shape[0])
        assertTrue(scores.toFloatArray()[0] >= 0.0f)

        scores.close()
        input.close()
    }

    @Test
    fun testTransRBatchForwardPass() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))

        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()

        val input =
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
        val scores = transR.model(input, entities, edges, matrix)

        assertNotNull(scores)
        assertEquals(3, scores.shape[0])

        val scoresArray = scores.toFloatArray()
        for (score in scoresArray) {
            assertTrue(score >= 0.0f, "TransR L1 distance should be non-negative")
        }

        scores.close()
        input.close()
    }

    @Test
    fun testTransRGetOutputShapes() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)

        val inputShapes = arrayOf(Shape(3, TRIPLE))
        val outputShapes = transR.getOutputShapes(inputShapes)

        assertEquals(1, outputShapes.size)
        assertEquals(3, outputShapes[0].size())
        assertEquals(3L, outputShapes[0][0])
    }

    @Test
    fun testTransRGetOutputShapesWithTwoInputs() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)

        val inputShapes = arrayOf(Shape(3, TRIPLE), Shape(3, TRIPLE))
        val outputShapes = transR.getOutputShapes(inputShapes)

        assertEquals(2, outputShapes.size)
        assertEquals(3, outputShapes[0].size())
        assertEquals(3, outputShapes[1].size())
    }

    @Test
    fun testTransRNormalize() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        transR.normalize()

        val entities = transR.getEntities()
        val edges = transR.getEdges()

        val entNorms = entities.norm(intArrayOf(1), true).toFloatArray()
        for (norm in entNorms) {
            assertTrue(norm <= 1.0f + 1e-5f, "Entity norm should be close to 1 after normalization")
            assertTrue(norm >= 1.0f - 1e-5f, "Entity norm should be close to 1 after normalization")
        }

        val edgeNorms = edges.norm(intArrayOf(1), true).toFloatArray()
        for (norm in edgeNorms) {
            assertTrue(norm <= 1.0f + 1e-5f, "Edge norm should be close to 1 after normalization")
            assertTrue(norm >= 1.0f - 1e-5f, "Edge norm should be close to 1 after normalization")
        }

        entNorms.let { manager.create(it).close() }
        edgeNorms.let { manager.create(it).close() }
    }

    @Test
    fun testTransREntitiesAndEdgesAccessible() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        assertNotNull(entities)
        assertArrayEquals(longArrayOf(numEnt, entDim), entities.shape.shape)

        val edges = transR.getEdges()
        assertNotNull(edges)
        assertArrayEquals(longArrayOf(numEdge, relDim), edges.shape.shape)

        val matrix = transR.getMatrix()
        assertNotNull(matrix)
        assertArrayEquals(longArrayOf(numEdge, relDim, entDim), matrix.shape.shape)
    }

    @Test
    fun testTransRModelWithZeroVector() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = manager.zeros(Shape(numEnt, entDim))
        val edges = manager.zeros(Shape(numEdge, relDim))
        val matrix = manager.zeros(Shape(numEdge, relDim, entDim))

        val input = manager.create(longArrayOf(0, 0, 1))
        val scores = transR.model(input, entities, edges, matrix)

        assertNotNull(scores)
        assertEquals(0.0f, scores.toFloatArray()[0], 1e-6f)

        scores.close()
        input.close()
        matrix.close()
        edges.close()
        entities.close()
    }

    @Test
    fun testTransRModelWithIdenticalEntities() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()

        val input = manager.create(longArrayOf(0, 0, 0))
        val scores = transR.model(input, entities, edges, matrix)

        assertNotNull(scores)
        assertTrue(scores.toFloatArray()[0] >= 0.0f)

        scores.close()
        input.close()
    }

    @Test
    fun testTransRDifferentDimensionsSizes() {
        val testCases =
            listOf(
                Triple(10L, 16L, 8L),
                Triple(100L, 32L, 16L),
                Triple(5L, 64L, 32L),
                Triple(20L, 8L, 8L),
            )

        for ((numEnt, entDim, relDim) in testCases) {
            val transR = TransR(numEnt, 5L, entDim, relDim)
            transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

            val entities = transR.getEntities()
            val edges = transR.getEdges()
            val matrix = transR.getMatrix()

            assertEquals(numEnt, entities.shape[0])
            assertEquals(entDim, entities.shape[1])
            assertEquals(relDim, edges.shape[1])
            assertEquals(relDim, matrix.shape[1])
            assertEquals(entDim, matrix.shape[2])
        }
    }

    @Test
    fun testTransRMultipleTriplesBatch() {
        val numEnt = 20L
        val numEdge = 5L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(5, TRIPLE))

        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()

        val input =
            manager.create(
                longArrayOf(
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                ),
            )
        val scores = transR.model(input, entities, edges, matrix)

        assertNotNull(scores)
        assertEquals(5, scores.shape[0])

        scores.close()
        input.close()
    }

    @Test
    fun testTransRScoresAreConsistent() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        val edges = transR.getEdges()
        val matrix = transR.getMatrix()

        val input = manager.create(longArrayOf(0, 0, 1))
        val scores1 = transR.model(input, entities, edges, matrix)
        val scores2 = transR.model(input, entities, edges, matrix)

        assertArrayEquals(
            scores1.toFloatArray(),
            scores2.toFloatArray(),
            1e-6f,
            "Scores should be consistent for same input",
        )

        scores2.close()
        scores1.close()
        input.close()
    }

    @Test
    fun testTransRFloat16Initialization() {
        val numEnt = 10L
        val numEdge = 3L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT16, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        assertNotNull(entities)

        val edges = transR.getEdges()
        assertNotNull(edges)

        val matrix = transR.getMatrix()
        assertNotNull(matrix)
    }

    @Test
    fun testTransRLargeEntitySpace() {
        val numEnt = 1000L
        val numEdge = 50L
        val entDim = 16L
        val relDim = 8L

        val transR = TransR(numEnt, numEdge, entDim, relDim)
        transR.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))

        val entities = transR.getEntities()
        assertEquals(numEnt, entities.shape[0])

        val edges = transR.getEdges()
        assertEquals(numEdge, edges.shape[0])

        val matrix = transR.getMatrix()
        assertEquals(numEdge, matrix.shape[0])
    }
}
