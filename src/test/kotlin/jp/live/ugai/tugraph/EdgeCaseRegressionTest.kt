package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.eval.*
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

/**
 * Edge case and regression tests for knowledge graph embedding models.
 *
 * Tests boundary conditions, special cases, and potential regression scenarios
 * to ensure robustness of the embedding models.
 */
class EdgeCaseRegressionTest {
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
    fun testTransEWithMinimalEntitySpace() {
        val numEntities = 2L
        val numEdges = 1L
        val dim = 4L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransRWithMinimalDimensions() {
        val numEntities = 2L
        val numEdges = 1L
        val entDim = 2L
        val relDim = 2L

        val transr =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transr").also { it.block = transr }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testSelfLoopTriple() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val selfLoopTriple = manager.create(longArrayOf(0, 0, 0))
        val prediction = predictor.predict(NDList(selfLoopTriple))

        assertNotNull(prediction)

        selfLoopTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testMaxEntityIndices() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val maxIndexTriple = manager.create(longArrayOf(9, 2, 9))
        val prediction = predictor.predict(NDList(maxIndexTriple))

        assertNotNull(prediction)

        maxIndexTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testDistMultSymmetry() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val distmult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(2, TRIPLE))
            }

        val model = Model.newInstance("distmult").also { it.block = distmult }
        val predictor = model.newPredictor(NoopTranslator())

        val triple1 = manager.create(longArrayOf(0, 0, 1))
        val triple2 = manager.create(longArrayOf(1, 0, 0))

        val pred1 = predictor.predict(NDList(triple1)).singletonOrThrow().toFloatArray()[0]
        val pred2 = predictor.predict(NDList(triple2)).singletonOrThrow().toFloatArray()[0]

        assertEquals(
            pred1,
            pred2,
            1e-5f,
            "DistMult should be symmetric: score(h,r,t) == score(t,r,h)",
        )

        triple2.close()
        triple1.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExWithOddDimensionStillWorks() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 7L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = complex.getEntities()

        assertEquals(
            0,
            entities.shape[1] % 2,
            "ComplEx doubles the dimension internally, so output should be even",
        )
    }

    @Test
    fun testResultEvalWithSingleEntity() {
        val numEntities = 1L
        val numEdges = 1L
        val dim = 4L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 0))

        val resultEval =
            ResultEvalTransE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()

        assertEquals(
            1.0f,
            tailResult["HIT@1"]!!,
            1e-5f,
            "With single entity, correct prediction should always rank 1",
        )
        assertEquals(1.0f, tailResult["MRR"]!!, 1e-5f, "MRR should be 1.0 when rank is always 1")

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransENormalizationPreservesUnitNorm() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        transe.normalize()

        val entities = transe.getEntities()
        val norms = entities.norm(intArrayOf(1), true).toFloatArray()

        for (norm in norms) {
            assertEquals(1.0f, norm, 1e-4f, "Entity embeddings should have unit L2 norm after normalization")
        }

        manager.create(norms).close()
    }

    @Test
    fun testTransRNormalizationPreservesUnitNorm() {
        val numEntities = 10L
        val numEdges = 3L
        val entDim = 16L
        val relDim = 8L

        val transr =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        transr.normalize()

        val entities = transr.getEntities()
        val edges = transr.getEdges()

        val entNorms = entities.norm(intArrayOf(1), true).toFloatArray()
        val edgeNorms = edges.norm(intArrayOf(1), true).toFloatArray()

        for (norm in entNorms) {
            assertEquals(1.0f, norm, 1e-4f, "Entity embeddings should have unit norm")
        }

        for (norm in edgeNorms) {
            assertEquals(1.0f, norm, 1e-4f, "Edge embeddings should have unit norm")
        }

        manager.create(entNorms).close()
        manager.create(edgeNorms).close()
    }

    @Test
    fun testBatchSizeOne() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))

        val resultEval =
            ResultEvalTransE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertEquals(4, tailResult.size, "Should have 4 metrics")

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testLargeBatchEvaluation() {
        val numEntities = 20L
        val numEdges = 5L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = mutableListOf<LongArray>()
        for (i in 0 until 20) {
            inputList.add(
                longArrayOf(
                    (i % 10).toLong(),
                    (i % 5).toLong(),
                    ((i + 1) % 20).toLong(),
                ),
            )
        }

        val resultEval =
            ResultEvalTransE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHigherIsBetterFlagAffectsRanking() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val distmult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("distmult").also { it.block = distmult }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))

        val resultEvalHigher =
            ResultEvalDistMult(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                distMult = distmult,
            )

        val tailResultHigher = resultEvalHigher.getTailResult()
        assertNotNull(tailResultHigher)
        assertTrue(tailResultHigher["MRR"]!! >= 0.0f && tailResultHigher["MRR"]!! <= 1.0f)

        resultEvalHigher.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testRepeatedTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(0, 0, 1),
                longArrayOf(0, 0, 1),
            )

        val resultEval =
            ResultEvalTransE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testAllDifferentRelations() {
        val numEntities = 10L
        val numEdges = 5L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
                longArrayOf(2, 2, 3),
                longArrayOf(3, 3, 4),
                longArrayOf(4, 4, 5),
            )

        val resultEval =
            ResultEvalTransE(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExEmbeddingDimensionIsDoubled() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 8L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val entities = complex.getEntities()
        assertEquals(
            dim * 2,
            entities.shape[1],
            "ComplEx should double dimension for real and imaginary parts",
        )

        val edges = complex.getEdges()
        assertEquals(dim * 2, edges.shape[1], "ComplEx edges should also have doubled dimension")
    }

    @Test
    fun testTransRMatrixShapeCorrectness() {
        val numEntities = 10L
        val numEdges = 3L
        val entDim = 16L
        val relDim = 8L

        val transr =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val matrix = transr.getMatrix()

        assertEquals(numEdges, matrix.shape[0], "Matrix should have one entry per relation")
        assertEquals(relDim, matrix.shape[1], "Matrix second dimension should be relDim")
        assertEquals(entDim, matrix.shape[2], "Matrix third dimension should be entDim")
    }
}
