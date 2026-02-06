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
 * Integration tests for knowledge graph embedding models.
 *
 * Tests the complete workflow of model initialization, forward pass, prediction,
 * and evaluation for TransE, TransR, DistMult, and ComplEx models.
 */
class EmbeddingModelsIntegrationTest {
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
    fun testTransEModelWorkflow() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)

        val score = prediction.singletonOrThrow().toFloatArray()[0]
        assertTrue(score >= 0.0f, "TransE score should be non-negative (L1 distance)")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransRModelWorkflow() {
        val numEntities = 10L
        val numEdges = 3L
        val entDim = 16L
        val relDim = 8L

        val transr =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transr").also { it.block = transr }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)

        val score = prediction.singletonOrThrow().toFloatArray()[0]
        assertTrue(score >= 0.0f, "TransR score should be non-negative (L1 distance)")

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testDistMultModelWorkflow() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val distmult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("distmult").also { it.block = distmult }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)
        assertEquals(1, prediction.size)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExModelWorkflow() {
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

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransEWithMultipleTriples() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(3, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
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
            assertTrue(score >= 0.0f, "All scores should be non-negative")
        }

        testTriples.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransEWithEvaluation() {
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
                longArrayOf(1, 1, 2),
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
    fun testDistMultWithEvaluation() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val distmult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("distmult").also { it.block = distmult }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEvalDistMult(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                distMult = distmult,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testComplExWithEvaluation() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complex =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("complex").also { it.block = complex }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEvalComplEx(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                complEx = complex,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransRWithEvaluation() {
        val numEntities = 10L
        val numEdges = 3L
        val entDim = 16L
        val relDim = 8L

        val transr =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transr").also { it.block = transr }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEvalTransR(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = false,
                transR = transr,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testTransENormalizationMaintainsScores() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(0, 0, 1))
        val predictionBefore = predictor.predict(NDList(testTriple)).singletonOrThrow().toFloatArray()[0]

        transe.normalize()

        val predictionAfter = predictor.predict(NDList(testTriple)).singletonOrThrow().toFloatArray()[0]

        assertTrue(predictionBefore >= 0.0f)
        assertTrue(predictionAfter >= 0.0f)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testModelsHandleLargeEntitySpace() {
        val numEntities = 100L
        val numEdges = 10L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val testTriple = manager.create(longArrayOf(50, 5, 75))
        val prediction = predictor.predict(NDList(testTriple))

        assertNotNull(prediction)

        testTriple.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testModelPredictionConsistency() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
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
    fun testDifferentDataTypes() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        for (dataType in listOf(DataType.FLOAT32, DataType.FLOAT16)) {
            val transe =
                TransE(numEntities, numEdges, dim).also {
                    it.initialize(manager, dataType, Shape(1, TRIPLE))
                }

            val entities = transe.getEntities()
            assertNotNull(entities)

            val edges = transe.getEdges()
            assertNotNull(edges)
        }
    }

    @Test
    fun testComplExEvenDimensionRequirement() {
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
            "ComplEx embedding dimension must be even (real + imaginary)",
        )
    }

    @Test
    fun testModelResourceCleanup() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        predictor.close()
        model.close()
    }
}
