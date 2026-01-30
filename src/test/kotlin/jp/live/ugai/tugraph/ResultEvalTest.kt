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
 * Test class for ResultEval ranking and evaluation methods.
 *
 * Tests the evaluation metrics computation including HIT@1, HIT@10, HIT@100, and MRR
 * for both head and tail prediction tasks across different embedding models.
 */
class ResultEvalTest {
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
    fun testResultEvalWithTransE() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model =
            Model.newInstance("transe").also {
                it.block = transe
            }

        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
                longArrayOf(2, 0, 3),
            )

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = false,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertTrue(tailResult.containsKey("HIT@1"))
        assertTrue(tailResult.containsKey("HIT@10"))
        assertTrue(tailResult.containsKey("HIT@100"))
        assertTrue(tailResult.containsKey("MRR"))

        assertTrue(tailResult["HIT@1"]!! >= 0.0f && tailResult["HIT@1"]!! <= 1.0f)
        assertTrue(tailResult["HIT@10"]!! >= 0.0f && tailResult["HIT@10"]!! <= 1.0f)
        assertTrue(tailResult["HIT@100"]!! >= 0.0f && tailResult["HIT@100"]!! <= 1.0f)
        assertTrue(tailResult["MRR"]!! >= 0.0f && tailResult["MRR"]!! <= 1.0f)

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)
        assertTrue(headResult.containsKey("HIT@1"))
        assertTrue(headResult.containsKey("HIT@10"))
        assertTrue(headResult.containsKey("HIT@100"))
        assertTrue(headResult.containsKey("MRR"))

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testResultEvalWithDistMult() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val distMult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model =
            Model.newInstance("distmult").also {
                it.block = distMult
            }

        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                distMult = distMult,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertEquals(4, tailResult.size)

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)
        assertEquals(4, headResult.size)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testResultEvalWithComplEx() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val complEx =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model =
            Model.newInstance("complex").also {
                it.block = complEx
            }

        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                complEx = complEx,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertTrue(tailResult.containsKey("HIT@1"))

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)
        assertTrue(headResult.containsKey("MRR"))

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testResultEvalWithTransR() {
        val numEntities = 10L
        val numEdges = 3L
        val entDim = 16L
        val relDim = 16L

        val transR =
            TransR(numEntities, numEdges, entDim, relDim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model =
            Model.newInstance("transr").also {
                it.block = transR
            }

        val predictor = model.newPredictor(NoopTranslator())

        val inputList =
            listOf(
                longArrayOf(0, 0, 1),
                longArrayOf(1, 1, 2),
            )

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = false,
                transR = transR,
            )

        val tailResult = resultEval.getTailResult()
        assertNotNull(tailResult)
        assertEquals(4, tailResult.size)

        val headResult = resultEval.getHeadResult()
        assertNotNull(headResult)
        assertEquals(4, headResult.size)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testMetricsAreInValidRange() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val result = resultEval.getTailResult()

        assertTrue(result["HIT@1"]!! in 0.0f..1.0f, "HIT@1 must be in [0, 1]")
        assertTrue(result["HIT@10"]!! in 0.0f..1.0f, "HIT@10 must be in [0, 1]")
        assertTrue(result["HIT@100"]!! in 0.0f..1.0f, "HIT@100 must be in [0, 1]")
        assertTrue(result["MRR"]!! in 0.0f..1.0f, "MRR must be in [0, 1]")

        assertTrue(result["HIT@1"]!! <= result["HIT@10"]!!, "HIT@1 <= HIT@10")
        assertTrue(result["HIT@10"]!! <= result["HIT@100"]!!, "HIT@10 <= HIT@100")

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testEmptyInputList() {
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = emptyList<LongArray>()

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertTrue(tailResult["HIT@1"]!!.isNaN() || tailResult["HIT@1"]!! == 0.0f)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testSingleTripleEvaluation() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        val tailResult = resultEval.getTailResult()
        assertEquals(4, tailResult.size)

        val headResult = resultEval.getHeadResult()
        assertEquals(4, headResult.size)

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testMultipleTriplesEvaluation() {
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
                longArrayOf(2, 0, 3),
                longArrayOf(3, 2, 4),
                longArrayOf(4, 1, 5),
            )

        val resultEval =
            ResultEval(
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
    fun testInvalidEntityIdThrowsException() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 100))

        val resultEval =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                transE = transe,
            )

        assertThrows<IllegalArgumentException> {
            resultEval.getTailResult()
        }

        resultEval.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testHigherIsBetterFlag() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val distMult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("distmult").also { it.block = distMult }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))

        val resultEvalHigher =
            ResultEval(
                inputList,
                manager.newSubManager(),
                predictor,
                numEntities,
                higherIsBetter = true,
                distMult = distMult,
            )

        val tailResultHigher = resultEvalHigher.getTailResult()
        assertNotNull(tailResultHigher)

        resultEvalHigher.close()
        predictor.close()
        model.close()
    }

    @Test
    fun testResultEvalClose() {
        val numEntities = 5L
        val numEdges = 2L
        val dim = 8L

        val transe =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }

        val model = Model.newInstance("transe").also { it.block = transe }
        val predictor = model.newPredictor(NoopTranslator())

        val inputList = listOf(longArrayOf(0, 0, 1))
        val subManager = manager.newSubManager()

        val resultEval =
            ResultEval(
                inputList,
                subManager,
                predictor,
                numEntities,
                transE = transe,
            )

        resultEval.close()

        predictor.close()
        model.close()
    }
}