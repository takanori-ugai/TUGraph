package jp.live.ugai.tugraph

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

/**
 * Test class for EvalQuatE internal utility functions.
 *
 * Tests the quaternion slicing, Hamilton product computation, scoring functions,
 * and evaluation utility methods used for QuatE model evaluation.
 */
class EvalQuatETest {
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
    fun testSliceQuat() {
        val batchSize = 2
        val dim = 4L
        val embeddingData = FloatArray(batchSize * 16) { it.toFloat() }
        val embeddings = manager.create(embeddingData).reshape(batchSize.toLong(), 16)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val parts = EvalQuatE.sliceQuat(embeddings, rIndex, iIndex, jIndex, kIndex, batchManager)

        assertNotNull(parts)
        assertNotNull(parts.r)
        assertNotNull(parts.i)
        assertNotNull(parts.j)
        assertNotNull(parts.k)

        assertEquals(batchSize.toLong(), parts.r.shape[0])
        assertEquals(dim, parts.r.shape[1])
        assertEquals(batchSize.toLong(), parts.i.shape[0])
        assertEquals(dim, parts.i.shape[1])
        assertEquals(batchSize.toLong(), parts.j.shape[0])
        assertEquals(dim, parts.j.shape[1])
        assertEquals(batchSize.toLong(), parts.k.shape[0])
        assertEquals(dim, parts.k.shape[1])

        parts.r.close()
        parts.i.close()
        parts.j.close()
        parts.k.close()
        batchManager.close()
        embeddings.close()
    }

    @Test
    fun testComputeATail() {
        val batchSize = 2
        val dim = 4L

        val fixedData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val relData = FloatArray(batchSize * dim.toInt() * 4) { 0.5f }

        val fixedEmb = manager.create(fixedData).reshape(batchSize.toLong(), dim * 4)
        val relEmb = manager.create(relData).reshape(batchSize.toLong(), dim * 4)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val fixedParts = EvalQuatE.sliceQuat(fixedEmb, rIndex, iIndex, jIndex, kIndex, batchManager)
        val relParts = EvalQuatE.sliceQuat(relEmb, rIndex, iIndex, jIndex, kIndex, batchManager)

        val aTail = EvalQuatE.computeATail(fixedParts, relParts, batchManager)

        assertNotNull(aTail)
        assertNotNull(aTail.r)
        assertNotNull(aTail.i)
        assertNotNull(aTail.j)
        assertNotNull(aTail.k)

        assertEquals(batchSize.toLong(), aTail.r.shape[0])
        assertEquals(dim, aTail.r.shape[1])

        fixedParts.r.close()
        fixedParts.i.close()
        fixedParts.j.close()
        fixedParts.k.close()
        relParts.r.close()
        relParts.i.close()
        relParts.j.close()
        relParts.k.close()
        aTail.r.close()
        aTail.i.close()
        aTail.j.close()
        aTail.k.close()
        batchManager.close()
        fixedEmb.close()
        relEmb.close()
    }

    @Test
    fun testComputeAHead() {
        val numEntities = 10L
        val batchSize = 2
        val dim = 4L

        val entitiesData = FloatArray((numEntities * dim * 4).toInt()) { it.toFloat() }
        val entities = manager.create(entitiesData).reshape(numEntities, dim * 4)

        val relData = FloatArray(batchSize * dim.toInt() * 4) { 0.5f }
        val relEmb = manager.create(relData).reshape(batchSize.toLong(), dim * 4)

        val trueIds = manager.create(longArrayOf(0, 1))

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val relParts = EvalQuatE.sliceQuat(relEmb, rIndex, iIndex, jIndex, kIndex, batchManager)

        val aHead =
            EvalQuatE.computeAHead(
                relParts,
                trueIds,
                entities,
                rIndex,
                iIndex,
                jIndex,
                kIndex,
                batchManager,
            )

        assertNotNull(aHead)
        assertNotNull(aHead.r)
        assertNotNull(aHead.i)
        assertNotNull(aHead.j)
        assertNotNull(aHead.k)

        assertEquals(batchSize.toLong(), aHead.r.shape[0])

        relParts.r.close()
        relParts.i.close()
        relParts.j.close()
        relParts.k.close()
        aHead.r.close()
        aHead.i.close()
        aHead.j.close()
        aHead.k.close()
        batchManager.close()
        entities.close()
        relEmb.close()
        trueIds.close()
    }

    @Test
    fun testComputeTrueScoreTail() {
        val numEntities = 10L
        val batchSize = 2
        val dim = 4L

        val entitiesData = FloatArray((numEntities * dim * 4).toInt()) { it.toFloat() }
        val entities = manager.create(entitiesData).reshape(numEntities, dim * 4)

        val aData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val aEmb = manager.create(aData).reshape(batchSize.toLong(), dim * 4)

        val trueIds = manager.create(longArrayOf(0, 1))

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val aParts = EvalQuatE.sliceQuat(aEmb, rIndex, iIndex, jIndex, kIndex, batchManager)

        val scores =
            EvalQuatE.computeTrueScoreTail(
                aParts,
                trueIds,
                entities,
                rIndex,
                iIndex,
                jIndex,
                kIndex,
                batchSize,
                batchManager,
            )

        assertNotNull(scores)
        assertEquals(batchSize.toLong(), scores.shape[0])
        assertEquals(1L, scores.shape[1])

        aParts.r.close()
        aParts.i.close()
        aParts.j.close()
        aParts.k.close()
        scores.close()
        batchManager.close()
        entities.close()
        aEmb.close()
        trueIds.close()
    }

    @Test
    fun testComputeTrueScoreHead() {
        val numEntities = 10L
        val batchSize = 2
        val dim = 4L

        val entitiesData = FloatArray((numEntities * dim * 4).toInt()) { it.toFloat() }
        val entities = manager.create(entitiesData).reshape(numEntities, dim * 4)

        val aData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val aEmb = manager.create(aData).reshape(batchSize.toLong(), dim * 4)

        val trueIds = manager.create(longArrayOf(0, 1))

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val aParts = EvalQuatE.sliceQuat(aEmb, rIndex, iIndex, jIndex, kIndex, batchManager)

        val scores =
            EvalQuatE.computeTrueScoreHead(
                aParts,
                trueIds,
                entities,
                rIndex,
                iIndex,
                jIndex,
                kIndex,
                batchSize,
                batchManager,
            )

        assertNotNull(scores)
        assertEquals(batchSize.toLong(), scores.shape[0])
        assertEquals(1L, scores.shape[1])

        aParts.r.close()
        aParts.i.close()
        aParts.j.close()
        aParts.k.close()
        scores.close()
        batchManager.close()
        entities.close()
        aEmb.close()
        trueIds.close()
    }

    @Test
    fun testAccumulateCountBetter() {
        val numEntities = 100
        val batchSize = 2
        val dim = 4L

        val entitiesData = FloatArray(numEntities * dim.toInt() * 4) { (it % 10).toFloat() }
        val entities = manager.create(entitiesData).reshape(numEntities.toLong(), dim * 4)

        val aData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val aEmb = manager.create(aData).reshape(batchSize.toLong(), dim * 4)

        val trueScoreData = floatArrayOf(10.0f, 20.0f)
        val trueScore = manager.create(trueScoreData).reshape(batchSize.toLong(), 1)

        val countBetter = manager.zeros(Shape(batchSize.toLong()), DataType.INT64)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val aParts = EvalQuatE.sliceQuat(aEmb, rIndex, iIndex, jIndex, kIndex, manager)

        EvalQuatE.accumulateCountBetter(
            entities,
            rIndex,
            iIndex,
            jIndex,
            kIndex,
            aParts,
            trueScore,
            countBetter,
            higherIsBetter = true,
            chunkSize = 50,
            numEntitiesInt = numEntities,
        )

        assertNotNull(countBetter)
        assertEquals(batchSize.toLong(), countBetter.shape[0])

        // Count better should be non-negative
        val countValues = countBetter.toLongArray()
        for (value in countValues) {
            assertTrue(value >= 0L, "Count better should be non-negative")
        }

        aParts.r.close()
        aParts.i.close()
        aParts.j.close()
        aParts.k.close()
        entities.close()
        aEmb.close()
        trueScore.close()
        countBetter.close()
    }

    @Test
    fun testQuaternionPartsDataClass() {
        val r = manager.zeros(Shape(2, 4), DataType.FLOAT32)
        val i = manager.zeros(Shape(2, 4), DataType.FLOAT32)
        val j = manager.zeros(Shape(2, 4), DataType.FLOAT32)
        val k = manager.zeros(Shape(2, 4), DataType.FLOAT32)

        val parts = EvalQuatE.Parts(r, i, j, k)

        assertNotNull(parts.r)
        assertNotNull(parts.i)
        assertNotNull(parts.j)
        assertNotNull(parts.k)

        assertEquals(r, parts.r)
        assertEquals(i, parts.i)
        assertEquals(j, parts.j)
        assertEquals(k, parts.k)

        r.close()
        i.close()
        j.close()
        k.close()
    }

    @Test
    fun testSliceQuatWithDifferentDimensions() {
        val batchSize = 3
        val dim = 8L
        val embeddingData = FloatArray(batchSize * 32) { it.toFloat() }
        val embeddings = manager.create(embeddingData).reshape(batchSize.toLong(), 32)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val parts = EvalQuatE.sliceQuat(embeddings, rIndex, iIndex, jIndex, kIndex, batchManager)

        assertNotNull(parts)
        assertEquals(batchSize.toLong(), parts.r.shape[0])
        assertEquals(dim, parts.r.shape[1])

        parts.r.close()
        parts.i.close()
        parts.j.close()
        parts.k.close()
        batchManager.close()
        embeddings.close()
    }

    @Test
    fun testComputeATailWithZeroRelation() {
        val batchSize = 2
        val dim = 4L

        val fixedData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val relData = FloatArray(batchSize * dim.toInt() * 4) { 0.0f }

        val fixedEmb = manager.create(fixedData).reshape(batchSize.toLong(), dim * 4)
        val relEmb = manager.create(relData).reshape(batchSize.toLong(), dim * 4)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val batchManager = manager.newSubManager()
        val fixedParts = EvalQuatE.sliceQuat(fixedEmb, rIndex, iIndex, jIndex, kIndex, batchManager)
        val relParts = EvalQuatE.sliceQuat(relEmb, rIndex, iIndex, jIndex, kIndex, batchManager)

        val aTail = EvalQuatE.computeATail(fixedParts, relParts, batchManager)

        assertNotNull(aTail)
        // With zero relation, result should be zero
        val rValues = aTail.r.toFloatArray()
        assertTrue(rValues.all { it == 0.0f }, "Result should be zero for zero relation")

        fixedParts.r.close()
        fixedParts.i.close()
        fixedParts.j.close()
        fixedParts.k.close()
        relParts.r.close()
        relParts.i.close()
        relParts.j.close()
        relParts.k.close()
        aTail.r.close()
        aTail.i.close()
        aTail.j.close()
        aTail.k.close()
        batchManager.close()
        fixedEmb.close()
        relEmb.close()
    }

    @Test
    fun testAccumulateCountBetterWithLowerIsBetter() {
        val numEntities = 50
        val batchSize = 2
        val dim = 4L

        val entitiesData = FloatArray(numEntities * dim.toInt() * 4) { (it % 10).toFloat() }
        val entities = manager.create(entitiesData).reshape(numEntities.toLong(), dim * 4)

        val aData = FloatArray(batchSize * dim.toInt() * 4) { 1.0f }
        val aEmb = manager.create(aData).reshape(batchSize.toLong(), dim * 4)

        val trueScoreData = floatArrayOf(10.0f, 20.0f)
        val trueScore = manager.create(trueScoreData).reshape(batchSize.toLong(), 1)

        val countBetter = manager.zeros(Shape(batchSize.toLong()), DataType.INT64)

        val rIndex = NDIndex(":, 0:$dim")
        val iIndex = NDIndex(":, $dim:${dim * 2}")
        val jIndex = NDIndex(":, ${dim * 2}:${dim * 3}")
        val kIndex = NDIndex(":, ${dim * 3}:")

        val aParts = EvalQuatE.sliceQuat(aEmb, rIndex, iIndex, jIndex, kIndex, manager)

        EvalQuatE.accumulateCountBetter(
            entities,
            rIndex,
            iIndex,
            jIndex,
            kIndex,
            aParts,
            trueScore,
            countBetter,
            higherIsBetter = false,
            chunkSize = 25,
            numEntitiesInt = numEntities,
        )

        assertNotNull(countBetter)
        val countValues = countBetter.toLongArray()
        for (value in countValues) {
            assertTrue(value >= 0L, "Count better should be non-negative")
        }

        aParts.r.close()
        aParts.i.close()
        aParts.j.close()
        aParts.k.close()
        entities.close()
        aEmb.close()
        trueScore.close()
        countBetter.close()
    }
}
