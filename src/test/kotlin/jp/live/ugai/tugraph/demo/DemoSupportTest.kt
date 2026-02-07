package jp.live.ugai.tugraph.demo

import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class DemoSupportTest {
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
    fun buildTriplesInfoWith2dInput() {
        val input =
            manager.create(
                longArrayOf(
                    0,
                    0,
                    1,
                    2,
                    1,
                    3,
                ),
                Shape(2, 3),
            )

        val info = buildTriplesInfo(input)

        assertEquals(2, info.numTriples)
        assertEquals(4L, info.numEntities)
        assertEquals(2L, info.numEdges)
        assertEquals(2, info.inputList.size)
        assertEquals(listOf(0L, 0L, 1L), info.inputList[0].toList())
        assertEquals(listOf(2L, 1L, 3L), info.inputList[1].toList())
        assertEquals(input, info.input)
    }

    @Test
    fun buildTriplesInfoWithFlatInput() {
        val input =
            manager.create(
                longArrayOf(
                    1,
                    2,
                    3,
                    3,
                    0,
                    4,
                ),
            )

        val info = buildTriplesInfo(input)

        assertEquals(2, info.numTriples)
        assertEquals(5L, info.numEntities)
        assertEquals(3L, info.numEdges)
        assertEquals(listOf(1L, 2L, 3L), info.inputList[0].toList())
        assertEquals(listOf(3L, 0L, 4L), info.inputList[1].toList())
    }

    @Test
    fun buildTriplesInfoRejectsInvalidSize() {
        val input = manager.create(longArrayOf(0, 1, 2, 3, 4))

        assertThrows(IllegalArgumentException::class.java) {
            buildTriplesInfo(input)
        }
    }
}
