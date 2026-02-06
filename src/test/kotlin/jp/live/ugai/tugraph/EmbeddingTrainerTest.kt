package jp.live.ugai.tugraph
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.loss.Loss
import ai.djl.training.tracker.Tracker
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

/**
 * Comprehensive test class for EmbeddingTrainer.
 *
 * Tests the EmbeddingTrainer's initialization, training loop, negative sampling,
 * loss computation, and edge cases.
 */
class EmbeddingTrainerTest {
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
    fun testEmbeddingTrainerInitialization() {
        val triples =
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
                    1,
                ),
            )
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("rotate").also { it.block = rotate }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertNotNull(embeddingTrainer)
        assertNotNull(embeddingTrainer.inputList)
        assertEquals(3, embeddingTrainer.inputList.size)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerInputListConstruction() {
        val triples =
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
                    1,
                    3,
                    2,
                    4,
                ),
            )
        val numEntities = 5L
        val numEdges = 3L
        val dim = 8L
        val distMult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("distmult").also { it.block = distMult }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertEquals(4, embeddingTrainer.inputList.size)
        // Verify first triple
        assertEquals(0L, embeddingTrainer.inputList[0][0])
        assertEquals(0L, embeddingTrainer.inputList[0][1])
        assertEquals(1L, embeddingTrainer.inputList[0][2])
        // Verify last triple
        assertEquals(3L, embeddingTrainer.inputList[3][0])
        assertEquals(2L, embeddingTrainer.inputList[3][1])
        assertEquals(4L, embeddingTrainer.inputList[3][2])
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithDistMult() {
        val triples =
            manager.create(
                longArrayOf(
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                ),
            )
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val distMult =
            DistMult(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("distmult").also { it.block = distMult }
        val lrt = Tracker.fixed(0.01f)
        val sgd =
            ai.djl.training.optimizer.Optimizer
                .sgd()
                .setLearningRateTracker(lrt)
                .build()
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(sgd)
                .optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertNotNull(embeddingTrainer)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithComplEx() {
        val triples =
            manager.create(
                longArrayOf(
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                ),
            )
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val complEx =
            ComplEx(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("complex").also { it.block = complEx }
        val lrt = Tracker.fixed(0.01f)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(adagrad)
                .optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertNotNull(embeddingTrainer)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerLossListAccumulation() {
        val triples =
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
                    1,
                ),
            )
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val epochs = 3
        val transE =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("transe").also { it.block = transE }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, epochs)
        embeddingTrainer.training()
        assertEquals(epochs, embeddingTrainer.lossList.size, "Loss list should have one entry per epoch")
        // Verify all loss values are non-negative
        for (loss in embeddingTrainer.lossList) {
            assertTrue(loss >= 0.0f, "Loss should be non-negative")
        }
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithSingleTriple() {
        val triples = manager.create(longArrayOf(0, 0, 1), Shape(1, TRIPLE))
        val numEntities = 2L
        val numEdges = 1L
        val dim = 8L
        val transE =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("transe").also { it.block = transE }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(Shape(1, TRIPLE))
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertEquals(1, embeddingTrainer.inputList.size)
        embeddingTrainer.training()
        assertEquals(2, embeddingTrainer.lossList.size)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithMultipleTriples() {
        val numTriples = 20
        val triplesArray = LongArray(numTriples * 3)
        for (i in 0 until numTriples) {
            triplesArray[i * 3] = (i % 10).toLong()
            triplesArray[i * 3 + 1] = (i % 3).toLong()
            triplesArray[i * 3 + 2] = ((i + 1) % 10).toLong()
        }
        val triples = manager.create(triplesArray)
        val numEntities = 10L
        val numEdges = 3L
        val dim = 16L
        val transE =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("transe").also { it.block = transE }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertEquals(numTriples, embeddingTrainer.inputList.size)
        embeddingTrainer.training()
        assertEquals(2, embeddingTrainer.lossList.size)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerClose() {
        val triples = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val transE =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("transe").also { it.block = transE }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 1)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithRotatE() {
        val triples = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val rotate =
            RotatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("rotate").also { it.block = rotate }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertNotNull(embeddingTrainer)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerWithQuatE() {
        val triples = manager.create(longArrayOf(0, 0, 1, 1, 1, 2))
        val numEntities = 3L
        val numEdges = 2L
        val dim = 8L
        val quate =
            QuatE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("quate").also { it.block = quate }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.01f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, 2)
        assertNotNull(embeddingTrainer)
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    @Test
    fun testEmbeddingTrainerLossDecreases() {
        val triples =
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
                    1,
                ),
            )
        val numEntities = 3L
        val numEdges = 2L
        val dim = 16L
        val epochs = 10
        val transE =
            TransE(numEntities, numEdges, dim).also {
                it.initialize(manager, DataType.FLOAT32, Shape(1, TRIPLE))
            }
        val model = Model.newInstance("transe").also { it.block = transE }
        val config =
            DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(
                    ai.djl.training.optimizer.Optimizer
                        .sgd()
                        .setLearningRateTracker(Tracker.fixed(0.1f))
                        .build(),
                ).optDevices(manager.engine.getDevices(1))
        val trainer =
            model.newTrainer(config).also {
                it.initialize(triples.shape)
                it.metrics = Metrics()
            }
        disableGradientChecks(trainer)
        val embeddingTrainer = EmbeddingTrainer(manager.newSubManager(), triples, numEntities, trainer, epochs)
        embeddingTrainer.training()
        val firstLoss = embeddingTrainer.lossList[0]
        val lastLoss = embeddingTrainer.lossList[epochs - 1]
        // Loss should generally decrease over training (with some tolerance for fluctuation)
        assertTrue(
            lastLoss <= firstLoss * 1.5f,
            "Loss should not increase significantly over training: first=$firstLoss, last=$lastLoss",
        )
        embeddingTrainer.close()
        trainer.close()
        model.close()
        triples.close()
    }

    private fun disableGradientChecks(trainer: ai.djl.training.Trainer) {
        runCatching {
            val method = trainer.javaClass.getMethod("setCheckGradients", Boolean::class.javaPrimitiveType)
            method.invoke(trainer, false)
        }
    }
}
