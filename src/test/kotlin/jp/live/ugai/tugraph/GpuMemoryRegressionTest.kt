package jp.live.ugai.tugraph

import ai.djl.Device
import ai.djl.Model
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.util.cuda.CudaUtils
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Test

class GpuMemoryRegressionTest {
    @Test
    fun transETrainingDoesNotGrowGpuMemory() {
        assumeTrue(CudaUtils.hasCuda())
        assumeTrue(CudaUtils.getGpuCount() > 0)

        val device = Device.gpu(0)
        val manager = NDManager.newBaseManager(device)
        var model: Model? = null
        var trainer: Trainer? = null
        try {
            val numEnt = 128L
            val numRel = 16L
            val dim = 16L
            val triples = manager.create(longArrayOf(0, 0, 1), Shape(1, TRIPLE))
            val negatives = manager.create(longArrayOf(0, 0, 2), Shape(1, TRIPLE))

            val transe =
                TransE(numEnt, numRel, dim).also {
                    it.initialize(manager, DataType.FLOAT32, triples.shape)
                }
            model = Model.newInstance("transe").also { it.block = transe }

            val config =
                DefaultTrainingConfig(Loss.l1Loss())
                    .optOptimizer(
                        Optimizer
                            .sgd()
                            .setLearningRateTracker(Tracker.fixed(0.01f))
                            .build(),
                    ).optDevices(arrayOf(device))
            trainer =
                model.newTrainer(config).also {
                    it.initialize(triples.shape)
                    it.metrics = Metrics()
                }

            val warmupSteps = 100
            repeat(warmupSteps) {
                trainer.newGradientCollector().use { gc ->
                    val out = trainer.forward(NDList(triples, negatives))
                    val baseLoss = out[0].mean()
                    val entL2 = transe.getEntities().mul(transe.getEntities()).mean()
                    val edgeL2 = transe.getEdges().mul(transe.getEdges()).mean()
                    val l2 = entL2.add(edgeL2)
                    val reg = l2.mul(1.0e-6f)
                    val loss = baseLoss.add(reg)
                    gc.backward(loss)
                    loss.close()
                    reg.close()
                    l2.close()
                    edgeL2.close()
                    entL2.close()
                    baseLoss.close()
                    out.close()
                }
                trainer.step()
                transe.normalize()
            }

            val before = CudaUtils.getGpuMemory(device).used
            val steps = 2000
            repeat(steps) {
                trainer.newGradientCollector().use { gc ->
                    val out = trainer.forward(NDList(triples, negatives))
                    val baseLoss = out[0].mean()
                    val entL2 = transe.getEntities().mul(transe.getEntities()).mean()
                    val edgeL2 = transe.getEdges().mul(transe.getEdges()).mean()
                    val l2 = entL2.add(edgeL2)
                    val reg = l2.mul(1.0e-6f)
                    val loss = baseLoss.add(reg)
                    gc.backward(loss)
                    loss.close()
                    reg.close()
                    l2.close()
                    edgeL2.close()
                    entL2.close()
                    baseLoss.close()
                    out.close()
                }
                trainer.step()
                transe.normalize()
            }
            val after = CudaUtils.getGpuMemory(device).used
            val delta = after - before

            // Allow some growth due to allocator caching.
            val maxGrowth = 200L * 1024L * 1024L
            assertTrue(
                delta <= maxGrowth,
                "GPU memory grew by ${delta / (1024 * 1024)} MiB (limit: ${maxGrowth / (1024 * 1024)} MiB)",
            )
        } finally {
            trainer?.close()
            model?.close()
            manager.close()
        }
    }
}
