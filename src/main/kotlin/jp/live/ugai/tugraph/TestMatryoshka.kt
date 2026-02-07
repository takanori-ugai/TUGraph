package jp.live.ugai.tugraph

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.ParameterStore
import ai.djl.training.listener.EpochTrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.tracker.Tracker
import ai.djl.translate.NoopTranslator
import jp.live.ugai.tugraph.demo.buildTriplesInfo
import jp.live.ugai.tugraph.demo.newTrainer
import jp.live.ugai.tugraph.eval.ResultEvalQuatE
import jp.live.ugai.tugraph.train.DenseAdagrad
import jp.live.ugai.tugraph.train.EmbeddingTrainer
import jp.live.ugai.tugraph.train.HingeLossLoggingListener

/**
 * Runs an end-to-end example that trains, inspects, predicts with, and evaluates a QuatE embedding model using "data/sample.csv".
 *
 * Reads triples from the CSV, derives dataset sizes (entities/relations), initializes a QuatE block and a Model, configures and runs
 * an embedding training loop, computes Matryoshka QuatE scores for a small batch of triples, performs a single prediction, evaluates
 * head and tail prediction quality using ResultEvalQuatE, and prints training results, Matryoshka scores, a sample prediction, and
 * evaluation summaries. All created DJL/NDManager resources are closed before exit.
 */
fun main() {
    NDManager.newBaseManager().use { manager ->
        val csvReader = CsvToNdarray(manager)
        val input = csvReader.read("data/sample.csv")
        println(input)
        val triples = buildTriplesInfo(input)

        val quate =
            QuatE(triples.numEntities, triples.numEdges, DIMENSION).also {
                it.initialize(manager, DataType.FLOAT32, input.shape)
            }
        val model =
            Model.newInstance("quate-matryoshka").also {
                it.block = quate
            }

        val lrt = Tracker.fixed(ADAGRAD_LEARNING_RATE)
        val adagrad = DenseAdagrad.builder().optLearningRateTracker(lrt).build()

        val config =
            DefaultTrainingConfig(Loss.l1Loss()) // Placeholder loss; EmbeddingTrainer computes its own.
                .optOptimizer(adagrad) // Optimizer
                .optDevices(manager.engine.getDevices(1)) // single GPU
                .addTrainingListeners(EpochTrainingListener(), HingeLossLoggingListener()) // Hinge loss logging

        val trainer = newTrainer(model, config, Shape(BATCH_SIZE.toLong(), TRIPLE))

        val eTrainer =
            EmbeddingTrainer(manager.newSubManager(), input, triples.numEntities, trainer, NEPOCH, useMatryoshkaOverride = true)
        eTrainer.training()
        eTrainer.close()
        println(trainer.trainingResult)

        val predictor = model.newPredictor(NoopTranslator())

        // Matryoshka scores using nested quaternion component dimensions.
        val matryoshkaDims = MATRYOSHKA_QUATE_DIMS
        val firstBatch = input.get("0:${minOf(4, triples.numTriples)}, :")
        firstBatch.use { batch ->
            manager.newSubManager().use { sm ->
                val parameterStore = ParameterStore(sm, true)
                val device = batch.device
                val entities = quate.getEntities(parameterStore, device, false)
                val edges = quate.getEdges(parameterStore, device, false)
                require(entities.shape[1] % 4L == 0L) {
                    "QuatE embedding width must be divisible by 4, was ${entities.shape[1]}."
                }
                val fullDim = entities.shape[1] / 4
                val componentDims = resolveQuatEComponentDims(matryoshkaDims, fullDim, entities.shape[1])
                for (dim in componentDims) {
                    val score = matryoshkaQuatEScore(batch, entities, edges, dim, fullDim)
                    try {
                        println("Matryoshka QuatE score (componentDim=$dim): $score")
                    } finally {
                        score.close()
                    }
                }
            }
        }

        val test = manager.create(longArrayOf(1, 1, 2))
        test.use {
            println(predictor.predict(NDList(it)).singletonOrThrow())
        }

        val result =
            ResultEvalQuatE(
                triples.inputList,
                manager.newSubManager(),
                predictor,
                triples.numEntities,
                quatE = quate,
                higherIsBetter = true,
            )
        println("Tail")
        result.getTailResult().forEach {
            println("${it.key} : ${it.value}")
        }
        println("Head")
        result.getHeadResult().forEach {
            println("${it.key} : ${it.value}")
        }
        result.close()
        predictor.close()
        trainer.close()
        model.close()
    }
}

private data class QuatView(
    val r: NDArray,
    val i: NDArray,
    val j: NDArray,
    val k: NDArray,
)

private fun resolveQuatEComponentDims(
    dims: LongArray,
    fullDim: Long,
    embDim: Long,
): List<Long> {
    val resolved = ArrayList<Long>(dims.size)
    val useTotalDims = dims.any { it > fullDim }
    for (d in dims) {
        if (d <= 0L) continue
        val compDim =
            if (useTotalDims) {
                if (d <= embDim && d % 4L == 0L) d / 4L else null
            } else {
                if (d <= fullDim) d else null
            }
        if (compDim != null && compDim <= fullDim) {
            resolved.add(compDim)
        }
    }
    require(resolved.isNotEmpty()) { "No valid Matryoshka dims for QuatE (fullDim=$fullDim, embDim=$embDim)." }
    return resolved
}

private fun matryoshkaQuatEScore(
    input: NDArray,
    entities: NDArray,
    edges: NDArray,
    componentDim: Long,
    fullDim: Long,
): NDArray {
    require(componentDim > 0L) { "componentDim must be > 0." }
    require(componentDim <= fullDim) { "componentDim must be <= fullDim ($fullDim)." }
    val numTriples = input.size() / TRIPLE
    val parent = input.manager
    return parent.newSubManager().use { sm ->
        val triples = input.reshape(numTriples, TRIPLE).also { it.attach(sm) }
        val headIds = triples.get(NDIndex(":, 0")).also { it.attach(sm) }
        val relationIds = triples.get(NDIndex(":, 1")).also { it.attach(sm) }
        val tailIds = triples.get(NDIndex(":, 2")).also { it.attach(sm) }

        val heads = entities.get(headIds).also { it.attach(sm) }
        val relations = edges.get(relationIds).also { it.attach(sm) }
        val tails = entities.get(tailIds).also { it.attach(sm) }

        val h = splitQuaternion(heads, componentDim, fullDim, attachTo = { it.attach(sm) })
        val r = splitQuaternion(relations, componentDim, fullDim, attachTo = { it.attach(sm) })
        val t = splitQuaternion(tails, componentDim, fullDim, attachTo = { it.attach(sm) })

        // Hamilton product r ⊗ t
        val rtR =
            r.r
                .mul(t.r)
                .sub(r.i.mul(t.i))
                .sub(r.j.mul(t.j))
                .sub(r.k.mul(t.k))
                .also { it.attach(sm) }
        val rtI =
            r.r
                .mul(t.i)
                .add(r.i.mul(t.r))
                .add(r.j.mul(t.k))
                .sub(r.k.mul(t.j))
                .also { it.attach(sm) }
        val rtJ =
            r.r
                .mul(t.j)
                .sub(r.i.mul(t.k))
                .add(r.j.mul(t.r))
                .add(r.k.mul(t.i))
                .also { it.attach(sm) }
        val rtK =
            r.r
                .mul(t.k)
                .add(r.i.mul(t.j))
                .sub(r.j.mul(t.i))
                .add(r.k.mul(t.r))
                .also { it.attach(sm) }

        val score =
            h.r
                .mul(rtR)
                .add(h.i.mul(rtI))
                .add(h.j.mul(rtJ))
                .add(h.k.mul(rtK))
                .sum(intArrayOf(1))
                .also { it.attach(parent) }
        score
    }
}

private fun splitQuaternion(
    base: NDArray,
    componentDim: Long,
    fullDim: Long,
    attachTo: (NDArray) -> Unit,
): QuatView =
    QuatView(
        base.get(NDIndex(":, 0:$componentDim")).also(attachTo),
        base.get(NDIndex(":, $fullDim:${fullDim + componentDim}")).also(attachTo),
        base.get(NDIndex(":, ${2 * fullDim}:${2 * fullDim + componentDim}")).also(attachTo),
        base.get(NDIndex(":, ${3 * fullDim}:${3 * fullDim + componentDim}")).also(attachTo),
    )

/** Marker class for TestMatryoshka example. */
class TestMatryoshka
