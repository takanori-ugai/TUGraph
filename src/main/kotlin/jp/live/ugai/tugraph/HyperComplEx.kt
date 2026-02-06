package jp.live.ugai.tugraph

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.UniformInitializer
import ai.djl.training.initializer.XavierInitializer
import ai.djl.util.PairList
import kotlin.math.sqrt

/**
 * HyperComplEx knowledge-graph embedding model.
 *
 * The model blends three scoring spaces—hyperbolic, complex, and Euclidean—then combines them with
 * relation-specific attention weights. Higher scores indicate more plausible triples.
 *
 * @property numEnt Number of entities in the graph.
 * @property numEdge Number of relations in the graph.
 * @property dimH Dimensionality of the hyperbolic embeddings.
 * @property dimC Dimensionality of the complex embeddings (per real/imag component).
 * @property dimE Dimensionality of the Euclidean embeddings.
 * @property curvature Positive curvature used for the Poincaré ball.
 * @property entityShards Number of shards used to store entity embeddings.
 * @property relationShards Number of shards used to store relation embeddings.
 * @property mixedPrecision Whether forward passes cast to FP16 for speed.
 * @property lossScale Mixed-precision loss scale used by training utilities.
 */
class HyperComplEx(
    /** Number of entities in the graph. */
    val numEnt: Long,
    /** Number of relations in the graph. */
    val numEdge: Long,
    /** Hyperbolic embedding dimension. */
    val dimH: Long,
    /** Complex embedding dimension (per real/imag component). */
    val dimC: Long,
    /** Euclidean embedding dimension. */
    val dimE: Long,
    /** Hyperbolic curvature. */
    val curvature: Float = HYPERCOMPLEX_CURVATURE,
    /** Number of shards for entity embeddings. */
    val entityShards: Int = HYPERCOMPLEX_ENTITY_SHARDS,
    /** Number of shards for relation embeddings. */
    val relationShards: Int = HYPERCOMPLEX_RELATION_SHARDS,
    /** Whether to use mixed precision during forward passes. */
    val mixedPrecision: Boolean = HYPERCOMPLEX_MIXED_PRECISION,
    /** Loss scale for mixed precision. */
    val lossScale: Float = HYPERCOMPLEX_LOSS_SCALE,
) : AbstractBlock() {
    private val entH: List<Parameter>
    private val entC: List<Parameter>
    private val entE: List<Parameter>
    private val relH: List<Parameter>
    private val relC: List<Parameter>
    private val relE: List<Parameter>
    private val attnW: Parameter
    private val headIndex = NDIndex(":, 0")
    private val relationIndex = NDIndex(":, 1")
    private val tailIndex = NDIndex(":, 2")
    private val sumAxis = intArrayOf(1)
    private val entityShardOffsets: LongArray
    private val entityShardSizes: LongArray
    private val relationShardOffsets: LongArray
    private val relationShardSizes: LongArray

    /** Convenience constructor that uses the same dimension for all three spaces. */
    constructor(
        numEnt: Long,
        numEdge: Long,
        dim: Long,
        curvature: Float = HYPERCOMPLEX_CURVATURE,
    ) : this(numEnt, numEdge, dim, dim, dim, curvature)

    init {
        val boundH = 1.0e-3f
        val entShardSpec = shardSpec(numEnt, entityShards)
        entityShardOffsets = entShardSpec.offsets
        entityShardSizes = entShardSpec.sizes
        val relShardSpec = shardSpec(numEdge, relationShards)
        relationShardOffsets = relShardSpec.offsets
        relationShardSizes = relShardSpec.sizes

        entH =
            createShards(
                "entH",
                entityShardSizes,
                dimH,
                UniformInitializer(boundH),
            )
        relH =
            createShards(
                "relH",
                relationShardSizes,
                dimH,
                UniformInitializer(boundH),
            )
        entC =
            createShards(
                "entC",
                entityShardSizes,
                dimC * 2,
                XavierInitializer(),
            )
        relC =
            createShards(
                "relC",
                relationShardSizes,
                dimC * 2,
                XavierInitializer(),
            )
        entE =
            createShards(
                "entE",
                entityShardSizes,
                dimE,
                XavierInitializer(),
            )
        relE =
            createShards(
                "relE",
                relationShardSizes,
                dimE,
                XavierInitializer(),
            )
        attnW =
            addParameter(
                Parameter
                    .builder()
                    .setName("attnW")
                    .setType(Parameter.Type.WEIGHT)
                    .optShape(Shape(numEdge, 3))
                    .build(),
            )
    }

    override fun initialize(
        manager: ai.djl.ndarray.NDManager,
        dataType: ai.djl.ndarray.types.DataType?,
        vararg inputShapes: Shape?,
    ) {
        super.initialize(manager, dataType, *inputShapes)
        val attnArr = getParameters().get("attnW").array
        val init =
            manager.randomUniform(
                0.0f,
                1.0f,
                attnArr.shape,
                attnArr.dataType,
                attnArr.device,
            )
        init.copyTo(attnArr)
        init.close()
    }

    /**
     * Compute scores for one or two batches of triples.
     *
     * The first NDArray in `inputs` must be shaped `(N, TRIPLE)` or `(N * TRIPLE)`. If a second
     * NDArray is present, it is scored independently and appended to the output list.
     *
     * @param inputs First element is a batch of triples; optional second element is another batch.
     * @param parameterStore Parameter store used to fetch embeddings.
     * @param training Whether parameters should be fetched in training mode.
     * @param params Additional parameters (unused).
     * @return An NDList containing one score vector per input batch.
     */
    @Override
    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val positive = inputs[0]
        val ret = NDList(scoreParts(positive, parameterStore, training).total)
        if (inputs.size > 1) {
            val negative = inputs[1]
            ret.add(scoreParts(negative, parameterStore, training).total)
        }
        return ret
    }

    data class ScoreParts(
        val total: NDArray,
        val hyperbolic: NDArray,
        val complex: NDArray,
        val euclidean: NDArray,
    )

    /**
     * Compute attention-weighted total scores and their hyperbolic, complex, and Euclidean components for a batch of triples.
     *
     * @param input Triples given either as shape `(N, TRIPLE)` or flattened `(N * TRIPLE)` where each triple is (head, relation, tail) ids.
     * @param training Whether parameters should be loaded in training mode.
     * @return A [ScoreParts] containing:
     *   - `total`: per-triple aggregate scores (higher = more plausible),
     *   - `hyperbolic`, `complex`, `euclidean`: per-triple score contributions from each subspace (all have the same shape as `total`).
     */
    fun scoreParts(
        input: NDArray,
        parameterStore: ParameterStore,
        training: Boolean,
    ): ScoreParts {
        val device = input.device
        val attnArr = parameterStore.getValue(attnW, device, training)
        val numTriples = input.size() / TRIPLE
        val parent = input.manager
        return parent.newSubManager().use { sm ->
            val triples = input.reshape(numTriples, TRIPLE).also { it.attach(sm) }
            val headIds = triples.get(headIndex).also { it.attach(sm) }
            val relationIds = triples.get(relationIndex).also { it.attach(sm) }
            val tailIds = triples.get(tailIndex).also { it.attach(sm) }

            val hH =
                gatherSharded(
                    headIds,
                    entH,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val rH =
                gatherSharded(
                    relationIds,
                    relH,
                    relationShardOffsets,
                    relationShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val tH =
                gatherSharded(
                    tailIds,
                    entH,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }

            val hC =
                gatherSharded(
                    headIds,
                    entC,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val rC =
                gatherSharded(
                    relationIds,
                    relC,
                    relationShardOffsets,
                    relationShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val tC =
                gatherSharded(
                    tailIds,
                    entC,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }

            val hE =
                gatherSharded(
                    headIds,
                    entE,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val rE =
                gatherSharded(
                    relationIds,
                    relE,
                    relationShardOffsets,
                    relationShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }
            val tE =
                gatherSharded(
                    tailIds,
                    entE,
                    entityShardOffsets,
                    entityShardSizes,
                    parameterStore,
                    device,
                    training,
                ).also { it.attach(sm) }

            val relAttn = attnArr.get(relationIds).also { it.attach(sm) }
            val alpha = relAttn.softmax(1).also { it.attach(sm) }
            val alphaH = alpha.get(NDIndex(":, 0")).also { it.attach(sm) }
            val alphaC = alpha.get(NDIndex(":, 1")).also { it.attach(sm) }
            val alphaE = alpha.get(NDIndex(":, 2")).also { it.attach(sm) }

            val mpH = maybeCast(hH, DataType.FLOAT16).also { it.attach(sm) }
            val mpR = maybeCast(rH, DataType.FLOAT16).also { it.attach(sm) }
            val mpT = maybeCast(tH, DataType.FLOAT16).also { it.attach(sm) }
            val mpHC = maybeCast(hC, DataType.FLOAT16).also { it.attach(sm) }
            val mpRC = maybeCast(rC, DataType.FLOAT16).also { it.attach(sm) }
            val mpTC = maybeCast(tC, DataType.FLOAT16).also { it.attach(sm) }
            val mpHE = maybeCast(hE, DataType.FLOAT16).also { it.attach(sm) }
            val mpRE = maybeCast(rE, DataType.FLOAT16).also { it.attach(sm) }
            val mpTE = maybeCast(tE, DataType.FLOAT16).also { it.attach(sm) }

            val hyper = hyperbolicScore(mpH, mpR, mpT).also { it.attach(sm) }
            val complex = complexScore(mpHC, mpRC, mpTC).also { it.attach(sm) }
            val euclid = euclideanScore(mpHE, mpRE, mpTE).also { it.attach(sm) }

            val total =
                alphaH
                    .mul(hyper)
                    .add(alphaC.mul(complex))
                    .add(alphaE.mul(euclid))
                    .also { it.attach(sm) }
            val totalOut = maybeCast(total, DataType.FLOAT32).also { it.attach(parent) }
            val hyperOut = maybeCast(hyper, DataType.FLOAT32).also { it.attach(parent) }
            val complexOut = maybeCast(complex, DataType.FLOAT32).also { it.attach(parent) }
            val euclidOut = maybeCast(euclid, DataType.FLOAT32).also { it.attach(parent) }
            ScoreParts(
                total = totalOut,
                hyperbolic = hyperOut,
                complex = complexOut,
                euclidean = euclidOut,
            )
        }
    }

    /**
     * Casts the given NDArray to the target data type only when mixed-precision mode is enabled and the array's type differs from the target.
     *
     * @param array The array to potentially cast.
     * @param target The desired data type.
     * @return The input `array` converted to `target` if mixed precision is enabled and types differ, otherwise the original `array`.
     */
    private fun maybeCast(
        array: NDArray,
        target: DataType,
    ): NDArray =
        if (mixedPrecision && array.dataType != target) {
            array.toType(target, false)
        } else {
            array
        }

    /**
     * Compute the hyperbolic-space score for a (head, relation, tail) triple.
     *
     * @param h Head entity embedding in the Poincaré ball.
     * @param r Relation embedding in the Poincaré ball.
     * @param t Tail entity embedding in the Poincaré ball.
     * @return The score defined as the negated hyperbolic distance between `mobiusAdd(h, r)` and `t`.
     */
    private fun hyperbolicScore(
        h: NDArray,
        r: NDArray,
        t: NDArray,
    ): NDArray {
        val hPlusR = mobiusAdd(h, r)
        return hyperbolicDistance(hPlusR, t).neg()
    }

    private fun complexScore(
        h: NDArray,
        r: NDArray,
        t: NDArray,
    ): NDArray {
        val realIndex = NDIndex(":, 0:$dimC")
        val imagIndex = NDIndex(":, $dimC:")
        val hRe = h.get(realIndex)
        val hIm = h.get(imagIndex)
        val rRe = r.get(realIndex)
        val rIm = r.get(imagIndex)
        val tRe = t.get(realIndex)
        val tIm = t.get(imagIndex)

        val term1 = hRe.mul(rRe).sub(hIm.mul(rIm))
        val term2 = hRe.mul(rIm).add(hIm.mul(rRe))
        val sum1 = term1.mul(tRe)
        val sum2 = term2.mul(tIm)
        return sum1.add(sum2).sum(sumAxis)
    }

    /**
     * Computes the Euclidean-space score for a triple as the negative squared distance between (h + r) and t.
     *
     * @param h Head entity embedding.
     * @param r Relation embedding.
     * @param t Tail entity embedding.
     * @return The negated sum of squared differences between (h + r) and t computed over the embedding axis.
     */
    private fun euclideanScore(
        h: NDArray,
        r: NDArray,
        t: NDArray,
    ): NDArray =
        h
            .add(r)
            .sub(t)
            .pow(2)
            .sum(sumAxis)
            .neg()

    /**
     * Compute the Möbius addition of two batches of vectors in the Poincaré ball using the block's curvature.
     *
     * @param x Batch of hyperbolic vectors with shape (batchSize, dim).
     * @param y Batch of hyperbolic vectors with shape (batchSize, dim).
     * @return The Möbius sum of `x` and `y`, with shape (batchSize, dim).
     */
    private fun mobiusAdd(
        x: NDArray,
        y: NDArray,
    ): NDArray {
        val c = curvature
        val xy = x.mul(y).sum(sumAxis).reshape(x.shape[0], 1)
        val x2 = x.pow(2).sum(sumAxis).reshape(x.shape[0], 1)
        val y2 = y.pow(2).sum(sumAxis).reshape(x.shape[0], 1)
        val twoC = 2.0f * c
        val numLeft = x.mul(xy.mul(twoC).add(y2.mul(c)).add(1f))
        val numRight = y.mul(x2.mul(-c).add(1f))
        val numerator = numLeft.add(numRight)
        val denominator =
            xy.mul(twoC).add(x2.mul(y2).mul(c * c)).add(1f)
        return numerator.div(denominator)
    }

    private fun hyperbolicDistance(
        x: NDArray,
        y: NDArray,
    ): NDArray {
        val c = curvature
        val diff2 = x.sub(y).pow(2).sum(sumAxis)
        val x2 = x.pow(2).sum(sumAxis)
        val y2 = y.pow(2).sum(sumAxis)
        val denom = x2.mul(-c).add(1f).mul(y2.mul(-c).add(1f))
        val term = diff2.mul(2f).div(denom).add(1f)
        val termClamped = term.maximum(1.0f + 1.0e-6f)
        val acosh = acosh(termClamped)
        return acosh.div(sqrt(c.toDouble()).toFloat())
    }

    private fun acosh(x: NDArray): NDArray {
        val zMinus1 = x.sub(1f)
        val zPlus1 = x.add(1f)
        val sqrtProd = zMinus1.mul(zPlus1).sqrt()
        return x.add(sqrtProd).log()
    }

    /**
     * Project hyperbolic embeddings back into the Poincaré ball for numerical stability.
     *
     * This should be called after updates to keep norms within the valid hyperbolic radius.
     */
    fun projectHyperbolic() {
        val maxNorm = (1.0f - 1.0e-5f) / sqrt(curvature.toDouble()).toFloat()
        for (param in entH) {
            projectToBall(param.array, maxNorm)
        }
        for (param in relH) {
            projectToBall(param.array, maxNorm)
        }
    }

    /**
     * Projects each row vector in `x` onto the Euclidean ball of radius `maxNorm`, modifying `x` in place.
     *
     * If a row's Euclidean norm is greater than `maxNorm`, it is scaled down to have norm equal to `maxNorm`;
     * rows with norm less than or equal to `maxNorm` are left unchanged.
     *
     * @param x NDArray containing row vectors with shape (N, D); the array is modified in place.
     * @param maxNorm Maximum allowed Euclidean norm for each row vector.
     */
    private fun projectToBall(
        x: NDArray,
        maxNorm: Float,
    ) {
        val sq = x.pow(2)
        val summed = sq.sum(sumAxis)
        sq.close()
        val sqrtVal = summed.sqrt()
        summed.close()
        val norms = sqrtVal.reshape(x.shape[0], 1)
        sqrtVal.close()
        val maxNormTensor = x.manager.full(norms.shape, maxNorm, norms.dataType, norms.device)
        val ratio = maxNormTensor.div(norms)
        val scale = ratio.minimum(1f)
        x.muli(scale)
        norms.close()
        maxNormTensor.close()
        ratio.close()
        scale.close()
    }

    /**
     * Compute output shapes for the provided input shapes.
     *
     * Each output is a 1D shape with length equal to the number of triples, computed as
     * `inputs[0].size() / TRIPLE`. A second output shape is returned only when a second input exists.
     *
     * @param inputs Array of input shapes; the first element represents triples of width `TRIPLE`.
     * @return Output shapes, one per input.
     */
    @Override
    override fun getOutputShapes(inputs: Array<Shape>): Array<Shape> {
        val numTriples = inputs[0].size() / TRIPLE
        val outShape = Shape(numTriples)
        return if (inputs.size > 1) {
            arrayOf(outShape, outShape)
        } else {
            arrayOf(outShape)
        }
    }

    /** Retrieve the hyperbolic entity embeddings array (caller must close). */
    fun getEntitiesH(): NDArray = concatParamShards(entH)

    /** Retrieve the complex entity embeddings array (caller must close). */
    fun getEntitiesC(): NDArray = concatParamShards(entC)

    /** Retrieve the Euclidean entity embeddings array (caller must close). */
    fun getEntitiesE(): NDArray = concatParamShards(entE)

    /** Retrieve the hyperbolic relation embeddings array (caller must close). */
    fun getEdgesH(): NDArray = concatParamShards(relH)

    /** Retrieve the complex relation embeddings array (caller must close). */
    fun getEdgesC(): NDArray = concatParamShards(relC)

    /**
     * Concatenates and exposes the Euclidean embeddings for all relations.
     *
     * The returned NDArray is a newly allocated view combining all relation shards; the caller is responsible for closing it.
     *
     * @return An NDArray containing all relation Euclidean embeddings stacked along axis 0.
     */
    fun getEdgesE(): NDArray = concatParamShards(relE)

    /**
     * Live attention weight matrix for relations; the returned NDArray is the live parameter and must not be closed.
     *
     * @return An NDArray of shape (numEdge, 3) containing per-relation attention weights. The caller must not close this NDArray.
     */
    fun getAttention(): NDArray = getParameters().get("attnW").array

    /** Retrieve hyperbolic entity embeddings on the requested device. */
    fun getEntitiesH(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(entH.map { parameterStore.getValue(it, device, training) })

    /** Retrieve complex entity embeddings on the requested device. */
    fun getEntitiesC(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(entC.map { parameterStore.getValue(it, device, training) })

    /** Retrieve Euclidean entity embeddings on the requested device. */
    fun getEntitiesE(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(entE.map { parameterStore.getValue(it, device, training) })

    /** Retrieve hyperbolic relation embeddings on the requested device. */
    fun getEdgesH(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(relH.map { parameterStore.getValue(it, device, training) })

    /** Retrieve complex relation embeddings on the requested device. */
    fun getEdgesC(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(relC.map { parameterStore.getValue(it, device, training) })

    /** Retrieve Euclidean relation embeddings on the requested device. */
    fun getEdgesE(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = concatNdShards(relE.map { parameterStore.getValue(it, device, training) })

    /** Retrieve attention weights on the requested device. */
    fun getAttention(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray = parameterStore.getValue(attnW, device, training)

    /**
     * Compute the mean L2 regularization loss over all entity and relation embeddings.
     *
     * The returned scalar is the sum of squared values of all embedding parameters divided by
     * (numEnt + numEdge) to make the regularization scale invariant to graph size.
     *
     * @return A scalar NDArray containing the averaged L2 regularization loss.
     * @throws IllegalArgumentException if no embedding parameters are present to compute the loss.
     */
    fun l2RegLoss(
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray {
        var sum: NDArray? = null

        /**
         * Adds the parameter's squared L2 norm to the running sum.
         *
         * Retrieves the parameter's array for the current device/training mode, computes the sum of its squared elements,
         * and accumulates that value into the enclosing `sum` variable.
         *
         * @param param The parameter whose squared L2 norm will be accumulated.
         */
        fun addParam(param: Parameter) {
            val arr = parameterStore.getValue(param, device, training)
            val term = arr.pow(2).use { it.sum() }
            sum = sum?.use { s -> term.use { t -> s.add(t) } } ?: term
        }
        for (param in entH) addParam(param)
        for (param in entC) addParam(param)
        for (param in entE) addParam(param)
        for (param in relH) addParam(param)
        for (param in relC) addParam(param)
        for (param in relE) addParam(param)
        val denom = (numEnt + numEdge).toFloat().coerceAtLeast(1f)
        val total = requireNotNull(sum) { "No parameters found for L2 loss." }
        val result = total.div(denom)
        total.close()
        return result
    }

    /** Unscale gradients for mixed-precision training. */
    fun unscaleGradients(scale: Float) {
        if (scale == 0.0f) return
        val inv = 1.0f / scale
        for (param in allParameters()) {
            val arr = param.array
            if (arr.hasGradient()) {
                arr.getGradient().muli(inv)
            }
        }
    }

    /**
     * Collects every model parameter (entity and relation shards across spaces, plus attention) into a single list.
     *
     * @return A list containing all `Parameter` instances: entity hyperbolic/complex/euclidean shards, relation hyperbolic/complex/euclidean shards, and the attention weight parameter.
     */
    private fun allParameters(): List<Parameter> {
        val params = ArrayList<Parameter>(entH.size + entC.size + entE.size + relH.size + relC.size + relE.size + 1)
        params.addAll(entH)
        params.addAll(entC)
        params.addAll(entE)
        params.addAll(relH)
        params.addAll(relC)
        params.addAll(relE)
        params.add(attnW)
        return params
    }

    private data class ShardSpec(
        val offsets: LongArray,
        val sizes: LongArray,
    )

    /**
     * Compute a balanced shard specification for dividing `total` items into up to `shards` parts.
     *
     * The returned `ShardSpec` contains `offsets` and `sizes` arrays describing each shard. Sizes
     * sum to `total` and differ by at most one between any two shards. If `shards` is less than 1
     * it is treated as 1; if `shards` is greater than `total`, the number of shards will be reduced
     * to at most `total`.
     *
     * @param total The total number of items to split across shards.
     * @param shards The desired number of shards.
     * @return A `ShardSpec` whose `offsets` and `sizes` partition `total` items into balanced shards.
     */
    private fun shardSpec(
        total: Long,
        shards: Int,
    ): ShardSpec {
        val shardCount = maxOf(1, minOf(shards, total.toInt()))
        val sizes = LongArray(shardCount)
        val offsets = LongArray(shardCount)
        val base = total / shardCount
        val rem = (total % shardCount).toInt()
        var offset = 0L
        for (i in 0 until shardCount) {
            val extra = if (i < rem) 1L else 0L
            val size = base + extra
            sizes[i] = size
            offsets[i] = offset
            offset += size
        }
        return ShardSpec(offsets, sizes)
    }

    /**
     * Creates a list of parameter shards named with the given prefix and indexed suffixes.
     *
     * Each shard is a weight Parameter with shape (sizes[i], dim) and the provided initializer.
     *
     * @param prefix Base name used for each shard; final names are `"$prefix-0"`, `"$prefix-1"`, etc.
     * @param sizes Array of row counts for each shard.
     * @param dim Number of columns (embedding dimension) for every shard.
     * @param initializer Initializer applied to each shard parameter.
     * @return A list of created Parameter objects, one per entry in `sizes`, in the same order.
     */
    private fun createShards(
        prefix: String,
        sizes: LongArray,
        dim: Long,
        initializer: ai.djl.training.initializer.Initializer,
    ): List<Parameter> {
        val params = ArrayList<Parameter>(sizes.size)
        for (i in sizes.indices) {
            val name = "$prefix-$i"
            val param =
                addParameter(
                    Parameter
                        .builder()
                        .setName(name)
                        .setType(Parameter.Type.WEIGHT)
                        .optInitializer(initializer)
                        .optShape(Shape(sizes[i], dim))
                        .build(),
                )
            params.add(param)
        }
        return params
    }

    private fun gatherSharded(
        ids: NDArray,
        shards: List<Parameter>,
        offsets: LongArray,
        sizes: LongArray,
        parameterStore: ParameterStore,
        device: Device,
        training: Boolean,
    ): NDArray {
        val idsLong = ids.toLongArray()
        val shardCount = shards.size
        val positions = Array(shardCount) { LongArray(0) }
        val locals = Array(shardCount) { LongArray(0) }
        val posLists = Array(shardCount) { ArrayList<Long>() }
        val localLists = Array(shardCount) { ArrayList<Long>() }

        for (i in idsLong.indices) {
            val id = idsLong[i]
            var shardIdx = 0
            while (shardIdx < shardCount - 1 && id >= offsets[shardIdx] + sizes[shardIdx]) {
                shardIdx++
            }
            posLists[shardIdx].add(i.toLong())
            localLists[shardIdx].add(id - offsets[shardIdx])
        }
        for (i in 0 until shardCount) {
            val p = posLists[i]
            val l = localLists[i]
            positions[i] = LongArray(p.size) { p[it] }
            locals[i] = LongArray(l.size) { l[it] }
        }

        val manager = ids.manager
        var output: NDArray? = null
        val outDim = shards.first().shape[1]
        for (i in 0 until shardCount) {
            if (positions[i].isEmpty()) continue
            val posNd = manager.create(positions[i]).reshape(positions[i].size.toLong(), 1)
            val localNd = manager.create(locals[i])
            val shardArr = parameterStore.getValue(shards[i], device, training)
            val gathered = shardArr.get(localNd)
            val currentOut =
                output ?: manager.zeros(Shape(ids.shape[0], outDim), shardArr.dataType, device)
            output = currentOut.scatter(posNd, gathered, 0)
            posNd.close()
            localNd.close()
            gathered.close()
            if (currentOut !== output) {
                currentOut.close()
            }
        }
        return requireNotNull(output) { "Failed to gather sharded embeddings." }
    }

    /**
     * Concatenates a list of NDArrays along axis 0.
     *
     * @param shards The list of NDArrays to join; must not be empty.
     * @return The concatenated NDArray along axis 0. If `shards` contains a single array, that array is returned unchanged.
     */
    private fun concatNdShards(shards: List<NDArray>): NDArray {
        require(shards.isNotEmpty()) { "No shards to concat." }
        return if (shards.size == 1) {
            shards[0]
        } else {
            NDArrays.concat(NDList(shards), 0)
        }
    }

    /**
     * Concatenates a list of parameter shards into a single NDArray.
     *
     * @param shards Parameter shards whose underlying arrays will be concatenated along axis 0.
     * @return An NDArray containing all shard arrays stacked along the first (0th) axis.
     */
    private fun concatParamShards(shards: List<Parameter>): NDArray = concatNdShards(shards.map { it.array })
}
