package jp.live.ugai.tugraph

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.ParameterTracker
import ai.djl.training.tracker.Tracker
import java.util.concurrent.ConcurrentHashMap

/**
 * Dense Adagrad optimizer implementation that avoids sparse gradient conversion.
 */
class DenseAdagrad(
    private val learningRateTracker: ParameterTracker,
    private val epsilon: Float,
    builder: Optimizer.OptimizerBuilder<*>,
) : Optimizer(builder) {
    private val history: MutableMap<String, MutableMap<Device, NDArray>> = ConcurrentHashMap()

    /**
     * Updates the given parameter `weight` in-place using the Dense Adagrad optimizer with the provided `grad`.
     *
     * The method:
     * - Validates the computed learning rate and configured weight decay and throws IllegalStateException if either is NaN or infinite.
     * - Optionally rescales and clips `grad`, and optionally adds a weight-decay term before applying updates.
     * - Updates the optimizer's per-parameter, per-device accumulated squared-gradient state.
     * - Computes the denominator as sqrt(accumulated_state) + epsilon and applies the Adagrad update: weight -= (lr * processedGrad) / denom.
     *
     * @param name Identifier for the parameter; used to read and update the optimizer state for this parameter and device.
     * @param weight The parameter NDArray to update in-place.
     * @param grad The gradient NDArray for the parameter.
     * @throws IllegalStateException if the learning rate or weight decay is NaN or infinite.
     */
    override fun update(
        name: String,
        weight: NDArray,
        grad: NDArray,
    ) {
        val count = updateCount(name)
        val lr = learningRateTracker.getNewValue(name, count)
        val wd = weightDecay
        if (lr.isNaN() || wd.isNaN() || lr.isInfinite() || wd.isInfinite()) {
            throw IllegalStateException("learning rate or weight decay is nan or infinite")
        }

        grad.manager.newSubManager().use { tempManager ->
            var gradWork: NDArray = grad
            if (rescaleGrad != 1.0f) {
                gradWork = gradWork.mul(rescaleGrad).also { it.attach(tempManager) }
            }
            if (clipGrad > 0f) {
                gradWork = gradWork.clip(-clipGrad, clipGrad).also { it.attach(tempManager) }
            }
            if (wd != 0f) {
                val wdTerm = weight.mul(wd).also { it.attach(tempManager) }
                gradWork = gradWork.add(wdTerm).also { it.attach(tempManager) }
            }

            val state =
                withDefaultState(history, name, weight.device) {
                    weight.zerosLike()
                }
            gradWork.mul(gradWork).use { gradSq ->
                state.addi(gradSq)
            }
            state.sqrt().use { sqrtState ->
                sqrtState.add(epsilon).use { denom ->
                    gradWork.mul(lr).use { scaled ->
                        scaled.div(denom).use { update ->
                            weight.subi(update)
                        }
                    }
                }
            }
        }
    }

    class Builder : Optimizer.OptimizerBuilder<Builder>() {
        private var learningRateTracker: ParameterTracker = Tracker.fixed(0.001f)
        private var epsilon: Float = 1.0e-8f

        /**
         * Sets the tracker used to obtain per-parameter learning rates.
         *
         * @param tracker The ParameterTracker to use for learning rate scheduling.
         * @return This Builder instance.
         */
        fun optLearningRateTracker(tracker: ParameterTracker): Builder {
            this.learningRateTracker = tracker
            return this
        }

        /**
         * Sets the epsilon used for numerical stability in denominator computations.
         *
         * @param value The epsilon value added to the denominator to avoid division by zero.
         * @return This Builder.
         */
        fun optEpsilon(value: Float): Builder {
            this.epsilon = value
            return this
        }

        /**
 * Provide the current builder instance for method chaining.
 *
 * @return This builder instance.
 */
override fun self(): Builder = this

        /**
 * Create a DenseAdagrad optimizer configured with the builder's current settings.
 *
 * @return A new DenseAdagrad instance configured with the builder's learningRateTracker and epsilon.
 */
fun build(): DenseAdagrad = DenseAdagrad(learningRateTracker, epsilon, this)
    }

    companion object {
        /**
 * Create a new Builder for configuring and constructing a DenseAdagrad optimizer.
 *
 * @return A new Builder instance.
 */
fun builder(): Builder = Builder()
    }
}