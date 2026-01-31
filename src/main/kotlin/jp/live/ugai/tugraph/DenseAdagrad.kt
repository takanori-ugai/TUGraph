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

        var gradWork: NDArray = grad
        var ownsGrad = false
        try {
            if (rescaleGrad != 1.0f) {
                gradWork = gradWork.mul(rescaleGrad)
                ownsGrad = true
            }
            if (clipGrad > 0f) {
                val clipped = gradWork.clip(-clipGrad, clipGrad)
                if (ownsGrad) {
                    gradWork.close()
                }
                gradWork = clipped
                ownsGrad = true
            }
            if (wd != 0f) {
                val wdTerm = weight.mul(wd)
                val withDecay = gradWork.add(wdTerm)
                wdTerm.close()
                if (ownsGrad) {
                    gradWork.close()
                }
                gradWork = withDecay
                ownsGrad = true
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
        } finally {
            if (ownsGrad) {
                gradWork.close()
            }
        }
    }

    class Builder : Optimizer.OptimizerBuilder<Builder>() {
        private var learningRateTracker: ParameterTracker = Tracker.fixed(0.001f)
        private var epsilon: Float = 1.0e-8f

        fun optLearningRateTracker(tracker: ParameterTracker): Builder {
            this.learningRateTracker = tracker
            return this
        }

        fun optEpsilon(value: Float): Builder {
            this.epsilon = value
            return this
        }

        override fun self(): Builder = this

        fun build(): DenseAdagrad = DenseAdagrad(learningRateTracker, epsilon, this)
    }

    companion object {
        fun builder(): Builder = Builder()
    }
}
