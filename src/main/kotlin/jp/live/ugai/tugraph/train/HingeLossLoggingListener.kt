package jp.live.ugai.tugraph.train

import ai.djl.training.Trainer
import ai.djl.training.listener.EvaluatorTrainingListener
import java.util.concurrent.ConcurrentHashMap

/**
 * Logs hinge-loss values populated by EmbeddingTrainer into TrainingResult.
 */
class HingeLossLoggingListener : EvaluatorTrainingListener() {
    private var epoch = 0
    private val latest = ConcurrentHashMap<String, Float>()
    private val logger = org.slf4j.LoggerFactory.getLogger(HingeLossLoggingListener::class.java)

    override fun onTrainingBegin(trainer: Trainer) {
        epoch = 0
    }

    override fun onEpoch(trainer: Trainer) {
        epoch += 1
        val train = latest["train_L1Loss"]
        val validate = latest["validate_L1Loss"]
        logger.debug("Epoch {} hinge loss: train={}, validate={}", epoch, train, validate)
    }

    fun updateEvaluations(
        train: Float,
        validate: Float,
    ) {
        latest["train_L1Loss"] = train
        latest["validate_L1Loss"] = validate
    }

    override fun getLatestEvaluations(): Map<String, Float> = latest
}
