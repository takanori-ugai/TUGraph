package jp.live.ugai.tugraph;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class FillMaskTranslator2 implements Translator<String, Classifications> {

  private HuggingFaceTokenizer tokenizer;
  private String maskToken;
  private long maskTokenId;
  private int topK;
  private Batchifier batchifier;

  public FillMaskTranslator2(
      HuggingFaceTokenizer tokenizer, String maskToken, int topK, Batchifier batchifier) {
    this.tokenizer = tokenizer;
    this.maskToken = maskToken;
    this.topK = topK;
    this.batchifier = batchifier;
    Encoding encoding = tokenizer.encode(maskToken, false);
    maskTokenId = encoding.getIds()[0];
  }

  /** {@inheritDoc} */
  @Override
  public Batchifier getBatchifier() {
    return batchifier;
  }

  /** {@inheritDoc} */
  @Override
  public NDList processInput(TranslatorContext ctx, String input) throws TranslateException {
    Encoding encoding = tokenizer.encode(input);
    long[] indices = encoding.getIds();
    int maskIndex = getMaskIndex(indices, maskToken, maskTokenId);
    ctx.setAttachment("maskIndex", maskIndex);
    return encoding.toNDList(ctx.getNDManager(), false);
  }

  /** {@inheritDoc} */
  @Override
  public Classifications processOutput(TranslatorContext ctx, NDList list) {
    int maskIndex = (int) ctx.getAttachment("maskIndex");
    System.out.println("TopK::: " + topK);
    return toClassifications(tokenizer, list, maskIndex, topK);
  }

  /** {@inheritDoc} */
  @Override
  public FillMaskBatchTranslator2 toBatchTranslator(Batchifier batchifier) {
    tokenizer.enableBatch();
    return new FillMaskBatchTranslator2(tokenizer, maskToken, topK, batchifier);
  }

  static int getMaskIndex(long[] indices, String maskToken, long maskTokenId)
      throws TranslateException {
    int maskIndex = -1;
    for (int i = 0; i < indices.length; ++i) {
      if (indices[i] == maskTokenId) {
        if (maskIndex != -1) {
          throw new TranslateException("Only one mask supported.");
        }
        maskIndex = i;
      }
    }
    if (maskIndex == -1) {
      throw new TranslateException("Mask token " + maskToken + " not found.");
    }
    return maskIndex;
  }

  static Classifications toClassifications(
      HuggingFaceTokenizer tokenizer, NDList output, int maskIndex, int topK) {
    NDArray prob = output.get(0).get(maskIndex).softmax(0);
    NDArray array = prob.argSort(0, false);
    long[] classIds = new long[topK];
    List<Double> probabilities = new ArrayList<>(topK);
    for (int i = 0; i < topK; ++i) {
      classIds[i] = array.getLong(i);
      probabilities.add((double) prob.getFloat(classIds[i]));
    }

    System.out.println(classIds.length);
    String[] classes = tokenizer.decode(classIds).trim().split(" ");
    System.out.println(classes.length);
    return new Classifications(Arrays.asList(classes), probabilities);
  }

  /**
   * Creates a builder to build a {@code FillMaskTranslator}.
   *
   * @param tokenizer the tokenizer
   * @return a new builder
   */
  public static Builder builder(HuggingFaceTokenizer tokenizer) {
    return new Builder(tokenizer);
  }

  /**
   * Creates a builder to build a {@code FillMaskTranslator}.
   *
   * @param tokenizer the tokenizer
   * @param arguments the models' arguments
   * @return a new builder
   */
  public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
    Builder builder = builder(tokenizer);
    builder.configure(arguments);

    return builder;
  }

  /** The builder for fill mask translator. */
  public static final class Builder {

    private HuggingFaceTokenizer tokenizer;
    private String maskedToken = "[MASK]";
    private int topK = 5;
    private Batchifier batchifier = Batchifier.STACK;

    Builder(HuggingFaceTokenizer tokenizer) {
      this.tokenizer = tokenizer;
    }

    /**
     * Sets the id of the mask {@link Translator}.
     *
     * @param maskedToken the id of the mask
     * @return this builder
     */
    public Builder optMaskToken(String maskedToken) {
      this.maskedToken = maskedToken;
      return this;
    }

    /**
     * Set the topK number of classes to be displayed.
     *
     * @param topK the number of top classes to return
     * @return this builder
     */
    public Builder optTopK(int topK) {
      this.topK = topK;
      return this;
    }

    /**
     * Sets the {@link Batchifier} for the {@link Translator}.
     *
     * @param batchifier true to include token types
     * @return this builder
     */
    public Builder optBatchifier(Batchifier batchifier) {
      this.batchifier = batchifier;
      return this;
    }

    /**
     * Configures the builder with the model arguments.
     *
     * @param arguments the model arguments
     */
    public void configure(Map<String, ?> arguments) {
      optMaskToken(ArgumentsUtil.stringValue(arguments, "maskToken", "[MASK]"));
      optTopK(ArgumentsUtil.intValue(arguments, "topK", 5));
      String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
      optBatchifier(Batchifier.fromString(batchifierStr));
    }

    /**
     * Builds the translator.
     *
     * @return the new translator
     * @throws IOException if I/O error occurs
     */
    public FillMaskTranslator2 build() throws IOException {
      return new FillMaskTranslator2(tokenizer, maskedToken, topK, batchifier);
    }
  }
}
