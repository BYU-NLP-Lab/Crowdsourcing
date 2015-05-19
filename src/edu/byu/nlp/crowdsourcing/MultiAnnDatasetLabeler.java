/**
 * Copyright 2013 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing;

import java.io.PrintWriter;

import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.ByteStreams;

import edu.byu.nlp.classify.data.DatasetBuilder;
import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.classify.util.ModelTraining;
import edu.byu.nlp.classify.util.ModelTraining.IntermediatePredictionLogger;
import edu.byu.nlp.crowdsourcing.MultiAnnModelBuilders.MultiAnnModelBuilder;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath;
import edu.byu.nlp.crowdsourcing.models.gibbs.BlockCollapsedMultiAnnModelMath.DiagonalizationMethod;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.util.Matrices;

/**
 * @author pfelt
 */
public class MultiAnnDatasetLabeler implements DatasetLabeler{
  private static final Logger logger = LoggerFactory.getLogger(ModelTraining.class);

  private Dataset data;
  private MultiAnnModelBuilder builder;
  private boolean predictSingleLastSample;
  private String trainingOperations;
  private boolean diagonalizationWithFullConfusionMatrix;
  private DiagonalizationMethod diagonalizationMethod;
  private int goldInstancesForDiagonalization;
  private PrintWriter statsOut;
  private String unannotatedDocumentWeight;
  private RandomGenerator rnd;
  private PrintWriter debugOut;
  private MultiAnnModel model;

  private IntermediatePredictionLogger predictionLogger;


  public static enum DocWeightAlgorithm{
    STATIC,
    BINARY_CLASSIFIER;
  }
  
  private static class DocumentWeightCalculator{
    private DocWeightAlgorithm strategy;
    private double staticValue;
    private Dataset data;
    private DatasetBuilder binaryDatasetBuilder;
    public static DocumentWeightCalculator buildCalculator(String strategy, Dataset data){
      DocumentWeightCalculator calc = new DocumentWeightCalculator();
      calc.data=data;
      // parse a number
      try{
        calc.strategy=DocWeightAlgorithm.STATIC;
        calc.staticValue = Double.parseDouble(strategy);
      }
      catch(NumberFormatException e){
        // parse an enum value
        try{
          calc.strategy = DocWeightAlgorithm.valueOf(strategy.toUpperCase());
        }
        catch(IllegalArgumentException e2){
          throw new IllegalArgumentException("Document weights must either be a number, "
              + "or else a valid value of the MultiAnnDtasetLabel.DocWeightAlgorithm enum",e2);
        }
      }
      
      return calc;
    }
    public double weightFor(DatasetInstance instance){
      
      switch(strategy){
      case STATIC:
        return staticValue;
      
      case BINARY_CLASSIFIER:
        throw new UnsupportedOperationException("not implemented");
        // train a binary classifier to tell the difference between 
        // labeled and unlabeled data.
//        nb = new NaiveBayesLearner().learnFrom(binaryData(data.allInstances()));
//        
//        // weight each doc by the probability that it is in the annotated set
//        double prob = Math.exp(nb.given(instance.getData()).logProbabilityOf(1));
//        return prob * instance.getData().getNumActiveFeatures();
        
      default:
        throw new IllegalArgumentException(strategy+" not implemented");
      }
    }
    
//    /**
//     * create a dataset where labeled examples are class=1 and unlabeled examples
//     * are class=0
//     */
//    private edu.byu.nlp.al.classify2.Dataset binaryData(Collection<DatasetInstance> instances) {
//      if (binaryDatasetBuilder == null) {
//        binaryDatasetBuilder = new DatasetBuilder(new LabelChooser<Integer>() {
//          @Override
//          public Integer labelFor(Multimap<Long, TimedAnnotation<Integer>> annotations) {
//            return annotations.size() == 0 ? 0 : 1;
//          }
//        }, 2, data.getNumFeatures());
//      }
//      return binaryDatasetBuilder.buildDataset(instances, null);
//    }
  }

  public MultiAnnDatasetLabeler(MultiAnnModel model, 
    PrintWriter debugOut,
    boolean predictSingleLastSample, 
    DiagonalizationMethod diagonalizationMethod,
    boolean diagonalizationWithFullConfusionMatrix,
    int goldInstancesForDiagonalization, Dataset trainingData,
    RandomGenerator algRnd) {
    this.model = model;
    this.predictSingleLastSample = predictSingleLastSample;
    this.diagonalizationMethod=diagonalizationMethod;
    this.diagonalizationWithFullConfusionMatrix=diagonalizationWithFullConfusionMatrix;
    this.goldInstancesForDiagonalization=goldInstancesForDiagonalization;
    this.statsOut=(statsOut==null)? new PrintWriter(ByteStreams.nullOutputStream()): statsOut;
    this.debugOut=(debugOut==null)? new PrintWriter(ByteStreams.nullOutputStream()): debugOut;
    this.rnd=algRnd;
    
    this.data=trainingData; // should get rid of this after we figure out how to pass instances into the label() method
  }
  /**
   * @param gold for computing confusion matrices for debugging
   */
  public MultiAnnDatasetLabeler(MultiAnnModelBuilder multiannModelBuilder, 
    PrintWriter debugOut,
		boolean predictSingleLastSample, String trainingOperations,
		DiagonalizationMethod diagonalizationMethod,
		boolean diagonalizationWithFullConfusionMatrix,
		int goldInstancesForDiagonalization, Dataset trainingData,
		String unannotatedDocumentWeight,
		IntermediatePredictionLogger predictionLogger,
		RandomGenerator algRnd) {
    // TODO (pfelt): in the future we'll probably want to NOT pass in the builder, but create a new builder 
    // for every label() request. The only problem with that right now is that it's not clear how
    // to build a Dataset from a collection of instances--it requires indexes generated from the original data???
    this.builder = multiannModelBuilder;
    this.predictSingleLastSample = predictSingleLastSample;
    this.trainingOperations = trainingOperations;
    this.diagonalizationMethod=diagonalizationMethod;
    this.diagonalizationWithFullConfusionMatrix=diagonalizationWithFullConfusionMatrix;
    this.goldInstancesForDiagonalization=goldInstancesForDiagonalization;
    this.statsOut=(statsOut==null)? new PrintWriter(ByteStreams.nullOutputStream()): statsOut;
    this.debugOut=(debugOut==null)? new PrintWriter(ByteStreams.nullOutputStream()): debugOut;
    this.unannotatedDocumentWeight=unannotatedDocumentWeight;
    this.predictionLogger=predictionLogger;
    this.rnd=algRnd;
    
    this.data=trainingData; // should get rid of this after we figure out how to pass instances into the label() method
  }


/** {@inheritDoc} */
  @Override
  public Predictions label(
      Dataset trainingData, 
      Dataset heldoutData) {

    // FIXME: figure out how to use the instances passed in here before this is usable by MultiAnnSim
    // for now, the builder has access to the data and knows about new annotation because the training
    // set is mutated
//    MultiAnnModelBuilder builder = MultiAnnModel.newModelBuilderWithUniform(priors, trainingData, rnd); 
    
    
    // Train a new model (if necessary)
    if (this.model==null){
      // calculate a weight for each document
      double[] docWeights = new double[data.getInfo().getNumDocuments()];
      calculateDocumentWeights(unannotatedDocumentWeight, docWeights);
      builder.setDocumentWeights(docWeights);
      // train
      model = builder.build();
      ModelTraining.doOperations(trainingOperations, model, predictionLogger);
    }
    
    // record model
    model.getCurrentState().longDescription(debugOut);
    
    
    // Use it to predict
    return predict(model, data, heldoutData);
  }

  /**
   * @return
   */
  private double[] calculateDocumentWeights(String unannotatedStrategy, double[] docWeights) {

    DocumentWeightCalculator passThroughWeightCalc = DocumentWeightCalculator.buildCalculator("1", data);
    DocumentWeightCalculator unannotatedWeightCalc = DocumentWeightCalculator.buildCalculator(unannotatedStrategy, data);
    
    // assign each document a weight (1 indicates no scaling). Must be that 0<=weight<=1
    int i=0;
    for (DatasetInstance instance: data){
      DocumentWeightCalculator weightCalc = instance.getInfo().getNumAnnotations()>0? passThroughWeightCalc: unannotatedWeightCalc;
      docWeights[i++] = weightCalc.weightFor(instance);
    }
    for (double w: docWeights){
      if (w<0 || w>1){
        throw new IllegalStateException("doc weights must be between 0 and 1 (inclusive). Not "+w);
      }
    }
    return docWeights;
  }

	// TODO: public for sanity test convenience. Should refactor at some point
	public Predictions predict(MultiAnnModel model, Dataset data, Dataset heldoutData) {
		logConfusions("Before Diagonalizing ", model.getCurrentState());
		MultiAnnModelPredictor predictor = new MultiAnnModelPredictor(model, data, predictSingleLastSample,
				diagonalizationMethod, goldInstancesForDiagonalization, diagonalizationWithFullConfusionMatrix, gold);

		// label switching fixed in these params
		MultiAnnState sample = predictor.getFinalPredictiveParameters(); 
		logConfusions("After Diagonalizing ", sample);

		// internally fixes label switching
		return predictor.predict(heldoutData);
	}



  private MultiAnnState goldsample = null; // cache gold labels based on sample identity--a minor optimization
  private int[] gold = null;
  
  private void logConfusions(String prefix, MultiAnnState sample){
    // cache gold labels based on sample identity--a minor optimization
    if (goldsample==null || goldsample!=sample){
      gold = Datasets.concealedLabels(sample.getData(), sample.getInstanceIndices());
    }
  
    statsOut.println("");
    statsOut.println(prefix+" Y Confusion\n"+Matrices.toString(
        BlockCollapsedMultiAnnModelMath.confusionMatrix(null, gold, sample.getY(), sample.getNumLabels(), data),
        100,100,0));
    statsOut.println(prefix+" M Confusion\n"+Matrices.toString(
        BlockCollapsedMultiAnnModelMath.confusionMatrix(null, gold, sample.getM(), sample.getNumLabels(), data),
        100,100,0));
//      statsOut.println("Temp="+temp+" Sample="+s+" Labeled Confusion\n"+Matrices.toString(model.confusionMatrix(true),100,100,0));
//      statsOut.println("Temp="+temp+" Sample="+s+" Unlabeled Confusion\n"+Matrices.toString(model.confusionMatrix(false),100,100,0));
    statsOut.println("");
  }
  
}
