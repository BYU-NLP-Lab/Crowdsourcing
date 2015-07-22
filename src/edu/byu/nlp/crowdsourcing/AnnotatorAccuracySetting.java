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

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;

import edu.byu.nlp.data.annotators.SimulatedAnnotator;
import edu.byu.nlp.data.annotators.SimulatedAnnotators;
import edu.byu.nlp.io.Files2;
import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.Indexer;
import edu.byu.nlp.util.Indexers;
import edu.byu.nlp.util.Matrices;

public enum AnnotatorAccuracySetting {
  
  GOOD (new double[] { 0.99999, 0.99999, 0.99999, 0.99999, 0.99999 }, 1e100), 
  HIGH (new double[] { 0.90, 0.85, 0.80, 0.75, 0.70 }, 1e100), 
  MED (new double[] { 0.70, 0.65, 0.60, 0.55, 0.50 }, 1e100),
  LOW (new double[] { 0.50, 0.40, 0.30, 0.20, 0.10 }, 1e100),
  MANYLOW (new double[] { 0.70, 0.67, 0.64, 0.61, 0.58, 0.55, 0.52, 0.49, 0.46, 0.43, 0.40, 0.37, 0.34, 0.31, 0.28, 0.25, 0.22, 0.19, 0.16, 0.13}, 1),
  CONFLICT (new double[] { 0.50, 0.40, 0.30, 0.20, 0.10 }, 0.1),
  CONFLICT_MILD (new double[] { 0.50, 0.40, 0.30, 0.20, 0.10 }, 1),
  EXPERT (new double[] { 0.90, 0.91, 0.93, 0.95, 0.97 }, 1e100),
  INDEPENDENT (new double[]{-1,-1,-1,-1,-1}, 0.1),
  FILE (null, -1),
  CFBETA (new double[100], 1), // accuracy values will be drawn from Beta(shape1,shape2) parameters MLE-fit from CFGroups data
  ;

  private static Logger logger = LoggerFactory.getLogger(AnnotatorAccuracySetting.class);
  
  // when this is very large, the off-diagonal is uniform
  private final double symmetricDirichletParam;
  private double[] accuracies;
  double[][][] confusionMatrices;
  double [] annotatorRates;
  private AnnotatorAccuracySetting(double[] accuracies, double symmetricDirichletParam){
    this.accuracies=accuracies;
    this.symmetricDirichletParam=symmetricDirichletParam;
  }
  public int getNumAnnotators(){
    Preconditions.checkNotNull(confusionMatrices, "call generateConfusionMatrices() first");
    return confusionMatrices.length;
  }
  public int getNumClasses(){
    Preconditions.checkNotNull(confusionMatrices, "call generateConfusionMatrices() first");
    return confusionMatrices[0].length;
  }
  public double[] getAccuracies(){
    Preconditions.checkNotNull(confusionMatrices, "call generateConfusionMatrices() first");
    // construct accuracies from confusion matrices, if necessary (e.g., if they were read from file)
    if (accuracies==null){
      accuracies = new double[confusionMatrices.length];
      for (int i=0; i<confusionMatrices.length; i++){
        accuracies[i] = Matrices.trace(confusionMatrices[i]) / confusionMatrices[0].length;
      }
    }
    return accuracies;
  }
  public double[][][] getConfusionMatrices(){
    Preconditions.checkNotNull(confusionMatrices, "call generateConfusionMatrices() first");
    return confusionMatrices;
  }
  public double[] getAnnotatorRates(){
    Preconditions.checkNotNull(annotatorRates, "call generateConfusionMatrices() first");
    return annotatorRates;
  }
  public void generateConfusionMatrices(RandomGenerator rnd, boolean varyAnnotatorRates, int numLabels, String filename){
    
    if (confusionMatrices==null){
      switch (this) {
      
      // read annotator confusion matrices from file
      case FILE:
        try {
          List<SimulatedAnnotator> annotators = SimulatedAnnotators.deserialize(Files2.toString(filename, Charsets.UTF_8));
          annotatorRates = SimulatedAnnotators.annotationRatesOf(annotators);
          confusionMatrices = SimulatedAnnotators.confusionsOf(annotators);
        } catch (IOException e) {
          throw new IllegalArgumentException("could not parse annotator file: "+filename);
        }
        // ensure that the simulated annotators we are reading in have the same number of classes as the dataset we are labeling
        Preconditions.checkState(confusionMatrices[0].length==numLabels,"mismatch between the number of label classes "
            + "in the simulated annotator file "+confusionMatrices[0].length+" and the number in the dataset "+numLabels);
        break;
        
      // annotators are drawn from independent dirichlets
      case INDEPENDENT:
        annotatorRates = uniformAnnotatorRates(accuracies.length);
        confusionMatrices = new double[accuracies.length][numLabels][numLabels];
        // a matrix where all rows are sampled from a dirichlet
        for (int a = 0; a < accuracies.length; a++) {
          for (int i = 0; i < numLabels; i++) {
            confusionMatrices[a][i] = DirichletDistribution.sampleSymmetric(symmetricDirichletParam, numLabels, rnd);
          }
        }
        break;
        
      // annotator accuracies drawn from Beta(shape1,shape2) parameters MLE-fit to CFGroups data
      case CFBETA:
        annotatorRates = uniformAnnotatorRates(accuracies.length);
        BetaDistribution accGen = new BetaDistribution(rnd, 3.6, 5.1);
        accuracies = accGen.sample(accuracies.length); // sample accuracies
        logger.info("sampled "+accuracies.length+" simulated annotator accuracies from a Beta(3.6,5.1). min="+DoubleArrays.min(accuracies)+" max="+DoubleArrays.max(accuracies)+" mean="+DoubleArrays.mean(accuracies));
        confusionMatrices = confusionMatricesFromAccuracyAndDirichlet(accuracies, numLabels, symmetricDirichletParam, rnd);
        break;
        
      // predetermined annotator accuracies 
      default:
        annotatorRates = uniformAnnotatorRates(accuracies.length);
        confusionMatrices = confusionMatricesFromAccuracyAndDirichlet(accuracies, numLabels, symmetricDirichletParam, rnd);
        break;
      }
      
      if (varyAnnotatorRates && this!=FILE){
        throw new IllegalStateException("Varying annotator rates are ONLY implemented for FILE annotators. Not "+this);
      }
      
      // force uniform annotator rates
      if (!varyAnnotatorRates){
          annotatorRates = uniformAnnotatorRates(confusionMatrices.length);
      }
    }
  }
  /**
   * A trivial identity indexer where id=index
   */
  public Indexer<String> getAnnotatorIdIndexer(){
    return Indexers.indexerOfStrings(getNumAnnotators());
  }
  
  private static double[] uniformAnnotatorRates(int numAnnotators){
    return DoubleArrays.of(1.0/numAnnotators, numAnnotators);
  }
  
  private static double[][][] confusionMatricesFromAccuracyAndDirichlet(double[] accuracies, int numLabels, double symmetricDirichletParam, RandomGenerator rnd){
    double[][][] confusionMatrices = new double[accuracies.length][numLabels][numLabels];
    // a matrix where non-diag entries
    // are sampled and scaled to make each row sum to 1
    for (int a = 0; a < accuracies.length; a++) {
      double rowDiag = accuracies[a];
      for (int i = 0; i < numLabels; i++) {
        // off-diag elements are Dirichlet. Note that when 
        // symmetricDirichletParam is very large, off-diag elements are uniform
        double[] offDiag = DirichletDistribution.sampleSymmetric(symmetricDirichletParam, numLabels - 1, rnd);
        // scale offDiag so row sums to 1
        DoubleArrays.multiplyToSelf(offDiag, 1.0 - rowDiag); 
        Iterator<Double> offDiagItr = DoubleArrays.iterator(offDiag);
        for (int j = 0; j < numLabels; j++) {
          confusionMatrices[a][i][j] = i == j ? rowDiag : offDiagItr.next();
        }
      }
    }
    return confusionMatrices;
  }
}
